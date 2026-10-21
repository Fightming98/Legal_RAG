"""知识库索引核心 — 确定性 ID、内容指纹与幂等写入

`scripts/import_data.py`（CLI 批量导入）与 `app/services/kb_service.py`
（前端上传 / 重建索引）共用本模块，保证同一条数据无论从哪条路径写入，
都会得到相同的分块 ID 与元数据，从而可以互相去重、断点续传。

幂等原理:
1. 分块 ID = uuid5(命名空间, 导入版本 + 文档类型 + 来源相对路径 + 位置)，
   同一文件在同一位置的分块 ID 固定不变；
2. 写入前按 ID 批量查询已有分块，并校验导入版本、Embedding 模型、
   上下文分块开关与内容哈希；全部一致才跳过，否则用相同 ID 覆盖更新；
3. 单个文件写入完成后，按 source_file 清理该文件已不存在的旧分块，
   避免文件内容变短后残留过期向量。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from langchain_core.documents import Document

from app.config import settings
from app.core.logging import get_logger
from app.core.model_config import resolve_embedding_model
from app.core.vectorstore import close_client, reset_store_cache
from app.utils.legal_chunker import split_legal_document

logger = get_logger(__name__)

# 导入协议版本：分块策略、元数据结构或清洗逻辑变化时递增。
# 递增后旧分块的元数据校验不通过，会在下次导入时按相同 ID 重新 Embedding（覆盖更新）。
IMPORT_VERSION = "v1"

# 确定性 ID 的命名空间。固定不变，保证同一分块在不同机器/不同运行中得到相同 ID。
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "legal-rag/import")

# Chroma 单次 get 的 ID 数量，分批查询避免请求体过大。
_GET_BATCH_SIZE = 1000

# 案例文本最小长度，过滤空/过短的案情，避免无效分块。
_MIN_CASE_LENGTH = 50

# 判断已有分块是否仍与当前导入配置一致时，需要逐项比对的元数据键。
_FINGERPRINT_KEYS = ("import_version", "embedding_model", "contextual_chunking")

# 进度输出回调：CLI 传 print，服务层传 logger.info
Progress = Callable[[str], None]


class LegacyIndexError(RuntimeError):
    """集合中存在旧版随机 ID 数据，无法与确定性 ID 去重，需要先重建索引"""


def _emit(log: Progress | None, message: str) -> None:
    """统一进度输出；未指定回调时写入应用日志"""
    if log is not None:
        log(message)
    else:
        logger.info(message)


# ===================== 文件读取与遍历 =====================

def read_file(filepath: str) -> str:
    """尝试多种编码读取文件"""
    # 依次尝试常见中文编码，直到某一种能成功解码
    for enc in ("utf-8", "gbk", "gb2312", "utf-16"):
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue  # 当前编码解码失败，换下一种继续
    raise ValueError(f"无法解码: {filepath}")  # 所有编码都失败则报错


def collect_files(dir_path: str, extensions: set[str]) -> list[str]:
    """递归收集目录下指定后缀的文件"""
    results: list[str] = []
    # os.walk 递归遍历 dir_path 下所有子目录
    for root, _dirs, files in os.walk(dir_path):
        for fname in sorted(files):  # 排序保证导入顺序稳定
            if fname.startswith("."):  # 跳过隐藏文件
                continue
            ext = os.path.splitext(fname)[1].lower()  # 取小写扩展名
            if ext in extensions:  # 扩展名匹配才收集
                results.append(os.path.join(root, fname))
    return results


def rel_path(filepath: str, base_dir: str) -> str:
    """返回相对 base_dir 的路径，并统一使用 / 分隔符（跨平台稳定）"""
    return os.path.relpath(filepath, base_dir).replace(os.sep, "/")


# ===================== 确定性 ID 与元数据 =====================

def hash_text(text: str) -> str:
    """分块内容哈希，用于检测同一位置的内容是否发生变化"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_metadata() -> dict:
    """当前导入配置的指纹，用于判断已有分块是否仍然有效

    使用 resolve_embedding_model() 而不是 settings.EMBEDDING_MODEL：
    前端允许用户通过 X-Embedding-Model 选择模型，这里必须记录本次实际
    使用的模型，否则换模型后会误判旧向量仍然有效。
    """
    return {
        "import_version": IMPORT_VERSION,
        "embedding_model": resolve_embedding_model(),
        "contextual_chunking": settings.CONTEXTUAL_CHUNKING,
    }


def law_chunk_id(rel_file: str, chunk_index: int) -> str:
    """法律分块的确定性 ID：来源相对路径 + 文件内块序号"""
    return str(
        uuid.uuid5(_ID_NAMESPACE, f"{IMPORT_VERSION}|law|{rel_file}|{chunk_index}")
    )


def case_chunk_id(rel_file: str, record_no: int, chunk_index: int) -> str:
    """案例分块的确定性 ID：来源相对路径 + 记录序号 + 案例内块序号

    record_no 的取值：
    - JSONL：行号（从 1 开始）
    - 整文件 JSON 对象：0
    - 整文件 JSON 数组：元素下标 + 1
    - .txt/.md 案例：0
    """
    return str(
        uuid.uuid5(
            _ID_NAMESPACE,
            f"{IMPORT_VERSION}|case|{rel_file}|{record_no}|{chunk_index}",
        )
    )


def _stamp(chunk: Document, base_metadata: dict, **metadata) -> None:
    """写入通用元数据（导入版本、Embedding 模型、内容哈希等）"""
    chunk.metadata.update(base_metadata)
    chunk.metadata.update(metadata)
    chunk.metadata["content_hash"] = hash_text(chunk.page_content)


def build_law_chunks(
    text: str,
    rel_file: str,
    filename: str,
    base_metadata: dict | None = None,
) -> list[Document]:
    """把一部法律文本切分成带确定性 ID 的分块"""
    base = base_metadata or import_metadata()
    chunks: list[Document] = []
    for index, chunk in enumerate(split_legal_document(text, doc_type="law", filename=filename)):
        if not chunk.page_content.strip():  # 空块不写入，避免浪费 Embedding
            continue
        chunk.id = law_chunk_id(rel_file, index)
        # source_file 使用相对路径，避免不同子目录的同名文件互相覆盖
        _stamp(chunk, base, source_file=rel_file)
        chunks.append(chunk)
    return chunks


def build_case_chunks(
    text: str,
    rel_file: str,
    record_no: int,
    case_index: int,
    filename: str,
    extra_metadata: dict | None = None,
    base_metadata: dict | None = None,
) -> list[Document]:
    """把一条案例文本切分成带确定性 ID 的分块"""
    base = base_metadata or import_metadata()
    chunks: list[Document] = []
    for index, chunk in enumerate(split_legal_document(text, doc_type="case", filename=filename)):
        if not chunk.page_content.strip():
            continue
        chunk.id = case_chunk_id(rel_file, record_no, index)
        if extra_metadata:
            chunk.metadata.update(extra_metadata)  # 附加罪名、法条、刑期等
        _stamp(
            chunk,
            base,
            source_file=rel_file,
            case_line=record_no,
            case_index=case_index,
            case_title=f"{filename}:案例{case_index}",
        )
        chunks.append(chunk)
    return chunks


# ===================== 案例记录解析 =====================

def _extract_case_text(obj: dict) -> tuple[str, dict]:
    """从 JSON 对象中提取案例文本和额外元数据

    支持多种格式：
    - CAIL2019-SCM: {"A": "...", "B": "...", "C": "...", "label": "..."}
    - CAIL2018:     {"fact": "...", "meta": {"accusation": [...], ...}}
    - 通用JSONL:    {"fact": "...", ...} 或 {"text": "...", ...} 或 {"content": "...", ...}
    """
    extra_meta: dict = {}

    # CAIL2018 格式
    if "fact" in obj:
        text = obj["fact"]  # 案情描述
        meta = obj.get("meta", {})
        if meta.get("accusation"):
            extra_meta["accusation"] = "；".join(meta["accusation"])  # 罪名列表用中文分号拼接
        if meta.get("relevant_articles"):
            extra_meta["relevant_articles"] = "；".join(
                str(a) for a in meta["relevant_articles"]  # 法条号转字符串后拼接
            )
        if meta.get("criminals"):
            extra_meta["criminals"] = "；".join(meta["criminals"])
        # 处理来自 prepare_datasets 转换后的扁平格式（罪名直接放在顶层字段）
        if "accusation" in obj and not extra_meta.get("accusation"):
            extra_meta["accusation"] = obj["accusation"]
        if "sentence" in obj:
            extra_meta["sentence"] = obj["sentence"]  # 刑期描述
        return text, extra_meta

    # CAIL2019-SCM 格式：A 字段即案情文本
    if "A" in obj:
        return obj["A"], extra_meta

    # 通用格式：依次尝试常见文本字段名
    for key in ("text", "content", "body"):
        if key in obj and obj[key]:
            return obj[key], extra_meta

    return "", extra_meta


def iter_case_records(filepath: str) -> Iterator[tuple[int, str, dict]]:
    """遍历案例文件，产出 (记录序号, 文本, 附加元数据)

    优先按 JSONL 逐行解析；若一行都没解析成功，再尝试把整个文件当作
    标准 JSON 读取（兼容格式化后的单个 JSON 对象或 JSON 数组）。
    """
    parsed_any = False
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                obj = json.loads(line)  # 每行解析为一个 JSON 对象
            except json.JSONDecodeError:
                continue  # 跳过无法解析的损坏行
            if not isinstance(obj, dict):
                continue
            parsed_any = True
            text, extra_meta = _extract_case_text(obj)
            if not text:
                continue
            yield line_no, text, extra_meta

    if parsed_any:
        return

    # 整文件 JSON 兜底（例如 pretty-print 的对象/数组，逐行解析必然失败）
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload if isinstance(payload, list) else [payload]
    start_index = 1 if isinstance(payload, list) else 0
    for offset, obj in enumerate(records):
        if not isinstance(obj, dict):
            continue
        text, extra_meta = _extract_case_text(obj)
        if not text:
            continue
        yield start_index + offset, text, extra_meta


# ===================== 查重与写入 =====================

@dataclass
class FileIndexResult:
    """单个文件的索引结果"""

    total_chunks: int = 0          # 文件中解析出的分块总数
    added_chunks: int = 0          # 本次实际写入（新增或覆盖）的分块数
    skipped_chunks: int = 0        # 已存在且配置一致、无需重写的分块数
    removed_chunks: int = 0        # 清理掉的过期分块数
    new_cases: int = 0             # 本次写入的案例条数
    scanned_cases: int = 0         # 扫描到的有效案例条数
    completed: bool = False        # 文件是否完整扫描（未因配额中断）


def _is_ready(existing: dict | None, chunk: Document) -> bool:
    """判断库中已有分块是否可以直接复用"""
    existing = existing or {}
    new = chunk.metadata or {}

    for key in _FINGERPRINT_KEYS:
        if existing.get(key) != new.get(key):
            return False

    old_hash = existing.get("content_hash")
    new_hash = new.get("content_hash")
    # 兼容旧版确定性数据：当时还没有 content_hash，不能因此判定为失效，
    # 否则升级后会把整库重新 Embedding 一遍。
    if old_hash and new_hash and old_hash != new_hash:
        return False
    return True


def existing_ready_ids(store, chunks: list[Document]) -> set[str]:
    """返回库中已存在且与当前导入配置一致的分块 ID"""
    if not chunks:
        return set()

    by_id = {chunk.id: chunk for chunk in chunks}
    ids = list(by_id)
    ready: set[str] = set()

    for i in range(0, len(ids), _GET_BATCH_SIZE):
        batch = ids[i : i + _GET_BATCH_SIZE]
        try:
            # 只取 ID + metadatas，不取 documents/embeddings，避免大字段传输
            result = store._collection.get(ids=batch, include=["metadatas"])
        except Exception:
            # 兼容不支持 include 参数的旧版 chromadb
            result = store._collection.get(ids=batch)

        for chunk_id, metadata in zip(
            result.get("ids") or [], result.get("metadatas") or []
        ):
            chunk = by_id.get(chunk_id)
            if chunk is not None and _is_ready(metadata, chunk):
                ready.add(chunk_id)
    return ready


def write_chunks(store, chunks: list[Document], batch_size: int = 5000) -> tuple[int, int, set[str]]:
    """分批写入，跳过已存在且未变化的分块

    Returns:
        (写入块数, 跳过块数, 本次写入的 ID 集合)
    """
    if not chunks:
        return 0, 0, set()

    added = 0
    skipped = 0
    added_ids: set[str] = set()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if any(not chunk.id for chunk in batch):
            raise ValueError("分块缺少确定性 ID，拒绝写入")

        ready = existing_ready_ids(store, batch)
        new_docs = [chunk for chunk in batch if chunk.id not in ready]
        skipped += len(batch) - len(new_docs)

        if new_docs:
            # 显式传 ids，避免 LangChain 自动生成随机 UUID。
            # Chroma 的 add_texts 内部使用 upsert：同 ID 会覆盖旧内容。
            store.add_texts(
                texts=[chunk.page_content for chunk in new_docs],
                metadatas=[chunk.metadata for chunk in new_docs],
                ids=[chunk.id for chunk in new_docs],
            )
            added += len(new_docs)
            added_ids.update(chunk.id for chunk in new_docs)

    return added, skipped, added_ids


def _source_ids(store, source_file: str) -> list[str]:
    """查询某个来源文件在库中的全部分块 ID"""
    try:
        result = store._collection.get(where={"source_file": source_file}, include=[])
    except Exception:
        # 兼容不支持 include=[] 的旧版 chromadb
        result = store._collection.get(where={"source_file": source_file})
    return list(result.get("ids") or [])


def prune_source(store, source_file: str, keep_ids: set[str]) -> int:
    """删除某个来源文件下已不存在的旧分块，返回清理数量

    清理属于收尾工作：即使失败也不应让已经成功的索引写入报错，
    因此这里只记录日志，下次导入会再次尝试。
    """
    try:
        existing = _source_ids(store, source_file)
    except Exception:
        logger.warning("查询 %s 的旧分块失败，跳过清理", source_file, exc_info=True)
        return 0

    stale = [chunk_id for chunk_id in existing if chunk_id not in keep_ids]
    if not stale:
        return 0
    try:
        store._collection.delete(ids=stale)
    except Exception:
        logger.warning("清理 %s 的旧分块失败", source_file, exc_info=True)
        return 0
    return len(stale)


def delete_source(store, source_file: str) -> int:
    """删除某个来源文件的全部向量分块，返回删除数量"""
    ids = _source_ids(store, source_file)
    if ids:
        store._collection.delete(ids=ids)
    return len(ids)


# ===================== 单文件索引 =====================

def index_law_file(store, filepath: str, base_dir: str, *, log: Progress | None = None) -> FileIndexResult:
    """索引一个法律文件（.txt/.md）"""
    rel_file = rel_path(filepath, base_dir)
    filename = os.path.basename(filepath)
    text = read_file(filepath)
    chunks = build_law_chunks(text, rel_file, filename)

    added, skipped, _ = write_chunks(store, chunks)
    removed = prune_source(store, rel_file, {chunk.id for chunk in chunks})
    _emit(
        log,
        f"  ✓ {rel_file}: 新增 {added} 块，跳过 {skipped} 块"
        + (f"，清理 {removed} 块旧数据" if removed else ""),
    )
    return FileIndexResult(
        total_chunks=len(chunks),
        added_chunks=added,
        skipped_chunks=skipped,
        removed_chunks=removed,
        completed=True,
    )


def index_case_file(store, filepath: str, base_dir: str, *, log: Progress | None = None) -> FileIndexResult:
    """索引一个整文件案例（.txt/.md），整个文件视为一条案例"""
    rel_file = rel_path(filepath, base_dir)
    filename = os.path.basename(filepath)
    text = read_file(filepath)
    chunks = build_case_chunks(text, rel_file, 0, 1, filename)

    added, skipped, added_ids = write_chunks(store, chunks)
    removed = prune_source(store, rel_file, {chunk.id for chunk in chunks})
    _emit(
        log,
        f"  ✓ {rel_file}: 新增 {added} 块，跳过 {skipped} 块"
        + (f"，清理 {removed} 块旧数据" if removed else ""),
    )
    return FileIndexResult(
        total_chunks=len(chunks),
        added_chunks=added,
        skipped_chunks=skipped,
        removed_chunks=removed,
        new_cases=1 if added_ids else 0,
        scanned_cases=1,
        completed=True,
    )


def index_case_jsonl_file(
    store,
    filepath: str,
    base_dir: str,
    *,
    max_new_cases: int | None = None,
    log: Progress | None = None,
) -> FileIndexResult:
    """索引一个 JSONL/JSON 案例文件，支持按新增案例数限制本次工作量

    已存在案例不占用 max_new_cases 额度，因此中断后重跑可以直接续传。

    注意：本函数不捕获读取/写入异常；调用方负责按文件维度记录日志并继续。
    """
    rel_file = rel_path(filepath, base_dir)
    filename = os.path.basename(filepath)
    base_metadata = import_metadata()

    result = FileIndexResult()
    batch_chunks: list[Document] = []
    case_ranges: list[tuple[int, int]] = []  # 每条案例在 batch_chunks 中的 [start, end)
    keep_ids: set[str] = set()               # 文件完整扫描后用于清理过期分块
    case_index = 0

    def flush() -> None:
        """写入当前缓冲的分块，并统计其中新增了多少条案例"""
        nonlocal batch_chunks, case_ranges
        if not batch_chunks:
            return

        added, skipped, added_ids = write_chunks(store, batch_chunks, batch_size=200)
        result.total_chunks += len(batch_chunks)
        result.added_chunks += added
        result.skipped_chunks += skipped
        # 一条案例只要有一个分块被写入，就计为本次新增案例
        result.new_cases += sum(
            1
            for start, end in case_ranges
            if any(chunk.id in added_ids for chunk in batch_chunks[start:end])
        )
        batch_chunks = []
        case_ranges = []

    try:
        for record_no, text, extra_meta in iter_case_records(filepath):
            if max_new_cases is not None and result.new_cases >= max_new_cases:
                break  # 达到本次新增上限，文件未扫描完，不执行清理
            if not text or len(text) < _MIN_CASE_LENGTH:
                continue

            case_index += 1
            result.scanned_cases += 1
            chunks = build_case_chunks(
                text, rel_file, record_no, case_index, filename, extra_meta, base_metadata
            )
            if not chunks:
                continue

            keep_ids.update(chunk.id for chunk in chunks)
            start = len(batch_chunks)
            batch_chunks.extend(chunks)
            case_ranges.append((start, len(batch_chunks)))

            # 及时落盘：达到 200 块、50 条案例，或剩余新增额度不足时
            remaining = None if max_new_cases is None else max_new_cases - result.new_cases
            flush_due = (
                len(batch_chunks) >= 200
                or len(case_ranges) >= 50
                or (remaining is not None and len(case_ranges) >= remaining)
            )
            if flush_due:
                flush()

        flush()  # 文件结尾写入剩余分块
        result.completed = True
    except Exception:
        # 已处理的分块尽量落盘，避免中断后重做；原始异常继续抛给调用方
        try:
            flush()
        except Exception:
            logger.warning("案例文件 %s 收尾写入失败", rel_file, exc_info=True)
        raise

    if result.completed:
        result.removed_chunks = prune_source(store, rel_file, keep_ids)
    _emit(
        log,
        f"  ✓ {rel_file}: 扫描 {result.scanned_cases} 条案例，"
        f"新增 {result.new_cases} 条 / {result.added_chunks} 块，"
        f"跳过 {result.skipped_chunks} 块"
        + (f"，清理 {result.removed_chunks} 块旧数据" if result.removed_chunks else "")
        + ("" if result.completed else "（达到本次上限，文件未扫描完）"),
    )
    return result


# ===================== 目录级导入 =====================

def import_laws(store, dir_path: str, *, log: Progress | None = None) -> int:
    """导入法律条文 (.txt/.md 文件)，返回本次写入的分块数"""
    files = collect_files(dir_path, {".txt", ".md"})
    if not files:
        _emit(log, "  目录为空或不存在")
        return 0

    total_added = 0
    total_skipped = 0
    total_removed = 0

    for fpath in files:
        rel_file = rel_path(fpath, dir_path)
        try:
            result = index_law_file(store, fpath, dir_path, log=log)
        except Exception as exc:
            _emit(log, f"  ✗ {rel_file}: {exc}")  # 单个文件失败不中断整体导入
            continue
        total_added += result.added_chunks
        total_skipped += result.skipped_chunks
        total_removed += result.removed_chunks

    _emit(
        log,
        f"  法律条文合计: 新增 {total_added} 块，跳过 {total_skipped} 块"
        + (f"，清理 {total_removed} 块旧数据" if total_removed else ""),
    )
    return total_added


def import_cases_jsonl(store, dir_path: str, max_cases: int | None = 5000, *, log: Progress | None = None) -> int:
    """导入 JSONL/JSON 格式的案例文件，返回本次写入的分块数

    自动识别多种格式（CAIL2019-SCM, CAIL2018, 通用JSONL）。

    Args:
        max_cases: 本次运行最多新增的案例数（已存在的案例不占用额度）；
                   None 表示不限制。该上限用于把长时间导入拆成多次执行。
    """
    json_files = collect_files(dir_path, {".json", ".jsonl"})
    if not json_files:
        _emit(log, "  无 JSON 文件")
        return 0

    total_added = 0
    new_cases = 0
    stopped = False

    for fpath in json_files:
        if max_cases is not None and new_cases >= max_cases:
            stopped = True
            break

        rel_file = rel_path(fpath, dir_path)
        _emit(log, f"  处理 {rel_file}...")
        remaining = None if max_cases is None else max_cases - new_cases
        try:
            result = index_case_jsonl_file(
                store, fpath, dir_path, max_new_cases=remaining, log=log
            )
        except Exception as exc:
            _emit(log, f"  ✗ {rel_file}: {exc}")  # 单个文件失败不中断整体导入
            continue

        total_added += result.added_chunks
        new_cases += result.new_cases

        if not result.completed:
            stopped = True
            break

    if stopped and max_cases is not None:
        _emit(
            log,
            f"  已达本次新增上限 {max_cases} 条（已存在案例不占用额度），停止导入",
        )
    return total_added


def import_case_files(store, dir_path: str, *, log: Progress | None = None) -> int:
    """导入整文件案例 (.txt/.md)，返回本次写入的分块数"""
    files = collect_files(dir_path, {".txt", ".md"})
    if not files:
        _emit(log, "  无 TXT/MD 案例文件")
        return 0

    total_added = 0
    total_skipped = 0
    total_removed = 0

    for fpath in files:
        rel_file = rel_path(fpath, dir_path)
        try:
            result = index_case_file(store, fpath, dir_path, log=log)
        except Exception as exc:
            _emit(log, f"  ✗ {rel_file}: {exc}")
            continue
        total_added += result.added_chunks
        total_skipped += result.skipped_chunks
        total_removed += result.removed_chunks

    _emit(
        log,
        f"  整文件案例合计: 新增 {total_added} 块，跳过 {total_skipped} 块"
        + (f"，清理 {total_removed} 块旧数据" if total_removed else ""),
    )
    return total_added


# ===================== 集合维护 =====================

def drop_collections(*, log: Progress | None = None) -> bool:
    """删除 laws/cases 集合（全量重建使用），返回是否全部删除成功"""
    import chromadb

    close_client()  # 先释放本进程缓存的客户端，避免删除时被占用
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    ok = True
    for name in (settings.LAWS_COLLECTION, settings.CASES_COLLECTION):
        try:
            client.delete_collection(name)
            _emit(log, f"  ✓ 已删除集合: {name}")
        except chromadb.errors.NotFoundError:
            _emit(log, f"  - 集合不存在，跳过: {name}")
        except Exception as exc:
            ok = False
            _emit(log, f"  ✗ 删除集合 {name} 失败: {exc}")
    try:
        client.close()
    except Exception:
        logger.debug("关闭 Chroma 客户端失败（忽略）", exc_info=True)
    reset_store_cache()
    return ok


def has_legacy_data(store, label: str, *, log: Progress | None = None) -> bool:
    """检测集合中是否存在旧版本（无 import_version）的数据

    旧版使用随机 UUID，无法与确定性 ID 去重，必须先全量重建。
    这里抽样检查，避免每次上传都扫描整库。
    """
    try:
        result = store._collection.get(limit=100, include=["metadatas"])
    except Exception:
        return False
    metadatas = result.get("metadatas") or []
    if metadatas and any("import_version" not in (metadata or {}) for metadata in metadatas):
        _emit(
            log,
            f"  ⚠ {label} 集合中存在旧版数据（随机 ID），旧数据无法与新机制去重。",
        )
        return True
    return False
