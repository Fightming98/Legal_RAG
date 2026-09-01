"""数据导入脚本

用法:
    cd backend
    python -m scripts.import_data

支持:
- data/laws/ 下的 .txt 文件（递归子目录）
- data/cases/ 下的 .json (JSONL) 文件（递归子目录）
"""

import os
import sys
import json

# 将 backend 目录加入 Python path，保证 `python -m scripts.import_data` 能导入 app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.core.vectorstore import get_vectorstore, reset_store_cache
from app.utils.legal_chunker import split_legal_document


def read_file(filepath: str) -> str:
    """尝试多种编码读取文件"""
    # 依次尝试常见中文编码，直到某一种能成功解码
    for enc in ["utf-8", "gbk", "gb2312"]:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue  # 当前编码解码失败，换下一种继续
    raise ValueError(f"无法解码: {filepath}")  # 所有编码都失败则报错


def collect_files(dir_path: str, extensions: set[str]) -> list[str]:
    """递归收集目录下指定后缀的文件"""
    results = []
    # os.walk 递归遍历 dir_path 下所有子目录
    for root, _dirs, files in os.walk(dir_path):
        for fname in sorted(files):  # 排序保证导入顺序稳定
            if fname.startswith("."):  # 跳过隐藏文件
                continue
            ext = os.path.splitext(fname)[1].lower()  # 取小写扩展名
            if ext in extensions:  # 扩展名匹配才收集
                results.append(os.path.join(root, fname))
    return results


def _batch_add_documents(store, chunks, batch_size: int = 5000):
    """分批添加文档，避免超出 ChromaDB 单批限制"""
    # 按固定步长切片分批写入，防止一次写入过多导致内存/批次超限
    for i in range(0, len(chunks), batch_size):
        store.add_documents(chunks[i:i + batch_size])


def import_laws(store, dir_path: str) -> int:
    """导入法律条文 (.txt 文件)"""
    files = collect_files(dir_path, {".txt", ".md"})
    if not files:
        print("  目录为空或不存在")
        return 0

    total_chunks = 0
    for fpath in files:
        fname = os.path.basename(fpath)  # 文件名将写入分块元数据的 source_file
        try:
            text = read_file(fpath)  # 读取法律全文
            chunks = split_legal_document(text, doc_type="law", filename=fname)  # 按法律条文结构分块
            if chunks:
                _batch_add_documents(store, chunks)  # 分批写入向量库
                total_chunks += len(chunks)
                print(f"  ✓ {fname}: {len(chunks)} 个分块")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")  # 单个文件失败不中断整体导入
    return total_chunks


def _extract_case_text(obj: dict) -> tuple[str, dict]:
    """从 JSON 对象中提取案例文本和额外元数据

    支持多种格式：
    - CAIL2019-SCM: {"A": "...", "B": "...", "C": "...", "label": "..."}
    - CAIL2018:     {"fact": "...", "meta": {"accusation": [...], ...}}
    - 通用JSONL:    {"fact": "...", ...} 或 {"text": "...", ...} 或 {"content": "...", ...}
    """
    extra_meta = {}

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


def import_cases_jsonl(store, dir_path: str, max_cases: int = 5000) -> int:
    """导入 JSONL 格式的案例文件

    自动识别多种格式（CAIL2019-SCM, CAIL2018, 通用JSONL）。
    """
    json_files = collect_files(dir_path, {".json", ".jsonl"})
    if not json_files: 
        print("  无 JSON 文件")
        return 0

    total_chunks = 0
    case_count = 0  # 已导入案例总数（跨文件累计，用于上限判断）

    for fpath in json_files:
        fname = os.path.basename(fpath)
        print(f"  处理 {fname}...")

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                batch_chunks = []  # 先攒够一批再写入，减少向量库 IO 次数
                for line_no, line in enumerate(f, 1):
                    if case_count >= max_cases:  # 达到全局上限立即停止
                        break
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    try:
                        obj = json.loads(line)  # 每行解析为一个 JSON 对象
                    except json.JSONDecodeError:
                        continue  # 跳过无法解析的损坏行

                    text, extra_meta = _extract_case_text(obj)
                    if not text or len(text) < 50:  # 过滤空/过短的案情，避免无效分块
                        continue

                    case_label = f"案例{case_count + 1}"  # 序号标签拼入来源文件名
                    chunks = split_legal_document(
                        text, doc_type="case", filename=f"{fname}:{case_label}"
                    )
                    # 附加额外元数据（罪名、法条、刑期等）
                    for chunk in chunks:
                        chunk.metadata.update(extra_meta)

                    batch_chunks.extend(chunks)
                    case_count += 1

                    # 批量写入
                    if len(batch_chunks) >= 200:  # 攒满 200 个分块即写入向量库
                        store.add_documents(batch_chunks)
                        total_chunks += len(batch_chunks)
                        print(f"    已导入 {case_count} 条案例 ({total_chunks} 个分块)")
                        batch_chunks = []

                # 写入剩余
                if batch_chunks:  # 文件结尾不足 200 块的剩余分块
                    store.add_documents(batch_chunks)
                    total_chunks += len(batch_chunks)

                print(f"  ✓ {fname}: {case_count} 条案例, {total_chunks} 个分块")

        except Exception as e:
            print(f"  ✗ {fname}: {e}")  # 单个文件失败不中断整体导入

        if case_count >= max_cases:  # 已达上限则不再处理后续文件
            print(f"  已达上限 {max_cases} 条，停止导入")
            break

    return total_chunks


def main():
    print("=" * 55)
    print("法律 RAG 系统 — 数据导入")
    print("=" * 55)

    reset_store_cache()  # 清空向量库缓存，确保从磁盘重新打开最新数据

    # --- 法律条文 ---
    print(f"\n[1/2] 导入法律条文 ({settings.LAWS_DIR})")
    laws_store = get_vectorstore(settings.LAWS_COLLECTION)  # 获取/创建 laws 集合
    laws_count = import_laws(laws_store, settings.LAWS_DIR)

    # --- 案例 ---
    print(f"\n[2/2] 导入案例 ({settings.CASES_DIR})")
    cases_store = get_vectorstore(settings.CASES_COLLECTION)  # 获取/创建 cases 集合
    cases_count = import_cases_jsonl(cases_store, settings.CASES_DIR, max_cases=5000)

    print(f"\n{'=' * 55}")
    print(f"导入完成: 法律条文 {laws_count} 块, 案例 {cases_count} 块")
    print(f"向量数据库: {settings.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
