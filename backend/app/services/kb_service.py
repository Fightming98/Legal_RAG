"""知识库管理服务"""

import os

from app.core.vectorstore import get_vectorstore
from app.services.kb_importer import (
    LegacyIndexError,
    collect_files,
    delete_source,
    drop_collections,
    has_legacy_data,
    import_case_files,
    import_cases_jsonl,
    import_laws,
    index_case_file,
    index_case_jsonl_file,
    index_law_file,
)
from app.models.schemas import KnowledgeFileInfo, KnowledgeStats
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ===================== 路径与文件工具 =====================

def _detect_doc_type(filepath: str) -> str:
    """根据路径判断文档类型"""
    path_lower = filepath.lower()
    if "case" in path_lower or "案例" in path_lower:
        return "case"
    return "law"


def _safe_filename(filename: str) -> str:
    """只保留文件名，阻止 ../ 或绝对路径写入数据目录之外"""
    name = os.path.basename(filename.replace("\\", "/")).strip()
    if not name or name in (".", ".."):
        raise ValueError("文件名无效")
    return name


def _safe_join(base_dir: str, filename: str) -> str | None:
    """把 filename 拼到 base_dir 下，并确保结果仍在 base_dir 内"""
    base = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base, filename))
    try:
        if os.path.commonpath([base, target]) != base:
            return None
    except ValueError:
        return None
    return target


def _count_files(dir_path: str) -> int:
    """递归统计目录下的文件数"""
    count = 0
    if not os.path.isdir(dir_path):
        return 0
    for root, _dirs, files in os.walk(dir_path):
        count += sum(1 for f in files if not f.startswith("."))
    return count


# ===================== 知识库统计 =====================

def get_kb_stats() -> KnowledgeStats:
    """获取知识库统计信息"""
    stats = KnowledgeStats()
    for name in [settings.LAWS_COLLECTION, settings.CASES_COLLECTION]:
        try:
            store = get_vectorstore(name)
            count = store._collection.count()
            if name == settings.LAWS_COLLECTION:
                stats.laws_chunks = count
            else:
                stats.cases_chunks = count
            stats.total_chunks += count
            stats.collections.append(name)
        except Exception:
            logger.warning("统计集合 %s 失败，已跳过", name, exc_info=True)
            continue

    stats.total_files = _count_files(settings.LAWS_DIR) + _count_files(settings.CASES_DIR)
    return stats


# ===================== 上传文档 =====================

async def upload_document(
    filename: str,
    content: bytes,
    doc_type: str | None = None,
) -> KnowledgeFileInfo:
    """上传并索引一个文档

    与 import_data 共用确定性 ID：同一个文件重复上传只会在首次写入，
    文件内容变化时按相同 ID 覆盖更新，并清理变短后残留的旧分块。
    """
    safe_name = _safe_filename(filename)
    if doc_type is None:
        doc_type = _detect_doc_type(safe_name)

    target_dir = settings.CASES_DIR if doc_type == "case" else settings.LAWS_DIR
    collection_name = (
        settings.CASES_COLLECTION if doc_type == "case" else settings.LAWS_COLLECTION
    )
    store = get_vectorstore(collection_name)

    # 旧版随机 ID 数据无法与确定性 ID 去重，先要求用户重建索引
    if has_legacy_data(store, collection_name, log=logger.warning):
        raise LegacyIndexError(
            "知识库中存在旧版随机 ID 数据，请先在「知识库管理」页点击「重建索引」，再上传文档。"
        )

    # 保存文件
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, safe_name)
    with open(filepath, "wb") as f:
        f.write(content)

    # 读取并写入向量库（JSON/JSONL 案例按记录逐条索引，与批量导入保持一致）
    ext = os.path.splitext(safe_name)[1].lower()
    if doc_type == "case" and ext in (".json", ".jsonl"):
        result = index_case_jsonl_file(
            store, filepath, target_dir, max_new_cases=None, log=logger.info
        )
    elif doc_type == "case":
        result = index_case_file(store, filepath, target_dir, log=logger.info)
    else:
        result = index_law_file(store, filepath, target_dir, log=logger.info)

    if result.total_chunks == 0:
        raise ValueError(f"未能从 {safe_name} 中解析出有效分块")

    return KnowledgeFileInfo(
        filename=safe_name,
        doc_type=doc_type,
        size_bytes=len(content),
        chunk_count=result.total_chunks,
    )


# ===================== 列出知识库文件 =====================

def list_documents() -> list[KnowledgeFileInfo]:
    """列出所有已上传的文档（递归子目录）"""
    files: list[KnowledgeFileInfo] = []
    for dir_path, dtype, exts in [
        (settings.LAWS_DIR, "law", {".txt", ".md"}),
        (settings.CASES_DIR, "case", {".txt", ".md", ".json", ".jsonl"}),
    ]:
        for fpath in collect_files(dir_path, exts):
            fname = os.path.relpath(fpath, dir_path)
            files.append(KnowledgeFileInfo(
                filename=fname,
                doc_type=dtype,
                size_bytes=os.path.getsize(fpath),
            ))
    return files


# ===================== 删除文档 =====================

def delete_document(filename: str) -> bool:
    """删除文档文件，并同步清理向量库中的对应分块"""
    rel_file = filename.replace("\\", "/")
    for dir_path, collection_name in [
        (settings.LAWS_DIR, settings.LAWS_COLLECTION),
        (settings.CASES_DIR, settings.CASES_COLLECTION),
    ]:
        fpath = _safe_join(dir_path, rel_file)
        if not fpath or not os.path.isfile(fpath):
            continue

        os.remove(fpath)
        source_file = os.path.relpath(fpath, dir_path).replace(os.sep, "/")
        try:
            removed = delete_source(get_vectorstore(collection_name), source_file)
            logger.info("已删除 %s 及 %d 个向量分块", source_file, removed)
        except Exception:
            logger.warning(
                "已删除文件 %s，但清理向量分块失败，请重建索引", source_file, exc_info=True
            )
        return True
    return False


# ===================== 重建索引 =====================

async def rebuild_index() -> KnowledgeStats:
    """从 data/ 目录重建全部向量索引

    复用 import_data 的确定性 ID / 查重逻辑，重建结果与 CLI 导入完全一致。
    """
    if not drop_collections(log=logger.info):
        raise RuntimeError(
            "重建失败：无法删除旧集合，请先停止其他导入/重建任务后重试。"
        )

    # --- 法律条文 ---
    laws_store = get_vectorstore(settings.LAWS_COLLECTION)
    import_laws(laws_store, settings.LAWS_DIR, log=logger.info)

    # --- 案例（JSONL/JSON + 整文件 TXT/MD）---
    cases_store = get_vectorstore(settings.CASES_COLLECTION)
    import_cases_jsonl(
        cases_store, settings.CASES_DIR, max_cases=None, log=logger.info
    )
    import_case_files(cases_store, settings.CASES_DIR, log=logger.info)

    return get_kb_stats()
