"""ChromaDB 向量数据库"""

import chromadb
from langchain_community.vectorstores import Chroma
from app.core.embeddings import get_embeddings
from app.core.logging import get_logger
from app.config import settings

logger = get_logger(__name__)

# 缓存已加载的向量库实例
_store_cache: dict[str, Chroma] = {}
# 缓存 Chroma 客户端（按持久化目录缓存，复用同一底层连接）
_client_cache: dict[str, chromadb.PersistentClient] = {}


def get_vectorstore(collection_name: str) -> Chroma:
    """获取（或创建）指定 collection 的 Chroma 向量库"""
    if collection_name in _store_cache:  # 命中缓存直接返回，避免重复加载
        return _store_cache[collection_name]

    # 注意：必须显式传入 chromadb.PersistentClient，而不是用 Chroma 的
    # persist_directory 参数。后者内部走 chromadb.Client(Settings(is_persistent=True))
    # 创建 client，在 chromadb>=1.5 下不会把 HNSW 索引文件（.bin）落盘，只写
    # index_metadata.pickle，导致集合重开/compaction 时报
    # "Error loading hnsw index"。
    client = _client_cache.get(settings.CHROMA_PERSIST_DIR)
    if client is None:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        _client_cache[settings.CHROMA_PERSIST_DIR] = client

    # 说明：chromadb 1.5.x（Rust 内核）默认 hnsw:sync_threshold=1000，即每积压 1000 条
    # 就触发一次 persistDirty()，反复 resize + 相对 seek 写 .bin，索引漂移损坏，
    # 导致重开时 "Error loading hnsw index"（已知 bug，见 chroma-core/chroma#5925）。
    # 这里把 sync_threshold / batch_size 调大，导入期间完全不触发 compaction，
    # 数据留在 sqlite WAL（embeddings_queue）里，重开时重放 WAL 构建索引，可正常使用。
    # 注意：collection_metadata 只在集合"首次创建"时生效，因此重建库前必须删掉旧库目录。
    store = Chroma(
        collection_name=collection_name,  # 集合名：laws / cases
        embedding_function=get_embeddings(),  # 使用共享的 BGE-M3 embedding 实例
        client=client,  # 复用持久化客户端
        collection_metadata={
            "hnsw:sync_threshold": 200000,
            "hnsw:batch_size": 200000,
        },
    )
    _store_cache[collection_name] = store  # 缓存实例，下次直接复用
    return store


def reset_store_cache():
    """清空向量库缓存（在重建索引后调用）"""
    _store_cache.clear()  # 下次 get_vectorstore 会从磁盘重新打开向量库
    _client_cache.clear()


def close_client():
    """显式关闭持久化客户端，触发最终落盘（在导入脚本结束时调用）"""
    for client in _client_cache.values():
        try:
            client.close()
        except Exception:
            # 关闭失败通常说明连接已被释放，属于正常收尾，仅 debug 级别记录
            logger.debug("关闭 Chroma 客户端失败（忽略）", exc_info=True)
    _store_cache.clear()
    _client_cache.clear()
