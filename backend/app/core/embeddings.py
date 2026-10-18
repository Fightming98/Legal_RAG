"""Embedding 初始化 — 按「用户 Key + 模型」缓存实例

Chroma 向量库实例会被进程全局缓存，其 ``embedding_function`` 会长期持有。
如果直接把某个用户的 Embedding 客户端绑上去，后续其他用户查询时就会用错
Key。因此 Chroma 实际持有的是 ``RequestScopedEmbeddings`` 代理：每次真正
发起 embedding 调用时，才按当前请求上下文解析 Key 并取对应客户端。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.core.api_key import resolve_api_key

_MAX_CACHE_ENTRIES = 64  # 最多缓存 64 个客户端，超出后淘汰最久未用条目

# 键：(Key 指纹, 模型名)；值：OpenAIEmbeddings 实例
_embed_cache: "OrderedDict[tuple[str, str], OpenAIEmbeddings]" = OrderedDict()


def _key_id(api_key: str) -> str:
    """缓存键中只保留 Key 的哈希指纹，不落明文"""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    """获取当前请求上下文对应 Key 的 Embedding 客户端"""
    api_key = resolve_api_key()  # 当前请求的 Key（无请求上下文时回退服务端 .env）
    _model = model or settings.EMBEDDING_MODEL  # 未指定则使用配置默认模型（BAAI/bge-m3）
    cache_key = (_key_id(api_key), _model)

    cached = _embed_cache.get(cache_key)
    if cached is not None:  # 命中缓存：刷新 LRU 位置并复用连接
        _embed_cache.move_to_end(cache_key)
        return cached

    embeddings = OpenAIEmbeddings(
        model=_model,
        base_url=settings.SILICONFLOW_BASE_URL,  # 硅基流动 OpenAI 兼容端点
        api_key=api_key,  # 用户自己的硅基流动 API 密钥
    )
    _embed_cache[cache_key] = embeddings
    while len(_embed_cache) > _MAX_CACHE_ENTRIES:  # 超出上限淘汰最久未使用条目
        _embed_cache.popitem(last=False)
    return embeddings


class RequestScopedEmbeddings(Embeddings):
    """请求级 Embedding 代理。

    本身不持有任何密钥或连接，只在被调用时按当前上下文解析出真正的
    Embedding 客户端。Chroma 的 ``embedding_function`` 可以安全地全局复用。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return get_embeddings().embed_query(text)


_request_embeddings = RequestScopedEmbeddings()


def get_request_embeddings() -> Embeddings:
    """返回全局共享的请求级 Embedding 代理（供 Chroma 长期持有）"""
    return _request_embeddings


def clear_embed_cache(api_key: str | None = None):
    """清空缓存（测试时使用）；传入 api_key 时只清除该 Key 的客户端"""
    if api_key is None:
        _embed_cache.clear()
        return
    key_id = _key_id(api_key)
    for cache_key in [k for k in _embed_cache if k[0] == key_id]:
        del _embed_cache[cache_key]
