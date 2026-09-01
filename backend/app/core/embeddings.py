"""Embedding 初始化 — 单例缓存"""

from langchain_openai import OpenAIEmbeddings
from app.config import settings

_embed_cache: dict[str, OpenAIEmbeddings] = {}  # 按模型名缓存 Embedding 实例


def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    _model = model or settings.EMBEDDING_MODEL  # 未指定则使用配置默认模型（BAAI/bge-m3）
    if _model not in _embed_cache:  # 未命中缓存才创建新实例
        _embed_cache[_model] = OpenAIEmbeddings(
            model=_model,
            base_url=settings.SILICONFLOW_BASE_URL,  # 硅基流动 OpenAI 兼容端点
            api_key=settings.SILICONFLOW_API_KEY,  # 硅基流动 API 密钥
        )
    return _embed_cache[_model]  # 返回缓存的 Embedding 实例
