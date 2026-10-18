"""LLM 初始化 — 按「用户 Key + 模型 + 温度」缓存实例

公网部署时每个请求可能携带不同用户的 SiliconFlow Key，因此缓存不能再只按
(model, temperature) 命中，否则会把 A 用户的客户端（含 A 的密钥）复用给
B 用户。这里用 Key 指纹参与缓存键，并限制缓存条目上限，避免不同 Key 无限
堆积。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from langchain_openai import ChatOpenAI

from app.config import settings
from app.core.api_key import resolve_api_key

_MAX_CACHE_ENTRIES = 64  # 最多缓存 64 个客户端，超出后淘汰最久未用条目

# 键：(Key 指纹, 模型名, 温度)；值：ChatOpenAI 实例
_llm_cache: "OrderedDict[tuple[str, str, float], ChatOpenAI]" = OrderedDict()


def _key_id(api_key: str) -> str:
    """缓存键中只保留 Key 的哈希指纹，不落明文"""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """获取当前请求上下文对应 Key 的 LLM 客户端"""
    api_key = resolve_api_key()  # 当前请求的 Key（无请求上下文时回退服务端 .env）
    _model = model or settings.LLM_MODEL  # 未指定则使用配置默认模型
    _temp = temperature if temperature is not None else settings.LLM_TEMPERATURE  # 未指定则使用配置默认温度
    cache_key = (_key_id(api_key), _model, _temp)  # 组合成缓存键

    cached = _llm_cache.get(cache_key)
    if cached is not None:  # 命中缓存：刷新 LRU 位置并复用连接
        _llm_cache.move_to_end(cache_key)
        return cached

    llm = ChatOpenAI(
        model=_model,
        temperature=_temp,
        base_url=settings.SILICONFLOW_BASE_URL,  # 硅基流动 OpenAI 兼容端点
        api_key=api_key,  # 用户自己的硅基流动 API 密钥
        timeout=settings.LLM_TIMEOUT,  # 单次请求超时（秒）
    )
    _llm_cache[cache_key] = llm
    while len(_llm_cache) > _MAX_CACHE_ENTRIES:  # 超出上限淘汰最久未使用条目
        _llm_cache.popitem(last=False)
    return llm


def clear_llm_cache(api_key: str | None = None):
    """清空缓存（模型切换或测试时使用）。

    传入 api_key 时只清除该 Key 对应的客户端；不传则全部清空。
    """
    if api_key is None:
        _llm_cache.clear()
        return
    key_id = _key_id(api_key)
    for cache_key in [k for k in _llm_cache if k[0] == key_id]:
        del _llm_cache[cache_key]
