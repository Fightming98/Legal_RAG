"""LLM 初始化 — 实例缓存，避免重复创建 HTTP 连接"""

from langchain_openai import ChatOpenAI
from app.config import settings

# 按 (model, temperature) 缓存 LLM 实例，同一参数组合复用同一连接，
# 减少到硅基流动平台侧的会话/连接开销
_llm_cache: dict[tuple[str, float], ChatOpenAI] = {}  # 实例缓存字典：键为 (model, temperature)


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    _model = model or settings.LLM_MODEL  # 未指定则使用配置默认模型
    _temp = temperature if temperature is not None else settings.LLM_TEMPERATURE  # 未指定则使用配置默认温度
    cache_key = (_model, _temp)  # 组合成缓存键

    if cache_key not in _llm_cache:  # 未命中缓存才创建新实例
        _llm_cache[cache_key] = ChatOpenAI(
            model=_model,
            temperature=_temp,
            base_url=settings.SILICONFLOW_BASE_URL,  # 硅基流动 OpenAI 兼容端点
            api_key=settings.SILICONFLOW_API_KEY,  # 硅基流动 API 密钥
            timeout=settings.LLM_TIMEOUT,  # 单次请求超时（秒）
        )
    return _llm_cache[cache_key]  # 返回缓存的 LLM 实例


def clear_llm_cache():
    """清空缓存（模型切换或测试时使用）"""
    _llm_cache.clear()  # 清空所有缓存的 LLM 实例
