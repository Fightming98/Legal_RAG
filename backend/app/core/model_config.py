"""请求级模型配置上下文

「模型设置」页面允许用户自定义 LLM / 嵌入 / 重排序模型。模型名通过请求头
``X-LLM-Model`` / ``X-Embedding-Model`` / ``X-Rerank-Model`` 传入，由
``model_config_dependency`` 写入请求级 ContextVar；``get_llm`` /
``get_embeddings`` / 重排序服务在真正发起调用时读取。

这样既能做到“每个访客使用自己选择的模型”，又不会污染进程级全局配置；
请求未携带模型名时回退 ``settings`` 中的默认值（DeepSeek-V3.2 / BGE-M3 /
BGE-Reranker-V2-M3）。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from fastapi import Header, HTTPException, status

from app.config import settings

# 前端携带模型配置的请求头名称
LLM_MODEL_HEADER = "X-LLM-Model"
EMBEDDING_MODEL_HEADER = "X-Embedding-Model"
RERANK_MODEL_HEADER = "X-Rerank-Model"

# 硅基流动模型 ID 形如 deepseek-ai/DeepSeek-V3.2、BAAI/bge-m3、Qwen/Qwen3-Embedding-8B
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:/@+\-]{1,200}$")

# 当前请求选择的模型；None / 缺失表示使用 settings 默认值
_models_var: ContextVar[dict[str, str | None]] = ContextVar("request_models", default={})


def normalize_model_id(raw: str | None, header: str) -> str | None:
    """校验请求头中的模型 ID；空值视为未指定，格式非法直接 422"""
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if not _MODEL_ID_PATTERN.fullmatch(value):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{header} 中的模型名称格式不正确（只允许字母、数字与 . _ : / @ + -）。",
        )
    return value


def set_current_models(
    llm: str | None = None,
    embedding: str | None = None,
    rerank: str | None = None,
) -> Token:
    """写入当前上下文，返回用于复位的 Token"""
    return _models_var.set({"llm": llm, "embedding": embedding, "rerank": rerank})


def reset_current_models(token: Token) -> None:
    """恢复写入前的上下文"""
    _models_var.reset(token)


def current_models() -> dict[str, str | None]:
    """读取当前请求选择的模型（未指定时对应值为 None）"""
    return _models_var.get()


def resolve_llm_model() -> str:
    """本次调用应使用的 LLM 模型（请求头优先，其次 .env 默认值）"""
    return current_models().get("llm") or settings.LLM_MODEL


def resolve_embedding_model() -> str:
    """本次调用应使用的嵌入模型（请求头优先，其次 .env 默认值）"""
    return current_models().get("embedding") or settings.EMBEDDING_MODEL


def resolve_rerank_model() -> str:
    """本次调用应使用的重排序模型（请求头优先，其次 .env 默认值）"""
    return current_models().get("rerank") or settings.RERANK_MODEL


async def model_config_dependency(
    x_llm_model: str | None = Header(
        default=None,
        alias=LLM_MODEL_HEADER,
        description="用户选择的 LLM 模型 ID",
    ),
    x_embedding_model: str | None = Header(
        default=None,
        alias=EMBEDDING_MODEL_HEADER,
        description="用户选择的嵌入模型 ID",
    ),
    x_rerank_model: str | None = Header(
        default=None,
        alias=RERANK_MODEL_HEADER,
        description="用户选择的重排序模型 ID",
    ),
) -> AsyncIterator[None]:
    """挂在所有消耗模型的接口上：把用户选择的模型写入请求级上下文"""
    token = set_current_models(
        llm=normalize_model_id(x_llm_model, LLM_MODEL_HEADER),
        embedding=normalize_model_id(x_embedding_model, EMBEDDING_MODEL_HEADER),
        rerank=normalize_model_id(x_rerank_model, RERANK_MODEL_HEADER),
    )
    try:
        yield
    finally:
        reset_current_models(token)


@contextmanager
def model_config_context(
    llm: str | None = None,
    embedding: str | None = None,
    rerank: str | None = None,
):
    """非 HTTP 场景（脚本 / 单元测试）显式指定模型的上下文管理器"""
    token = set_current_models(llm=llm, embedding=embedding, rerank=rerank)
    try:
        yield
    finally:
        reset_current_models(token)


# ===================== 模型分类（供模型设置页分组展示） =====================

_RERANK_MARKERS = ("rerank",)

_EMBEDDING_MARKERS = (
    "embedding",
    "embed-",
    "-embed",
    "bge-m3",
    "bge-large",
    "bge-small",
    "bce-",
    "gte-",
    "m3e",
    "text2vec",
)

# 图像 / 语音 / 视频等非文本模型，不放入 LLM 候选，避免误选
_NON_TEXT_MARKERS = (
    "stable-diffusion",
    "sdxl",
    "sd3",
    "flux",
    "kolors",
    "dall-e",
    "image",
    "cosyvoice",
    "fish-speech",
    "speech",
    "whisper",
    "tts",
    "asr",
    "audio",
    "wan2",
    "t2v",
    "i2v",
    "video",
    "vl-",
)


def classify_model_id(model_id: str) -> str:
    """按模型 ID 推断用途分类：llm / embedding / rerank / other"""
    lowered = (model_id or "").lower()
    if any(marker in lowered for marker in _RERANK_MARKERS):
        return "rerank"
    if any(marker in lowered for marker in _EMBEDDING_MARKERS):
        return "embedding"
    if any(marker in lowered for marker in _NON_TEXT_MARKERS):
        return "other"
    return "llm"
