"""请求级 SiliconFlow API Key 上下文

公网部署时，希望每个访客使用自己的 SiliconFlow 额度，而不是消耗部署者的
密钥。前端在请求头 ``X-SiliconFlow-API-Key`` 中携带用户自己的 Key，本模块
负责：

1. 校验 Key 格式；
2. 通过 FastAPI 依赖把 Key 写入请求级 ContextVar；
3. 为 ``get_llm`` / ``get_embeddings`` 提供解析入口。

ContextVar 在同一个请求的异步调用链中自动可见（包括服务层里同步调用的
Embedding 代码），请求结束后由依赖的 ``finally`` 复位，不会串到其他用户。
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from fastapi import Header, HTTPException, status

from app.config import settings

# 当前请求携带的用户 Key；None 表示请求未携带（或当前不在 HTTP 请求中）
_api_key_var: ContextVar[str | None] = ContextVar("siliconflow_api_key", default=None)

_MIN_KEY_LENGTH = 20  # 硅基流动 Key 形如 sk- + 长串，低于该长度必然是残缺值
_MAX_KEY_LENGTH = 256  # 上限防止超长请求头打爆日志/缓存


def normalize_api_key(raw: str | None) -> str | None:
    """校验并规范化请求头中的 API Key；格式非法直接 401"""
    if raw is None:
        return None
    key = raw.strip()
    if not key:
        return None
    if (
        not key.startswith("sk-")
        or len(key) < _MIN_KEY_LENGTH
        or len(key) > _MAX_KEY_LENGTH
        or any(ch.isspace() for ch in key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="SiliconFlow API Key 格式不正确，应形如 sk-xxxxxxxx。",
        )
    return key


def key_fingerprint(key: str) -> str:
    """Key 指纹：用作客户端缓存键，避免在缓存字典里留存明文密钥"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def mask_api_key(key: str) -> str:
    """脱敏展示（日志与接口返回中不出现完整 Key）"""
    if len(key) <= 12:
        return "sk-****"
    return f"{key[:6]}...{key[-4:]}"


def set_current_api_key(key: str | None) -> Token:
    """写入当前上下文，返回用于复位的 Token"""
    return _api_key_var.set(key)


def reset_current_api_key(token: Token) -> None:
    """恢复写入前的上下文"""
    _api_key_var.reset(token)


def current_api_key() -> str | None:
    """读取当前请求携带的 Key（未携带则为 None）"""
    return _api_key_var.get()


@contextmanager
def api_key_context(key: str | None):
    """非 HTTP 场景（脚本 / 单元测试）显式指定 Key 的上下文管理器"""
    token = set_current_api_key(key)
    try:
        yield
    finally:
        reset_current_api_key(token)


def resolve_api_key() -> str:
    """解析本次模型调用应使用的 Key。

    优先级：
    1. 当前请求头中的用户 Key；
    2. 服务端 .env 的 SILICONFLOW_API_KEY —— 仅作为非请求场景
       （导入脚本、健康检查、单元测试）的兜底。

    注意：HTTP 请求是否允许走第 2 级，由 ``api_key_dependency`` 按
    ``REQUIRE_USER_API_KEY`` 控制；默认开启时，未携带 Key 的模型请求
    在进入业务逻辑前就会收到 401。
    """
    key = current_api_key()
    if key:
        return key
    server_key = (settings.SILICONFLOW_API_KEY or "").strip()
    if server_key:
        return server_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="缺少 SiliconFlow API Key，请先在页面「API 设置」中填入你自己的密钥。",
    )


async def api_key_dependency(
    x_siliconflow_api_key: str | None = Header(
        default=None,
        alias=settings.API_KEY_HEADER,
        description="用户自己的 SiliconFlow API Key（sk- 开头）",
    ),
) -> AsyncIterator[None]:
    """挂在所有消耗模型的接口上：校验 Key 并写入请求级上下文。

    默认 ``REQUIRE_USER_API_KEY=True``，未携带 Key 直接 401，保证部署者的
    密钥不会被公网访客消耗。
    """
    key = normalize_api_key(x_siliconflow_api_key)
    if key is None and settings.REQUIRE_USER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "本站要求使用你自己的 SiliconFlow API Key。"
                "请点击左下角「API 设置」填入密钥后再试。"
            ),
        )
    token = set_current_api_key(key)
    try:
        yield
    finally:
        reset_current_api_key(token)
