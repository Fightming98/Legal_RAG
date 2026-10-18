"""API Key 设置 API — 密钥策略查询与连通性校验"""

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.core.api_key import (
    api_key_dependency,
    mask_api_key,
    resolve_api_key,
)
from app.models.schemas import APIResponse

router = APIRouter(prefix="/settings", tags=["设置"])


@router.get("/key-policy", response_model=APIResponse)
async def key_policy():
    """公开的密钥策略（不涉及任何密钥内容），供前端决定是否强制弹窗配置"""
    return APIResponse(data={
        "require_user_api_key": settings.REQUIRE_USER_API_KEY,  # 是否必须自带 Key
        "header": settings.API_KEY_HEADER,  # 前端应携带的请求头名称
        "server_key_configured": bool((settings.SILICONFLOW_API_KEY or "").strip()),  # 服务端是否配置了兜底 Key（不返回内容）
        "base_url": settings.SILICONFLOW_BASE_URL,
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
    })


@router.post("/verify-key", response_model=APIResponse)
async def verify_key(_: None = Depends(api_key_dependency)):
    """用当前请求携带的 Key 调用硅基流动 /models 端点，验证是否可用"""
    api_key = resolve_api_key()  # 依赖已写入上下文；无请求时回退服务端 .env
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.SILICONFLOW_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"无法连接硅基流动服务: {e}") from e

    if resp.status_code in (401, 403):
        raise HTTPException(
            status_code=401,
            detail="API Key 无效或已过期，请检查后重新输入。",
        )
    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="硅基流动接口限流，请稍后再试。")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"硅基流动接口返回异常状态码 {resp.status_code}。",
        )

    try:
        models = resp.json().get("data", []) or []
    except ValueError:
        models = []

    return APIResponse(data={
        "valid": True,
        "key_masked": mask_api_key(api_key),  # 只回传脱敏后的 Key
        "model_count": len(models),
        "models": [m.get("id", "") for m in models][:20],  # 最多展示 20 个模型
    })
