"""模型与密钥设置 API — 密钥策略、连接测试、模型列表"""

import asyncio
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.core.api_key import (
    api_key_dependency,
    mask_api_key,
    resolve_api_key,
)
from app.core.model_config import (
    classify_model_id,
    model_config_dependency,
    resolve_embedding_model,
    resolve_llm_model,
    resolve_rerank_model,
)
from app.models.schemas import APIResponse

router = APIRouter(prefix="/settings", tags=["设置"])


def _auth_headers(api_key: str) -> dict[str, str]:
    """硅基流动使用 Bearer Token 鉴权"""
    return {"Authorization": f"Bearer {api_key}"}


def _extract_upstream_message(resp: httpx.Response) -> str:
    """从上游错误响应中尽量提取原始报错文本（不包含任何密钥内容）"""
    try:
        payload = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:500]

    if not isinstance(payload, dict):
        return str(payload)[:500]

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("msg") or error.get("detail")
        if message:
            return str(message)[:500]
    elif isinstance(error, str) and error:
        return error[:500]

    for field in ("message", "msg", "detail", "error_message"):
        value = payload.get(field)
        if value:
            return str(value)[:500]

    code = payload.get("code")
    if code not in (None, 0, 200):
        return f"错误码 {code}: {str(payload)[:400]}"
    return str(payload)[:500]


def _upstream_error(resp: httpx.Response) -> str:
    """把上游错误响应转成可调试的中文提示（保留原始报错信息，不含密钥）"""
    reason = {
        400: "请求参数或模型名称不被支持",
        401: "API Key 无效或已过期",
        403: "API Key 无权限访问该模型",
        404: "模型不存在或接口路径错误",
        429: "触发硅基流动限流或账户余额不足",
    }.get(resp.status_code)
    prefix = f"HTTP {resp.status_code}" + (f"（{reason}）" if reason else "")
    message = _extract_upstream_message(resp)
    return f"{prefix}: {message}" if message else prefix


def _exception_detail(exc: Exception) -> str:
    """网络异常转成含异常类型的可调试文本"""
    message = str(exc).strip() or repr(exc)
    return f"连接失败（{type(exc).__name__}）: {message}"


@router.get("/key-policy", response_model=APIResponse)
async def key_policy():
    """公开的密钥与默认模型策略（不涉及任何密钥内容）"""
    return APIResponse(data={
        "require_user_api_key": settings.REQUIRE_USER_API_KEY,  # 是否必须自带 Key
        "header": settings.API_KEY_HEADER,  # 前端应携带的请求头名称
        "server_key_configured": bool((settings.SILICONFLOW_API_KEY or "").strip()),  # 服务端是否配置了兜底 Key
        "base_url": settings.SILICONFLOW_BASE_URL,
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "rerank_model": settings.RERANK_MODEL,
    })


@router.post("/verify-key", response_model=APIResponse)
async def verify_key(_: None = Depends(api_key_dependency)):
    """用当前请求携带的 Key 调用硅基流动 /models 端点，验证是否可用"""
    api_key = resolve_api_key()  # 依赖已写入上下文；无请求时回退服务端 .env
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.SILICONFLOW_BASE_URL}/models",
                headers=_auth_headers(api_key),
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


@router.get("/models", response_model=APIResponse, dependencies=[Depends(api_key_dependency)])
async def list_models():
    """获取当前 Key 可用的完整模型列表，并按 LLM / 嵌入 / 重排分组"""
    api_key = resolve_api_key()
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{settings.SILICONFLOW_BASE_URL}/models",
                headers=_auth_headers(api_key),
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"无法连接硅基流动服务: {e}") from e

    if resp.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="API Key 无效或已过期。")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=_upstream_error(resp))

    try:
        raw_models = resp.json().get("data", []) or []
    except ValueError:
        raw_models = []

    model_ids = sorted({
        str(item.get("id"))
        for item in raw_models
        if isinstance(item, dict) and item.get("id")
    })
    groups: dict[str, list[str]] = {"llm": [], "embedding": [], "rerank": [], "other": []}
    models = []
    for model_id in model_ids:
        category = classify_model_id(model_id)
        groups[category].append(model_id)
        models.append({"id": model_id, "category": category})

    return APIResponse(data={
        "models": models,
        "groups": groups,
        "count": len(models),
        "latency_ms": round((time.time() - started) * 1000, 1),
        "defaults": {
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "rerank_model": settings.RERANK_MODEL,
        },
    })


# ===================== 连接测试 =====================

async def _check_llm(client: httpx.AsyncClient, api_key: str, model: str) -> dict:
    """最小化 LLM 调用：只生成 1 个 token，验证模型可用且鉴权通过"""
    started = time.time()
    try:
        resp = await client.post(
            f"{settings.SILICONFLOW_BASE_URL}/chat/completions",
            headers=_auth_headers(api_key),
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
        )
    except httpx.HTTPError as e:
        return {
            "key": "llm", "label": "LLM 模型", "model": model,
            "ok": False, "latency_ms": None,
            "error_type": type(e).__name__,
            "detail": _exception_detail(e),
        }

    latency = round((time.time() - started) * 1000, 1)
    if resp.status_code == 200:
        return {
            "key": "llm", "label": "LLM 模型", "model": model,
            "ok": True, "latency_ms": latency, "detail": "模型响应正常",
        }
    return {
        "key": "llm", "label": "LLM 模型", "model": model,
        "ok": False, "latency_ms": latency,
        "status_code": resp.status_code,
        "detail": _upstream_error(resp),
    }


async def _check_embedding(client: httpx.AsyncClient, api_key: str, model: str) -> dict:
    """调用嵌入接口并报告向量维度"""
    started = time.time()
    try:
        resp = await client.post(
            f"{settings.SILICONFLOW_BASE_URL}/embeddings",
            headers=_auth_headers(api_key),
            json={"model": model, "input": "连接测试"},
        )
    except httpx.HTTPError as e:
        return {
            "key": "embedding", "label": "嵌入模型", "model": model,
            "ok": False, "latency_ms": None,
            "error_type": type(e).__name__,
            "detail": _exception_detail(e),
        }

    latency = round((time.time() - started) * 1000, 1)
    if resp.status_code != 200:
        return {
            "key": "embedding", "label": "嵌入模型", "model": model,
            "ok": False, "latency_ms": latency,
            "status_code": resp.status_code,
            "detail": _upstream_error(resp),
        }

    try:
        items = resp.json().get("data") or []
        vector = items[0].get("embedding") if items else None
    except (ValueError, AttributeError):
        vector = None
    dimensions = len(vector) if isinstance(vector, list) else None
    return {
        "key": "embedding", "label": "嵌入模型", "model": model,
        "ok": dimensions is not None,
        "latency_ms": latency,
        "dimensions": dimensions,
        "detail": (
            f"向量维度 {dimensions}"
            if dimensions
            else f"接口返回成功，但未解析到向量，响应: {(resp.text or '')[:300]}"
        ),
    }


async def _check_rerank(client: httpx.AsyncClient, api_key: str, model: str) -> dict:
    """用一对查询/文档调用重排序接口，验证模型可用"""
    started = time.time()
    try:
        resp = await client.post(
            f"{settings.SILICONFLOW_BASE_URL}/rerank",
            headers=_auth_headers(api_key),
            json={
                "model": model,
                "query": "合同",
                "documents": ["合同依法成立"],
                "top_n": 1,
                "return_documents": False,
            },
        )
    except httpx.HTTPError as e:
        return {
            "key": "rerank", "label": "重排序模型", "model": model,
            "ok": False, "latency_ms": None,
            "error_type": type(e).__name__,
            "detail": _exception_detail(e),
        }

    latency = round((time.time() - started) * 1000, 1)
    if resp.status_code != 200:
        return {
            "key": "rerank", "label": "重排序模型", "model": model,
            "ok": False, "latency_ms": latency,
            "status_code": resp.status_code,
            "detail": _upstream_error(resp),
        }

    try:
        results = resp.json().get("results") or []
        score = results[0].get("relevance_score") if results else None
    except (ValueError, AttributeError):
        score = None
    return {
        "key": "rerank", "label": "重排序模型", "model": model,
        "ok": score is not None,
        "latency_ms": latency,
        "score": score,
        "detail": (
            f"相关性分数 {score}"
            if score is not None
            else f"接口返回成功，但未解析到分数，响应: {(resp.text or '')[:300]}"
        ),
    }


@router.post(
    "/test-connection",
    response_model=APIResponse,
    dependencies=[Depends(api_key_dependency), Depends(model_config_dependency)],
)
async def test_connection():
    """测试 API Key 与三类模型是否可用（模型取自请求头，未携带时用默认值）"""
    api_key = resolve_api_key()
    llm_model = resolve_llm_model()
    embedding_model = resolve_embedding_model()
    rerank_model = resolve_rerank_model()

    checks: list[dict] = []
    async with httpx.AsyncClient(timeout=20) as client:
        # 第一步：验证 Key 是否有效，同时拿到可用模型数量
        started = time.time()
        try:
            resp = await client.get(
                f"{settings.SILICONFLOW_BASE_URL}/models",
                headers=_auth_headers(api_key),
            )
            latency = round((time.time() - started) * 1000, 1)
            if resp.status_code == 200:
                try:
                    count = len(resp.json().get("data", []) or [])
                except ValueError:
                    count = 0
                checks.append({
                    "key": "key", "label": "API Key",
                    "ok": True, "latency_ms": latency,
                    "detail": f"连接成功，共 {count} 个模型",
                })
            else:
                checks.append({
                    "key": "key", "label": "API Key",
                    "ok": False, "latency_ms": latency,
                    "status_code": resp.status_code,
                    "detail": _upstream_error(resp),
                })
        except httpx.HTTPError as e:
            checks.append({
                "key": "key", "label": "API Key",
                "ok": False, "latency_ms": None,
                "error_type": type(e).__name__,
                "detail": _exception_detail(e),
            })

        # Key 无效时不再发起模型调用，直接标记为跳过，避免无意义的失败请求
        if not checks[-1]["ok"]:
            for key, label, model in (
                ("llm", "LLM 模型", llm_model),
                ("embedding", "嵌入模型", embedding_model),
                ("rerank", "重排序模型", rerank_model),
            ):
                checks.append({
                    "key": key, "label": label, "model": model,
                    "ok": False, "skipped": True, "latency_ms": None,
                    "detail": "API Key 校验未通过，已跳过",
                })
            return APIResponse(data={"ok": False, "checks": checks})

        # 三项互不依赖，并行执行，避免连接测试串行等待
        checks.extend(await asyncio.gather(
            _check_llm(client, api_key, llm_model),
            _check_embedding(client, api_key, embedding_model),
            _check_rerank(client, api_key, rerank_model),
        ))

    return APIResponse(data={"ok": all(check["ok"] for check in checks), "checks": checks})
