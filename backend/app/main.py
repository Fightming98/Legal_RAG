"""FastAPI 应用入口"""

import openai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.api import chat, knowledge, performance
from app.api import settings as settings_api

app = FastAPI(
    title="法律 RAG 系统",
    description="基于 LangChain + DeepSeek-V3.2 + BGE-M3（硅基流动 API）的法律智能问答系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix=settings.API_PREFIX)
app.include_router(knowledge.router, prefix=settings.API_PREFIX)
app.include_router(performance.router, prefix=settings.API_PREFIX)
app.include_router(settings_api.router, prefix=settings.API_PREFIX)


# ===================== 模型服务异常 → 明确 HTTP 语义 =====================
# 用户自带 Key 后，认证失败/余额不足是常见情况；返回可读的 401/429，
# 前端据此提示用户重新配置，而不是笼统的 500。

@app.exception_handler(openai.AuthenticationError)
async def handle_authentication_error(request: Request, exc: openai.AuthenticationError):
    """Key 无效 / 被禁用"""
    return JSONResponse(
        status_code=401,
        content={"detail": "SiliconFlow API Key 无效或已过期，请在左下角「API 设置」中重新配置。"},
    )


@app.exception_handler(openai.RateLimitError)
async def handle_rate_limit_error(request: Request, exc: openai.RateLimitError):
    """触发限流或账户余额不足"""
    return JSONResponse(
        status_code=429,
        content={"detail": "硅基流动额度不足或请求过于频繁，请检查你账户的余额后重试。"},
    )


@app.exception_handler(openai.APIConnectionError)
async def handle_api_connection_error(request: Request, exc: openai.APIConnectionError):
    """网络不可达（服务器无法访问硅基流动）"""
    return JSONResponse(
        status_code=502,
        content={"detail": "服务器无法连接硅基流动服务，请稍后重试。"},
    )

# 注：模型改用硅基流动云端 API 后，无需启动时预热（无 Ollama 本地加载到
# GPU 的冷启动问题），首次请求即命中远端模型。


@app.get("/")
async def root():
    return {
        "name": "法律 RAG 系统",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "require_user_api_key": settings.REQUIRE_USER_API_KEY,
    }


@app.get("/health")
async def health_check():
    """全局健康检查 — 包含模型服务（硅基流动）状态"""
    from app.services.kb_service import get_kb_stats
    from app.services.perf_service import get_provider_status
    stats = get_kb_stats()
    provider = await get_provider_status()
    return {
        "status": "healthy",
        "models": {
            "llm": settings.LLM_MODEL,
            "embedding": settings.EMBEDDING_MODEL,
        },
        "provider": provider,
        "knowledge_base": stats,
    }
