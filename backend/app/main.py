"""FastAPI 应用入口"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import chat, knowledge, performance

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

# 注：模型改用硅基流动云端 API 后，无需启动时预热（无 Ollama 本地加载到
# GPU 的冷启动问题），首次请求即命中远端模型。


@app.get("/")
async def root():
    return {
        "name": "法律 RAG 系统",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
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
