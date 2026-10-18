"""问答 API"""

import json

import openai
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from app.core.api_key import api_key_dependency
from app.models.schemas import ChatRequest, ChatResponse, APIResponse
from app.services.rag_service import rag_query
from app.services.stream_service import stream_rag_query
from app.services.perf_service import get_system_info
from app.services.report_service import (
    save_chat_record,
    list_chat_records,
    get_chat_record_filepath,
)

router = APIRouter(prefix="/chat", tags=["问答"])

# 流式响应开始后 HTTP 状态码无法再修改，异常统一通过 error 事件下发
_KEY_INVALID_MSG = "SiliconFlow API Key 无效或已过期，请在左下角「API 设置」中重新配置。"
_RATE_LIMIT_MSG = "硅基流动额度不足或请求过于频繁，请检查你账户的余额后重试。"
_CONNECTION_MSG = "服务器无法连接硅基流动服务，请稍后重试。"


def _sse(event: str, data: dict) -> str:
    """格式化为 SSE 帧（JSON 数据中的换行已被转义，不会截断事件）"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=APIResponse, dependencies=[Depends(api_key_dependency)])
async def chat(req: ChatRequest):
    """RAG 问答接口，支持系统监控和质量评估（必须携带用户自己的 API Key）"""
    try:
        # 系统快照（问答前）
        system_before = None
        if req.monitor_system:
            system_before = get_system_info()

        # RAG 查询
        result: ChatResponse = await rag_query(
            question=req.question,
            use_rerank=req.use_rerank,
            use_query_rewrite=req.use_query_rewrite,
            top_k=req.top_k,
            collection=req.collection,
            query_transform=req.query_transform,
            rerank_strategy=req.rerank_strategy,
            generation_strategy=req.generation_strategy,
            use_kg=req.use_kg,
        )

        # 系统快照（问答后）
        system_after = None
        if req.monitor_system:
            system_after = get_system_info()

        # 质量评估
        quality = None
        if req.evaluate_quality:
            from app.services.quality_service import evaluate_single_query
            quality = await evaluate_single_query(
                query=req.question,
                answer=result.answer,
                sources=result.sources,
            )

        result.system_before = system_before
        result.system_after = system_after
        result.quality = quality

        return APIResponse(data=result.model_dump())
    except HTTPException:
        raise  # 401（缺 Key/Key 非法）等语义化异常直接透传
    except (openai.AuthenticationError, openai.RateLimitError, openai.APIConnectionError):
        raise  # 交给 main.py 的全局处理器转成 401 / 429 / 502
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答服务异常: {e}")


@router.post("/stream", dependencies=[Depends(api_key_dependency)])
async def chat_stream(req: ChatRequest):
    """流式 RAG 问答（SSE）：分别推送思考过程、回答正文与检索来源"""
    async def event_stream():
        try:
            async for event, data in stream_rag_query(
                question=req.question,
                use_rerank=req.use_rerank,
                use_query_rewrite=req.use_query_rewrite,
                top_k=req.top_k,
                collection=req.collection,
                query_transform=req.query_transform,
                rerank_strategy=req.rerank_strategy,
                generation_strategy=req.generation_strategy,
                use_kg=req.use_kg,
                enable_thinking=req.enable_thinking,
                monitor_system=req.monitor_system,
                evaluate_quality=req.evaluate_quality,
            ):
                yield _sse(event, data)
        except HTTPException as exc:
            # 依赖已完成校验；此处主要兜底生成阶段抛出的语义化异常
            yield _sse("error", {"detail": exc.detail, "status": exc.status_code})
        except openai.AuthenticationError:
            yield _sse("error", {"detail": _KEY_INVALID_MSG, "status": 401})
        except openai.RateLimitError:
            yield _sse("error", {"detail": _RATE_LIMIT_MSG, "status": 429})
        except openai.APIConnectionError:
            yield _sse("error", {"detail": _CONNECTION_MSG, "status": 502})
        except Exception as e:
            yield _sse("error", {"detail": f"问答服务异常: {e}", "status": 500})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # 禁止中间层缓存
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 等反向代理的缓冲
        },
    )


@router.post("/save-record", response_model=APIResponse)
async def save_record(record: dict):
    """保存一次问答的性能记录"""
    try:
        record_id = save_chat_record(record)
        return APIResponse(data={"record_id": record_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存记录失败: {e}")


@router.get("/records", response_model=APIResponse)
async def get_records():
    """列出问答性能记录"""
    metas = list_chat_records()
    return APIResponse(data=[m.model_dump() for m in metas])


@router.get("/records/{record_id}")
async def download_record(record_id: str):
    """下载问答性能记录 JSON"""
    filepath = get_chat_record_filepath(record_id)
    if filepath is None:
        raise HTTPException(status_code=404, detail=f"记录 {record_id} 不存在")
    return FileResponse(
        path=str(filepath),
        media_type="application/json",
        filename=f"chat_record_{record_id}.json",
    )
