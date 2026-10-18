"""流式 RAG 服务 — 将管线执行过程转换为可序列化的事件流

事件类型：
- stage:          管线阶段提示（检索、生成、反思、质量评估）
- sources:        检索完成，附带来源、查询变换结果与检索阶段耗时
- reasoning:      模型思考内容增量
- answer:         回答正文增量
- answer_replace: 自我反思修正后的完整回答（替换此前的流式内容）
- metrics:        最终阶段耗时
- quality:        质量评估结果
- done:           流结束元数据（系统前后快照、生成策略等）
- error:          流内异常（由 API 层负责转换）
"""

import time
from collections.abc import AsyncIterator

from app.config import settings
from app.models.schemas import StageMetrics
from app.services.perf_service import get_system_info
from app.services.pipeline import (
    GenerationStrategy,
    PipelineConfig,
    QueryTransformStrategy,
    RAGPipeline,
    RerankStrategy,
    resolve_collections,
)


async def stream_rag_query(
    question: str,
    use_rerank: bool = True,
    use_query_rewrite: bool = False,
    top_k: int = 5,
    collection: str = "all",
    query_transform: str = "none",
    rerank_strategy: str = "simple",
    generation_strategy: str = "standard",
    use_kg: bool = False,
    enable_thinking: bool = True,
    monitor_system: bool = False,
    evaluate_quality: bool = False,
) -> AsyncIterator[tuple[str, dict]]:
    """执行流式 RAG 查询，逐事件产出 (事件名, 数据)"""
    total_start = time.time()  # 端到端计时起点

    # 遗留参数向新策略的映射（与 rag_service.rag_query 保持一致）
    effective_rerank = rerank_strategy if use_rerank else "none"
    effective_transform = query_transform
    if use_query_rewrite and query_transform == "none":
        effective_transform = "multi_query"

    config = PipelineConfig(
        query_transform=QueryTransformStrategy(effective_transform),
        rerank_strategy=RerankStrategy(effective_rerank),
        generation_strategy=GenerationStrategy(generation_strategy),
        use_kg=use_kg,
        top_k=top_k,
        collection_names=resolve_collections(collection),
    )
    pipeline = RAGPipeline(config)

    system_before = get_system_info() if monitor_system else None  # 问答前系统快照

    # Stage 1-3: 查询变换 / 检索 / KG / 重排
    yield "stage", {"stage": "prepare", "message": "正在理解问题并检索资料…"}
    prepared = await pipeline.prepare(question)

    yield "sources", {
        "sources": [s.model_dump() for s in prepared.sources],
        "rewritten_queries": prepared.rewritten_queries,
        "kg_entities": prepared.kg_entities or None,
        "system_before": system_before.model_dump() if system_before else None,
        # 检索阶段的局部耗时，前端可先展示，最终以 metrics 事件为准
        "metrics": {
            "query_rewrite_ms": pipeline.metrics.get("query_rewrite_ms"),
            "retrieval_ms": pipeline.metrics.get("retrieval_ms"),
            "kg_lookup_ms": pipeline.metrics.get("kg_lookup_ms"),
            "rerank_ms": pipeline.metrics.get("rerank_ms"),
        },
    }

    # Stage 4: 流式生成（思考内容 + 回答正文）
    yield "stage", {"stage": "generation", "message": "正在生成回答…"}
    answer_parts: list[str] = []
    async for kind, text in pipeline.stream_generate(
        question,
        prepared.docs,
        enable_thinking=enable_thinking,
        context=prepared.context,
    ):
        if kind == "answer":
            answer_parts.append(text)
        yield kind, {"delta": text}

    answer = "".join(answer_parts)
    pipeline.metrics["llm_calls"] += 1  # 生成至少 1 次

    # 自我反思：流式输出初答后，若被修正则发送完整替换内容
    if config.generation_strategy == GenerationStrategy.SELF_REFLECT:
        yield "stage", {"stage": "self_reflect", "message": "正在校验并修正回答…"}
        t_reflect = time.time()
        from app.services.self_reflect import self_reflect_and_correct

        final_answer, was_corrected = await self_reflect_and_correct(
            question=question,
            initial_answer=answer,
            context=prepared.context,
            max_iterations=settings.SELF_REFLECT_MAX_ITER,
        )
        pipeline.metrics["self_reflect_ms"] = round((time.time() - t_reflect) * 1000, 1)
        pipeline.metrics["was_corrected"] = was_corrected
        if was_corrected:
            pipeline.metrics["llm_calls"] += 2  # 反思验证 + 修正
        if final_answer != answer:
            answer = final_answer
            yield "answer_replace", {"answer": final_answer, "was_corrected": True}

    pipeline.metrics["total_ms"] = round((time.time() - total_start) * 1000, 1)
    yield "metrics", StageMetrics(**pipeline.metrics).model_dump()

    # 质量评估（可选，位于回答完成之后）
    if evaluate_quality:
        yield "stage", {"stage": "quality", "message": "正在评估回答质量…"}
        from app.services.quality_service import evaluate_single_query

        quality = await evaluate_single_query(
            query=question,
            answer=answer,
            sources=prepared.sources,
        )
        yield "quality", quality.model_dump()

    system_after = get_system_info() if monitor_system else None  # 问答后系统快照
    yield "done", {
        "generation_strategy": config.generation_strategy.value,
        "pipeline_config": config.to_dict(),
        "system_before": system_before.model_dump() if system_before else None,
        "system_after": system_after.model_dump() if system_after else None,
        "was_corrected": pipeline.metrics.get("was_corrected", False),
    }
