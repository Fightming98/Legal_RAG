"""RAG 可插拔管线 — 编排查询变换、检索、重排序、生成等阶段"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum

from langchain_core.documents import Document
from langchain_core.messages import convert_to_openai_messages
from openai import AsyncOpenAI

from app.core.api_key import resolve_api_key
from app.core.llm import get_llm
from app.core.model_config import (
    resolve_embedding_model,
    resolve_llm_model,
    resolve_rerank_model,
)
from app.core.retriever import get_hybrid_retriever, rrf_fuse_ranked_lists
from app.services.reranker import simple_rerank, llm_rerank, bge_rerank
from app.services.query_rewriter import multi_query_rewrite
from app.services.prompts import (
    LEGAL_QA_PROMPT,
    LEGAL_COT_PROMPT,
    LEGAL_STRUCTURED_PROMPT,
)
from app.utils.metadata import format_source_display
from app.models.schemas import ChatResponse, SourceDocument, StageMetrics
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ===================== 策略枚举 =====================

class QueryTransformStrategy(str, Enum):
    """查询变换策略"""
    NONE = "none"  # 不做变换，直接使用原问题检索
    MULTI_QUERY = "multi_query"  # 多查询重写：扩展为多个角度的检索查询
    HYDE = "hyde"  # 假设文档：LLM 先生成假设文档再用于向量检索
    DECOMPOSE = "decompose"  # 查询分解：拆分为多个子问题分别检索
    MULTI_QUERY_HYDE = "multi_query_hyde"  # 组合策略：多查询重写 + HyDE 并行执行


class RerankStrategy(str, Enum):
    """重排序策略"""
    NONE = "none"  # 不重排，直接取检索结果前 top_k
    SIMPLE = "simple"  # 简单重排：基于 jieba 关键词 Jaccard 相似度（零 LLM 开销）
    LLM = "llm"  # LLM 重排：批量评分，更准确但消耗一次 LLM 调用
    BGE = "bge"  # BGE 精排：硅基流动 BAAI/bge-reranker-v2-m3，按相关性分数排序（默认）


class GenerationStrategy(str, Enum):
    """生成策略"""
    STANDARD = "standard"  # 标准法律问答
    COT = "chain_of_thought"  # 链式推理
    SELF_REFLECT = "self_reflect"  # 自我反思修正
    STRUCTURED = "structured_legal"  # 结构化法律回答


# ===================== 管线配置 =====================

@dataclass
class PipelineConfig:
    """RAG 管线各阶段的策略配置"""
    query_transform: QueryTransformStrategy = QueryTransformStrategy.NONE  # 查询变换策略
    rerank_strategy: RerankStrategy = RerankStrategy.BGE  # 重排序策略（默认 BGE 精排）
    generation_strategy: GenerationStrategy = GenerationStrategy.SELF_REFLECT  # 生成策略
    use_kg: bool = False  # 是否启用犯罪知识图谱增强
    top_k: int = 5  # 最终返回给生成阶段的最大文档数
    collection_names: list[str] = field(default_factory=lambda: ["laws", "cases"])  # 检索的向量库集合

    def to_dict(self) -> dict:
        """将配置序列化为字典（返回给前端 / 写入 pipeline_config 字段）"""
        return {
            "query_transform": self.query_transform.value,
            "rerank_strategy": self.rerank_strategy.value,
            "generation_strategy": self.generation_strategy.value,
            "use_kg": self.use_kg,
            "top_k": self.top_k,
            "collection_names": self.collection_names,
            # 记录本次请求实际使用的模型，便于前端展示与结果回溯
            "models": {
                "llm": resolve_llm_model(),
                "embedding": resolve_embedding_model(),
                "rerank": resolve_rerank_model(),
            },
        }


@dataclass
class PreparedContext:
    """检索侧管线（Stage 1-3）的产物，供生成 / 流式生成阶段复用"""
    context: str  # 拼装后的参考资料文本
    docs: list[Document]  # 重排后的文档
    sources: list[SourceDocument]  # 返回给前端的来源摘要
    kg_entities: list[str]  # KG 命中的罪名实体
    rewritten_queries: list[str] | None  # 查询变换结果


# ===================== RAG 管线 =====================

class RAGPipeline:
    """可插拔 RAG 管线，按配置调度各阶段策略"""

    def __init__(self, config: PipelineConfig):
        self.config = config  # 保存管线配置
        # 各阶段耗时与效率指标（单位 ms）；None 表示该阶段未启用
        self.metrics: dict = {
            "query_rewrite_ms": None,  # 查询变换耗时
            "retrieval_ms": 0,  # 检索耗时
            "rerank_ms": None,  # 重排序耗时
            "generation_ms": 0,  # 生成耗时
            "total_ms": 0,  # 端到端总耗时
            "kg_lookup_ms": None,  # KG 查找耗时
            "self_reflect_ms": None,  # 自我反思耗时
            "was_corrected": False,  # 是否发生了自我修正
            "llm_calls": 0,  # 实际 LLM 调用次数
            "llm_calls_saved": 0,  # 通过优化节省的 LLM 调用次数
        }

    async def execute(self, question: str) -> ChatResponse:
        """执行完整 RAG 管线：查询变换 → 检索 → KG → 重排 → 生成"""
        total_start = time.time()  # 端到端计时起点

        # Stage 1-3: 查询变换 / 检索 / KG / 重排
        prepared = await self.prepare(question)

        # Stage 4: 生成（选择生成策略并调用 LLM 产出答案）
        answer, was_corrected = await self._generate(question, prepared.docs, prepared.context)
        self.metrics["was_corrected"] = was_corrected
        self.metrics["llm_calls"] += 1  # 生成至少 1 次
        if was_corrected:
            self.metrics["llm_calls"] += 2  # 反思验证 + 修正

        # 总耗时
        self.metrics["total_ms"] = round((time.time() - total_start) * 1000, 1)

        # 组装最终响应
        return ChatResponse(
            answer=answer,
            sources=prepared.sources,
            metrics=StageMetrics(**self.metrics),
            rewritten_queries=prepared.rewritten_queries,  # 重写/分解后的查询（供前端展示）
            kg_entities=prepared.kg_entities or None,  # KG 实体（无则返回 None）
            generation_strategy=self.config.generation_strategy.value,
            pipeline_config=self.config.to_dict(),
        )

    async def prepare(self, question: str) -> PreparedContext:
        """执行检索侧管线（Stage 1-3），返回生成阶段所需的上下文"""
        # Stage 1: 查询变换（可选，按配置生成多个检索查询 / HyDE 文档）
        search_queries, hyde_doc, rewritten_queries = await self._query_transform(question)
        self.metrics["llm_calls"] += self._count_transform_calls()  # 计入查询变换阶段的 LLM 调用

        # Stage 2: 检索（纯检索，零 LLM 调用）
        all_docs = await self._retrieve(search_queries, hyde_doc)

        # Stage 2.5: KG 查找（并入检索结果，增强刑事类问题的结构化知识）
        kg_entities: list[str] = []  # 识别到的罪名实体列表
        if self.config.use_kg:
            kg_entities, kg_docs = await self._kg_lookup(question)
            # KG 精确匹配命中时节省 1 次 LLM 调用
            if kg_docs:
                self.metrics["llm_calls_saved"] += 1
                # 用内容哈希去重，避免 KG 文档与检索结果重复
                existing_keys = {hash(d.page_content[:200]) for d in all_docs}
                for kd in kg_docs:
                    key = hash(kd.page_content[:200])
                    if key not in existing_keys:
                        all_docs.insert(0, kd)  # KG 文档优先置于最前
                        existing_keys.add(key)

        # Stage 3: 重排序（按策略精排，取前 top_k）
        reranked_docs = await self._rerank(question, all_docs)
        if self.config.rerank_strategy == RerankStrategy.LLM:
            self.metrics["llm_calls"] += 1  # LLM 重排计入一次调用
            # 批量重排序节省 N-1 次调用（相比逐文档评分）
            self.metrics["llm_calls_saved"] += max(0, len(all_docs) - 1)

        # 构建来源：截取每个文档前 500 字返回给前端
        sources = [
            SourceDocument(
                content=doc.page_content[:500],
                metadata=doc.metadata,
                score=None,
            )
            for doc in reranked_docs
        ]

        return PreparedContext(
            context=build_context(reranked_docs),
            docs=reranked_docs,
            sources=sources,
            kg_entities=kg_entities,
            rewritten_queries=rewritten_queries,
        )

    # ---- Stage 1: 查询变换 ----

    def _count_transform_calls(self) -> int:
        """统计查询变换阶段的 LLM 调用次数"""
        s = self.config.query_transform
        if s == QueryTransformStrategy.NONE:
            return 0
        if s == QueryTransformStrategy.MULTI_QUERY_HYDE:
            return 2  # 并行但各调一次
        return 1  # MULTI_QUERY / HYDE / DECOMPOSE 各一次

    async def _query_transform(self, question: str) -> tuple[list[str], str | None, list[str] | None]:
        """返回 (search_queries, hyde_doc_or_none, rewritten_queries_or_none)"""
        strategy = self.config.query_transform
        if strategy == QueryTransformStrategy.NONE:
            return [question], None, None  # 无变换：仅原问题

        t0 = time.time()  # 查询变换计时起点
        search_queries = [question]  # 用于检索的查询列表（默认含原问题）
        hyde_doc = None  # HyDE 假设文档
        rewritten_queries = None  # 重写后的查询（返回给前端展示）

        if strategy == QueryTransformStrategy.MULTI_QUERY:
            search_queries = await multi_query_rewrite(question)  # 多查询重写
            rewritten_queries = search_queries

        elif strategy == QueryTransformStrategy.HYDE:
            from app.services.hyde import hyde_transform
            original, hypo = await hyde_transform(question)  # 生成假设文档
            search_queries = [original]  # 原始查询用于 BM25 检索
            hyde_doc = hypo  # 假设文档用于向量检索

        elif strategy == QueryTransformStrategy.DECOMPOSE:
            from app.services.query_rewriter import decompose_query
            sub_qs = await decompose_query(question)  # 分解为子问题
            search_queries = sub_qs
            rewritten_queries = sub_qs

        elif strategy == QueryTransformStrategy.MULTI_QUERY_HYDE:
            import asyncio
            from app.services.hyde import hyde_transform
            # 并行执行多查询重写和 HyDE 生成，降低整体延迟
            mq_task = asyncio.create_task(multi_query_rewrite(question))
            hyde_task = asyncio.create_task(hyde_transform(question))
            mq, (_, hypo) = await asyncio.gather(mq_task, hyde_task)  # 等待两个协程同时完成
            search_queries = mq
            hyde_doc = hypo
            rewritten_queries = mq

        self.metrics["query_rewrite_ms"] = round((time.time() - t0) * 1000, 1)
        return search_queries, hyde_doc, rewritten_queries

    # ---- Stage 2: 检索 ----

    async def _retrieve(self, search_queries: list[str], hyde_doc: str | None) -> list[Document]:
        """执行检索：HyDE / 多查询各自作为一路召回，最后用 RRF 融合多路结果"""
        t0 = time.time()  # 检索计时起点
        retriever = get_hybrid_retriever(self.config.collection_names)  # 混合检索器（BM25+向量）

        ranked_lists: list[list[Document]] = []  # 多路召回的已排序结果

        if hyde_doc:
            # HyDE 作为独立一路：BM25 用原始查询，向量用假设文档
            ranked_lists.append(
                retriever.search_with_split_queries(
                    bm25_query=search_queries[0],
                    vector_query=hyde_doc,
                )
            )

        # 常规多查询检索：每个查询作为一路召回
        for q in search_queries:
            ranked_lists.append(retriever.invoke(q))

        # 多路召回之间做 RRF 融合（而非按查询顺序简单拼接）
        all_docs = rrf_fuse_ranked_lists(ranked_lists, rrf_k=settings.RECALL_RRF_K)

        self.metrics["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        return all_docs

    # ---- Stage 2.5: KG 查找 ----

    async def _kg_lookup(self, question: str) -> tuple[list[str], list[Document]]:
        """从问题中提取罪名并在犯罪知识图谱中查找，返回 (实体列表, KG 文档列表)"""
        t0 = time.time()  # KG 查找计时起点
        try:
            from app.services.kg_service import extract_crime_entities, kg_lookup
            entities = await extract_crime_entities(question)  # 提取罪名实体（先精确匹配，再 LLM 模糊匹配）
            docs = kg_lookup(entities) if entities else []  # 按罪名查询图谱
            self.metrics["kg_lookup_ms"] = round((time.time() - t0) * 1000, 1)
            return entities, docs
        except Exception:
            self.metrics["kg_lookup_ms"] = round((time.time() - t0) * 1000, 1)
            logger.warning("KG 查找失败，已跳过（不影响主流程）", exc_info=True)
            return [], []  # KG 异常时不阻断主流程，返回空

    # ---- Stage 3: 重排序 ----

    async def _rerank(self, question: str, all_docs: list[Document]) -> list[Document]:
        """按配置的重排序策略精排文档，返回前 top_k"""
        strategy = self.config.rerank_strategy
        top_k = self.config.top_k

        if strategy == RerankStrategy.NONE or not all_docs:
            return all_docs[:top_k]  # 不重排或无可排文档时直接截取

        t0 = time.time()  # 重排计时起点
        try:
            if strategy == RerankStrategy.SIMPLE:
                scored = simple_rerank(question, all_docs, top_k=top_k)  # 轻量关键词相似度重排
                reranked = [doc for doc, _ in scored]
            elif strategy == RerankStrategy.LLM:
                scored = await llm_rerank(question, all_docs, top_k=top_k)  # LLM 批量评分重排
                reranked = [doc for doc, _ in scored]
            elif strategy == RerankStrategy.BGE:
                scored = await bge_rerank(question, all_docs, top_k=top_k)  # BGE 交叉编码器精排
                reranked = [doc for doc, _ in scored]
            else:
                reranked = all_docs[:top_k]
            self.metrics["rerank_ms"] = round((time.time() - t0) * 1000, 1)
            return reranked
        except Exception:
            # 重排异常时回退为直接截取，保证主流程可用
            logger.warning("重排序失败，回退为直接截取 top_k", exc_info=True)
            self.metrics["rerank_ms"] = round((time.time() - t0) * 1000, 1)
            return all_docs[:top_k]

    # ---- Stage 4: 生成 ----

    def _select_prompt(self):
        """按生成策略选择提示词模板"""
        strategy = self.config.generation_strategy
        if strategy == GenerationStrategy.COT:
            return LEGAL_COT_PROMPT  # 链式推理模板
        if strategy == GenerationStrategy.STRUCTURED:
            return LEGAL_STRUCTURED_PROMPT  # 结构化法律回答模板
        return LEGAL_QA_PROMPT  # 标准问答模板（含 SELF_REFLECT 的初答阶段）

    async def _generate(
        self,
        question: str,
        docs: list[Document],
        context: str | None = None,
    ) -> tuple[str, bool]:
        """按生成策略调用 LLM 产出答案，返回 (answer, was_corrected)"""
        t0 = time.time()  # 生成计时起点
        context = context if context is not None else build_context(docs)  # 复用检索阶段的上下文
        prompt = self._select_prompt()
        strategy = self.config.generation_strategy

        llm = get_llm()
        chain = prompt | llm  # LCEL 链：提示词模板 + 模型
        response = await chain.ainvoke({"context": context, "question": question})
        answer = response.content
        was_corrected = False  # 是否被反思修正

        # 自我反思
        if strategy == GenerationStrategy.SELF_REFLECT:
            gen_ms = round((time.time() - t0) * 1000, 1)
            self.metrics["generation_ms"] = gen_ms

            t_reflect = time.time()  # 反思阶段计时起点
            from app.services.self_reflect import self_reflect_and_correct
            # 验证初答并修正，最多迭代 SELF_REFLECT_MAX_ITER 轮
            answer, was_corrected = await self_reflect_and_correct(
                question=question,
                initial_answer=answer,
                context=context,
                max_iterations=settings.SELF_REFLECT_MAX_ITER,
            )
            self.metrics["self_reflect_ms"] = round((time.time() - t_reflect) * 1000, 1)
            return answer, was_corrected

        self.metrics["generation_ms"] = round((time.time() - t0) * 1000, 1)
        return answer, was_corrected

    async def stream_generate(
        self,
        question: str,
        docs: list[Document],
        enable_thinking: bool = True,
        context: str | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """流式生成：逐段产出 ("reasoning" | "answer", 文本增量)

        直接使用 OpenAI 兼容客户端（而非 ChatOpenAI），因为 langchain-openai
        会丢弃非标准字段 ``reasoning_content``，导致思考过程不可见。
        """
        t0 = time.time()  # 生成计时起点
        context = context if context is not None else build_context(docs)
        prompt = self._select_prompt()
        messages = convert_to_openai_messages(
            prompt.format_messages(context=context, question=question)
        )

        client = AsyncOpenAI(
            api_key=resolve_api_key(),  # 当前用户自己的 Key
            base_url=settings.SILICONFLOW_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
        )
        request_kwargs = {
            "model": resolve_llm_model(),
            "messages": messages,
            "stream": True,
            # 硅基流动 DeepSeek-V3.2 的非标准参数：开启后返回 reasoning_content
            "extra_body": {"enable_thinking": bool(enable_thinking)},
        }
        if not enable_thinking:
            # 思考模式不支持 temperature，仅在非思考模式下传入
            request_kwargs["temperature"] = settings.LLM_TEMPERATURE

        try:
            stream = await client.chat.completions.create(**request_kwargs)
            try:
                async for chunk in stream:
                    if not chunk.choices:  # 末尾 usage chunk 等无 choices 的分片
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        yield "reasoning", reasoning
                    if delta.content:
                        yield "answer", delta.content
            finally:
                if hasattr(stream, "close"):
                    await stream.close()
        finally:
            self.metrics["generation_ms"] = round((time.time() - t0) * 1000, 1)
            await client.close()


# ===================== 公共辅助函数 =====================

def resolve_collections(collection: str) -> list[str]:
    """将前端传入的 collection 参数解析为实际的集合名列表"""
    if collection == "laws":
        return [settings.LAWS_COLLECTION]  # 仅法律
    elif collection == "cases":
        return [settings.CASES_COLLECTION]  # 仅案例
    else:
        return [settings.LAWS_COLLECTION, settings.CASES_COLLECTION]  # all：法律 + 案例


def build_context(docs: list[Document], max_length: int = 4000) -> str:
    """构建上下文文本 — 在本地 8B 模型有限的上下文窗口(8192 tokens)下，
    显式按文档优先级分配预算：KG 结构化知识 > 法律条文 > 案例。"""
    if not docs:
        return "未找到相关参考资料。"

    # 按 doc_type 优先级排序：kg > law/statute > case > other
    PRIORITY = {"kg": 0, "law": 1, "statute": 1, "case": 2}  # 数值越小优先级越高
    sorted_docs = sorted(docs, key=lambda d: PRIORITY.get(d.metadata.get("doc_type", ""), 3))

    parts = []  # 已纳入上下文的文本段
    total_len = 0  # 累计长度，用于截断控制
    for i, doc in enumerate(sorted_docs, 1):
        source_label = format_source_display(doc.metadata)  # 格式化来源标签（法律名/案号等）
        segment = f"[来源{i}] {source_label}\n{doc.page_content}\n"
        if total_len + len(segment) > max_length:  # 超过预算则停止拼接
            break
        parts.append(segment)
        total_len += len(segment)

    return "\n".join(parts)
