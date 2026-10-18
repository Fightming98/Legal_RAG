"""混合检索器 — BM25 稀疏检索 + 向量稠密检索 + RRF 融合

- ``HybridRetriever`` 负责单次查询内部的多通道融合（BM25 + 各集合向量检索）；
- ``rrf_fuse_ranked_lists`` 负责多查询/多路召回之间的 RRF 融合（供管线层调用）。
"""

import jieba
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from app.core.logging import get_logger
from app.core.vectorstore import get_vectorstore
from app.config import settings

logger = get_logger(__name__)


def _is_dimension_mismatch(exc: Exception) -> bool:
    """判断异常是否由「嵌入模型维度与集合维度不一致」引起"""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "dimension" in name or "dimension" in text or "dimensionality" in text


class BM25ChineseRetriever(BaseRetriever):
    """基于 jieba 分词的 BM25 中文检索器"""

    documents: list[Document] = []
    tokenized_corpus: list[list[str]] = []
    bm25: BM25Okapi | None = None
    k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, documents: list[Document], k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.documents = documents  # 原始文档列表
        self.k = k  # 返回的 top-k 条数
        # jieba 分词构建语料库
        self.tokenized_corpus = [  # 分词后的语料库（每篇文档对应一个词列表）
            list(jieba.cut(doc.page_content)) for doc in documents
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)  # BM25 索引对象

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = list(jieba.cut(query))  # 对查询进行 jieba 分词
        scores = self.bm25.get_scores(tokenized_query)  # 查询与各文档的 BM25 相似度分数
        # 获取 top-k 索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[  # 按分数降序取 top-k 的文档索引
            : self.k
        ]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


class HybridRetriever(BaseRetriever):
    """混合检索器：BM25 + 向量检索 + RRF 融合"""

    bm25_retriever: BM25ChineseRetriever | None = None  # BM25 子检索器（懒加载，加载语料后赋值）
    collection_names: list[str] = ["laws", "cases"]  # 参与混合检索的集合名列表
    bm25_weight: float = settings.BM25_WEIGHT  # BM25（稀疏）检索融合权重
    vector_weight: float = settings.VECTOR_WEIGHT  # 向量（稠密）检索融合权重
    k: int = settings.RETRIEVAL_TOP_K  # 最终返回的 top-k 条数
    all_documents: list[Document] = []  # 全部文档缓存（用于构建 BM25 索引）

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, collection_names: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        if collection_names:
            self.collection_names = collection_names  # 覆盖默认检索集合
        self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        """从向量库加载文档构建 BM25 索引"""
        all_docs = []  # 累积各集合分页读取的全部文档
        BATCH = 1000  # 分页大小：一次性 get 全部记录会撞上 SQLite 绑定变量上限
        # （too many SQL variables），必须分页读取。如 laws 集合 3.8 万条记录
        # 全量 get 会抛 chromadb.errors.InternalError。
        for name in self.collection_names:
            try:
                store = get_vectorstore(name)  # 获取该集合对应的向量库
                offset = 0  # 分页偏移量（从第 0 条开始）
                while True:
                    # 分页读取：limit=1000 一批，offset 逐批推进
                    result = store._collection.get(  # 分页查询本批文档
                        include=["documents", "metadatas"],
                        limit=BATCH,
                        offset=offset,
                    )
                    docs = result.get("documents") or []  # 本批文档文本
                    if not docs:  # 没有更多数据，退出循环
                        break
                    metas = result.get("metadatas") or [{}] * len(docs)  # 本批文档元数据（缺省补空字典）
                    for doc_text, meta in zip(docs, metas):
                        all_docs.append(Document(page_content=doc_text, metadata=meta or {}))
                    offset += BATCH
                    if len(docs) < BATCH:  # 最后一批不足 BATCH 条，说明取完了
                        break
            except Exception:
                # 记录日志暴露问题，避免异常被静默吞掉导致该集合被整体跳过
                logger.warning("加载集合 %s 的 BM25 语料失败，已跳过", name, exc_info=True)
                continue
        self.all_documents = all_docs  # 保存全部加载的文档
        if all_docs:
            self.bm25_retriever = BM25ChineseRetriever(documents=all_docs, k=self.k)  # 基于全部文档构建 BM25 检索器

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """RRF (Reciprocal Rank Fusion) 混合检索"""
        results_map: dict[str, dict] = {}  # content_hash -> {doc, rrf_score}
        rrf_k = 60  # RRF 常数

        # BM25 检索
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(query)  # BM25 检索结果
            for rank, doc in enumerate(bm25_results):
                # 以文档内容前 200 字符的哈希作为去重键：同一文档同时被
                # BM25 与向量检索命中时，累加二者的 RRF 分数而非重复返回
                key = hash(doc.page_content[:200])  # 文档去重键
                score = self.bm25_weight / (rrf_k + rank + 1)  # BM25 检索贡献的 RRF 分数
                if key in results_map:
                    results_map[key]["score"] += score
                else:
                    results_map[key] = {"doc": doc, "score": score}

        # 向量检索
        for name in self.collection_names:
            try:
                vec_results = get_vectorstore(name).similarity_search(query, k=self.k)  # 向量检索结果
                for rank, doc in enumerate(vec_results):
                    key = hash(doc.page_content[:200])  # 文档去重键
                    score = self.vector_weight / (rrf_k + rank + 1)  # 向量检索贡献的 RRF 分数
                    if key in results_map:
                        results_map[key]["score"] += score
                    else:
                        results_map[key] = {"doc": doc, "score": score}
            except Exception as exc:
                if _is_dimension_mismatch(exc):
                    # 用户切换嵌入模型后维度可能与旧索引不一致；静默跳过会让
                    # 检索退化成纯 BM25，这里直接抛出可读错误引导用户处理。
                    raise RuntimeError(
                        "当前嵌入模型与知识库索引的向量维度不一致。"
                        "请在「模型设置」中恢复默认嵌入模型（BAAI/bge-m3），"
                        "或使用当前嵌入模型重建知识库索引。"
                    ) from exc
                logger.warning("向量检索集合 %s 失败，已跳过该集合", name, exc_info=True)
                continue

        # 按 RRF 分数排序
        sorted_results = sorted(results_map.values(), key=lambda x: x["score"], reverse=True)  # 按 RRF 分数降序排列
        return [item["doc"] for item in sorted_results[: self.k]]

    def search_with_split_queries(
        self, bm25_query: str, vector_query: str
    ) -> list[Document]:
        """分离式检索：BM25 使用一个查询，向量检索使用另一个查询

        用于 HyDE 场景：bm25_query 为原始问题，vector_query 为假设文档。
        """
        results_map: dict[int, dict] = {}  # content_hash -> {doc, rrf_score}
        rrf_k = 60  # RRF 常数

        # BM25 检索（使用原始查询）
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(bm25_query)  # BM25 检索结果（原始查询）
            for rank, doc in enumerate(bm25_results):
                key = hash(doc.page_content[:200])  # 文档去重键
                score = self.bm25_weight / (rrf_k + rank + 1)  # BM25 检索贡献的 RRF 分数
                if key in results_map:
                    results_map[key]["score"] += score
                else:
                    results_map[key] = {"doc": doc, "score": score}

        # 向量检索（使用假设文档 / HyDE 文本）
        for name in self.collection_names:
            try:
                vec_results = get_vectorstore(name).similarity_search(vector_query, k=self.k)  # 向量检索结果（HyDE 文本）
                for rank, doc in enumerate(vec_results):
                    key = hash(doc.page_content[:200])  # 文档去重键
                    score = self.vector_weight / (rrf_k + rank + 1)  # 向量检索贡献的 RRF 分数
                    if key in results_map:
                        results_map[key]["score"] += score
                    else:
                        results_map[key] = {"doc": doc, "score": score}
            except Exception as exc:
                if _is_dimension_mismatch(exc):
                    raise RuntimeError(
                        "当前嵌入模型与知识库索引的向量维度不一致。"
                        "请在「模型设置」中恢复默认嵌入模型（BAAI/bge-m3），"
                        "或使用当前嵌入模型重建知识库索引。"
                    ) from exc
                logger.warning("分离式向量检索集合 %s 失败，已跳过该集合", name, exc_info=True)
                continue

        sorted_results = sorted(results_map.values(), key=lambda x: x["score"], reverse=True)  # 按 RRF 分数降序排列
        return [item["doc"] for item in sorted_results[: self.k]]


def get_hybrid_retriever(
    collection_names: list[str] | None = None,
) -> HybridRetriever:
    """获取混合检索器"""
    names = collection_names or ["laws", "cases"]  # 默认检索法条与案例两个集合
    return HybridRetriever(collection_names=names)


def rrf_fuse_ranked_lists(
    ranked_lists: list[list[Document]],
    *,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[Document]:
    """对多路召回结果做加权 RRF（倒数排名融合）

    每个子列表代表一路召回（例如多查询改写中的每个查询、HyDE 检索），
    列表内部已按相关性降序。对第 rank 名（从 0 开始）的文档贡献
    ``weight / (rrf_k + rank + 1)``；同一文档被多路命中时分数累加。
    返回按融合分数降序、内容去重后的文档列表。

    Args:
        ranked_lists: 多路召回的文档列表，每个列表内部已按相关性降序。
        rrf_k: RRF 常数，越大排名差异越平滑。
        weights: 每路召回的权重；缺省时所有路权重相同（1.0）。
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused: dict[int, dict] = {}  # content_hash -> {doc, rrf_score}
    for route_idx, docs in enumerate(ranked_lists):
        # 权重按路对齐；缺失时按 1.0 处理，避免调用方传入短列表时报错
        weight = weights[route_idx] if route_idx < len(weights) else 1.0
        for rank, doc in enumerate(docs):
            key = hash(doc.page_content[:200])  # 文档去重键（与混合检索器保持一致）
            score = weight / (rrf_k + rank + 1)
            if key in fused:
                fused[key]["score"] += score  # 多路命中：累加 RRF 分数
            else:
                fused[key] = {"doc": doc, "score": score}

    sorted_results = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results]
