"""重排序服务

提供三种重排序策略：
1. simple_rerank — 基于 jieba 分词的 Jaccard 相似度 + 元数据加分（零额外资源）
2. llm_rerank   — 使用 DeepSeek-V3.2 批量评分（单次 LLM 调用，替代逐文档调用）
3. bge_rerank   — 调用硅基流动 BAAI/bge-reranker-v2-m3 精排，
                  只按交叉编码器的相关性分数排序
"""

import re

import httpx
import jieba
from langchain_core.documents import Document

from app.config import settings
from app.core.api_key import resolve_api_key
from app.core.llm import get_llm
from app.core.logging import get_logger

logger = get_logger(__name__)


# ===================== 简单重排序 =====================

def simple_rerank(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """基于关键词重叠 + 元数据加分的轻量重排序"""
    query_tokens = set(jieba.cut(query))

    scored: list[tuple[Document, float]] = []
    for doc in documents:
        doc_tokens = set(jieba.cut(doc.page_content))

        # Jaccard 相似度
        intersection = query_tokens & doc_tokens
        union = query_tokens | doc_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        # 元数据加分
        meta = doc.metadata
        bonus = 0.0
        law_name = meta.get("law_name", "")
        if law_name and law_name in query:
            bonus += 0.2
        article_num = meta.get("article_number", "")
        if article_num and article_num in query:
            bonus += 0.3
        guiding = meta.get("guiding_number", "")
        if guiding and guiding in query:
            bonus += 0.2

        scored.append((doc, jaccard + bonus))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ===================== LLM 重排序（批量单次调用） =====================

_BATCH_RERANK_PROMPT = """请判断以下多段法律文本与用户问题的相关性，为每段文本打分。

用户问题：{query}

{doc_list}

请严格按以下格式输出每段的评分（0-10整数），每行一个，不要输出其他内容：
1: 评分
2: 评分
...

评分标准：0=完全无关，5=部分相关，10=高度相关且直接回答了问题"""


async def llm_rerank(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """使用 LLM 批量评分重排序（单次调用替代逐文档调用）

    优化策略：
    1. 先用 simple_rerank 预筛选到 top_k*2 篇，减少送入 LLM 的文档数
    2. 将所有候选文档拼入一个 prompt，单次 LLM 调用完成评分
    """
    # 预筛选：用 simple_rerank 取 top_k*2 篇候选
    pre_k = min(len(documents), top_k * 2, 8)
    pre_scored = simple_rerank(query, documents, top_k=pre_k)
    candidates = [doc for doc, _ in pre_scored]

    if not candidates:
        return []

    # 构建批量 prompt
    doc_parts = []
    for i, doc in enumerate(candidates, 1):
        text = doc.page_content[:300]
        meta = doc.metadata
        label = meta.get("law_name", "") or meta.get("guiding_number", "") or meta.get("source_file", "")
        doc_parts.append(f"[文本{i}] {label}\n{text}")

    doc_list = "\n\n".join(doc_parts)
    prompt = _BATCH_RERANK_PROMPT.format(query=query, doc_list=doc_list)

    llm = get_llm(temperature=0)
    try:
        resp = await llm.ainvoke(prompt)
        scores = _parse_batch_scores(resp.content, len(candidates))
    except Exception:
        # LLM 重排失败时回退到预筛选排序，保证结果可用
        logger.warning("LLM 重排序失败，回退为简单重排结果", exc_info=True)
        return pre_scored[:top_k]

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _parse_batch_scores(text: str, expected_count: int) -> list[float]:
    """解析批量评分输出，如 '1: 8\n2: 5\n3: 2'"""
    scores = [5.0] * expected_count  # 默认值

    for line in text.strip().split("\n"):
        match = re.match(r"(\d+)\s*[:：]\s*(\d+)", line.strip())
        if match:
            idx = int(match.group(1)) - 1
            score = float(match.group(2))
            if 0 <= idx < expected_count:
                scores[idx] = min(max(score, 0), 10)

    return scores


# ===================== BGE 精排（硅基流动 rerank 模型） =====================

async def _siliconflow_rerank(
    query: str,
    documents: list[Document],
) -> list[tuple[int, float]]:
    """调用硅基流动 rerank 接口，返回按相关性降序的 (文档下标, 相关性分数)"""
    api_key = resolve_api_key()  # 当前用户自己的 Key
    url = f"{settings.SILICONFLOW_BASE_URL.rstrip('/')}/rerank"
    payload = {
        "model": settings.RERANK_MODEL,
        "query": query,
        # 过长文档截断，避免超出 reranker 的上下文窗口
        "documents": [doc.page_content[: settings.RERANK_DOC_MAX_CHARS] for doc in documents],
        "top_n": len(documents),
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=settings.RERANK_TIMEOUT) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            # 记录响应体便于排查 Key / 额度 / 模型名问题
            logger.warning("重排序接口返回 %s: %s", resp.status_code, resp.text[:300])
            resp.raise_for_status()
        data = resp.json()

    ranked: list[tuple[int, float]] = []
    seen: set[int] = set()  # 防止接口返回重复下标
    for item in data.get("results") or []:
        try:
            idx = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(documents) or idx in seen:
            continue
        seen.add(idx)
        ranked.append((idx, float(item.get("relevance_score") or 0.0)))

    if not ranked:
        raise ValueError("重排序接口未返回有效结果")
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


async def bge_rerank(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """BGE 精排：只按 ``BAAI/bge-reranker-v2-m3`` 的相关性分数排序

    召回层的多路 RRF 融合已完成候选筛选，这里让交叉编码器独立决定最终顺序，
    不再与检索排名做二次融合。接口异常时回退为 simple_rerank，保证主流程可用。
    """
    if not documents:
        return []

    try:
        model_ranking = await _siliconflow_rerank(query, documents)
    except Exception:
        # 网络 / Key / 额度异常时回退，不阻断问答主流程
        logger.warning("BGE 精排失败，回退为简单重排", exc_info=True)
        return simple_rerank(query, documents, top_k=top_k)

    # model_ranking 已按 relevance_score 降序，直接取前 top_k
    return [(documents[idx], score) for idx, score in model_ranking[:top_k]]
