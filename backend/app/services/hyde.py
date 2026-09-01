"""HyDE (Hypothetical Document Embeddings) 查询变换

生成假设性法律文档用于向量检索，原始查询用于 BM25 检索。
"""

from app.core.llm import get_llm
from app.core.logging import get_logger
from app.services.prompts import HYDE_PROMPT
from app.config import settings

logger = get_logger(__name__)


async def hyde_transform(question: str) -> tuple[str, str | None]:
    """
    生成假设性法律文档

    Returns:
        (original_question, hypothetical_document)
    """
    try:
        llm = get_llm(temperature=settings.HYDE_TEMPERATURE)
        chain = HYDE_PROMPT | llm
        response = await chain.ainvoke({"question": question})
        hypothetical_doc = response.content.strip()
        if len(hypothetical_doc) < 10:
            return question, None
        return question, hypothetical_doc
    except Exception:
        # HyDE 只是增强手段，失败时回退为仅用原始查询，不阻断主流程
        logger.warning("HyDE 生成失败，回退为仅用原始查询", exc_info=True)
        return question, None
