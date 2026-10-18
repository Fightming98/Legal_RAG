"""请求级模型配置测试（不依赖外部 API）"""

import pytest


def test_resolve_model_defaults():
    """未指定请求头时，解析结果应等于 .env / 代码中的默认模型"""
    from app.config import settings
    from app.core.model_config import (
        resolve_embedding_model,
        resolve_llm_model,
        resolve_rerank_model,
    )

    assert resolve_llm_model() == settings.LLM_MODEL
    assert resolve_embedding_model() == settings.EMBEDDING_MODEL
    assert resolve_rerank_model() == settings.RERANK_MODEL


def test_model_config_context_overrides_and_resets():
    """上下文中的自定义模型优先，退出上下文后恢复默认值"""
    from app.config import settings
    from app.core.model_config import (
        model_config_context,
        resolve_embedding_model,
        resolve_llm_model,
        resolve_rerank_model,
    )

    with model_config_context(
        llm="Qwen/Qwen3-8B",
        embedding="BAAI/bge-large-zh-v1.5",
        rerank="BAAI/bge-reranker-v2-m3",
    ):
        assert resolve_llm_model() == "Qwen/Qwen3-8B"
        assert resolve_embedding_model() == "BAAI/bge-large-zh-v1.5"
        assert resolve_rerank_model() == "BAAI/bge-reranker-v2-m3"

    assert resolve_llm_model() == settings.LLM_MODEL
    assert resolve_embedding_model() == settings.EMBEDDING_MODEL
    assert resolve_rerank_model() == settings.RERANK_MODEL


def test_normalize_model_id():
    """模型 ID 去除首尾空格；空值视为未指定；非法字符报 422"""
    from fastapi import HTTPException

    from app.core.model_config import normalize_model_id

    assert normalize_model_id(" deepseek-ai/DeepSeek-V3.2 ", "X") == "deepseek-ai/DeepSeek-V3.2"
    assert normalize_model_id("", "X") is None
    assert normalize_model_id(None, "X") is None
    with pytest.raises(HTTPException) as exc_info:
        normalize_model_id("bad model", "X")
    assert exc_info.value.status_code == 422


def test_classify_model_id():
    """模型设置页按 ID 推断用途分类"""
    from app.core.model_config import classify_model_id

    assert classify_model_id("deepseek-ai/DeepSeek-V3.2") == "llm"
    assert classify_model_id("Qwen/Qwen3-8B") == "llm"
    assert classify_model_id("BAAI/bge-m3") == "embedding"
    assert classify_model_id("Qwen/Qwen3-Embedding-8B") == "embedding"
    assert classify_model_id("BAAI/bge-reranker-v2-m3") == "rerank"
    assert classify_model_id("Kwai-Kolors/Kolors") == "other"
