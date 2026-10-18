"""用户自带 API Key 相关测试"""

import pytest
from fastapi import HTTPException

from app.core.api_key import (
    api_key_context,
    current_api_key,
    key_fingerprint,
    mask_api_key,
    normalize_api_key,
    resolve_api_key,
)


def test_normalize_rejects_bad_key():
    """非 sk- 开头的 Key 直接 401"""
    with pytest.raises(HTTPException) as exc:
        normalize_api_key("not-a-key")
    assert exc.value.status_code == 401


def test_normalize_accepts_valid_key():
    key = "sk-" + "a" * 40
    assert normalize_api_key(f"  {key}  ") == key  # 去除首尾空白
    assert normalize_api_key("") is None  # 空值视为未携带
    assert normalize_api_key(None) is None


def test_resolve_prefers_request_key():
    """请求上下文中的 Key 优先于服务端 .env"""
    request_key = "sk-" + "b" * 40
    with api_key_context(request_key):
        assert current_api_key() == request_key
        assert resolve_api_key() == request_key
    assert current_api_key() is None  # 上下文退出后复位


def test_cache_fingerprint_isolated_per_key():
    """不同 Key 的指纹不同，避免缓存串用客户端"""
    a = "sk-" + "a" * 40
    b = "sk-" + "b" * 40
    assert key_fingerprint(a) != key_fingerprint(b)
    assert len(key_fingerprint(a)) == 16


def test_mask_api_key_hides_body():
    key = "sk-1234567890abcdef"
    masked = mask_api_key(key)
    assert "1234567890abcdef" not in masked
    assert masked.startswith("sk-123")


def test_chat_requires_user_api_key(monkeypatch):
    """缺少 Key 时 /api/chat 在进入业务逻辑前返回 401"""
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "REQUIRE_USER_API_KEY", True)
    client = TestClient(app)
    resp = client.post("/api/chat", json={"question": "民法典合同编的规定？"})
    assert resp.status_code == 401
    assert "SiliconFlow API Key" in resp.json()["detail"]


def test_llm_clients_are_isolated_per_key():
    """不同用户的 Key 不能命中同一个 LLM 客户端缓存"""
    from app.core.llm import clear_llm_cache, get_llm

    key_a = "sk-" + "a" * 40
    key_b = "sk-" + "b" * 40
    clear_llm_cache()
    try:
        with api_key_context(key_a):
            a1 = get_llm()
            a2 = get_llm()  # 同一 Key 复用实例
        with api_key_context(key_b):
            b1 = get_llm()  # 不同 Key 必须是新实例
        assert a1 is a2
        assert a1 is not b1
    finally:
        clear_llm_cache()


def test_embedding_clients_are_isolated_per_key():
    """不同用户的 Key 不能命中同一个 Embedding 客户端缓存"""
    from app.core.embeddings import clear_embed_cache, get_embeddings

    key_a = "sk-" + "a" * 40
    key_b = "sk-" + "b" * 40
    clear_embed_cache()
    try:
        with api_key_context(key_a):
            a1 = get_embeddings()
            a2 = get_embeddings()
        with api_key_context(key_b):
            b1 = get_embeddings()
        assert a1 is a2
        assert a1 is not b1
    finally:
        clear_embed_cache()


def test_dependency_propagates_key_to_endpoint():
    """FastAPI 依赖写入的 ContextVar 必须能被同一请求的处理函数读到"""
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from app.core.api_key import api_key_dependency

    probe = FastAPI()

    @probe.get("/probe", dependencies=[Depends(api_key_dependency)])
    async def probe_endpoint():
        return {"key": current_api_key()}

    key = "sk-" + "c" * 40
    client = TestClient(probe)
    resp = client.get("/probe", headers={"X-SiliconFlow-API-Key": key})
    assert resp.status_code == 200
    assert resp.json()["key"] == key


def test_key_policy_is_public(monkeypatch):
    """密钥策略接口无需携带 Key 即可访问"""
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "REQUIRE_USER_API_KEY", True)
    client = TestClient(app)
    resp = client.get("/api/settings/key-policy")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["require_user_api_key"] is True
    assert data["header"] == settings.API_KEY_HEADER
