"""流式问答 SSE 测试（不访问真实模型）"""

from fastapi.testclient import TestClient

import app.api.chat as chat_module
from app.config import settings
from app.main import app

VALID_KEY = "sk-" + "a" * 40


async def _fake_stream_rag_query(**kwargs):
    """模拟流式管线：依次产出各类事件"""
    yield "stage", {"stage": "prepare", "message": "正在检索资料…"}
    yield "sources", {
        "sources": [],
        "rewritten_queries": None,
        "kg_entities": None,
        "system_before": None,
        "metrics": {},
    }
    yield "reasoning", {"delta": "先分析法律依据"}
    yield "answer", {"delta": "应承担赔偿责任"}
    yield "metrics", {"total_ms": 1.0}
    yield "done", {"generation_strategy": "standard"}


def test_stream_requires_user_api_key(monkeypatch):
    """缺少 Key 时 /api/chat/stream 直接返回 401"""
    monkeypatch.setattr(settings, "REQUIRE_USER_API_KEY", True)
    client = TestClient(app)
    resp = client.post("/api/chat/stream", json={"question": "测试问题"})
    assert resp.status_code == 401
    assert "SiliconFlow API Key" in resp.json()["detail"]


def test_stream_emits_sse_events(monkeypatch):
    """携带 Key 时按 SSE 协议推送思考与回答事件"""
    monkeypatch.setattr(settings, "REQUIRE_USER_API_KEY", True)
    captured = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        async for item in _fake_stream_rag_query(**kwargs):
            yield item

    monkeypatch.setattr(chat_module, "stream_rag_query", fake_stream)

    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "测试问题", "enable_thinking": True},
        headers={settings.API_KEY_HEADER: VALID_KEY},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())

    assert "event: reasoning" in body
    assert "先分析法律依据" in body
    assert "event: answer" in body
    assert "应承担赔偿责任" in body
    assert "event: done" in body
    assert captured["enable_thinking"] is True  # 请求参数正确透传到流式服务


def test_user_key_visible_during_stream(monkeypatch):
    """流式响应执行期间，依赖写入的请求级 Key 仍然有效（未被提前复位）"""
    from app.core.api_key import current_api_key

    async def fake_stream(**kwargs):
        yield "answer", {"delta": current_api_key() or "MISSING"}
        yield "done", {}

    monkeypatch.setattr(chat_module, "stream_rag_query", fake_stream)

    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "测试问题"},
        headers={settings.API_KEY_HEADER: VALID_KEY},
    ) as resp:
        body = "".join(resp.iter_text())

    assert VALID_KEY in body  # 用户 Key 在生成器内部可见
    assert "MISSING" not in body


def test_sse_frame_escapes_newlines():
    """SSE 帧以空行结尾，数据内的换行被 JSON 转义"""
    frame = chat_module._sse("answer", {"delta": "第一行\n第二行"})
    assert frame.startswith("event: answer\ndata: ")
    assert frame.endswith("\n\n")
    assert "\\n" in frame  # JSON 中的换行转义，不会截断事件
