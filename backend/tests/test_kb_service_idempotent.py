"""kb_service 上传/重建与 import_data 共用确定性 ID 的测试

全部使用假 Embedding + 内存 Chroma，不访问外部 API。
"""
import asyncio
import json
import sys
import uuid
from pathlib import Path

import chromadb
import pytest
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.services import kb_service
from app.services.kb_importer import (
    LegacyIndexError,
    build_law_chunks,
    import_cases_jsonl,
    import_laws,
)


class FakeEmbeddings(Embeddings):
    """固定维度的假 Embedding，避免测试依赖 SiliconFlow API"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text) % 11), float(i % 7), 1.0]
            for i, text in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 11), 0.0, 1.0]


def _make_store(collection_name: str, client=None) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=FakeEmbeddings(),
        client=client or chromadb.EphemeralClient(),
    )


def _patch_dirs(monkeypatch, tmp_path):
    laws_dir = tmp_path / "laws"
    cases_dir = tmp_path / "cases"
    laws_dir.mkdir()
    cases_dir.mkdir()
    monkeypatch.setattr(kb_service.settings, "LAWS_DIR", str(laws_dir))
    monkeypatch.setattr(kb_service.settings, "CASES_DIR", str(cases_dir))
    return laws_dir, cases_dir


def _use_unique_collections(monkeypatch):
    """每个测试使用独立集合名，避免内存 Chroma 在测试之间互相污染"""
    suffix = uuid.uuid4().hex[:8]
    monkeypatch.setattr(kb_service.settings, "LAWS_COLLECTION", f"test_laws_{suffix}")
    monkeypatch.setattr(kb_service.settings, "CASES_COLLECTION", f"test_cases_{suffix}")


def _patch_stores(monkeypatch):
    _use_unique_collections(monkeypatch)
    stores = {}

    def fake_get_vectorstore(name):
        if name not in stores:
            stores[name] = _make_store(name)
        return stores[name]

    monkeypatch.setattr(kb_service, "get_vectorstore", fake_get_vectorstore)
    return stores


def test_upload_law_is_idempotent_and_matches_import(monkeypatch, tmp_path):
    laws_dir, _ = _patch_dirs(monkeypatch, tmp_path)
    _patch_stores(monkeypatch)
    store = kb_service.get_vectorstore(kb_service.settings.LAWS_COLLECTION)

    content = (
        "中华人民共和国测试法\n"
        "第一条 为了测试上传幂等，制定本法。\n"
        "第二条 本法自公布之日起施行。"
    ).encode("utf-8")

    first = asyncio.run(kb_service.upload_document("测试法.txt", content, "law"))
    assert first.chunk_count > 0
    count_after_first = store._collection.count()

    second = asyncio.run(kb_service.upload_document("测试法.txt", content, "law"))
    assert second.chunk_count == first.chunk_count
    assert store._collection.count() == count_after_first

    # 上传与批量导入必须得到相同的 ID：再跑一次 import_laws 不应新增任何分块
    assert import_laws(store, str(laws_dir)) == 0

    # 内容变短后重新上传：相同位置的块覆盖更新，多出来的旧块应被清理
    old_ids = set(store._collection.get(include=[])["ids"])
    shorter = "中华人民共和国测试法\n第一条 为了测试上传幂等，制定本法。"
    result = asyncio.run(kb_service.upload_document("测试法.txt", shorter.encode("utf-8"), "law"))
    expected = len(build_law_chunks(shorter, "测试法.txt", "测试法.txt"))
    assert result.chunk_count == expected
    assert store._collection.count() == expected
    assert set(store._collection.get(include=[])["ids"]) <= old_ids


def test_upload_jsonl_case_matches_batch_import(monkeypatch, tmp_path):
    _, cases_dir = _patch_dirs(monkeypatch, tmp_path)
    _patch_stores(monkeypatch)
    store = kb_service.get_vectorstore(kb_service.settings.CASES_COLLECTION)

    rows = [
        {
            "fact": f"被告人张三实施第{i}起盗窃行为。" + "案情描述。" * 20,
            "accusation": "盗窃",
        }
        for i in range(2)
    ]
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows).encode("utf-8")

    first = asyncio.run(kb_service.upload_document("案例.json", content, "case"))
    assert first.chunk_count > 0
    count_after_first = store._collection.count()

    # 重复上传不新增；随后批量导入也应全部跳过
    asyncio.run(kb_service.upload_document("案例.json", content, "case"))
    assert store._collection.count() == count_after_first
    assert import_cases_jsonl(store, str(cases_dir)) == 0

    # 文件变短（删掉第二条案例）后重新上传：第一条保留，第二条的块被清理
    old_ids = set(store._collection.get(include=[])["ids"])
    one_row = json.dumps(rows[0], ensure_ascii=False).encode("utf-8")
    result = asyncio.run(kb_service.upload_document("案例.json", one_row, "case"))
    new_ids = set(store._collection.get(include=[])["ids"])
    assert result.chunk_count > 0
    assert new_ids < old_ids
    assert import_cases_jsonl(store, str(cases_dir)) == 0


def test_upload_rejects_legacy_random_id_data(monkeypatch, tmp_path):
    laws_dir, _ = _patch_dirs(monkeypatch, tmp_path)
    _patch_stores(monkeypatch)
    store = kb_service.get_vectorstore(kb_service.settings.LAWS_COLLECTION)
    store.add_texts(["旧版随机 ID 数据"], ids=["legacy-random-id"])

    with pytest.raises(LegacyIndexError):
        asyncio.run(
            kb_service.upload_document(
                "测试法.txt", "中华人民共和国测试法\n第一条 测试。".encode("utf-8"), "law"
            )
        )
    # 检测失败时不应写入文件
    assert not (laws_dir / "测试法.txt").exists()


def test_rebuild_uses_deterministic_ids(monkeypatch, tmp_path):
    _use_unique_collections(monkeypatch)
    laws_dir, cases_dir = _patch_dirs(monkeypatch, tmp_path)
    (laws_dir / "测试法.txt").write_text(
        "中华人民共和国测试法\n"
        "第一条 为了测试重建索引，制定本法。\n"
        "第二条 本法自公布之日起施行。",
        encoding="utf-8",
    )
    (cases_dir / "cases.jsonl").write_text(
        json.dumps(
            {"fact": "被告人李四实施盗窃行为。" + "案情描述。" * 20, "accusation": "盗窃"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    client = chromadb.EphemeralClient()
    stores = {}

    def fake_get_vectorstore(name):
        if name not in stores:
            stores[name] = _make_store(name, client)
        return stores[name]

    def fake_drop_collections(*, log=None):
        for name in list(stores):
            try:
                client.delete_collection(name)
            except Exception:
                pass
            stores.pop(name, None)
        return True

    monkeypatch.setattr(kb_service, "get_vectorstore", fake_get_vectorstore)
    monkeypatch.setattr(kb_service, "drop_collections", fake_drop_collections)

    # 预置一条旧版随机 ID 数据，重建后必须消失
    fake_get_vectorstore(kb_service.settings.LAWS_COLLECTION).add_texts(
        ["旧版随机 ID 数据"], ids=["legacy-random-id"]
    )

    stats = asyncio.run(kb_service.rebuild_index())
    assert stats.laws_chunks > 0
    assert stats.cases_chunks > 0

    laws_store = fake_get_vectorstore(kb_service.settings.LAWS_COLLECTION)
    result = laws_store._collection.get(include=["metadatas"])
    assert "legacy-random-id" not in result["ids"]
    assert all((metadata or {}).get("import_version") for metadata in result["metadatas"])

    # 再次重建结果一致（不产生重复）
    stats_again = asyncio.run(kb_service.rebuild_index())
    assert stats_again.laws_chunks == stats.laws_chunks
    assert stats_again.cases_chunks == stats.cases_chunks
