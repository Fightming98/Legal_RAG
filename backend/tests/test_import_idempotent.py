"""import_data 幂等/断点续传测试（使用假 Embedding，不访问外部 API）"""
import json
import sys
from pathlib import Path

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from scripts import import_data
from scripts.import_data import import_cases_jsonl, import_laws


class FakeEmbeddings(Embeddings):
    """固定维度的假 Embedding，避免测试依赖 SiliconFlow API"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text) % 11), float(i % 7), 1.0]
            for i, text in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 11), 0.0, 1.0]


def _make_store(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=FakeEmbeddings(),
        client=chromadb.EphemeralClient(),
    )


def test_laws_import_is_idempotent(tmp_path):
    laws_dir = tmp_path / "laws"
    laws_dir.mkdir()
    (laws_dir / "测试法.txt").write_text(
        "中华人民共和国测试法\n"
        "第一条 为了测试幂等导入，制定本法。\n"
        "第二条 本法自公布之日起施行。",
        encoding="utf-8",
    )

    store = _make_store("test_laws_idempotent")
    first = import_laws(store, str(laws_dir))
    assert first > 0
    count_after_first = store._collection.count()

    second = import_laws(store, str(laws_dir))
    assert second == 0
    assert store._collection.count() == count_after_first


def test_laws_import_repairs_missing_chunk(tmp_path):
    laws_dir = tmp_path / "laws"
    laws_dir.mkdir()
    (laws_dir / "测试法.txt").write_text(
        "中华人民共和国测试法\n"
        "第一条 为了测试断点续传，制定本法。\n"
        "第二条 本法自公布之日起施行。",
        encoding="utf-8",
    )

    store = _make_store("test_laws_repair")
    import_laws(store, str(laws_dir))
    count_before = store._collection.count()

    # 模拟上次导入在写入过程中中断：删掉一个分块后重跑，应只补回缺失的块
    one_id = store._collection.get(limit=1, include=[])["ids"][0]
    store._collection.delete(ids=[one_id])

    added = import_laws(store, str(laws_dir))
    assert added == 1
    assert store._collection.count() == count_before


def test_rebuild_drops_collections(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(import_data.settings, "CHROMA_PERSIST_DIR", str(chroma_dir))

    client = chromadb.PersistentClient(path=str(chroma_dir))
    client.get_or_create_collection("laws")
    client.get_or_create_collection("cases")
    client.close()

    import_data._drop_collections()

    client = chromadb.PersistentClient(path=str(chroma_dir))
    names = {
        collection.name if hasattr(collection, "name") else collection
        for collection in client.list_collections()
    }
    client.close()
    assert "laws" not in names
    assert "cases" not in names


def test_cases_import_resumes_without_duplicates(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    rows = [
        {
            "fact": f"被告人张三实施第{i}起盗窃行为。" + "案情描述。" * 20,
            "accusation": "盗窃",
        }
        for i in range(5)
    ]
    (cases_dir / "cases.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    store = _make_store("test_cases_resume")
    first = import_cases_jsonl(store, str(cases_dir), max_cases=3)
    assert first > 0
    count_after_first = store._collection.count()

    # 模拟上次运行只导入了 3 条后中断：重跑应跳过已有案例，只补剩余 2 条
    second = import_cases_jsonl(store, str(cases_dir), max_cases=3)
    assert second > 0
    count_after_second = store._collection.count()
    assert count_after_second > count_after_first

    # 所有案例都已存在，再跑不应新增
    third = import_cases_jsonl(store, str(cases_dir), max_cases=3)
    assert third == 0
    assert store._collection.count() == count_after_second
