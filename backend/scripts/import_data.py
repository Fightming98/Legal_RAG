"""数据导入脚本（断点续传 / 幂等）

用法:
    cd backend
    python -m scripts.import_data                 # 增量导入：已存在的分块自动跳过
    python -m scripts.import_data --rebuild       # 先清空 laws/cases 集合，再全量重建
    python -m scripts.import_data --max-cases 1000
    python -m scripts.import_data --all-cases     # 本次不限制新增案例数

支持:
- data/laws/ 下的 .txt/.md 文件（递归子目录）
- data/cases/ 下的 .json/.jsonl (JSONL) 文件（递归子目录）
- data/cases/ 下的 .txt/.md 整文件案例

幂等/续传原理:
- 每个分块使用「来源相对路径 + 分块位置」生成确定性 ID（uuid5）；
- 每批写入前先查询库中已存在的 ID，只对缺失或内容变化的分块调用 Embedding；
- 因此中断后直接重跑即可从断点继续，不会重复写入，也不会重复消耗 Embedding 额度。

实现说明:
- 核心逻辑位于 app/services/kb_importer.py，与前端「上传文档 / 重建索引」
  共用同一套 ID、元数据指纹和查重逻辑；
- 分块策略/元数据结构变更时，请递增 kb_importer.IMPORT_VERSION；
- 若集合中存在旧版本（随机 ID）数据，首次升级请加 --rebuild 清库；
- --rebuild 会删除 laws/cases 两个集合，请先停止正在运行的后端服务。
"""

import argparse
import os
import sys

# 将 backend 目录加入 Python path，保证 `python -m scripts.import_data` 能导入 app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.core.vectorstore import close_client, get_vectorstore, reset_store_cache
from app.services.kb_importer import (
    IMPORT_VERSION,
    drop_collections,
    has_legacy_data,
    import_case_files,
    import_cases_jsonl,
    import_laws,
    import_metadata,
)


def _drop_collections() -> bool:
    """删除 laws/cases 集合（--rebuild 使用），返回是否全部删除成功"""
    return drop_collections(log=print)


def _has_legacy_data(store, label: str) -> bool:
    """检测集合中是否存在旧版本（无 import_version）的数据"""
    return has_legacy_data(store, label, log=print)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="法律 RAG 数据导入（确定性 ID + 断点续传/幂等）"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="导入前删除 laws/cases 集合，全量重建（会清空现有向量）",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=5000,
        help="本次最多新增的案例数，已存在案例不占用额度（默认 5000）",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="本次不限制新增案例数（等价于 --max-cases 无限大）",
    )
    args = parser.parse_args(argv)

    if args.max_cases < 0:
        parser.error("--max-cases 不能为负数")
    max_cases = None if args.all_cases else args.max_cases

    print("=" * 55)
    print("法律 RAG 系统 — 数据导入（断点续传 / 幂等）")
    print("=" * 55)
    metadata = import_metadata()
    print(
        f"导入版本: {IMPORT_VERSION}，"
        f"Embedding: {metadata['embedding_model']}"
    )
    print("已存在的分块会自动跳过，中断后可直接重跑。")

    if args.rebuild:
        print("\n[重建] 删除 laws/cases 集合...")
        if not _drop_collections():
            print("\n重建中止：集合删除失败，请先停止后端服务后重试（避免旧数据残留）。")
            return
    else:
        reset_store_cache()  # 清空向量库缓存，确保从磁盘重新打开最新数据

    laws_store = get_vectorstore(settings.LAWS_COLLECTION)  # 获取/创建 laws 集合
    cases_store = get_vectorstore(settings.CASES_COLLECTION)  # 获取/创建 cases 集合

    # 旧版数据使用随机 UUID，无法与确定性 ID 去重；直接续导会产生重复向量
    if not args.rebuild and (
        _has_legacy_data(laws_store, "laws")
        | _has_legacy_data(cases_store, "cases")
    ):
        print("\n检测到旧版数据，已中止导入。请先执行以下任一操作后重试：")
        print("  1) python -m scripts.import_data --rebuild   # 清空集合并全量重建")
        print("  2) 前端「知识库管理」→「重建索引」           # 使用同一套确定性 ID 重建")
        print("  3) python scripts/cleanup_chroma.py          # 先备份/清理整个 chroma_db")
        close_client()
        return

    # --- 法律条文 ---
    print(f"\n[1/3] 导入法律条文 ({settings.LAWS_DIR})")
    laws_count = import_laws(laws_store, settings.LAWS_DIR, log=print)

    # --- JSONL 案例 ---
    print(f"\n[2/3] 导入 JSONL 案例 ({settings.CASES_DIR})")
    cases_count = import_cases_jsonl(
        cases_store, settings.CASES_DIR, max_cases=max_cases, log=print
    )

    # --- 整文件案例 ---
    print(f"\n[3/3] 导入 TXT/MD 案例 ({settings.CASES_DIR})")
    case_files_count = import_case_files(cases_store, settings.CASES_DIR, log=print)

    print(f"\n{'=' * 55}")
    print(
        f"本次导入完成: 法律条文新增 {laws_count} 块，"
        f"案例新增 {cases_count + case_files_count} 块"
    )
    print(
        f"集合总数: laws={laws_store._collection.count()}，"
        f"cases={cases_store._collection.count()}"
    )
    print(f"向量数据库: {settings.CHROMA_PERSIST_DIR}")

    close_client()  # 显式关闭客户端，确保 sqlite WAL 检查点落盘


if __name__ == "__main__":
    main()
