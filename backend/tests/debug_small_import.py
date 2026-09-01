"""调试用最小数据集导入脚本

用途: 用少量真实测试数据（.txt x5、.md x5、.json x5）跑 import_data.py 的导入逻辑，
方便在 VS Code 里打断点调试，避免全量数据太大、执行太慢。

运行（在 backend 目录下）:
    1. 先抽取真实数据（只需一次）:
       python tests/extract_test_data.py
    2. 再跑调试导入:
       python tests/debug_small_import.py

说明:
    - 测试数据来自 backend/tests/data_test/，由 tests/extract_test_data.py 从
      backend/data 真实数据中抽取（非伪造）:
        data_test/laws/    .txt x5（国家法律）+ .md x5（法规/决定）
        data_test/cases/   .json x5（CAIL2018 真实案例，JSONL 格式）
    - 使用独立测试向量库 backend/tests/chroma_db_test/，不影响正式 chroma_db
    - 每次运行前自动清理测试向量库，可反复运行不会重复导入
    - 复用 import_data.py 中的 import_laws / import_cases_jsonl，
      所以在这些函数里打断点即可命中
"""
import os
import shutil
import sys
from pathlib import Path

# ---- 环境配置（必须在 import app.config 之前设置，覆盖默认路径）----
TESTS_DIR = Path(__file__).resolve().parent            # backend/tests/
BASE = TESTS_DIR.parent                                 # backend/
DATA_LAWS = TESTS_DIR / "data_test" / "laws"            # 真实法律数据（已抽取）
DATA_CASES = TESTS_DIR / "data_test" / "cases"          # 真实案例数据（已抽取）
TEST_CHROMA = TESTS_DIR / "chroma_db_test"              # 独立测试向量库

os.environ["LAWS_DIR"] = str(DATA_LAWS)
os.environ["CASES_DIR"] = str(DATA_CASES)
os.environ["CHROMA_PERSIST_DIR"] = str(TEST_CHROMA)
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))  # 使 import_data 可被导入

from app.config import settings
from app.core.vectorstore import get_vectorstore, reset_store_cache, close_client
from import_data import import_laws, import_cases_jsonl


def check_test_data() -> tuple[int, int, int]:
    """校验 data_test 里是否已有真实测试数据（缺数据时提示先运行抽取脚本）"""
    n_txt = len(list(DATA_LAWS.glob("*.txt"))) if DATA_LAWS.exists() else 0
    n_md = len(list(DATA_LAWS.glob("*.md"))) if DATA_LAWS.exists() else 0
    n_json = len(list(DATA_CASES.glob("*.json"))) if DATA_CASES.exists() else 0
    if n_txt == 0 or n_md == 0 or n_json == 0:
        print("=" * 55)
        print("缺少测试数据！请先运行抽取脚本:")
        print("  python tests/extract_test_data.py")
        print(f"  当前数据: laws/.txt x{n_txt}, laws/.md x{n_md}, cases/.json x{n_json}")
        print("=" * 55)
        sys.exit(1)
    return n_txt, n_md, n_json


def reset_test_chroma() -> None:
    """清理测试向量库（仅删测试库，不动正式 chroma_db）"""
    if TEST_CHROMA.exists():
        shutil.rmtree(TEST_CHROMA, ignore_errors=True)
        print(f"  已清理测试向量库: {TEST_CHROMA}")


# ===================== 主流程 =====================

def main() -> None:
    n_txt, n_md, n_json = check_test_data()
    print("=" * 55)
    print("调试用最小数据集导入（真实数据）")
    print(f"  法律测试数据: {DATA_LAWS}（.txt x{n_txt}, .md x{n_md}）")
    print(f"  案例测试数据: {DATA_CASES}（.json x{n_json}）")
    print(f"  测试向量库:   {TEST_CHROMA}")
    print("=" * 55)

    reset_test_chroma()  # 清理测试向量库（每次运行前都清理，避免重复导入）

    # 复用 import_data.py 的导入逻辑（这里打断点会命中）
    reset_store_cache()
    laws_store = get_vectorstore(settings.LAWS_COLLECTION)
    laws_count = import_laws(laws_store, settings.LAWS_DIR)
    print(f"\n法律导入完成: {laws_count} 块")

    cases_store = get_vectorstore(settings.CASES_COLLECTION)
    cases_count = import_cases_jsonl(cases_store, settings.CASES_DIR, max_cases=20)
    print(f"案例导入完成: {cases_count} 块")

    # ---- 验证 ----
    print(f"\n== 验证 ==")
    print(f"laws 集合总数:  {laws_store._collection.count()}")
    print(f"cases 集合总数: {cases_store._collection.count()}")

    # 检索测试（触发向量检索，验证查询链路）
    try:
        q = laws_store.similarity_search("车船税如何缴纳", k=3)
        print(f"laws 检索返回:  {len(q)} 条")
    except Exception as e:
        print(f"laws 检索失败:  {e}")

    close_client()
    print(f"\n完成。测试向量库: {TEST_CHROMA}")


if __name__ == "__main__":
    main()
