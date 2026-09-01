"""从 backend/data 真实数据中抽取少量样本到 backend/tests/data_test/（供调试用）

抽取规则（均为真实数据，不构造）:
    - .txt x5: data/laws/Chinese-Laws/ 中体积最小的 5 部国家法律
    - .md  x5: data/laws/HF_Chinese_Laws/markdown/ 中体积最小的 5 个地方法规
    - .json x5: data/cases/CAIL2018/cail2018_cases.jsonl 中抽取前 15 条真实案例，
                每 3 条写入一个 .json（JSONL 格式，CAIL2018 的 fact+meta 结构）

运行（在 backend 目录下）:
    python tests/extract_test_data.py
"""
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # backend/
DATA = BASE / "data"
DST_LAWS = BASE / "tests" / "data_test" / "laws"
DST_CASES = BASE / "tests" / "data_test" / "cases"

NUM_TXT = 5
NUM_MD = 5
NUM_JSON = 5
RECORDS_PER_JSON = 3


def clean(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


def extract_txt() -> None:
    src = DATA / "laws" / "Chinese-Laws"
    files = sorted(src.glob("*.txt"), key=lambda p: p.stat().st_size)[:NUM_TXT]
    for f in files:
        shutil.copy2(f, DST_LAWS / f.name)
        print(f"  [txt] {f.name} ({f.stat().st_size}B)")


def extract_md() -> None:
    src = DATA / "laws" / "HF_Chinese_Laws" / "markdown"
    files = sorted(src.glob("*.md"), key=lambda p: p.stat().st_size)[:NUM_MD]
    for f in files:
        shutil.copy2(f, DST_LAWS / f.name)
        print(f"  [md ] {f.name} ({f.stat().st_size}B)")


def extract_json() -> None:
    src = DATA / "cases" / "CAIL2018" / "cail2018_cases.jsonl"
    records = []
    with open(src, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= NUM_JSON * RECORDS_PER_JSON:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) < NUM_JSON * RECORDS_PER_JSON:
        print(f"  警告: 源 jsonl 只有 {len(records)} 条可读记录，不足 {NUM_JSON * RECORDS_PER_JSON} 条")
    for j in range(NUM_JSON):
        chunk = records[j * RECORDS_PER_JSON:(j + 1) * RECORDS_PER_JSON]
        dst = DST_CASES / f"案例{j + 1}.json"
        with open(dst, "w", encoding="utf-8") as f:
            for rec in chunk:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [json] {dst.name} ({len(chunk)} 条真实案例)")


def main() -> None:
    print("抽取真实数据样本 -> backend/tests/data_test/")
    clean(DST_LAWS)
    clean(DST_CASES)
    extract_txt()
    extract_md()
    extract_json()
    print(f"完成。测试数据目录: {BASE / 'tests' / 'data_test'}")


if __name__ == "__main__":
    main()
