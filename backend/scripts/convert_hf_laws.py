"""将 HuggingFace 法律法规库 (parquet) 转换为 Markdown 文本文件

用法:
    cd backend
    python -m scripts.convert_hf_laws                # 转换全部
    python -m scripts.convert_hf_laws --limit 100    # 只转前 100 条（测试用）

背景:
    import_data.py 只递归读取 data/laws/ 下的 .txt/.md 文件，而 HF 数据集
    twang2218/chinese-law-and-regulations 是 parquet 格式，无法被直接导入。
    本脚本将其 content 字段导出为逐部法规的 .md 文件，供 import_data.py 扫描。

输出:
    data/laws/HF_Chinese_Laws/markdown/*.md
"""

import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
HF_DIR = BASE_DIR / "data" / "laws" / "HF_Chinese_Laws"
PARQUET_GLOB = "data/train-*.parquet"
OUTPUT_DIR = HF_DIR / "markdown"

# Windows 文件名不允许的字符
_ILLEGAL = re.compile(r'[<>:"/\\|?*]')
# 文件名主干最大长度：部分法规 title 极长（可达 100+ 字符），
# 叠加较长的基础路径后会超过 Windows MAX_PATH(260)，故截断
MAX_STEM = 60


def sanitize_filename(title: str) -> str:
    """清理 title 中 Windows 文件名不允许的字符，并截断过长标题

    部分 title 含换行符等空白字符，或 <、>、/ 等非法字符，需折叠/替换。
    """
    # 折叠空白（含换行、制表符）为单个空格
    name = re.sub(r"\s+", " ", title).strip()
    name = _ILLEGAL.sub("_", name).rstrip(".")
    return name[:MAX_STEM]


def convert(limit: int | None = None) -> int:
    """读取 parquet 并逐条导出 .md 文件，返回生成的文件数"""
    parquet_files = sorted(HF_DIR.glob(PARQUET_GLOB))
    if not parquet_files:
        print(f"✗ 未找到 parquet 文件: {HF_DIR / PARQUET_GLOB}")
        return 0

    print(f"读取 {len(parquet_files)} 个 parquet 文件...")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    if limit:
        df = df.head(limit)

    # 统计同名 title 出现次数，用于重名时加年份/序号区分
    title_counts = df["title"].astype(str).value_counts().to_dict()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    written = 0
    skipped = 0

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title or title.lower() in ("nan", "none"):
            skipped += 1
            continue

        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            skipped += 1
            continue

        base = sanitize_filename(title)
        count = title_counts.get(title, 1)

        # 同名法规：用发布年份区分，仍冲突则由下方唯一性循环兜底
        if count > 1:
            year = None
            pd_ = row.get("publish_date")
            if pd.notna(pd_):
                try:
                    year = int(pd_.year)
                except (TypeError, ValueError, AttributeError):
                    year = None
            stem = f"{base}（{year}年）" if year else base
        else:
            stem = base

        # 确保文件名全局唯一
        candidate = stem
        i = 2
        while f"{candidate}.md" in used:
            candidate = f"{stem}_{i}"
            i += 1
        fname = f"{candidate}.md"
        used.add(fname)

        (OUTPUT_DIR / fname).write_text(content, encoding="utf-8")
        written += 1

    msg = f"✓ 转换完成: {written} 个法规 → {OUTPUT_DIR}"
    if skipped:
        msg += f"（跳过 {skipped} 条空内容）"
    print(msg)
    return written


def main():
    # 避免 Windows GBK 控制台输出 ✓/✗ 等字符时报 UnicodeEncodeError
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])
    convert(limit)


if __name__ == "__main__":
    main()
