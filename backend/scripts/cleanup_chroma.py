"""chroma_db 清理脚本 — 手动重跑导入前的清库工具

用法（在 backend 目录下执行）:
    python scripts/cleanup_chroma.py           # 把当前 chroma_db 改名备份（推荐，不删数据）
    python scripts/cleanup_chroma.py --force   # 直接删除当前 chroma_db（不留备份）
    python scripts/cleanup_chroma.py --list    # 只列出所有 chroma_db* 目录，不做任何操作

说明:
    - 只处理 backend/chroma_db（活动库）与各级备份（chroma_db.bak_* / chroma_db.broken*）
    - 默认"改名备份"而不是删除，避免误删；确认无误后可手动删除备份目录
    - 若检测到 import_data.py / _reimport.py 正在运行会警告，避免文件占用
"""
import os
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # backend/
CHROMA_DIR = BASE / "chroma_db"


def running_import_processes() -> list:
    """检测是否有导入进程正在运行（占用 chroma_db 会导致改名/删除失败）"""
    try:
        import psutil
    except ImportError:
        return []
    hits = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "import_data.py" in cmd or "_reimport" in cmd:
                hits.append(f"  PID {p.pid}: {cmd[:120]}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return hits


def list_dirs() -> list:
    return sorted(
        p for p in BASE.iterdir() if p.name.startswith("chroma_db")
    )


def main():
    force = "--force" in sys.argv
    only_list = "--list" in sys.argv

    dirs = list_dirs()
    print("当前 chroma_db* 目录:")
    if not dirs:
        print("  （无）")
    for p in dirs:
        print(f"  - {p.name}")

    if only_list:
        return

    runn = running_import_processes()
    if runn:
        print("\n警告: 检测到导入进程正在运行，建议先停止再清理:")
        for line in runn:
            print(line)

    live = CHROMA_DIR
    if not live.exists():
        print("\n当前 chroma_db 不存在，无需清理，可以直接重新导入。")
        return

    if force:
        try:
            shutil.rmtree(live, ignore_errors=False)
            print(f"\n已删除: {live.name}")
        except PermissionError as e:
            print(f"\n删除失败（文件被占用）: {e}")
            print("请先停止所有 import_data.py / _reimport.py 进程再重试。")
            sys.exit(1)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        dst = BASE / f"chroma_db.bak_{ts}"
        n = 1
        while dst.exists():  # 避免时间戳撞车
            dst = BASE / f"chroma_db.bak_{ts}_{n}"
            n += 1
        try:
            os.rename(live, dst)
            print(f"\n已改名备份: {live.name} -> {dst.name}")
        except PermissionError as e:
            print(f"\n改名失败（文件被占用）: {e}")
            print("请先停止所有 import_data.py / _reimport.py 进程再重试。")
            sys.exit(1)

    print("\n清理完成，可以重新导入了。")
    print("提示: 旧备份目录（chroma_db.broken / chroma_db.broken_v2 等）已无用，")
    print("确认不需要后可手动删除以释放磁盘空间。")


if __name__ == "__main__":
    main()
