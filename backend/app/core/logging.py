"""日志初始化 — 统一日志格式与级别（幂等）"""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """配置全局日志格式（幂等：根 logger 已有 handler 时静默跳过）"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块日志器；首次调用时自动完成全局日志配置"""
    setup_logging()
    return logging.getLogger(name)
