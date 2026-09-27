"""
日志初始化模块
集中配置 Python logging：同时输出到控制台与 logs/ 目录下的按日期文件

背景: backend 六个模块都用 logging.getLogger(__name__) 打日志，但项目原先
没有任何地方调用 logging.basicConfig()，导致所有日志（含"摄像头初始化失败"
这类关键诊断）都不会被输出。本模块负责把这条链路接通。

用法（在入口处调用一次即可，重复调用是安全的）:
    from backend.logging_setup import setup_logging
    setup_logging()
"""

import logging
import sys
from datetime import datetime

from backend.config import LOGS_DIR, BASE_DIR


# 模块级标记：保证多次 import/调用只配置一次
_CONFIGURED = False
_LOG_FILE_PATH = ""

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, log_to_file: bool = True) -> str:
    """
    配置全局日志。

    参数:
        level: 控制台与文件的最低日志级别（默认 INFO）
        log_to_file: 是否额外写入 logs/ 目录（默认 True）

    返回:
        str: 实际写入的日志文件路径；未写文件时返回空字符串
    """
    global _CONFIGURED, _LOG_FILE_PATH

    # Streamlit 的每次 rerun 都会重新执行页面脚本。日志只需在首次配置，
    # 否则控制台会随 rerun 次数重复堆积同一条日志。
    if _CONFIGURED:
        return _LOG_FILE_PATH

    file_path = ""
    try:
        import os
        os.makedirs(LOGS_DIR, exist_ok=True)
        file_path = os.path.join(
            LOGS_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log"
        )
    except Exception:
        log_to_file = False

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_to_file and file_path:
        try:
            handlers.append(logging.FileHandler(file_path, encoding="utf-8"))
        except Exception:
            # 写文件失败（如无权限）时仅保留控制台输出，不影响主流程
            pass

    logging.basicConfig(level=level, format=_LOG_FORMAT,
                        datefmt=_DATE_FORMAT, handlers=handlers, force=True)

    # 抑制第三方库的噪声日志
    for noisy in ("urllib3", "matplotlib", "PIL", "ultralytics", "streamlit"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    _LOG_FILE_PATH = file_path
    logging.getLogger(__name__).info(
        "日志系统已初始化 (level=%s, file=%s)",
        logging.getLevelName(level),
        file_path or "未启用文件日志",
    )
    return file_path


def is_configured() -> bool:
    """日志是否已初始化"""
    return _CONFIGURED
