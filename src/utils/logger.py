# src/utils/logger.py
import logging
from pathlib import Path


def get_logger(name="train", save_dir="outputs/logs", level=logging.INFO):
    """로그 출력을 위한 logger 객체 생성"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(save_dir) / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 중복 로그 방지

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
