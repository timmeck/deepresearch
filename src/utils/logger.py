"""Logging for DeepResearch."""
import logging, sys
from src.config import LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"deepresearch.{name}")
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
