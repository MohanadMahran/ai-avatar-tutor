"""
Structured logging utility with console and file output.
Provides a configured logger with rotating file handlers,
consistent formatting, and module-level logger creation.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_loggers: dict = {}
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a configured logger instance.
    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: INFO).
    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Prevent duplicate handlers
    if logger.handlers:
        _loggers[name] = logger
        return logger
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024, # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    # Prevent propagation to root logger
    logger.propagate = False
    _loggers[name] = logger
    return logger