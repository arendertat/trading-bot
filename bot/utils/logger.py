"""Structured logging setup"""

import logging
import sys
from pathlib import Path
from typing import Optional

from bot.config.models import LoggingConfig


def setup_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Configure structured logging for the bot.

    Args:
        config: LoggingConfig instance (if None, uses defaults)

    Returns:
        Logger instance
    """
    # Use defaults if no config provided
    if config is None:
        log_level = "INFO"
        log_dir = "./logs"
        event_log_file = "events.jsonl"
    else:
        log_level = config.log_level
        log_dir = config.log_dir
        event_log_file = config.event_log_file

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("trading_bot")
    logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler for events
    event_log_path = log_path / event_log_file
    file_handler = logging.FileHandler(event_log_path)
    file_handler.setLevel(getattr(logging, log_level))
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "trading_bot") -> logging.Logger:
    """Get logger instance by name"""
    return logging.getLogger(name)
