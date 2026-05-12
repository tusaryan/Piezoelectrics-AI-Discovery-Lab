"""
Piezo.AI — Structured Logging Configuration
==============================================
Dual-output logging:
- stdout/stderr: key info (startup, errors, warnings) — keeps terminal clean
- log files: detailed debug info in logs/ directory with rotation
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(project_root: Path | None = None) -> None:
    """Configure dual-output logging for Piezo.AI backend."""

    if project_root is None:
        project_root = Path(__file__).resolve().parents[4]  # app/core/ → project root

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create formatters
    # Detailed format for file output
    FILE_FORMAT = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    # Clean format for stdout
    CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"

    # Rotating file handler — captures ALL log levels to file
    file_handler = RotatingFileHandler(
        log_dir / "piezo-ai.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    # Separate error file
    error_handler = RotatingFileHandler(
        log_dir / "piezo-ai-errors.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    # Console handler — INFO and above only (keeps terminal clean)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))

    # Suppress verbose third-party loggers
    noisy_loggers = [
        "uvicorn.access",
        "uvicorn.config",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "asyncio",
        "shap",
        "numpy",
        "sklearn",
        "xgboost",
        "lightgbm",
        "matplotlib",
        "PIL",
        "fsspec",
        "distributed",
        "boto3",
        "botocore",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
        logging.getLogger(name).propagate = False

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # uvicorn access logs should go to file only, not stdout
    # (we keep our own console output clean)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)