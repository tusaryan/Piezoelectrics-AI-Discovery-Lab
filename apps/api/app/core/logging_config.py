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

    import os
    import datetime

    # Configurable log levels
    console_level_name = os.getenv("PZ_LOG_LEVEL", "INFO").upper()
    console_level = getattr(logging, console_level_name, logging.INFO)

    # Detailed session file handler for EVERYTHING
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file_handler = logging.FileHandler(
        log_dir / f"backend_detailed_{session_id}.log",
        encoding="utf-8"
    )
    session_file_handler.setLevel(logging.DEBUG)
    session_file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    # Console handler (clean terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))

    # Suppress verbose third-party loggers from console completely
    noisy_third_party = [
        "shap", "numpy", "sklearn", "matplotlib", "PIL", 
        "fsspec", "distributed", "boto3", "botocore"
    ]
    for name in noisy_third_party:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()
        logger.addHandler(session_file_handler)
        logger.addHandler(error_handler)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(session_file_handler)
    root_logger.addHandler(console_handler)

    # Route backend-heavy logs strictly to files unless PZ_LOG_LEVEL is DEBUG
    heavy_loggers = [
        "uvicorn.access",
        "sqlalchemy.engine",
        "piezo_ml",
        "xgboost",
        "lightgbm"
    ]
    for name in heavy_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()
        
        # Always save to our detailed files
        logger.addHandler(session_file_handler)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        
        # Only show in console if user asked for DEBUG
        if console_level == logging.DEBUG:
            logger.addHandler(console_handler)

    # Uvicorn error/startup logs should hit console to show server status
    logging.getLogger("uvicorn.error").propagate = True