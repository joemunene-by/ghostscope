"""Standard logging configuration for ghostscope."""

from __future__ import annotations

import logging

_LOGGER_NAME = "ghostscope"


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the ghostscope logger.

    Args:
        verbose: when True, set DEBUG level, otherwise INFO.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the shared ghostscope logger."""
    return logging.getLogger(_LOGGER_NAME)
