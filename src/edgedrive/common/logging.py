"""Structured logging configuration for EdgeDrive.

Uses structlog for structured JSON logging with a fallback to stdlib
logging if structlog is not available.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False

_configured = False


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, output JSON-formatted logs.
    """
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    if _HAS_STRUCTLOG:
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=log_level,
            force=True,
        )
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ]
        if json_output:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            level=log_level,
            force=True,
        )

    _configured = True


class _StdlogCompat:
    """Wrapper around stdlib logging.Logger that accepts structlog-style kwargs.

    When structlog is not installed, callers still use keyword arguments
    like ``logger.info("msg", model="x", count=3)``.  The stdlib
    ``logging.Logger.info`` method does not accept arbitrary kwargs, so
    this wrapper absorbs them and appends a readable ``key=value`` string
    to the log message.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _fmt(self, msg: str, kwargs: dict) -> str:
        if kwargs:
            pairs = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{msg} {pairs}"
        return msg

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(self._fmt(msg, kwargs))

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(self._fmt(msg, kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(self._fmt(msg, kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(self._fmt(msg, kwargs))

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._logger.exception(self._fmt(msg, kwargs))

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._logger.critical(self._fmt(msg, kwargs))

    def setLevel(self, level: int) -> None:
        self._logger.setLevel(level)

    @property
    def level(self) -> int:
        return self._logger.level


def get_logger(name: str = "edgedrive") -> Any:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually module or component name).

    Returns:
        A structlog logger if available, otherwise a compat wrapper
        around a stdlib logger that accepts structlog-style kwargs.
    """
    configure_logging()
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return _StdlogCompat(logging.getLogger(name))