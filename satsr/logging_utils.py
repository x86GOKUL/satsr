"""Centralised logging setup for the satsr framework."""
import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "satsr", level: str = "INFO") -> logging.Logger:
    """Return a configured logger. Idempotent — safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger("satsr")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False
        _CONFIGURED = True
    return logger.getChild(name) if name != "satsr" else logger
