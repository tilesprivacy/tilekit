"""
Basic logging setup for mlx_engine.

This module configures standard library logging to output to stderr.
Individual modules should get their own loggers using logging.getLogger(__name__).
"""

import logging
import sys


class MLXEngineStreamHandler(logging.StreamHandler):
    """Custom StreamHandler that suppresses errors locally instead of globally."""

    def handleError(self, record):
        """Swallow handler-specific exceptions."""
        pass


def setup_logging():
    """Setup basic logging configuration for mlx_engine."""
    # Configure root logger for mlx_engine
    logger = logging.getLogger("mlx_engine")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers.clear()

    # Create handler that writes to stderr
    handler = MLXEngineStreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)

    # Simple formatter with logger name and level
    formatter = logging.Formatter("[%(module)s][%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
