# src/logger.py

import logging
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_configured = False

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a configured logger. Safe to call multiple times.
    """
    global _configured
    if not _configured:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
        _configured = True
    return logging.getLogger(name if name else __name__)