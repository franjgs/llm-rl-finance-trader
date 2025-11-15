# src/logging_config.py
"""
Centralized logging configuration with verbose support.
Used by all scripts (data_fetch.py, ensemble.py, etc.)
"""
import logging
import sys
from pathlib import Path

def setup_logging(verbose: int = 1):
    """
    Configure logging based on verbose level.
    verbose = 0 → WARNING only
    verbose = 1 → INFO + above
    verbose = 2 → DEBUG + above
    """
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbose, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout
    )
    
    # Reduce noise from libraries
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logger = logging.getLogger("ensemble")
    logger.info(f"Logging initialized → verbose level = {verbose} ({logging.getLevelName(level)})")
    
    return logger