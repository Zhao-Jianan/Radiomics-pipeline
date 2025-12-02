import logging
import sys
from pathlib import Path


def _init_logger(name: str = __name__, level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Initialize a logger that outputs to console and optionally to a file.

    Args:
        name (str): Logger name.
        level (int): Logging level, e.g., logging.INFO.
        log_file (str, optional): Path to log file. If None, no file is created.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter for console and file
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------
# Instantiate global logger
# ---------------------------------------------------------
logger = _init_logger(__name__, level=logging.INFO, log_file=None)
