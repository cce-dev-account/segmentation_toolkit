"""
Centralized Logging Configuration for IRB Segmentation Framework

Provides structured logging with file and console output, respecting
the framework's verbose parameter for backward compatibility.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .config import SegmentationConfig


class IRBLogger:
    """
    Centralized logger for the IRB segmentation framework.

    Supports:
    - Console output (respects verbose flag)
    - File output (persistent logs)
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Structured formatting

    Example:
        >>> from irb_segmentation.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Model training started")
        >>> logger.warning("Monotonicity constraint violated")
    """

    _loggers: Dict[str, logging.Logger] = {}  # Cache for created loggers

    @staticmethod
    def setup_logger(
        name: str = 'irb_segmentation',
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        verbose: bool = True,
        log_format: Optional[str] = None
    ) -> logging.Logger:
        """
        Setup or retrieve a configured logger.

        Args:
            name: Logger name (typically __name__ of calling module)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file for persistent logging
            verbose: If False, suppress console output (only log to file)
            log_format: Optional custom log format string

        Returns:
            Configured logger instance
        """
        # Return cached logger if already configured
        if name in IRBLogger._loggers:
            return IRBLogger._loggers[name]

        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear any existing handlers (prevents duplicate logs)
        logger.handlers = []

        # Prevent propagation to root logger
        logger.propagate = False

        # Define formats
        if log_format is None:
            console_format = '%(levelname)s: %(message)s'
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            console_format = log_format
            file_format = log_format

        # Console handler (respects verbose flag)
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(console_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler (always active if log_file specified)
        if log_file:
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(file_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Log the session start
            logger.info("=" * 70)
            logger.info(f"Logging session started: {datetime.now().isoformat()}")
            logger.info("=" * 70)

        # Cache the logger
        IRBLogger._loggers[name] = logger

        return logger

    @staticmethod
    def reset_loggers() -> None:
        """
        Reset all cached loggers. Useful for testing or reconfiguration.
        """
        for logger in IRBLogger._loggers.values():
            logger.handlers = []
        IRBLogger._loggers = {}

    @staticmethod
    def set_level(name: str, level: int) -> None:
        """
        Change logging level for an existing logger.

        Args:
            name: Logger name
            level: New logging level
        """
        if name in IRBLogger._loggers:
            logger = IRBLogger._loggers[name]
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)


def get_logger(
    name: str = 'irb_segmentation',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    verbose: bool = True
) -> logging.Logger:
    """
    Convenience function to get a configured logger.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        log_file: Optional log file path
        verbose: Enable/disable console output

    Returns:
        Configured logger

    Example:
        >>> logger = get_logger(__name__, log_file='output/training.log')
        >>> logger.info("Training started")
    """
    return IRBLogger.setup_logger(name, level, log_file, verbose)


def configure_logging_from_config(config: 'SegmentationConfig') -> logging.Logger:
    """
    Configure logging based on SegmentationConfig object.

    Args:
        config: SegmentationConfig instance with logging settings

    Returns:
        Configured logger
    """
    # Check if config has logging settings
    logging_config = getattr(config, 'logging', None)

    if logging_config:
        return get_logger(
            name='irb_segmentation',
            level=getattr(logging, logging_config.level.upper(), logging.INFO),
            log_file=logging_config.log_file,
            verbose=config.verbose
        )
    else:
        # Fallback to verbose parameter
        return get_logger(
            name='irb_segmentation',
            verbose=config.verbose
        )


# Module-level logger for internal use
_default_logger = None

def get_default_logger() -> logging.Logger:
    """Get the default framework logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger('irb_segmentation')
    return _default_logger
