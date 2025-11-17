"""
Unit Tests for IRB Segmentation Logger

Tests cover:
- Logger creation and configuration
- Console and file handlers
- Verbose parameter behavior
- Log levels and formatting
- Logger caching and reset
- Multiple logger instances
"""

import pytest
import logging
import tempfile
from pathlib import Path
from io import StringIO
import sys

from irb_segmentation.logger import (
    IRBLogger,
    get_logger,
    get_default_logger
)


@pytest.mark.unit
class TestLoggerCreation:
    """Test basic logger creation and configuration."""

    def setup_method(self):
        """Reset loggers before each test."""
        IRBLogger.reset_loggers()

    def test_create_basic_logger(self):
        """Test creating a basic logger with defaults."""
        logger = get_logger('test_logger')

        assert logger is not None
        assert logger.name == 'test_logger'
        assert logger.level == logging.INFO
        assert not logger.propagate

    def test_logger_caching(self):
        """Test that loggers are cached and reused."""
        logger1 = get_logger('cached_logger')
        logger2 = get_logger('cached_logger')

        assert logger1 is logger2
        assert 'cached_logger' in IRBLogger._loggers

    def test_multiple_loggers(self):
        """Test creating multiple different loggers."""
        logger1 = get_logger('logger_one')
        logger2 = get_logger('logger_two')

        assert logger1 is not logger2
        assert logger1.name == 'logger_one'
        assert logger2.name == 'logger_two'

    def test_logger_levels(self):
        """Test creating loggers with different levels."""
        logger_debug = get_logger('debug_logger', level=logging.DEBUG)
        logger_warning = get_logger('warning_logger', level=logging.WARNING)
        logger_error = get_logger('error_logger', level=logging.ERROR)

        assert logger_debug.level == logging.DEBUG
        assert logger_warning.level == logging.WARNING
        assert logger_error.level == logging.ERROR


@pytest.mark.unit
class TestConsoleHandler:
    """Test console output handler behavior."""

    def setup_method(self):
        """Reset loggers before each test."""
        IRBLogger.reset_loggers()

    def test_verbose_true_has_console_handler(self):
        """Test that verbose=True adds console handler."""
        logger = get_logger('verbose_logger', verbose=True)

        # Should have at least one handler
        assert len(logger.handlers) > 0

        # Check if any handler is a StreamHandler
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
            for h in logger.handlers
        )
        assert has_stream_handler

    def test_verbose_false_no_console_handler(self):
        """Test that verbose=False removes console handler."""
        logger = get_logger('silent_logger', verbose=False)

        # Should have no console handlers
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
            for h in logger.handlers
        )
        assert not has_stream_handler

    def test_console_output_format(self, capsys):
        """Test console output format."""
        logger = get_logger('format_logger', verbose=True, level=logging.INFO)
        logger.info("Test info message")

        captured = capsys.readouterr()
        assert "INFO: Test info message" in captured.out

    def test_console_output_levels(self, capsys):
        """Test that console respects log levels."""
        logger = get_logger('level_logger', verbose=True, level=logging.WARNING)

        logger.debug("Debug message")  # Should not appear
        logger.info("Info message")    # Should not appear
        logger.warning("Warning message")  # Should appear
        logger.error("Error message")  # Should appear

        captured = capsys.readouterr()
        assert "Debug message" not in captured.out
        assert "Info message" not in captured.out
        assert "WARNING: Warning message" in captured.out
        assert "ERROR: Error message" in captured.out


@pytest.mark.unit
class TestFileHandler:
    """Test file output handler behavior."""

    def setup_method(self):
        """Reset loggers before each test."""
        IRBLogger.reset_loggers()

    def test_file_handler_creation(self, tmp_path):
        """Test that log file is created."""
        log_file = tmp_path / "test.log"
        logger = get_logger('file_logger', log_file=str(log_file))

        logger.info("Test message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_file_handler_creates_directory(self, tmp_path):
        """Test that log directory is created if missing."""
        log_file = tmp_path / "nested" / "dir" / "test.log"
        logger = get_logger('nested_logger', log_file=str(log_file))

        logger.info("Nested test")

        assert log_file.exists()
        assert log_file.parent.exists()

    def test_file_format_includes_timestamp(self, tmp_path):
        """Test that file format includes timestamp and metadata."""
        log_file = tmp_path / "timestamped.log"
        logger = get_logger('time_logger', log_file=str(log_file))

        logger.info("Timestamped message")

        content = log_file.read_text()
        # Should contain timestamp, logger name, level, and message
        assert "time_logger" in content
        assert "INFO" in content
        assert "Timestamped message" in content

    def test_file_append_mode(self, tmp_path):
        """Test that file handler appends to existing log."""
        log_file = tmp_path / "append.log"

        # First logger
        logger1 = get_logger('append_logger1', log_file=str(log_file))
        logger1.info("First message")

        # Reset and create new logger (should append)
        IRBLogger.reset_loggers()
        logger2 = get_logger('append_logger2', log_file=str(log_file))
        logger2.info("Second message")

        content = log_file.read_text()
        assert "First message" in content
        assert "Second message" in content

    def test_file_and_console_both_work(self, tmp_path, capsys):
        """Test that both file and console output work together."""
        log_file = tmp_path / "both.log"
        logger = get_logger(
            'both_logger',
            log_file=str(log_file),
            verbose=True
        )

        logger.info("Message to both")

        # Check console
        captured = capsys.readouterr()
        assert "INFO: Message to both" in captured.out

        # Check file
        content = log_file.read_text()
        assert "Message to both" in content

    def test_file_only_no_console(self, tmp_path, capsys):
        """Test file logging without console output."""
        log_file = tmp_path / "file_only.log"
        logger = get_logger(
            'file_only_logger',
            log_file=str(log_file),
            verbose=False
        )

        logger.info("File only message")

        # Should NOT appear in console
        captured = capsys.readouterr()
        assert "File only message" not in captured.out

        # Should appear in file
        content = log_file.read_text()
        assert "File only message" in content


@pytest.mark.unit
class TestCustomFormat:
    """Test custom log format configuration."""

    def setup_method(self):
        """Reset loggers before each test."""
        IRBLogger.reset_loggers()

    def test_custom_format(self, tmp_path):
        """Test using a custom log format."""
        log_file = tmp_path / "custom_format.log"
        custom_format = "%(levelname)s | %(message)s"

        logger = IRBLogger.setup_logger(
            'custom_logger',
            log_file=str(log_file),
            log_format=custom_format
        )

        logger.info("Custom formatted message")

        content = log_file.read_text()
        # Should use custom format
        assert "INFO | Custom formatted message" in content


@pytest.mark.unit
class TestLoggerReset:
    """Test logger reset functionality."""

    def test_reset_clears_cache(self):
        """Test that reset clears the logger cache."""
        logger = get_logger('reset_test')
        assert 'reset_test' in IRBLogger._loggers

        IRBLogger.reset_loggers()
        assert len(IRBLogger._loggers) == 0

    def test_reset_clears_handlers(self, tmp_path):
        """Test that reset removes all handlers."""
        log_file = tmp_path / "reset.log"
        logger = get_logger('handler_reset', log_file=str(log_file), verbose=True)

        initial_handlers = len(logger.handlers)
        assert initial_handlers > 0

        IRBLogger.reset_loggers()
        assert len(logger.handlers) == 0

    def test_logger_after_reset_is_new(self):
        """Test that getting logger after reset is reconfigured."""
        logger1 = get_logger('before_reset', verbose=True)

        # Should have console handler
        initial_handlers = len(logger1.handlers)
        assert initial_handlers > 0

        IRBLogger.reset_loggers()

        logger2 = get_logger('before_reset', verbose=False)

        # After reset, new configuration applied (no console handler)
        # Note: Python's logging.getLogger returns the same object by name,
        # but our reset() cleared handlers and reconfigured it
        has_console = any(
            isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
            for h in logger2.handlers
        )
        assert not has_console


@pytest.mark.unit
class TestSetLevel:
    """Test dynamic level adjustment."""

    def setup_method(self):
        """Reset loggers before each test."""
        IRBLogger.reset_loggers()

    def test_set_level_changes_logger(self, capsys):
        """Test that set_level changes logger level."""
        logger = get_logger('level_change', level=logging.INFO, verbose=True)

        # INFO level - debug should not show
        logger.debug("Debug before")
        logger.info("Info before")

        captured = capsys.readouterr()
        assert "Debug before" not in captured.out
        assert "INFO: Info before" in captured.out

        # Change to DEBUG level
        IRBLogger.set_level('level_change', logging.DEBUG)

        logger.debug("Debug after")
        captured = capsys.readouterr()
        assert "DEBUG: Debug after" in captured.out

    def test_set_level_changes_handlers(self, tmp_path):
        """Test that set_level changes handler levels."""
        log_file = tmp_path / "level_change.log"
        logger = get_logger(
            'handler_level',
            level=logging.WARNING,
            log_file=str(log_file)
        )

        logger.info("Info before")  # Should not appear

        # Change to INFO
        IRBLogger.set_level('handler_level', logging.INFO)
        logger.info("Info after")  # Should appear

        content = log_file.read_text()
        assert "Info before" not in content
        assert "Info after" in content

    def test_set_level_nonexistent_logger(self):
        """Test set_level with non-existent logger (should not error)."""
        # Should not raise exception
        IRBLogger.set_level('nonexistent', logging.DEBUG)


@pytest.mark.unit
class TestDefaultLogger:
    """Test default logger functionality."""

    def setup_method(self):
        """Reset loggers before each test."""
        IRBLogger.reset_loggers()
        # Reset the global default logger
        import irb_segmentation.logger
        irb_segmentation.logger._default_logger = None

    def test_get_default_logger(self):
        """Test getting the default logger."""
        logger = get_default_logger()

        assert logger is not None
        assert logger.name == 'irb_segmentation'

    def test_default_logger_cached(self):
        """Test that default logger is cached."""
        logger1 = get_default_logger()
        logger2 = get_default_logger()

        assert logger1 is logger2


@pytest.mark.unit
class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def setup_method(self):
        """Reset loggers before each test."""
        IRBLogger.reset_loggers()

    def test_multiple_modules_logging(self, tmp_path, capsys):
        """Test multiple modules logging to same file."""
        log_file = tmp_path / "multi_module.log"

        logger1 = get_logger('module1', log_file=str(log_file), verbose=True)
        logger2 = get_logger('module2', log_file=str(log_file), verbose=True)

        logger1.info("Message from module 1")
        logger2.warning("Warning from module 2")

        # Check console
        captured = capsys.readouterr()
        assert "Message from module 1" in captured.out
        assert "Warning from module 2" in captured.out

        # Check file has both
        content = log_file.read_text()
        assert "module1" in content
        assert "Message from module 1" in content
        assert "module2" in content
        assert "Warning from module 2" in content

    def test_all_log_levels(self, tmp_path):
        """Test all standard log levels."""
        log_file = tmp_path / "all_levels.log"
        logger = get_logger('all_levels', level=logging.DEBUG, log_file=str(log_file))

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        content = log_file.read_text()
        assert "DEBUG" in content
        assert "INFO" in content
        assert "WARNING" in content
        assert "ERROR" in content
        assert "CRITICAL" in content

    def test_logger_with_exception(self, tmp_path):
        """Test logging exceptions with traceback."""
        log_file = tmp_path / "exception.log"
        logger = get_logger('exception_logger', log_file=str(log_file))

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.error("Caught exception", exc_info=True)

        content = log_file.read_text()
        assert "Caught exception" in content
        assert "ValueError: Test exception" in content
        assert "Traceback" in content

    def test_logger_no_handlers_still_works(self):
        """Test that logger without handlers doesn't error."""
        logger = get_logger('no_handler', verbose=False)

        # Should not raise exception
        logger.info("Message with no handlers")
        logger.warning("Warning with no handlers")


@pytest.mark.unit
class TestLoggerEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset loggers before each test."""
        IRBLogger.reset_loggers()

    def test_empty_logger_name(self):
        """Test creating logger with empty name."""
        logger = get_logger('')
        assert logger is not None

    def test_very_long_message(self, tmp_path):
        """Test logging very long messages."""
        log_file = tmp_path / "long_message.log"
        logger = get_logger('long_logger', log_file=str(log_file))

        long_message = "A" * 10000
        logger.info(long_message)

        content = log_file.read_text()
        assert long_message in content

    def test_unicode_messages(self, tmp_path):
        """Test logging unicode messages."""
        log_file = tmp_path / "unicode.log"
        logger = get_logger('unicode_logger', log_file=str(log_file))

        logger.info("Unicode test: 你好世界 🚀 café")

        content = log_file.read_text(encoding='utf-8')
        assert "你好世界" in content
        assert "🚀" in content
        assert "café" in content

    def test_logging_with_special_characters(self, tmp_path):
        """Test logging messages with special characters."""
        log_file = tmp_path / "special_chars.log"
        logger = get_logger('special_logger', log_file=str(log_file))

        logger.info("Special: %s %d %f {} [] \n \t")

        content = log_file.read_text()
        assert "Special:" in content

    def test_invalid_log_file_path_creates_dirs(self, tmp_path):
        """Test that invalid nested paths are created."""
        log_file = tmp_path / "a" / "b" / "c" / "d" / "test.log"
        logger = get_logger('nested_path', log_file=str(log_file))

        logger.info("Deep nested message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Deep nested message" in content


@pytest.mark.unit
class TestLoggerPerformance:
    """Test logger performance characteristics."""

    def setup_method(self):
        """Reset loggers before each test."""
        IRBLogger.reset_loggers()

    def test_logger_caching_performance(self):
        """Test that cached loggers are reused."""
        # First call (creates logger and caches it)
        logger1 = get_logger('perf_test')

        # Verify it's cached
        assert 'perf_test' in IRBLogger._loggers

        # Second call (should return cached logger)
        logger2 = get_logger('perf_test')

        # Should be the same object (not just equal)
        assert logger1 is logger2

        # Verify cache hit
        assert IRBLogger._loggers['perf_test'] is logger1

    @pytest.mark.slow
    def test_many_log_messages(self, tmp_path):
        """Test handling many log messages."""
        log_file = tmp_path / "many_messages.log"
        logger = get_logger('many_logger', log_file=str(log_file), verbose=False)

        # Log 1000 messages
        for i in range(1000):
            logger.info(f"Message {i}")

        content = log_file.read_text()
        assert "Message 0" in content
        assert "Message 999" in content
        assert content.count("Message") == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
