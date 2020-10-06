"""utils functions."""
import logging


class DisableLogger:
    """Context manager to disable logging."""

    def __init__(self, log_level=logging.CRITICAL, logger=None):
        self.log_level = log_level
        self.logger = logger
        if self.logger is None:
            self.logger = logging

    def __enter__(self):
        self.logger.disable(self.log_level)

    def __exit__(self, *args):
        self.logger.disable(0)


def setup_logging(
    log_level=logging.DEBUG,
    log_file=None,
    log_file_level=None,
    log_format=None,
    date_format=None,
    logger=None,
):
    """Setup logging"""
    if logger is None:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

    # Setup logging formatter
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s -- %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Setup console logging handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(log_level)
    root.addHandler(console)

    # Setup file logging handler
    if log_file is not None:
        if log_file_level is None:
            log_file_level = log_level
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(log_file_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
