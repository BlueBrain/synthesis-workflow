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
