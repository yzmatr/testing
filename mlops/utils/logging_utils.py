import logging
import contextlib
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os
import sys


###################################################################################
# SETUP STANDARD LOGGER
###################################################################################

def setup_logger(name: str = None, verbose: bool = True) -> logging.Logger:
    """
    Set up a logger with consistent formatting. Call this in any class/module
    to get a properly configured logger.

    Args:
        name (str): Optional logger name (e.g., module or class name).
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    level = logging.INFO if verbose else logging.WARNING

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent duplicate logs if root logger is also active
    logger.propagate = False
    return logger

###################################################################################
# SUPPRESS OUTPUTS
###################################################################################
@contextmanager
def suppress_output(min_log_level=logging.WARNING, suppress_stdout=True, suppress_stderr=True):
    """
    Suppress stdout/stderr and raise logging levels, but preserve tqdm output.
    tqdm will continue to write to sys.__stdout__.
    """
    # Save original output streams
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    # Open null device
    with open(os.devnull, 'w') as devnull:
        # Setup redirections
        cm_out = redirect_stdout(devnull) if suppress_stdout else contextmanager(lambda: (yield))()
        cm_err = redirect_stderr(devnull) if suppress_stderr else contextmanager(lambda: (yield))()

        # Backup logging levels
        root_logger = logging.getLogger()
        loggers = [root_logger] + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        original_levels = {logger: logger.level for logger in loggers}

        try:
            # Temporarily raise log levels
            for logger in loggers:
                logger.setLevel(min_log_level)

            # Perform redirection
            with cm_out, cm_err:
                # ⬇️ Critical: make sure `sys.stdout` is NOT None or stale
                sys.stdout = sys.__stdout__  # ensure tqdm still writes correctly
                sys.stderr = sys.__stderr__
                yield

        finally:
            # Restore log levels
            for logger, lvl in original_levels.items():
                logger.setLevel(lvl)

            # Restore original output streams
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr