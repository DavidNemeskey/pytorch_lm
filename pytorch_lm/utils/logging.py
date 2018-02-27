#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Generic utility functions."""

import logging


def setup_stream_logger(logging_level, name='script'):
    """Sets a stream logger up."""
    sh = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    return setup_logger(logging_level, sh, name)


def setup_logger(logging_level, handler, name='script'):
    """Setups logging for scripts."""
    logger = logging.getLogger('pytorch_lm')
    # Remove old handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[-1])

    if logging_level:
        log_level = __get_logging_level(logging_level)
        # Set up root logger
        handler.setLevel(log_level)
        logger.addHandler(handler)
    else:
        # Don't log anything
        log_level = logging.CRITICAL + 1
    logger.setLevel(log_level)

    # Set up the specific logger requested
    logger = logging.getLogger('pytorch_lm.' + name)
    logger.setLevel(log_level)
    return logger


def __get_logging_level(logging_level):
    """Returns the logging level that corresponds to the parameter string."""
    if isinstance(logging_level, str):
        return getattr(logging, logging_level.upper())
    else:
        return logging_level
