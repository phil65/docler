"""Logging configuration for docler."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name."""
    return logging.getLogger(f"docler.{name}")
