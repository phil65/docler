from __future__ import annotations


class ConverterError(Exception):
    """Base exception for converter errors."""


class MissingConfigurationError(ConverterError):
    """Required configuration is missing."""
