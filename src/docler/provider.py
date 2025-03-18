"""Data models for document representation."""

from __future__ import annotations

import importlib.util
from typing import ClassVar


class BaseProvider:
    """Represents an image within a document."""

    REQUIRED_PACKAGES: ClassVar[set[str]] = set()
    """Packages required for this converter."""

    @classmethod
    def has_required_packages(cls) -> bool:
        """Check if all required packages are available.

        Returns:
            True if all required packages are installed, False otherwise
        """
        for package in cls.REQUIRED_PACKAGES:
            if not importlib.util.find_spec(package.replace("-", "_")):
                return False
        return True
