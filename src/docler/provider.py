"""Data models for document representation."""

from __future__ import annotations

import importlib.util
from typing import ClassVar, TypeVar

from pydantic import BaseModel


TConfig = TypeVar("TConfig", bound=BaseModel)


class BaseProvider[TConfig]:
    """Represents an image within a document."""

    Config: type[TConfig]

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

    @classmethod
    def from_config(cls, config: TConfig) -> BaseProvider[TConfig]:
        """Create an instance of the provider from a configuration object."""
        raise NotImplementedError

    def to_config(self) -> TConfig:
        """Extract configuration from the provider instance."""
        raise NotImplementedError
