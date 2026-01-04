"""Data models for document representation."""

from __future__ import annotations

from contextlib import AsyncExitStack
import importlib.util
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import BaseModel

from docler.log import get_logger


if TYPE_CHECKING:
    from docler_config.provider import ProviderConfig


class BaseProvider[TConfig: BaseModel]:
    """Base class for configurable providers."""

    Config: ClassVar[type[ProviderConfig]]
    REQUIRED_PACKAGES: ClassVar[set[str]] = set()
    """Packages required for this converter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.logger = get_logger(__name__)
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""

    @classmethod
    def get_available_providers(cls) -> list[type[Self]]:
        """Get a list of available provider classes."""
        return [kls for kls in cls.__subclasses__() if kls.has_required_packages()]

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


if __name__ == "__main__":
    from docler_config.chunker_configs import BaseChunkerConfig

    test = BaseChunkerConfig.resolve_type("")
