"""Data models for document representation."""

from __future__ import annotations

from contextlib import AsyncExitStack
import importlib.util
from typing import Any, ClassVar, Self, TypeVar

from pydantic import BaseModel, Field, HttpUrl, SecretStr, field_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict

from docler.log import get_logger


TConfig = TypeVar("TConfig", bound=BaseModel)


class ProviderConfig(BaseSettings):
    """Base configuration for document converters."""

    type: str = Field(init=False)
    """Type discriminator for provider configs."""

    model_config = SettingsConfigDict(
        frozen=True,
        use_attribute_docstrings=True,
        extra="forbid",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @classmethod
    def resolve_type(cls, name: str) -> type[Self]:  # type: ignore
        """Get a the config class of given type."""
        return next(kls for kls in cls.__subclasses__() if kls.type == name)

    @field_serializer("*", when_used="json-unless-none")
    def serialize_special_types(self, v, _info):
        match v:
            case SecretStr():
                return v.get_secret_value()
            case HttpUrl():
                return str(v)
            case _:
                return v

    def get_config_fields(self):
        return self.model_dump(exclude={"type"}, mode="json")

    def get_provider(self) -> BaseProvider:
        """Get the provider instance."""
        raise NotImplementedError


class BaseProvider[TConfig]:
    """Base class for configurable providers."""

    Config: ClassVar[type[ProviderConfig]]
    REQUIRED_PACKAGES: ClassVar[set[str]] = set()
    """Packages required for this converter."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.logger = get_logger(__name__)
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
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
    from docler.configs.chunker_configs import BaseChunkerConfig

    test = BaseChunkerConfig.resolve_type("")
