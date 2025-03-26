"""Data models for document representation."""

from __future__ import annotations

import importlib.util
from typing import ClassVar, TypeVar

from pydantic import BaseModel, Field, SecretStr, field_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @field_serializer("*", when_used="json-unless-none")
    def serialize_secrets(self, v, _info):
        if isinstance(v, SecretStr):
            return v.get_secret_value()
        return v

    def get_config_fields(self):
        return self.model_dump(exclude={"type"}, mode="json")

    def get_provider(self) -> BaseProvider:
        """Get the provider instance."""
        raise NotImplementedError


class BaseProvider[TConfig]:
    """Represents an image within a document."""

    Config: ClassVar[type[ProviderConfig]]

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
