"""Data models for document representation."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, HttpUrl, SecretStr, field_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseSettings):
    """Base configuration for document converters."""

    type: str = Field(init=False)
    """Type discriminator for provider configs."""

    model_config = SettingsConfigDict(
        frozen=True,
        use_attribute_docstrings=True,
        extra="forbid",
        env_file_encoding="utf-8",
    )

    @classmethod
    def resolve_type(cls, name: str) -> type[Self]:  # type: ignore
        """Get a the config class of given type."""
        return next(kls for kls in cls.__subclasses__() if kls.type == name)

    @field_serializer("*", when_used="json-unless-none")
    def serialize_special_types(self, v: Any, _info: Any) -> Any:
        match v:
            case SecretStr():
                return v.get_secret_value()
            case HttpUrl():
                return str(v)
            case _:
                return v

    def get_config_fields(self) -> dict[str, Any]:
        return self.model_dump(exclude={"type"}, mode="json")

    def get_provider(self) -> Any:
        """Get the provider instance."""
        raise NotImplementedError
