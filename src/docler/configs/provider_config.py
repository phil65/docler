"""Converter configuration."""

from __future__ import annotations

from pydantic import Field, SecretStr, field_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict


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
