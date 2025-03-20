"""Configuration models for embedding providers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class BaseEmbeddingConfig(BaseModel):
    """Base configuration for embedding providers."""

    type: str = Field(init=False)
    """Type identifier for the embedding provider."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    def get_embedding_provider(self):
        """Get the embedding provider instance."""
        raise NotImplementedError


class OpenAIEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for OpenAI embeddings."""

    type: Literal["openai"] = Field(default="openai", init=False)
    """Type discriminator for OpenAI embedding provider."""

    api_key: SecretStr
    """OpenAI API key."""

    model: str = "text-embedding-ada-002"
    """Model identifier for embeddings."""

    def get_embedding_provider(self):
        """Get the embedding provider instance."""
        from docler.embeddings.openai_provider import OpenAIEmbeddings

        return OpenAIEmbeddings(**self.model_dump(exclude={"type"}))


class BGEEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for BGE embeddings."""

    type: Literal["bge"] = Field(default="bge", init=False)
    """Type discriminator for BGE embedding provider."""

    model: str = "BAAI/bge-large-en-v1.5"
    """Model name or path."""

    def get_embedding_provider(self):
        """Get the embedding provider instance."""
        from docler.embeddings.bge_provider import BGEEmbeddings

        return BGEEmbeddings(**self.model_dump(exclude={"type"}))


class SentenceTransformerEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for Sentence Transformer embeddings."""

    type: Literal["sentence_transformer"] = Field(
        default="sentence_transformer", init=False
    )
    """Type discriminator for Sentence Transformer embedding provider."""

    model: str = "all-MiniLM-L6-v2"
    """Model name or path."""

    def get_embedding_provider(self):
        """Get the embedding provider instance."""
        from docler.embeddings.stf_provider import SentenceTransformerEmbeddings

        return SentenceTransformerEmbeddings(**self.model_dump(exclude={"type"}))


class LiteLLMEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for LiteLLM embeddings."""

    type: Literal["litellm"] = Field(default="litellm", init=False)
    """Type discriminator for LiteLLM embedding provider."""

    model: str
    """Model identifier (e.g., "text-embedding-ada-002", "mistral/mistral-embed")."""

    api_key: SecretStr | None = None
    """Optional API key for the provider."""

    dimensions: int | None = None
    """Optional number of dimensions for the embeddings."""

    extra_params: dict[str, str | float | bool | None] = Field(default_factory=dict)
    """Additional parameters to pass to LiteLLM."""

    def get_embedding_provider(self):
        """Get the embedding provider instance."""
        from docler.embeddings.litellm_provider import LiteLLMEmbeddings

        config = self.model_dump(exclude={"type", "extra_params"})
        return LiteLLMEmbeddings(**config, **self.extra_params)  # type: ignore


# Union type for embedding configs
EmbeddingConfig = Annotated[
    OpenAIEmbeddingConfig
    | BGEEmbeddingConfig
    | SentenceTransformerEmbeddingConfig
    | LiteLLMEmbeddingConfig,
    Field(discriminator="type"),
]
