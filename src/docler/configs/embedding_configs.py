"""Configuration models for embedding providers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr

from docler.provider import ProviderConfig


EmbeddingShorthand = Literal["openai", "bge", "sentence-transformer", "mistral-embed"]

LiteLLMInputType = Literal[
    "sentence-similarity",
    "search_document",
    "search_query",
    "classification",
    "clustering",
]
SentenceTransformerModel = Literal[
    # Multilingual models
    "paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    # Paraphrase models
    "paraphrase-albert-small-v2",
    "paraphrase-MiniLM-L3-v2",
    # Multi-QA models
    "multi-qa-mpnet-base-dot-v1",
    "multi-qa-distilbert-cos-v1",
    "multi-qa-MiniLM-L6-cos-v1",
    # Multilingual distil models
    "distiluse-base-multilingual-cased-v2",
    "distiluse-base-multilingual-cased-v1",
    # General purpose models
    "all-mpnet-base-v2",
    "all-distilroberta-v1",
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
]

OpenAIEmbeddingModel = Literal[
    "text-embedding-ada-002",
    "text-embedding-3-small",
    # "text-embedding-3-medium",
    "text-embedding-3-large",
]


class BaseEmbeddingConfig(ProviderConfig):
    """Base configuration for embedding providers."""

    def get_provider(self):
        """Get the embedding provider instance."""
        raise NotImplementedError


class OpenAIEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for OpenAI embeddings."""

    type: Literal["openai"] = Field(default="openai", init=False)
    """Type discriminator for OpenAI embedding provider."""

    api_key: SecretStr | None = None
    """OpenAI API key."""

    model: OpenAIEmbeddingModel = "text-embedding-3-small"
    """Model identifier for embeddings."""

    def get_provider(self):
        """Get the embedding provider instance."""
        from docler.embeddings.openai_provider import OpenAIEmbeddings

        return OpenAIEmbeddings(**self.get_config_fields())


class BGEEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for BGE embeddings."""

    type: Literal["bge"] = Field(default="bge", init=False)
    """Type discriminator for BGE embedding provider."""

    model: str = "BAAI/bge-large-en-v1.5"
    """Model name or path."""

    def get_provider(self):
        """Get the embedding provider instance."""
        from docler.embeddings.bge_provider import BGEEmbeddings

        return BGEEmbeddings(**self.get_config_fields())


class SentenceTransformerEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for Sentence Transformer embeddings."""

    type: Literal["sentence_transformer"] = Field(
        default="sentence_transformer", init=False
    )
    """Type discriminator for Sentence Transformer embedding provider."""

    model: SentenceTransformerModel = "all-MiniLM-L6-v2"
    """Model name or path."""

    def get_provider(self):
        """Get the embedding provider instance."""
        from docler.embeddings.stf_provider import SentenceTransformerEmbeddings

        return SentenceTransformerEmbeddings(**self.get_config_fields())


class LiteLLMEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for LiteLLM embeddings."""

    type: Literal["litellm"] = Field(default="litellm", init=False)
    """Type discriminator for LiteLLM embedding provider."""

    model: str = "mistral/mistral-embed"
    """Model identifier (e.g., "text-embedding-3-small", "mistral/mistral-embed")."""

    api_key: SecretStr | None = None
    """Optional API key for the provider."""

    dimensions: int | None = Field(default=None, gt=0)
    """Optional number of dimensions for the embeddings."""

    input_type: LiteLLMInputType | None = None
    """Optional input type for the embeddings."""

    base_url: HttpUrl | None = None
    """Optional base URL for the provider."""

    extra_params: dict[str, str | float | bool | None] = Field(default_factory=dict)
    """Additional parameters to pass to LiteLLM."""

    def get_provider(self):
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
