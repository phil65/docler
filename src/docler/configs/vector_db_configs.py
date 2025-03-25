"""Vector store configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, SecretStr
from pydantic.functional_validators import model_validator

from docler.provider import ProviderConfig
from docler.utils import get_api_key


Metric = Literal["cosine", "euclidean", "dotproduct"]
PineconeRegion = Literal[
    "us-east-1", "us-west-2", "eu-west-1", "europe-west4", "us-central1", "eastus2"
]
PineconeCloud = Literal["aws", "gcp", "azure"]
OpenAIChunkingStrategy = Literal["auto", "static"]


class BaseVectorStoreConfig(ProviderConfig):
    """Base configuration for vector stores."""

    collection_name: str = "default"
    """Name of the collection to use."""


class ChromaConfig(BaseVectorStoreConfig):
    """Configuration for ChromaDB vector store."""

    type: Literal["chroma"] = Field(default="chroma", init=False)

    persist_directory: str | None = None
    """Where to persist the database."""


class QdrantConfig(BaseVectorStoreConfig):
    """Configuration for Qdrant vector store."""

    type: Literal["qdrant"] = Field(default="qdrant", init=False)

    location: str | None = None
    """Path to local Qdrant storage. If None, uses memory."""

    url: str | None = None
    """URL for Qdrant server. If set, location is ignored."""

    api_key: SecretStr | None = None
    """API key for Qdrant cloud."""

    prefer_grpc: bool = True
    """Whether to prefer gRPC over HTTP."""

    @model_validator(mode="after")
    def validate_connection(self) -> QdrantConfig:
        """Ensure either location or url is set, but not both."""
        if self.location and self.url:
            msg = "Cannot specify both location and url"
            raise ValueError(msg)
        if self.api_key and not self.url:
            msg = "API key only valid with url"
            raise ValueError(msg)
        return self


class KdbAiConfig(BaseVectorStoreConfig):
    """Configuration for KDB.AI vector store."""

    type: Literal["kdbai"] = Field(default="kdbai", init=False)

    endpoint: str | None = None
    """Server endpoint to connect to."""

    api_key: SecretStr | None = None
    """API Key for authentication."""

    mode: Literal["rest", "qipc"] | None = None
    """Implementation method used for the session."""

    table_name: str = "vectors"
    """Name of the table to store vectors."""

    index_type: Literal["flat", "hnsw"] = "hnsw"
    """Type of index to use."""

    @model_validator(mode="after")
    def validate_config(self) -> KdbAiConfig:
        """Validate configuration."""
        if not self.endpoint and not self.api_key:
            msg = "Must specify either endpoint or api_key"
            raise ValueError(msg)
        return self


class PineconeConfig(BaseVectorStoreConfig):
    """Configuration for Pinecone vector store."""

    type: Literal["pinecone"] = Field(default="pinecone", init=False)
    """Type identifier for Pinecone."""

    api_key: SecretStr | None = None
    """Pinecone API key."""

    environment: str = "us-west1-gcp"
    """Pinecone environment."""

    cloud: PineconeCloud = "aws"
    """Cloud provider (aws, gcp, azure)."""

    region: PineconeRegion = "us-east-1"
    """Cloud region."""

    dimension: int = 1536
    """Vector dimension."""

    metric: Metric = "cosine"
    """Distance metric for similarity search."""


class OpenAIVectorConfig(BaseVectorStoreConfig):
    """Configuration for OpenAI Vector store."""

    type: Literal["openai"] = Field(default="openai", init=False)
    """Type identifier for OpenAI Vector."""

    api_key: SecretStr | None = None
    """OpenAI API key."""

    organization: str | None = None
    """OpenAI organization ID."""

    chunking_strategy: OpenAIChunkingStrategy = "auto"
    """Strategy for chunking text."""

    max_chunk_size: int = 1000
    """Maximum chunk size in tokens (fixed strategy)."""

    chunk_overlap: int = 200
    """Overlap between chunks in tokens (fixed strategy)."""

    @model_validator(mode="after")
    def validate_config(self) -> OpenAIVectorConfig:
        """Validate configuration."""
        if not self.api_key:
            get_api_key("OPENAI_API_KEY")
        return self


# Complete VectorStoreConfig union type with all implemented configurations
VectorStoreConfig = Annotated[
    ChromaConfig | QdrantConfig | KdbAiConfig | PineconeConfig | OpenAIVectorConfig,
    Field(discriminator="type"),
]
