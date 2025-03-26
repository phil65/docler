"""Unified configuration for the complete RAG pipeline."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, SecretStr

from docler.configs.chunker_configs import ChunkerConfig  # noqa: TC001
from docler.configs.converter_configs import ConverterConfig  # noqa: TC001
from docler.configs.embedding_configs import EmbeddingConfig  # noqa: TC001
from docler.configs.vector_db_configs import VectorStoreConfig  # noqa: TC001
from docler.provider import ProviderConfig
from docler.utils import get_api_key


# Shorthand literals for common configurations
ChunkerShorthand = Literal["markdown", "llamaindex", "ai"]

EmbeddingShorthand = Literal["openai", "bge", "sentence-transformer", "mistral-embed"]

VectorDBShorthand = Literal["chroma", "qdrant", "pinecone", "openai"]

ConverterShorthand = Literal[
    "docling", "marker", "mistral", "llamaparse", "datalab", "azure", "llm"
]


class FileDatabaseConfig(ProviderConfig):
    """Base configuration for file databases."""

    store_name: str = "default"
    """Name identifier for this file database."""

    def resolve(self) -> FileDatabaseConfig:
        """Resolve any shorthand configurations to full configurations.

        Returns:
            A fully resolved configuration with no shorthands
        """
        return self


class ComponentBasedConfig(FileDatabaseConfig):
    """Configuration for component-based file database."""

    type: Literal["component"] = "component"

    converter: ConverterConfig | ConverterShorthand = "marker"
    """Document converter configuration or shorthand."""

    chunker: ChunkerConfig | ChunkerShorthand = "markdown"
    """Text chunker configuration or shorthand."""

    embeddings: EmbeddingConfig | EmbeddingShorthand = "openai"
    """Embedding provider configuration or shorthand."""

    vector_store: VectorStoreConfig | VectorDBShorthand = "chroma"
    """Vector store configuration or shorthand."""

    batch_size: int = 8
    """Batch size for processing."""

    def resolve_converter(self) -> ConverterConfig:  # noqa: PLR0911
        """Get full converter config from shorthand or pass through existing."""
        if isinstance(self.converter, str):
            from docler.configs.converter_configs import (
                AzureConfig,
                DataLabConfig,
                DoclingConverterConfig,
                LlamaParseConfig,
                LLMConverterConfig,
                MarkerConfig,
                MistralConfig,
            )

            match self.converter:
                case "docling":
                    return DoclingConverterConfig()
                case "marker":
                    return MarkerConfig()
                case "mistral":
                    return MistralConfig()
                case "llamaparse":
                    return LlamaParseConfig()
                case "datalab":
                    return DataLabConfig()
                case "azure":
                    return AzureConfig()
                case "llm":
                    return LLMConverterConfig()
        return self.converter

    def resolve_chunker(self) -> ChunkerConfig:
        """Get full chunker config from shorthand or pass through existing."""
        if isinstance(self.chunker, str):
            from docler.configs.chunker_configs import (
                AiChunkerConfig,
                LlamaIndexChunkerConfig,
                MarkdownChunkerConfig,
            )

            match self.chunker:
                case "markdown":
                    return MarkdownChunkerConfig()
                case "llamaindex":
                    return LlamaIndexChunkerConfig()
                case "ai":
                    return AiChunkerConfig()
        return self.chunker

    def resolve_embeddings(self) -> EmbeddingConfig:
        """Get full embedding config from shorthand or pass through existing."""
        if isinstance(self.embeddings, str):
            from docler.configs.embedding_configs import (
                BGEEmbeddingConfig,
                LiteLLMEmbeddingConfig,
                OpenAIEmbeddingConfig,
                SentenceTransformerEmbeddingConfig,
            )

            match self.embeddings:
                case "openai":
                    return OpenAIEmbeddingConfig(
                        api_key=SecretStr(get_api_key("OPENAI_API_KEY"))
                    )
                case "bge":
                    return BGEEmbeddingConfig()
                case "sentence-transformer":
                    return SentenceTransformerEmbeddingConfig()
                case "mistral-embed":
                    return LiteLLMEmbeddingConfig(
                        model="mistral/mistral-embed",
                        api_key=SecretStr(get_api_key("MISTRAL_API_KEY")),
                    )
        return self.embeddings

    def resolve_vector_store(self) -> VectorStoreConfig:
        """Get full vector store config from shorthand or pass through existing."""
        if isinstance(self.vector_store, str):
            from docler.configs.vector_db_configs import (
                ChromaConfig,
                OpenAIVectorConfig,
                PineconeConfig,
                QdrantConfig,
            )

            match self.vector_store:
                case "chroma":
                    return ChromaConfig(collection_name=self.store_name)
                case "qdrant":
                    return QdrantConfig(collection_name=self.store_name)
                case "pinecone":
                    return PineconeConfig(
                        collection_name=self.store_name,
                        api_key=SecretStr(get_api_key("PINECONE_API_KEY")),
                    )
                case "openai":
                    return OpenAIVectorConfig(
                        api_key=SecretStr(get_api_key("OPENAI_API_KEY")),
                    )
        return self.vector_store

    def resolve(self) -> ComponentBasedConfig:
        """Resolve all shorthand configurations."""
        return ComponentBasedConfig(
            store_name=self.store_name,
            converter=self.resolve_converter(),
            chunker=self.resolve_chunker(),
            embeddings=self.resolve_embeddings(),
            vector_store=self.resolve_vector_store(),
            batch_size=self.batch_size,
        )


class OpenAIFileDatabaseConfig(FileDatabaseConfig):
    """Configuration for OpenAI file database."""

    type: Literal["openai"] = "openai"

    api_key: str | None = None
    """OpenAI API key (falls back to OPENAI_API_KEY env var)."""

    vector_store_id: str | None = None
    """ID of existing vector store (if None, creates a new one using store_name)."""

    # store_name is inherited from FileDatabaseConfig
    # and used to create a new vector store if vector_store_id is None

    chunking_strategy: Literal["auto", "static"] = "auto"
    """Chunking strategy to use."""

    max_chunk_size: int = 1000
    """Maximum chunk size in tokens (static strategy)."""

    chunk_overlap: int = 200
    """Chunk overlap in tokens (static strategy)."""

    def resolve(self) -> OpenAIFileDatabaseConfig:
        """No resolution needed for OpenAI config."""
        return self


# Union type for file database configs
FileDatabaseConfigUnion = Annotated[
    ComponentBasedConfig | OpenAIFileDatabaseConfig,
    Field(discriminator="type"),
]

# Shorthand literals for the complete configuration
DatabaseShorthand = Literal["openai", "component", "chroma", "qdrant", "pinecone"]


def resolve_database_config(
    config: FileDatabaseConfigUnion | DatabaseShorthand,
) -> FileDatabaseConfigUnion:
    """Resolve database configuration from shorthand or pass through existing.

    Args:
        config: Shorthand string or full configuration object

    Returns:
        Fully resolved configuration object
    """
    if isinstance(config, str):
        match config:
            case "openai":
                return OpenAIFileDatabaseConfig()
            case "component":
                return ComponentBasedConfig()
            case "chroma":
                return ComponentBasedConfig(vector_store="chroma")
            case "qdrant":
                return ComponentBasedConfig(vector_store="qdrant")
            case "pinecone":
                return ComponentBasedConfig(vector_store="pinecone")

    # Return the fully resolved config if it's already a config object
    return config.resolve()
