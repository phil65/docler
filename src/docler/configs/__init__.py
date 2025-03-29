"""Configuration models for Docler components."""

from __future__ import annotations

from docler.configs.annotator_configs import (
    AIDocumentAnnotatorConfig,
    AIImageAnnotatorConfig,
    AnnotatorConfig,
    BaseAnnotatorConfig,
    DEFAULT_DOC_PROMPT_TEMPLATE,
    DEFAULT_DOC_SYSTEM_PROMPT,
    DEFAULT_IMAGE_PROMPT_TEMPLATE,
    DEFAULT_IMAGE_SYSTEM_PROMPT,
)
from docler.configs.chunker_configs import (
    AiChunkerConfig,
    BaseChunkerConfig,
    ChunkerConfig,
    LlamaIndexChunkerConfig,
    MarkdownChunkerConfig,
    ChunkerShorthand,
    DEFAULT_CHUNKER_USER_TEMPLATE,
    DEFAULT_CHUNKER_SYSTEM_PROMPT,
)
from docler.configs.converter_configs import (
    AzureConfig,
    AzureFeatureFlag,
    AzureModel,
    BaseConverterConfig,
    ConverterConfig,
    DataLabConfig,
    DoclingConverterConfig,
    DoclingEngine,
    KreuzbergConfig,
    LlamaParseConfig,
    LlamaParseMode,
    LLMConverterConfig,
    MarkerConfig,
    MarkItDownConfig,
    MistralConfig,
    UpstageConfig,
)
from docler.configs.embedding_configs import (
    BaseEmbeddingConfig,
    BGEEmbeddingConfig,
    EmbeddingConfig,
    LiteLLMEmbeddingConfig,
    LiteLLMInputType,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingModel,
    SentenceTransformerEmbeddingConfig,
    SentenceTransformerModel,
)
from docler.configs.file_db_configs import (
    ComponentBasedConfig,
    ConverterShorthand,
    DatabaseShorthand,
    EmbeddingShorthand,
    FileDatabaseConfig,
    FileDatabaseConfigUnion,
    OpenAIFileDatabaseConfig,
    VectorDBShorthand,
    resolve_database_config,
)
from docler.configs.processor_configs import (
    BaseProcessorConfig,
    DEFAULT_PROOF_READER_PROMPT_TEMPLATE,
    DEFAULT_PROOF_READER_SYSTEM_PROMPT,
    LLMProofReaderConfig,
    ProcessorConfig,
)
from docler.configs.vector_db_configs import (
    BaseVectorStoreConfig,
    ChromaConfig,
    KdbAiConfig,
    Metric,
    OpenAIChunkingStrategy,
    OpenAIVectorConfig,
    PineconeCloud,
    PineconeConfig,
    PineconeRegion,
    QdrantConfig,
    VectorStoreConfig,
)

__all__ = [
    "DEFAULT_CHUNKER_SYSTEM_PROMPT",
    "DEFAULT_CHUNKER_USER_TEMPLATE",
    "DEFAULT_DOC_PROMPT_TEMPLATE",
    "DEFAULT_DOC_SYSTEM_PROMPT",
    "DEFAULT_IMAGE_PROMPT_TEMPLATE",
    "DEFAULT_IMAGE_SYSTEM_PROMPT",
    "DEFAULT_PROOF_READER_PROMPT_TEMPLATE",
    "DEFAULT_PROOF_READER_SYSTEM_PROMPT",
    "AIDocumentAnnotatorConfig",
    "AIImageAnnotatorConfig",
    "AiChunkerConfig",
    "AnnotatorConfig",
    "AzureConfig",
    "AzureFeatureFlag",
    "AzureModel",
    "BGEEmbeddingConfig",
    "BaseAnnotatorConfig",
    "BaseChunkerConfig",
    "BaseConverterConfig",
    "BaseEmbeddingConfig",
    "BaseProcessorConfig",
    "BaseVectorStoreConfig",
    "ChromaConfig",
    "ChunkerConfig",
    "ChunkerShorthand",
    "ComponentBasedConfig",
    "ConverterConfig",
    "ConverterShorthand",
    "DataLabConfig",
    "DatabaseShorthand",
    "DoclingConverterConfig",
    "DoclingEngine",
    "EmbeddingConfig",
    "EmbeddingShorthand",
    "FileDatabaseConfig",
    "FileDatabaseConfigUnion",
    "KdbAiConfig",
    "KreuzbergConfig",
    "LLMConverterConfig",
    "LLMProofReaderConfig",
    "LiteLLMEmbeddingConfig",
    "LiteLLMInputType",
    "LlamaIndexChunkerConfig",
    "LlamaParseConfig",
    "LlamaParseMode",
    "MarkItDownConfig",
    "MarkdownChunkerConfig",
    "MarkerConfig",
    "Metric",
    "MistralConfig",
    "OpenAIChunkingStrategy",
    "OpenAIEmbeddingConfig",
    "OpenAIEmbeddingModel",
    "OpenAIFileDatabaseConfig",
    "OpenAIVectorConfig",
    "PineconeCloud",
    "PineconeConfig",
    "PineconeRegion",
    "ProcessorConfig",
    "QdrantConfig",
    "SentenceTransformerEmbeddingConfig",
    "SentenceTransformerModel",
    "UpstageConfig",
    "VectorDBShorthand",
    "VectorStoreConfig",
    "resolve_database_config",
]
