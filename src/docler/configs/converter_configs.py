"""Converter configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from docler.common_types import SupportedLanguage  # noqa: TC001


if TYPE_CHECKING:
    from docler.converters.base import DocumentConverter


FormatterType = Literal["text", "json", "vtt", "srt"]
GoogleSpeechEncoding = Literal["LINEAR16", "FLAC", "MP3"]


def default_languages() -> list[SupportedLanguage]:
    return ["en"]


class BaseConverterConfig(BaseModel):
    """Base configuration for document converters."""

    type: str = Field(init=False)
    """Type discriminator for converter configs."""

    languages: list[SupportedLanguage] = Field(default_factory=default_languages)
    """List of supported languages for the converter."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        raise NotImplementedError


class DoclingConverterConfig(BaseConverterConfig):
    """Configuration for docling-based converter."""

    type: Literal["docling"] = Field("docling", init=False)
    """Type discriminator for docling converter."""

    image_scale: float = 2.0
    """Scale factor for image resizing."""

    generate_images: bool = True
    """Whether to generate images."""

    ocr_engine: str = "easy_ocr"
    """OCR engine to use."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.docling_provider import DoclingConverter

        return DoclingConverter(**self.model_dump(exclude={"type"}))


class MarkItDownConfig(BaseConverterConfig):
    """Configuration for MarkItDown-based converter."""

    type: Literal["markitdown"] = Field("markitdown", init=False)
    """Type discriminator for MarkItDown converter."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.markitdown_provider import MarkItDownConverter

        return MarkItDownConverter(**self.model_dump(exclude={"type"}))


class KreuzbergConfig(BaseConverterConfig):
    """Configuration for Kreuzberg document converter.

    Reference:
    https://docs.kreuzberg.ai/configuration
    """

    type: Literal["kreuzberg"] = Field("kreuzberg", init=False)
    """Type identifier for this converter."""

    force_ocr: bool = False
    """Whether to force OCR for all documents."""

    max_processes: int = Field(default=1, ge=1)
    """Maximum number of concurrent processes."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.kreuzberg_provider import KreuzbergConverter

        return KreuzbergConverter(**self.model_dump(exclude={"type"}))


class DataLabConfig(BaseConverterConfig):
    """Configuration for DataLab-based converter."""

    type: Literal["datalab"] = Field("datalab", init=False)
    """Type discriminator for DataLab converter."""

    api_key: str | None = None
    """DataLab API key. If None, will try env var DATALAB_API_KEY."""

    mode: Literal["marker", "table_rec", "ocr", "layout"] = "marker"
    """API endpoint to use."""

    force_ocr: bool = False
    """Whether to force OCR on every page."""

    use_llm: bool = False
    """Whether to use LLM for enhanced accuracy."""

    max_pages: int | None = None
    """Maximum number of pages to process."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.datalab_provider import DataLabConverter

        return DataLabConverter(**self.model_dump(exclude={"type"}))


class LLMConverterConfig(BaseConverterConfig):
    """Configuration for LLM-based converter."""

    type: Literal["llm"] = Field("llm", init=False)
    """Type discriminator for LLM converter."""

    model: str = "gemini/gemini-2.0-flash"
    """LLM model to use."""

    system_prompt: str | None = None
    """Optional system prompt to guide conversion."""

    user_prompt: str | None = None
    """Custom prompt for the conversion task."""

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    """Sampling temperature."""

    max_tokens: int | None = None
    """Maximum tokens in response."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.llm_provider import LLMConverter

        return LLMConverter(**self.model_dump(exclude={"type"}))


class MistralConfig(BaseConverterConfig):
    """Configuration for Mistral-based converter."""

    type: Literal["mistral"] = Field("mistral", init=False)
    """Type discriminator for Mistral converter."""

    api_key: str | None = None
    """Mistral API key. If None, will try env var MISTRAL_API_KEY."""

    ocr_model: str = "mistral-ocr-latest"
    """Mistral OCR model to use."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.mistral_provider import MistralConverter

        return MistralConverter(**self.model_dump(exclude={"type"}))


class LlamaParseConfig(BaseConverterConfig):
    """Configuration for LlamaParse-based converter."""

    type: Literal["llamaparse"] = Field("llamaparse", init=False)
    """Type discriminator for LlamaParse converter."""

    api_key: str | None = None
    """LlamaParse API key. Falls back to LLAMAPARSE_API_KEY env var."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.llamaparse_provider import LlamaParseConverter

        return LlamaParseConverter(**self.model_dump(exclude={"type"}))


class LlamaIndexConfig(BaseConverterConfig):
    """Configuration for LlamaIndex document converter."""

    type: Literal["llamaindex"] = Field("llamaindex", init=False)
    """Type discriminator for LlamaIndex converter."""

    llmsherpa_api_url: str = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    """API endpoint for LLMSherpa PDF parser."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.llamaindex_provider import LlamaIndexConverter

        return LlamaIndexConverter(**self.model_dump(exclude={"type"}))


class AzureConfig(BaseConverterConfig):
    """Configuration for Azure Document Intelligence converter."""

    type: Literal["azure"] = Field("azure", init=False)
    """Type discriminator for Azure converter."""

    endpoint: str | None = None
    """Azure endpoint URL. Falls back to AZURE_DOC_INTELLIGENCE_ENDPOINT envvar."""

    api_key: str | None = None
    """Azure API key. Falls back to AZURE_DOC_INTELLIGENCE_KEY env var."""

    model: Literal[
        "prebuilt-read",
        "prebuilt-layout",
        "prebuilt-document",
        "prebuilt-idDocument",
        "prebuilt-receipt",
    ] = "prebuilt-document"
    """Pre-trained model to use."""

    additional_features: list[str] = Field(default_factory=list)
    """Optional add-on capabilities like BARCODES, FORMULAS, OCR_HIGH_RESOLUTION etc."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.azure_provider import AzureConverter

        return AzureConverter(**self.model_dump(exclude={"type"}))


class MarkerConfig(BaseConverterConfig):
    """Configuration for Marker-based converter."""

    type: Literal["marker"] = Field("marker", init=False)
    """Type discriminator for Marker converter."""

    dpi: int = 192
    """DPI setting for image extraction."""

    llm_provider: Literal["gemini", "ollama", "vertex", "claude"] | None = None
    """Language model provider to use for OCR."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.marker_provider import MarkerConverter

        return MarkerConverter(**self.model_dump(exclude={"type"}))


ConverterConfig = Annotated[
    DataLabConfig
    | DoclingConverterConfig
    | KreuzbergConfig
    | LLMConverterConfig
    | MarkItDownConfig
    | MistralConfig
    | LlamaParseConfig
    | LlamaIndexConfig
    | AzureConfig
    | MarkerConfig,
    Field(discriminator="type"),
]
