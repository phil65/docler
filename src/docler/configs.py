"""Converter configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from docler.base import DocumentConverter
from docler.lang_code import SupportedLanguage


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
        from docler.docling_provider import DoclingConverter

        return DoclingConverter(**self.model_dump(exclude={"type"}))


class MarkItDownConfig(BaseConverterConfig):
    """Configuration for MarkItDown-based converter."""

    type: Literal["markitdown"] = Field("markitdown", init=False)
    """Type discriminator for MarkItDown converter."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.markitdown_provider import MarkItDownConverter

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
        from docler.kreuzberg_provider import KreuzbergConverter

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
        from docler.datalab_provider import DataLabConverter

        return DataLabConverter(**self.model_dump(exclude={"type"}))


class LiteLLMConfig(BaseConverterConfig):
    """Configuration for LiteLLM-based converter."""

    type: Literal["litellm"] = Field("litellm", init=False)
    """Type discriminator for LiteLLM converter."""

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
        from docler.litellm_provider import LiteLLMConverter

        return LiteLLMConverter(**self.model_dump(exclude={"type"}))


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
        from docler.mistral_provider import MistralConverter

        return MistralConverter(**self.model_dump(exclude={"type"}))


class OlmConfig(BaseConverterConfig):
    """Configuration for OLM-based converter."""

    type: Literal["olm"] = Field("olm", init=False)
    """Type discriminator for OLM converter."""

    model_name: str = "ollm/ollm-ocr-v1.0"
    """Name of the OLM model to use."""

    device: str | None = None
    """Device to run model on. If None, will use CUDA if available."""

    engine: Literal["pdftotext", "pdfium", "pypdf", "topcoherency", "pdfreport"] = (
        "pdfreport"
    )
    """PDF engine to use."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.olmocr_provider import OlmConverter

        return OlmConverter(**self.model_dump(exclude={"type"}))


ConverterConfig = Annotated[
    DataLabConfig
    | DoclingConverterConfig
    | KreuzbergConfig
    | LiteLLMConfig
    | MarkItDownConfig
    | MistralConfig
    | OlmConfig,
    Field(discriminator="type"),
]
