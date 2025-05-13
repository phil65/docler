"""Converter configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

# from docling.datamodel.pipeline_options import (
#     EasyOcrOptions,
#     OcrMacOptions,
#     RapidOcrOptions,
#     TesseractCliOcrOptions,
#     TesseractOcrOptions,
# )
from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import SettingsConfigDict
from schemez import ModelIdentifier  # noqa: TC002

from docler.common_types import DEFAULT_CONVERTER_MODEL, SupportedLanguage
from docler.provider import ProviderConfig


if TYPE_CHECKING:
    from docler.converters.base import DocumentConverter


ConverterShorthand = Literal[
    "docling",
    "docling_remote",
    "marker",
    "mistral",
    "llamaparse",
    "datalab",
    "azure",
    "llm",
]

DoclingEngine = Literal[
    "easy_ocr", "tesseract_cli_ocr", "tesseract_ocr", "ocr_mac", "rapid_ocr"
]

UpstageOCRType = Literal["auto", "force"]
UpstageOutputFormat = Literal["markdown", "text", "html"]
UpstageCategory = Literal["figure", "chart", "table", "paragraph"]

AzureModel = Literal[
    "prebuilt-read",
    "prebuilt-layout",
    "prebuilt-idDocument",
    "prebuilt-receipt",
]

AzureFeatureFlag = Literal[
    "ocrHighResolution",
    "languages",
    "barcodes",
    "formulas",
    "keyValuePairs",
    "styleFont",
    "queryFields",
]


LlamaParseMode = Literal[
    "parse_page_without_llm",
    "parse_page_with_llm",
    "parse_page_with_lvm",
    "parse_page_with_agent",
    "parse_document_with_llm",
]


MarkerLLMProvider = Literal["gemini", "ollama", "vertex", "claude"]


LLM_SYSTEM_PROMPT = """
You are a OCR engine that can parse PDF documents & images into markdown formatted text.
"""


LLM_USER_PROMPT = """
Convert this PDF document to markdown format
Preserve the original formatting and structure where possible.
Include any important tables or lists.
Describe any images you see in brackets.
Indicate page breaks with a specially formatted JSON comment,
like <!-- docler:page_break {"next_page":2} --> for page 2.
Only respond with the content, nothing else.
"""


def default_languages() -> set[SupportedLanguage]:
    return {"en"}


class BaseConverterConfig(ProviderConfig):
    """Base configuration for document converters."""

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        raise NotImplementedError


class DoclingConverterConfig(BaseConverterConfig):
    """Configuration for docling-based converter."""

    type: Literal["docling"] = Field("docling", init=False)
    """Type discriminator for docling converter."""

    languages: set[SupportedLanguage] = Field(default_factory=default_languages)
    """List of supported languages for the converter."""

    image_scale: float = Field(default=2.0, gt=0)
    """Scale factor for image resizing."""

    generate_images: bool = True
    """Whether to generate images."""

    ocr_engine: (
        DoclingEngine
        # | EasyOcrOptions
        # | TesseractCliOcrOptions
        # | TesseractOcrOptions
        # | OcrMacOptions
        # | RapidOcrOptions
    ) = "easy_ocr"
    """OCR engine to use."""

    model_config = SettingsConfigDict(env_prefix="DOCLING_")

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.docling_provider import DoclingConverter

        return DoclingConverter(**self.get_config_fields())


class MarkItDownConfig(BaseConverterConfig):
    """Configuration for MarkItDown-based converter."""

    type: Literal["markitdown"] = Field("markitdown", init=False)
    """Type discriminator for MarkItDown converter."""

    languages: set[SupportedLanguage] = Field(default_factory=default_languages)
    """List of supported languages for the converter."""

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.markitdown_provider import MarkItDownConverter

        return MarkItDownConverter(**self.get_config_fields())


class DataLabConfig(BaseConverterConfig):
    """Configuration for DataLab-based converter."""

    type: Literal["datalab"] = Field("datalab", init=False)
    """Type discriminator for DataLab converter."""

    languages: set[SupportedLanguage] = Field(default_factory=default_languages)
    """List of supported languages for the converter."""

    api_key: SecretStr | None = None
    """DataLab API key. If None, will try env var DATALAB_API_KEY."""

    force_ocr: bool = False
    """Whether to force OCR on every page."""

    use_llm: bool = False
    """Whether to use LLM for enhanced accuracy."""

    max_pages: int | None = Field(default=None, ge=1)
    """Maximum number of pages to process."""

    model_config = SettingsConfigDict(env_prefix="DATALAB_")

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.datalab_provider import DataLabConverter

        return DataLabConverter(**self.get_config_fields())


class LLMConverterConfig(BaseConverterConfig):
    """Configuration for LLM-based converter."""

    type: Literal["llm"] = Field("llm", init=False)
    """Type discriminator for LLM converter."""

    languages: set[SupportedLanguage] = Field(default_factory=default_languages)
    """List of supported languages for the converter."""

    model: ModelIdentifier = DEFAULT_CONVERTER_MODEL
    """LLM model to use."""

    system_prompt: str = LLM_SYSTEM_PROMPT
    """Optional system prompt to guide conversion."""

    user_prompt: str = LLM_USER_PROMPT
    """Custom prompt for the conversion task."""

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.llm_provider import LLMConverter

        return LLMConverter(**self.get_config_fields())


class MistralConfig(BaseConverterConfig):
    """Configuration for Mistral-based converter."""

    type: Literal["mistral"] = Field("mistral", init=False)
    """Type discriminator for Mistral converter."""

    languages: set[SupportedLanguage] = Field(default_factory=default_languages)
    """List of supported languages for the converter."""

    api_key: SecretStr | None = None
    """Mistral API key. If None, will try env var MISTRAL_API_KEY."""

    model_config = SettingsConfigDict(env_prefix="MISTRAL_")

    # right now there only is one model
    # ocr_model: str = "mistral-ocr-latest"
    # """Mistral OCR model to use."""

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.mistral_provider import MistralConverter

        return MistralConverter(**self.get_config_fields())


class LlamaParseConfig(BaseConverterConfig):
    """Configuration for LlamaParse-based converter."""

    type: Literal["llamaparse"] = Field("llamaparse", init=False)
    """Type discriminator for LlamaParse converter."""

    languages: set[SupportedLanguage] = Field(default_factory=default_languages)
    """List of supported languages for the converter."""

    api_key: SecretStr | None = None
    """LlamaParse API key. Falls back to LLAMAPARSE_API_KEY env var."""

    adaptive_long_table: bool = False
    """Whether to use adaptive long table."""

    parse_mode: LlamaParseMode = "parse_page_with_llm"
    """Parse mode, defaults to "parse_page_with_llm"."""

    skip_diagonal_text: bool = False
    """Whether to skip diagonal text."""

    disable_ocr: bool = False
    """Whether to disable OCR for images."""

    continuous_mode: bool = True
    """Whether to use continuous mode."""

    html_tables: bool = False
    """Whether to output HTML tables instead of markdown."""

    model_config = SettingsConfigDict(env_prefix="LLAMAPARSE_")

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.llamaparse_provider import LlamaParseConverter

        return LlamaParseConverter(**self.get_config_fields())


class AzureConfig(BaseConverterConfig):
    """Configuration for Azure Document Intelligence converter."""

    type: Literal["azure"] = Field("azure", init=False)
    """Type discriminator for Azure converter."""

    languages: set[SupportedLanguage] = Field(default_factory=default_languages)
    """List of supported languages for the converter."""

    endpoint: str | None = None
    """Azure endpoint URL. Falls back to AZURE_DOC_INTELLIGENCE_ENDPOINT envvar."""

    api_key: SecretStr | None = None
    """Azure API key. Falls back to AZURE_DOC_INTELLIGENCE_KEY env var."""

    model: AzureModel = "prebuilt-layout"
    """Pre-trained model to use."""

    additional_features: set[AzureFeatureFlag] = Field(default_factory=set)
    """Optional add-on capabilities like BARCODES, FORMULAS, OCR_HIGH_RESOLUTION etc."""

    model_config = SettingsConfigDict(env_prefix="LLAMAPARSE_")

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.azure_provider import AzureConverter

        return AzureConverter(**self.get_config_fields())


class MarkerConfig(BaseConverterConfig):
    """Configuration for Marker-based converter."""

    type: Literal["marker"] = Field("marker", init=False)
    """Type discriminator for Marker converter."""

    dpi: int = Field(default=192, gt=0)
    """DPI setting for image extraction."""

    llm_provider: MarkerLLMProvider | None = None
    """Language model provider to use for OCR."""

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.marker_provider import MarkerConverter

        return MarkerConverter(**self.get_config_fields())


class UpstageConfig(BaseConverterConfig):
    """Configuration for Upstage Document AI converter."""

    type: Literal["upstage"] = Field("upstage", init=False)
    """Type discriminator for Upstage converter."""

    api_key: SecretStr | None = None
    """Upstage API key. Falls back to UPSTAGE_API_KEY env var."""

    base_url: HttpUrl = HttpUrl("https://api.upstage.ai/v1/document-ai/document-parse")
    """API endpoint URL."""

    # model: str = "document-parse"
    # """Model name for document parsing."""

    ocr: UpstageOCRType = "auto"
    """OCR mode ('auto' or 'force')."""

    base64_categories: set[UpstageCategory] = Field(
        default_factory=lambda: {"figure", "chart"}  # type: ignore
    )
    """Element categories to encode in base64."""

    model_config = SettingsConfigDict(env_prefix="UPSTAGE_")

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.upstage_provider import UpstageConverter

        return UpstageConverter(**self.get_config_fields())


class DoclingRemoteConfig(BaseConverterConfig):
    """Configuration for remote Docling service."""

    type: Literal["docling_remote"] = Field("docling_remote", init=False)
    """Type discriminator for remote docling converter."""

    languages: set[SupportedLanguage] = Field(default_factory=default_languages)
    """List of supported languages for the converter."""

    api_key: SecretStr | None = None
    """Optional API key for auth."""

    endpoint: HttpUrl = HttpUrl("http://localhost:5001")
    """Base URL of the Docling service."""

    ocr_engine: DoclingEngine = "easy_ocr"
    """OCR engine to use."""

    pdf_backend: str = "dlparse_v4"
    """PDF backend to use."""

    table_mode: Literal["fast", "accurate"] = "fast"
    """Table mode to use."""

    force_ocr: bool = False
    """Whether to force OCR even on digital documents."""

    image_scale: float = Field(default=2.0, gt=0)
    """Scale factor for image resolution."""

    do_table_structure: bool = True
    """Extract table structure."""

    do_code_enrichment: bool = False
    """Enable code extraction."""

    do_formula_enrichment: bool = False
    """Enable formula extraction."""

    do_picture_classification: bool = False
    """Enable picture classification."""

    do_picture_description: bool = False
    """Enable picture description."""

    model_config = SettingsConfigDict(env_prefix="DOCLING_REMOTE_")

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.docling_remote_provider import DoclingRemoteConverter

        return DoclingRemoteConverter(**self.get_config_fields())


class AggregatedConverterConfig(BaseConverterConfig):
    """Configuration for the aggregated converter."""

    type: Literal["aggregated"] = Field(default="aggregated", init=False)
    """Type discriminator for aggregated converter."""

    converters: list[ConverterConfig] = Field(default_factory=list)
    """List of converter configurations to use."""

    mime_preferences: dict[str, str] = Field(default_factory=dict)
    """Mapping of MIME types or extensions to preferred converter names."""

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.aggregated_converter import AggregatedConverter

        return AggregatedConverter.from_config(self)


class UnstructuredConverterConfig(BaseConverterConfig):
    """Configuration for Unstructured-based converter."""

    type: Literal["unstructured"] = "unstructured"
    """Type discriminator for Unstructured converter."""

    languages: set[SupportedLanguage] = Field(default_factory=lambda: {"en"})  # type: ignore
    """List of supported languages for the converter."""

    api_key: str | None = None
    """API key for Unstructured (optional for local processing)."""

    strategy: str = "hi_res"
    """Strategy for document processing (hi_res, fast, etc.)."""

    extract_images: bool = True
    """Whether to extract and include images."""

    extract_tables: bool = True
    """Whether to extract tables as images."""

    local_mode: bool = True
    """Whether to use local processing or API."""

    def get_provider(self) -> DocumentConverter:
        """Get the converter instance."""
        from docler.converters.unstructured_provider import UnstructuredConverter

        return UnstructuredConverter(**self.get_config_fields())


ConverterConfig = Annotated[
    DataLabConfig
    | DoclingConverterConfig
    | DoclingRemoteConfig
    | LLMConverterConfig
    | MarkItDownConfig
    | MistralConfig
    | LlamaParseConfig
    | AzureConfig
    | UpstageConfig
    | UnstructuredConverterConfig
    | AggregatedConverterConfig
    | MarkerConfig,
    Field(discriminator="type"),
]
