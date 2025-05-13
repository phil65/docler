"""Document converter using Marker's PDF processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from mkdown import Document
import upath

from docler.configs.converter_configs import MarkerConfig
from docler.converters.base import DocumentConverter
from docler.converters.datalab_provider.utils import process_response
from docler.log import get_logger
from docler.utils import shift_page_range


logger = get_logger(__name__)


if TYPE_CHECKING:
    from marker.output import MarkdownOutput
    from schemez import MimeType

    from docler.common_types import StrPath, SupportedLanguage


ProviderType = Literal["gemini", "ollama", "vertex", "claude"]

PROVIDERS: dict[ProviderType, str] = {
    "gemini": "marker.services.gemini.GoogleGeminiService",
    "ollama": "marker.services.ollama.OllamaService",
    "vertex": "marker.services.vertex.GoogleVertexService",
    "claude": "marker.services.claude.ClaudeService",
}


class MarkerConverter(DocumentConverter[MarkerConfig]):
    """Document converter using Marker's PDF processing."""

    Config = MarkerConfig

    NAME = "marker"
    REQUIRED_PACKAGES: ClassVar = {"marker-pdf"}
    SUPPORTED_MIME_TYPES: ClassVar = {
        # PDF
        "application/pdf",
        # Images
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/tiff",
        # EPUB
        "application/epub+zip",
        # Office Documents
        "application/msword",  # doc
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
        "application/vnd.oasis.opendocument.text",  # odt
        # Spreadsheets
        "application/vnd.ms-excel",  # xls
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
        "application/vnd.oasis.opendocument.spreadsheet",  # ods
        # Presentations
        "application/vnd.ms-powerpoint",  # ppt
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx  # noqa: E501
        "application/vnd.oasis.opendocument.presentation",  # odp
        # HTML
        "text/html",
    }

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        page_range: str | None = None,
        dpi: int = 192,
        use_llm: bool = False,
        llm_provider: ProviderType | None = None,
    ):
        """Initialize the Marker converter.

        Args:
            page_range: Page range(s) to extract, like "1-5,7-10" (1-based)
            dpi: DPI setting for image extraction.
            languages: Languages to use for OCR.
            use_llm: Whether to use LLM for enhanced accuracy.
            llm_provider: Language model provider to use for OCR.
        """
        super().__init__(languages=languages, page_range=page_range)
        self.config = {
            "output_format": "markdown",
            "highres_image_dpi": dpi,
            "paginate_output": True,
        }
        if languages:
            self.config["languages"] = ",".join(languages)
        if llm_provider:
            self.config["use_llm"] = use_llm
        if page_range is not None:
            # Marker API expects 0-based page ranges
            rng = shift_page_range(page_range, -1) if page_range else None
            self.config["page_range"] = rng
        self.llm_provider = llm_provider

    def _convert_path_sync(self, file_path: StrPath, mime_type: MimeType) -> Document:
        """Implementation of abstract method."""
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        local_file = upath.UPath(file_path)
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
            llm_service=PROVIDERS.get(self.llm_provider) if self.llm_provider else None,  # pyright: ignore
            config=self.config,
        )
        rendered: MarkdownOutput = converter(str(local_file))
        content, images = process_response(rendered.model_dump())
        return Document(
            content=content,
            images=images,
            title=local_file.stem,
            source_path=str(local_file),
            mime_type=mime_type,
        )


if __name__ == "__main__":
    import anyenv

    pdf_path = "src/docler/resources/pdf_sample.pdf"
    output_dir = "E:/markdown-test/"
    converter = MarkerConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
