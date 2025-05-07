"""Document converter using Marker's PDF processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import upath

from docler.configs.converter_configs import MarkerConfig
from docler.converters.base import DocumentConverter
from docler.converters.datalab_provider.utils import process_response
from docler.log import get_logger
from docler.models import Document


logger = get_logger(__name__)


if TYPE_CHECKING:
    from marker.output import MarkdownOutput

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
        dpi: int = 192,
        llm_provider: ProviderType | None = None,
    ):
        """Initialize the Marker converter.

        Args:
            dpi: DPI setting for image extraction.
            languages: Languages to use for OCR.
            llm_provider: Language model provider to use for OCR.
        """
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        super().__init__(languages=languages)
        self.config = {
            "output_format": "markdown",
            "highres_image_dpi": dpi,
            "paginate_output": True,
        }
        if languages:
            self.config["languages"] = ",".join(languages)
        if llm_provider:
            self.config["use_llm"] = True
        model_dict = create_model_dict()
        llm_cls_path = PROVIDERS.get(llm_provider) if llm_provider else None
        self.converter = PdfConverter(
            artifact_dict=model_dict,
            llm_service=llm_cls_path,
            config=self.config,
        )

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Implementation of abstract method."""
        local_file = upath.UPath(file_path)
        rendered: MarkdownOutput = self.converter(str(local_file))
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
