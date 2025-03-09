"""Document converter using Marker's PDF processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from docler.base import DocumentConverter
from docler.models import Document, Image
from docler.utils import get_mime_from_pil, pil_to_bytes


if TYPE_CHECKING:
    from docler.common_types import StrPath
    from docler.lang_code import SupportedLanguage


ProviderType = Literal["gemini", "ollama", "vertex", "claude"]

PROVIDERS: dict[ProviderType, str] = {
    "gemini": "marker.services.gemini.GoogleGeminiService",
    "ollama": "marker.services.ollama.OllamaService",
    "vertex": "marker.services.vertex.GoogleVertexService",
    "claude": "marker.services.claude.ClaudeService",
}


class MarkerConverter(DocumentConverter):
    """Document converter using Marker's PDF processing."""

    NAME = "marker"
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
        }
        if languages:
            self.config["languages"] = ",".join(languages)
        if llm_provider:
            self.config["use_llm"] = True
        model_dict = create_model_dict()
        llm_cls_path = PROVIDERS.get(llm_provider) if llm_provider else None
        self.converter = PdfConverter(artifact_dict=model_dict, llm_service=llm_cls_path)

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Implementation of abstract method."""
        from marker.output import text_from_rendered
        import upath

        local_file = upath.UPath(file_path)

        # Convert using Marker (CPU-intensive)
        rendered = self.converter(str(local_file))
        content, _, pil_images = text_from_rendered(rendered)

        # Convert PIL images to our Image model
        images: list[Image] = []
        for img_name, pil_img in pil_images.items():
            image_data = pil_to_bytes(pil_img)
            image = Image(
                id=img_name,
                content=image_data,
                mime_type=get_mime_from_pil(pil_img),
                filename=img_name,
            )
            images.append(image)

        return Document(
            content=content,
            images=images,
            title=local_file.stem,
            source_path=str(local_file),
            mime_type=mime_type,
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    output_dir = "E:/markdown-test/"
    converter = MarkerConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
    print("PDF processed successfully.")
