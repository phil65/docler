"""Document converter using Marker's PDF processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import upath

from docler.configs.converter_configs import MarkerConfig
from docler.converters.base import DocumentConverter
from docler.models import Document, Image
from docler.utils import get_mime_from_pil, pil_to_bytes


if TYPE_CHECKING:
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

        self.config = {"output_format": "markdown", "highres_image_dpi": dpi}
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

        local_file = upath.UPath(file_path)
        rendered = self.converter(str(local_file))
        content, _, pil_images = text_from_rendered(rendered)
        images: list[Image] = []
        image_replacements = {}
        for img_name, pil_img in pil_images.items():
            # Create standardized image ID and filename
            image_count = len(images)
            id_ = f"img-{image_count}"
            filename = f"{id_}.png"
            image_replacements[img_name] = (id_, filename)
            image_data = pil_to_bytes(pil_img)
            mime = get_mime_from_pil(pil_img)
            image = Image(id=id_, content=image_data, mime_type=mime, filename=filename)
            images.append(image)
        # Replace image references in content
        # Match Marker's format: ![](_page_X_Picture_Y.jpeg)
        for old_name, (image_id, filename) in image_replacements.items():
            old_ref = f"![]({old_name})"
            new_ref = f"![{image_id}]({filename})"
            content = content.replace(old_ref, new_ref)
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
    print(result)
