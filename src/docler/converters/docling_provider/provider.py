"""Document converter using Docling's PDF processing."""

from __future__ import annotations

from io import BytesIO
import re
from typing import TYPE_CHECKING, ClassVar

from mkdown import Document, Image, create_image_reference, create_page_break
import upath

from docler.configs.converter_configs import DoclingConverterConfig, DoclingEngine
from docler.converters.base import DocumentConverter
from docler.converters.docling_provider.utils import _parse_page_range, convert_languages
from docler.log import get_logger
from docler.utils import pil_to_bytes


if TYPE_CHECKING:
    from collections.abc import Mapping

    from schemez import MimeType

    from docler.common_types import StrPath, SupportedLanguage

PAGE_BREAK_MARKER = "<!-- PageBreak -->"
logger = get_logger(__name__)


class DoclingConverter(DocumentConverter[DoclingConverterConfig]):
    """Document converter using Docling's processing."""

    Config = DoclingConverterConfig

    NAME = "docling"
    REQUIRED_PACKAGES: ClassVar = {"docling"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        page_range: str | None = None,
        image_scale: float = 2.0,
        delim: str = "\n\n",
        strict_text: bool = False,
        escaping_underscores: bool = True,
        indent: int = 4,
        text_width: int = -1,
        ocr_engine: DoclingEngine = "easy_ocr",
    ):
        """Initialize the Docling converter.

        Args:
            languages: List of supported languages.
            page_range: Page range(s) to extract, like "1-5,7-10" (1-based)
            image_scale: Scale factor for image resolution (1.0 = 72 DPI).
            delim: Delimiter for markdown sections.
            strict_text: Whether to use strict text processing.
            escaping_underscores: Whether to escape underscores.
            indent: Indentation level for markdown sections.
            text_width: Maximum width for text in markdown sections.
            ocr_engine: The OCR engine to use.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            OcrMacOptions,
            OcrOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
            TesseractCliOcrOptions,
            TesseractOcrOptions,
        )
        from docling.document_converter import (
            DocumentConverter as DoclingDocumentConverter,
            PdfFormatOption,
        )

        super().__init__(languages=languages, page_range=page_range)
        self.delim = delim
        self.strict_text = strict_text
        self.escaping_underscores = escaping_underscores
        self.indent = indent
        self.text_width = text_width

        opts: Mapping[DoclingEngine, type[OcrOptions]] = {
            "easy_ocr": EasyOcrOptions,
            "tesseract_cli_ocr": TesseractCliOcrOptions,
            "tesseract_ocr": TesseractOcrOptions,
            "ocr_mac": OcrMacOptions,
            "rapid_ocr": RapidOcrOptions,
        }
        # Configure pipeline options
        engine = opts.get(ocr_engine)
        assert engine
        ocr_opts = engine(lang=convert_languages(languages or ["en"], engine))  # type: ignore
        pipeline_options = PdfPipelineOptions(
            ocr_options=ocr_opts,
            generate_picture_images=True,
            images_scale=image_scale,
            generate_page_images=True,
        )
        fmt_opts = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        self.converter = DoclingDocumentConverter(format_options=fmt_opts)  # type: ignore

    def _convert_path_sync(self, file_path: StrPath, mime_type: MimeType) -> Document:
        """Convert a PDF file using Docling.

        Args:
            file_path: Path to the PDF file to process.
            mime_type: MIME type of the file (must be PDF).

        Returns:
            Converted document with extracted text and images.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a PDF.
        """
        from docling.datamodel.settings import DEFAULT_PAGE_RANGE
        from docling_core.types.doc.base import ImageRefMode
        from docling_core.types.io import DocumentStream

        pdf_path = upath.UPath(file_path)
        stream = BytesIO(pdf_path.read_bytes())
        source = DocumentStream(name=pdf_path.name, stream=stream)
        page_range = _parse_page_range(self.page_range)
        doc_result = self.converter.convert(
            source, page_range=page_range or DEFAULT_PAGE_RANGE
        )
        mk_content = doc_result.document.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED,
            delim=self.delim,
            indent=self.indent,
            text_width=self.text_width,
            escape_underscores=self.escaping_underscores,
            strict_text=self.strict_text,
            page_break_placeholder=PAGE_BREAK_MARKER,
        )
        page_num = 1

        def replace_marker(match: re.Match[str]) -> str:
            nonlocal page_num
            page_num += 1
            return create_page_break(next_page=page_num, newline_separators=1)

        mk_content = re.sub(PAGE_BREAK_MARKER, replace_marker, mk_content)

        first_page_marker = create_page_break(next_page=1, newline_separators=1).lstrip()
        mk_content = first_page_marker + mk_content.lstrip()

        images: list[Image] = []
        for i, picture in enumerate(doc_result.document.pictures):
            if not picture.image or not picture.image.pil_image:
                continue
            image_id = f"img-{i}"
            filename = f"{image_id}.png"
            mk_link = create_image_reference(image_id, filename)
            mk_content = mk_content.replace("<!-- image -->", mk_link, 1)
            content = pil_to_bytes(picture.image.pil_image)
            mime = "image/png"
            image = Image(id=image_id, content=content, mime_type=mime, filename=filename)
            images.append(image)

        return Document(
            content=mk_content,
            images=images,
            title=pdf_path.stem,
            source_path=str(pdf_path),
            mime_type=mime_type,
        )


if __name__ == "__main__":
    import anyenv

    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = DoclingConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
