"""Document converter using Docling's PDF processing."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from mkdown import Image, create_image_reference, create_page_break

from docler.converters.base import ConverterResult, DocumentConverter
from docler.converters.docling_provider.utils import convert_languages, parse_page_range
from docler.log import get_logger
from docler.utils import pil_to_bytes
from docler_config.converter_configs import DoclingConverterConfig


if TYPE_CHECKING:
    from io import BytesIO

    from docling.document_converter import FormatOption
    from schemez import MimeType

    from docler.common_types import PageRangeString, SupportedLanguage
    from docler_config.converter_configs import DoclingEngine

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
        page_range: PageRangeString | None = None,
        image_scale: float = 2.0,
        delim: str = "\n\n",
        strict_text: bool = False,
        escaping_underscores: bool = True,
        indent: int = 4,
        text_width: int = -1,
        ocr_engine: DoclingEngine = "easy_ocr",
    ) -> None:
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

        opts: dict[
            str,
            type[
                EasyOcrOptions
                | TesseractCliOcrOptions
                | TesseractOcrOptions
                | OcrMacOptions
                | RapidOcrOptions
            ],
        ] = {
            "easy_ocr": EasyOcrOptions,
            "tesseract_cli_ocr": TesseractCliOcrOptions,
            "tesseract_ocr": TesseractOcrOptions,
            "ocr_mac": OcrMacOptions,
            "rapid_ocr": RapidOcrOptions,
        }
        # Configure pipeline options
        engine = opts.get(ocr_engine)
        assert engine
        ocr_opts = engine(lang=convert_languages(languages or ["en"], engine))
        pipeline_options = PdfPipelineOptions(
            ocr_options=ocr_opts,
            generate_picture_images=True,
            images_scale=image_scale,
            generate_page_images=True,
        )
        fmt_opts: dict[InputFormat, FormatOption] = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        self.converter = DoclingDocumentConverter(format_options=fmt_opts)  # pyright: ignore[reportArgumentType]

    def _convert_sync(self, data: BytesIO, mime_type: MimeType) -> ConverterResult:
        """Convert a PDF file using Docling.

        Args:
            data: File content as BytesIO.
            mime_type: MIME type of the file (must be PDF).

        Returns:
            Intermediate conversion result.
        """
        from docling.datamodel.settings import DEFAULT_PAGE_RANGE
        from docling_core.types.doc.base import ImageRefMode
        from docling_core.types.io import DocumentStream

        source = DocumentStream(name="document.pdf", stream=data)
        page_range = parse_page_range(self.page_range) if self.page_range else None
        doc_result = self.converter.convert(source, page_range=page_range or DEFAULT_PAGE_RANGE)
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

        return ConverterResult(content=mk_content, images=images)


if __name__ == "__main__":
    import anyenv

    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = DoclingConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
