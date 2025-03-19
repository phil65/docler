"""Document converter using Docling's PDF processing."""

from __future__ import annotations

from io import BytesIO
import logging
from typing import TYPE_CHECKING, ClassVar

from docler.configs.converter_configs import DoclingConverterConfig
from docler.converters.base import DocumentConverter
from docler.converters.docling_provider.utils import convert_languages
from docler.models import Document, Image


if TYPE_CHECKING:
    from docler.common_types import StrPath, SupportedLanguage


logger = logging.getLogger(__name__)


class DoclingConverter(DocumentConverter[DoclingConverterConfig]):
    """Document converter using Docling's processing."""

    NAME = "docling"
    REQUIRED_PACKAGES: ClassVar = {"docling"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        image_scale: float = 2.0,
        generate_images: bool = True,
        delim: str = "\n\n",
        strict_text: bool = False,
        escaping_underscores: bool = True,
        indent: int = 4,
        text_width: int = -1,
        ocr_engine: str = "easy_ocr",
    ):
        """Initialize the Docling converter.

        Args:
            languages: List of supported languages.
            image_scale: Scale factor for image resolution (1.0 = 72 DPI).
            generate_images: Whether to generate and keep page images.
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

        super().__init__(languages=languages)
        self.delim = delim
        self.strict_text = strict_text
        self.escaping_underscores = escaping_underscores
        self.indent = indent
        self.text_width = text_width
        opts = dict(
            easy_ocr=EasyOcrOptions,
            tesseract_cli_ocr=TesseractCliOcrOptions,
            tesseract_ocr=TesseractOcrOptions,
            ocr_mac=OcrMacOptions,
            rapid_ocr=RapidOcrOptions,
        )
        # Configure pipeline options
        engine = opts.get(ocr_engine)
        assert engine
        ocr_opts = engine(lang=convert_languages(languages or ["en"], engine))  # type: ignore
        pipeline_options = PdfPipelineOptions(
            ocr_options=ocr_opts, generate_picture_images=True
        )
        pipeline_options.images_scale = image_scale
        pipeline_options.generate_page_images = generate_images
        fmt_opts = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        self.converter = DoclingDocumentConverter(format_options=fmt_opts)  # type: ignore

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
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
        from docling_core.types.doc.base import ImageRefMode
        import upath

        pdf_path = upath.UPath(file_path)

        # Convert using Docling
        doc_result = self.converter.convert(str(pdf_path))

        # Get markdown with placeholders
        mk_content = doc_result.document.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED,
            delim=self.delim,
            indent=self.indent,
            text_width=self.text_width,
            escaping_underscores=self.escaping_underscores,
            strict_text=self.strict_text,
        )

        # Process actual images from the document
        images: list[Image] = []
        for i, picture in enumerate(doc_result.document.pictures):
            if not picture.image or not picture.image.pil_image:
                continue
            image_id = f"img-{i}"
            filename = f"{image_id}.png"
            mk_link = f"![{image_id}]({filename})"
            mk_content = mk_content.replace("<!-- image -->", mk_link, 1)
            # Convert PIL image to bytes
            img_bytes = BytesIO()
            picture.image.pil_image.save(img_bytes, format="PNG")
            content = img_bytes.getvalue()
            mime = "image/png"
            image = Image(id=image_id, content=content, mime_type=mime, filename=filename)
            images.append(image)

        return Document(
            content=mk_content,
            images=images,
            title=pdf_path.stem,
            source_path=str(pdf_path),
            mime_type=mime_type,
            page_count=len(doc_result.pages),
        )


if __name__ == "__main__":
    import anyenv

    pdf_path = "C:/Users/phili/Downloads/2402.079271.pdf"
    converter = DoclingConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
