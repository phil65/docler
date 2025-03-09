"""Document converter using Docling's PDF processing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from docler.base import DocumentConverter
from docler.lang_code import SupportedLanguage, convert_languages
from docler.models import Document, Image


if TYPE_CHECKING:
    from docler.common_types import StrPath


logger = logging.getLogger(__name__)


class DoclingConverter(DocumentConverter):
    """Document converter using Docling's processing."""

    NAME = "docling"
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        image_scale: float = 2.0,
        generate_images: bool = True,
        ocr_engine: str = "easy_ocr",
    ) -> None:
        """Initialize the Docling converter.

        Args:
            languages: List of supported languages.
            image_scale: Scale factor for image resolution (1.0 = 72 DPI).
            generate_images: Whether to generate and keep page images.
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
        pipeline_options = PdfPipelineOptions(ocr_options=ocr_opts)
        pipeline_options.images_scale = image_scale
        pipeline_options.generate_page_images = generate_images

        # Create converter
        self.converter = DoclingDocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

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
        from io import BytesIO

        from docling.utils.export import generate_multimodal_pages
        from PIL import Image as PILImage
        import upath

        pdf_path = upath.UPath(file_path)

        # Convert using Docling
        conv_result = self.converter.convert(str(pdf_path))

        # Process all pages
        contents: list[str] = []
        images: list[Image] = []
        page_count = 0

        for content_text, content_md, _, _, _segments, page in generate_multimodal_pages(
            conv_result
        ):
            page_count += 1

            # Add page content (preferring markdown if available)
            contents.append(content_md or content_text)

            # Convert page image to our format
            if page.image:
                # Convert PIL image to bytes
                img_bytes = BytesIO()
                pil_image = PILImage.frombytes(
                    "RGB",
                    (page.image.width, page.image.height),
                    page.image.tobytes(),
                )
                pil_image.save(img_bytes, format="PNG")

                images.append(
                    Image(
                        id=f"page_{page.page_no}",
                        content=img_bytes.getvalue(),
                        mime_type="image/png",
                        filename=f"page_{page.page_no}.png",
                    )
                )

        return Document(
            content="\n\n".join(contents),
            images=images,
            title=pdf_path.stem,
            source_path=str(pdf_path),
            mime_type=mime_type,
            page_count=page_count,
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.INFO)

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    converter = DoclingConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
