"""Document converter using Docling's PDF processing."""

from __future__ import annotations

from io import BytesIO
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
        import re

        import upath

        pdf_path = upath.UPath(file_path)

        # Convert using Docling
        doc_result = self.converter.convert(str(pdf_path))

        # Get markdown content
        markdown_content = doc_result.document.export_to_markdown()

        # Find all image placeholders
        image_placeholders = re.findall(r"<!-- image -->", markdown_content)

        # Prepare images
        images: list[Image] = []
        image_replacements = []

        # Process each page with an image
        for page_item in doc_result.pages:
            if page_item.image:
                # Create image ID and filename
                image_count = len(images) + 1
                image_id = f"img-{image_count}"
                filename = f"{image_id}.png"

                # Prepare image replacement
                image_replacements.append((image_id, filename))

                # Convert PIL image to bytes
                img_bytes = BytesIO()
                pil_image = page_item.image
                pil_image.save(img_bytes, format="PNG")

                # Create image object
                image = Image(
                    id=image_id,
                    content=img_bytes.getvalue(),
                    mime_type="image/png",
                    filename=filename,
                )
                images.append(image)

        # Replace placeholders with actual image references
        for i, _placeholder in enumerate(image_placeholders):
            if i < len(image_replacements):
                image_id, filename = image_replacements[i]
                markdown_content = markdown_content.replace(
                    "<!-- image -->",
                    f"![{image_id}]({filename})",
                    1,  # Replace only the first occurrence
                )

        return Document(
            content=markdown_content,
            images=images,
            title=pdf_path.stem,
            source_path=str(pdf_path),
            mime_type=mime_type,
            page_count=len(doc_result.pages),
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.INFO)

    pdf_path = "C:/Users/phili/Downloads/2402.079271.pdf"
    converter = DoclingConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
