"""OCR functionality for processing PDF files using Mistral's API."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar

from docler.base import DocumentConverter
from docler.models import Document, Image


if TYPE_CHECKING:
    from docler.common_types import StrPath
    from docler.lang_code import SupportedLanguage


logger = logging.getLogger(__name__)


class MistralConverter(DocumentConverter):
    """Document converter using Mistral's OCR API."""

    NAME = "mistral"
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        api_key: str | None = None,
        ocr_model: str = "mistral-ocr-latest",
    ):
        """Initialize the Mistral converter.

        Args:
            languages: List of supported languages.
            api_key: Mistral API key. If None, will try to get from environment.
            ocr_model: Mistral OCR model to use. Defaults to "mistral-ocr-latest".

        Raises:
            ValueError: If MISTRAL_API_KEY environment variable is not set.
        """
        super().__init__(languages=languages)
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            msg = "MISTRAL_API_KEY environment variable is not set"
            raise ValueError(msg)
        self.model = ocr_model

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Implementation of abstract method."""
        from mistralai import DocumentURLChunk, Mistral
        from mistralai.models import File
        import upath

        pdf_file = upath.UPath(file_path)
        client = Mistral(api_key=self.api_key)
        logger.debug("Uploading file %s...", pdf_file.name)
        data = pdf_file.read_bytes()
        file_ = File(file_name=pdf_file.stem, content=data)
        uploaded = client.files.upload(file=file_, purpose="ocr")  # type: ignore
        signed_url = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
        logger.debug("Processing with OCR model...")
        doc = DocumentURLChunk(document_url=signed_url.url)
        r = client.ocr.process(document=doc, model=self.model, include_image_base64=True)

        # Convert response to our Document format
        images: list[Image] = []
        for page in r.pages:
            for img in page.images:
                # Skip images without required data
                if not img.id or not img.image_base64:
                    continue

                image_data = img.image_base64
                # Ensure proper base64 format
                if image_data.startswith("data:image/"):
                    image_data = image_data.split(",", 1)[1]

                # Determine mime type from filename
                ext = img.id.split(".")[-1].lower() if "." in img.id else "jpeg"
                image = Image(
                    id=img.id,
                    content=image_data,
                    mime_type=f"image/{ext}",
                    filename=img.id,
                )
                images.append(image)

        # Combine markdown content from all pages
        content = "\n\n".join(page.markdown for page in r.pages)
        return Document(
            content=content,
            images=images,
            title=pdf_file.stem,
            source_path=str(pdf_file),
            mime_type=mime_type,
            page_count=len(r.pages),
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    output_dir = "E:/markdown-test/"
    converter = MistralConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
    print("PDF processed successfully.")
