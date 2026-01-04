"""OCR functionality for processing documents using Mistral's API."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, ClassVar

from mkdown import Image, create_image_reference, create_page_break

from docler.converters.base import ConverterResult, DocumentConverter
from docler.converters.mistral_provider.utils import convert_image, get_images
from docler.pdf_utils import parse_page_range, shift_page_range
from docler.utils import get_api_key
from docler_config.converter_configs import MistralConfig


if TYPE_CHECKING:
    from io import BytesIO

    from schemez import MimeType

    from docler.common_types import PageRangeString, SupportedLanguage


# https://docs.mistral.ai/api/#tag/ocr


class MistralConverter(DocumentConverter[MistralConfig]):
    """Document converter using Mistral's OCR API."""

    Config = MistralConfig

    NAME = "mistral"
    REQUIRED_PACKAGES: ClassVar = {"mistralai"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/tiff",
    }

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        page_range: PageRangeString | None = None,
        api_key: str | None = None,
        ocr_model: str = "mistral-ocr-latest",
        image_min_size: int | None = None,
    ) -> None:
        """Initialize the Mistral converter.

        Args:
            languages: List of supported languages.
            page_range: Page range(s) to extract, like "1-5,7-10" (1-based)
            api_key: Mistral API key. If None, will try to get from environment.
            ocr_model: Mistral OCR model to use. Defaults to "mistral-ocr-latest".
            image_min_size: Minimum size of image in pixels.

        Raises:
            ValueError: If MISTRAL_API_KEY environment variable is not set.
        """
        super().__init__(languages=languages, page_range=page_range)
        self.api_key = api_key or get_api_key("MISTRAL_API_KEY")
        self.model = ocr_model
        self.image_min_size = image_min_size

    def _convert_sync(self, data: BytesIO, mime_type: MimeType) -> ConverterResult:
        """Implementation of abstract method."""
        file_data = data.read()

        if mime_type.startswith("image/"):
            result = self._process_image(file_data, mime_type)
            first_page_marker = create_page_break(next_page=1, newline_separators=1)
            result.content = first_page_marker.lstrip() + result.content.lstrip()
            return result
        # PDF processing handles page breaks internally
        return self._process_pdf(file_data, mime_type)

    def _process_pdf(
        self,
        file_data: bytes,
        mime_type: MimeType,
    ) -> ConverterResult:
        """Process a PDF file using Mistral OCR.

        Args:
            file_data: Raw PDF data
            mime_type: MIME type of the file

        Returns:
            Intermediate conversion result
        """
        from mistralai import Mistral
        from mistralai.models import File

        client = Mistral(api_key=self.api_key)
        self.logger.debug("Uploading PDF file...")

        file_ = File(file_name="document.pdf", content=file_data)
        uploaded = client.files.upload(file=file_, purpose="ocr")
        signed_url = client.files.get_signed_url(file_id=uploaded.id, expiry=60)

        self.logger.debug("Processing with OCR model...")
        rng = shift_page_range(self.page_range, -1) if self.page_range else None
        r = client.ocr.process(
            model=self.model,
            document={"type": "document_url", "document_url": signed_url.url},
            include_image_base64=True,
            image_min_size=self.image_min_size,
            pages=list(parse_page_range(rng)) if rng else None,
        )
        images = [
            convert_image(img)
            for page in r.pages
            for img in page.images
            if img.id and img.image_base64
        ]

        content_parts: list[str] = []
        if r.pages:
            # Always add marker for the first page
            first_page_marker = create_page_break(next_page=1, newline_separators=1)
            content_parts.append(first_page_marker.lstrip())
            content_parts.append(r.pages[0].markdown.lstrip())  # Add first page content
            page_num = 1
            for page in r.pages[1:]:
                page_num += 1  # Increment for the next page in the output
                # Use newline_separators=1 to potentially reduce vertical space
                comment = create_page_break(next_page=page_num, newline_separators=1)
                content_parts.append(comment)
                content_parts.append(page.markdown.lstrip())
        content = "\n\n".join(content_parts)  # Use double newline between parts

        return ConverterResult(content=content, images=images)

    def _process_image(
        self,
        file_data: bytes,
        mime_type: MimeType,
    ) -> ConverterResult:
        """Process an image file using Mistral OCR.

        Args:
            file_data: Raw image data
            mime_type: MIME type of the file

        Returns:
            Intermediate conversion result (without the initial page 1 marker, added later)
        """
        from mistralai import Mistral

        client = Mistral(api_key=self.api_key)
        self.logger.debug("Processing image with Mistral OCR...")
        img_b64 = base64.b64encode(file_data).decode("utf-8")
        img_url = f"data:{mime_type};base64,{img_b64}"

        # Process with OCR using the correct document format
        r = client.ocr.process(
            model=self.model,
            document={"type": "image_url", "image_url": img_url},
            include_image_base64=True,
            image_min_size=self.image_min_size,
        )

        # Extract the content (for images, we'll usually have just one page)
        content = "\n\n".join(page.markdown for page in r.pages)
        image_id = "img-0"
        # Use a generic filename since we don't have path info
        ext = mime_type.split("/")[-1]
        filename = f"image.{ext}"
        image = Image(
            id=image_id,
            content=file_data,
            mime_type=mime_type,
            filename=filename,
        )
        image_ref = create_image_reference(image_id, filename)
        content = image_ref + "\n\n" + content
        additional_images = get_images(r)
        return ConverterResult(content=content, images=[image, *additional_images])


if __name__ == "__main__":
    import devtools

    async def main() -> None:
        # # Example usage with PDF
        pdf_path = "/home/phil65/dev/oss/docler/tests/resources/pdf_sample_page_nums.pdf"
        converter = MistralConverter()
        result = await converter.convert_file(pdf_path)

        # Example usage with image
        # img_path = "E:/sap.png"
        # result = anyenv.run_sync(converter.convert_file(img_path))
        devtools.debug(result.content)
        await result.export_to_directory(".")

    import asyncio

    asyncio.run(main())
