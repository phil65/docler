"""OCR functionality for processing documents using Mistral's API."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, ClassVar

import upath

from docler.configs.converter_configs import MistralConfig
from docler.converters.base import DocumentConverter
from docler.converters.mistral_provider.utils import convert_image
from docler.models import Document, Image
from docler.utils import get_api_key


if TYPE_CHECKING:
    from docler.common_types import StrPath, SupportedLanguage


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
        api_key: str | None = None,
        ocr_model: str = "mistral-ocr-latest",
        image_min_size: int | None = None,
    ):
        """Initialize the Mistral converter.

        Args:
            languages: List of supported languages.
            api_key: Mistral API key. If None, will try to get from environment.
            ocr_model: Mistral OCR model to use. Defaults to "mistral-ocr-latest".
            image_min_size: Minimum size of image in pixels.

        Raises:
            ValueError: If MISTRAL_API_KEY environment variable is not set.
        """
        super().__init__(languages=languages)
        self.api_key = api_key or get_api_key("MISTRAL_API_KEY")
        self.model = ocr_model
        self.image_min_size = image_min_size

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Implementation of abstract method."""
        local_file = upath.UPath(file_path)
        data = local_file.read_bytes()

        if mime_type.startswith("image/"):
            return self._process_image(data, local_file, mime_type)
        return self._process_pdf(data, local_file, mime_type)

    def _process_pdf(
        self, file_data: bytes, file_path: upath.UPath, mime_type: str
    ) -> Document:
        """Process a PDF file using Mistral OCR.

        Args:
            file_data: Raw PDF data
            file_path: Path to the file (for metadata)
            mime_type: MIME type of the file

        Returns:
            Converted document
        """
        from mistralai import Mistral
        from mistralai.models import File

        client = Mistral(api_key=self.api_key)
        self.logger.debug("Uploading PDF file %s...", file_path.name)

        file_ = File(file_name=file_path.stem, content=file_data)
        uploaded = client.files.upload(file=file_, purpose="ocr")  # type: ignore
        signed_url = client.files.get_signed_url(file_id=uploaded.id, expiry=1)

        self.logger.debug("Processing with OCR model...")
        r = client.ocr.process(
            model=self.model,
            document={"type": "document_url", "document_url": signed_url.url},
            include_image_base64=True,
            image_min_size=self.image_min_size,
        )
        images = [
            convert_image(img)
            for page in r.pages
            for img in page.images
            if img.id and img.image_base64
        ]

        content = "\n\n".join(page.markdown for page in r.pages)
        return Document(
            content=content,
            images=images,
            title=file_path.stem,
            source_path=str(file_path),
            mime_type=mime_type,
            page_count=len(r.pages),
        )

    def _process_image(
        self, file_data: bytes, file_path: upath.UPath, mime_type: str
    ) -> Document:
        """Process an image file using Mistral OCR.

        Args:
            file_data: Raw image data
            file_path: Path to the file (for metadata)
            mime_type: MIME type of the file

        Returns:
            Converted document
        """
        from mistralai import Mistral

        client = Mistral(api_key=self.api_key)
        self.logger.debug("Processing image %s with Mistral OCR...", file_path.name)

        # Convert raw image to base64
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
        image = Image(
            id=image_id,
            content=file_data,  # Store the original image
            mime_type=mime_type,
            filename=file_path.name,
        )
        image_ref = f"\n\n![{image_id}]({file_path.name})\n\n"
        content = image_ref + content
        additional_images = []
        for page in r.pages:
            for idx, img in enumerate(page.images):
                if not img.id or not img.image_base64:
                    continue
                img_data = img.image_base64
                if img_data.startswith("data:image/"):
                    img_data = img_data.split(",", 1)[1]
                ext = img.id.split(".")[-1].lower() if "." in img.id else "jpeg"
                mime = f"image/{ext}"
                img_id = f"extracted-img-{idx}"
                filename = f"{img_id}.{ext}"
                obj = Image(
                    id=img_id, content=img_data, mime_type=mime, filename=filename
                )
                additional_images.append(obj)

        return Document(
            content=content,
            images=[image, *additional_images],
            title=file_path.stem,
            source_path=str(file_path),
            mime_type=mime_type,
            page_count=1,  # Images are single-page
        )


if __name__ == "__main__":
    import anyenv
    import devtools

    # # Example usage with PDF
    # pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = MistralConverter()
    # result = anyenv.run_sync(converter.convert_file(pdf_path))
    # print(f"PDF result: {len(result.content)} chars, {len(result.images)} images")

    # Example usage with image
    img_path = "E:/sap.png"
    result = anyenv.run_sync(converter.convert_file(img_path))
    devtools.debug(result)
