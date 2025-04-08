"""Document converter using Upstage's Document AI API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from docler.configs.converter_configs import (
    MistralConfig,
    UpstageCategory,
    UpstageOCRType,
    UpstageOutputFormat,
)
from docler.converters.base import DocumentConverter
from docler.models import Document, Image
from docler.utils import get_api_key


if TYPE_CHECKING:
    from docler.common_types import StrPath, SupportedLanguage


# API endpoints
DOCUMENT_PARSE_BASE_URL = "https://api.upstage.ai/v1/document-ai/document-parse"
DOCUMENT_PARSE_DEFAULT_MODEL = "document-parse"


class UpstageConverter(DocumentConverter[MistralConfig]):
    """Document converter using Upstage's Document AI API."""

    Config = MistralConfig

    NAME = "upstage"
    REQUIRED_PACKAGES: ClassVar = {"requests"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        api_key: str | None = None,
        base_url: str = DOCUMENT_PARSE_BASE_URL,
        model: str = DOCUMENT_PARSE_DEFAULT_MODEL,
        ocr: UpstageOCRType = "auto",
        output_format: UpstageOutputFormat = "markdown",
        base64_categories: set[UpstageCategory] | None = None,
    ):
        """Initialize the Upstage converter.

        Args:
            languages: List of supported languages
            api_key: Upstage API key (falls back to UPSTAGE_API_KEY env var)
            base_url: API endpoint URL
            model: Model name for document parsing
            ocr: OCR mode ('auto' or 'force')
            output_format: Output format ('markdown', 'text', or 'html')
            base64_categories: Element categories to encode in base64

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        super().__init__(languages=languages)
        self.api_key = api_key or get_api_key("UPSTAGE_API_KEY")
        self.base_url = base_url
        self.model = model
        self.ocr = ocr
        self.output_format = output_format
        self.base64_categories = base64_categories or {"figure", "chart"}

    @property
    def price_per_page(self) -> float:
        """Price per page in USD."""
        return 0.01

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a document using Upstage's Document AI API.

        Args:
            file_path: Path to the document file
            mime_type: MIME type of the file

        Returns:
            Converted document with extracted text and metadata

        Raises:
            ValueError: If conversion fails
        """
        import requests
        import upath

        path = upath.UPath(file_path)
        file_content = path.read_bytes()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"document": (path.name, file_content, mime_type)}
        data = {
            "ocr": self.ocr,
            "model": self.model,
            "output_formats": f"['{self.output_format}']",
            "base64_encoding": str(list(self.base64_categories)),
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data=data,
            )
            response.raise_for_status()
            result = response.json()
        except requests.HTTPError as e:
            msg = f"Upstage API error: {e.response.text if e.response else str(e)}"
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Failed to convert document: {e}"
            self.logger.exception(msg)
            raise ValueError(msg) from e

        # Extract content from response
        content = result.get("content", {}).get(self.output_format, "")
        if not content:
            msg = "No content found in API response"
            raise ValueError(msg)

        images: list[Image] = []
        image_counter = 0  # Dedicated counter for images only
        elements = result.get("elements", [])
        for element in elements:
            if element.get("category") not in self.base64_categories:
                continue

            # Skip elements without base64 encoding
            if not element.get("base64_encoding"):
                continue

            image_id = f"img-{image_counter}"
            image_counter += 1
            # Handle base64 encoded images
            img_data = element["base64_encoding"]
            if img_data.startswith("data:image/"):
                # Extract MIME type and remove prefix
                mime_parts = img_data.split(";")[0].split(":")
                img_mime_type = mime_parts[1] if len(mime_parts) > 1 else "image/png"
                img_data = img_data.split(",", 1)[1]
            else:
                img_mime_type = "image/png"

            # Determine file extension based on MIME type
            ext = img_mime_type.split("/")[-1]
            filename = f"{image_id}.{ext}"

            image = Image(
                id=image_id,
                content=img_data,
                mime_type=img_mime_type,
                filename=filename,
            )
            images.append(image)

            # Replace image placeholders with actual references in the content
            if "/image/placeholder" in content:
                img_ref = f"![{image_id}]({filename})"
                content = content.replace("![image](/image/placeholder)", img_ref, 1)
        # Create document with extracted information
        return Document(
            content=content,
            images=images,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
        )


if __name__ == "__main__":
    import anyenv

    pdf_path = "src/docler/resources/pdf_sample.pdf"

    # Initialize converter with custom base64 categories
    converter = UpstageConverter()

    # Convert document
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.images)
