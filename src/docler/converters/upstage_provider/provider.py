"""Document converter using Upstage's Document AI API."""

from __future__ import annotations

import base64
from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

import requests
import upath

from docler.configs.converter_configs import (
    UpstageCategory,
    UpstageConfig,
    UpstageOCRType,
)
from docler.converters.base import DocumentConverter
from docler.markdown_utils import PAGE_BREAK_TYPE, create_metadata_comment
from docler.models import Document, Image
from docler.utils import get_api_key


if TYPE_CHECKING:
    from docler.common_types import StrPath, SupportedLanguage


# API endpoints
DOCUMENT_PARSE_BASE_URL = "https://api.upstage.ai/v1/document-digitization"
DOCUMENT_PARSE_DEFAULT_MODEL = "document-parse"


class UpstageConverter(DocumentConverter[UpstageConfig]):
    """Document converter using Upstage's Document AI API."""

    Config = UpstageConfig

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
        base64_categories: set[UpstageCategory] | None = None,
    ):
        """Initialize the Upstage converter.

        Args:
            languages: List of supported languages (currently unused by Upstage API)
            api_key: Upstage API key (falls back to UPSTAGE_API_KEY env var)
            base_url: API endpoint URL
            model: Model name for document parsing
            ocr: OCR mode ('auto' or 'force')
            base64_categories: Element categories to encode in base64

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        super().__init__(languages=languages)
        self.api_key = api_key or get_api_key("UPSTAGE_API_KEY")
        self.base_url = base_url
        self.model = model
        self.ocr = ocr
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
            Converted document with extracted text, images, and page break markers.

        Raises:
            ValueError: If conversion fails or response is malformed.
        """
        path = upath.UPath(file_path)
        file_content = path.read_bytes()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"document": (path.name, file_content, mime_type)}
        data = {
            "ocr": self.ocr,
            "model": self.model,
            "output_formats": "['markdown']",
            "base64_encoding": str(list(self.base64_categories)),
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data=data,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
        except requests.HTTPError as e:
            msg = f"Upstage API error: {e.response.text if e.response else str(e)}"
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Failed to convert document via Upstage API: {e}"
            self.logger.exception(msg)
            raise ValueError(msg) from e

        content_data = result.get("content", {})
        initial_markdown = content_data.get("markdown")
        if not initial_markdown:
            msg = "No content found in Upstage API response."
            # self.logger.warning("Full Upstage response: %s", result)
            raise ValueError(msg)

        elements = result.get("elements", [])
        max_page = result.get("usage", {}).get("pages", 0)

        modified_markdown = initial_markdown  # Start with the original markdown
        if max_page > 1 and elements:
            elements_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for element in elements:
                page_num = element.get("page")
                if page_num is not None:
                    elements_by_page[page_num].append(element)

            for page_num in elements_by_page:  # noqa: PLC0206
                elements_by_page[page_num].sort(key=lambda x: x.get("id", 0))

            insertion_offset = 0
            for page_number in range(2, max_page + 1):
                if (
                    page_number not in elements_by_page
                    or not elements_by_page[page_number]
                ):
                    continue

                first_element = elements_by_page[page_number][0]
                first_element_md = first_element.get("content", {}).get("markdown", "")
                if not first_element_md:
                    for elem in elements_by_page[page_number][1:]:
                        first_element_md = elem.get("content", {}).get("markdown", "")
                        if first_element_md:
                            break
                    if not first_element_md:
                        msg = "Could not find non-empty element md anchor for page %d"
                        self.logger.warning(msg, page_number)
                        continue

                # Find the position using the offset
                search_start_index = 0
                found_index = -1
                temp_index = modified_markdown.find(first_element_md, search_start_index)
                while temp_index != -1:
                    if temp_index >= insertion_offset:
                        found_index = temp_index
                        break
                    search_start_index = temp_index + 1
                    temp_index = modified_markdown.find(
                        first_element_md, search_start_index
                    )

                if found_index != -1:
                    page_data = {"next_page": page_number}
                    marker = (
                        f"\n\n{create_metadata_comment(PAGE_BREAK_TYPE, page_data)}\n\n"
                    )
                    modified_markdown = (
                        modified_markdown[:found_index]
                        + marker
                        + modified_markdown[found_index:]
                    )
                    insertion_offset = found_index + len(marker)
                else:
                    self.logger.warning(
                        "Could not find insertion point for page break before page %d",
                        page_number,
                    )

        images: list[Image] = []
        image_counter = 0
        # Use a temporary variable for markdown content during image replacement
        # to avoid interfering with the page break logic's offset calculations if
        # image placeholders were part of the first element's markdown.
        content_for_image_replacement = modified_markdown

        for element in elements:
            category = element.get("category")
            base64_data = element.get("base64_encoding")

            if category in self.base64_categories and base64_data:
                image_id = f"img-{image_counter}"
                image_counter += 1

                if base64_data.startswith("data:image/"):
                    mime_parts = base64_data.split(";")[0].split(":")
                    img_mime_type = mime_parts[1] if len(mime_parts) > 1 else "image/png"
                    img_data = base64_data.split(",", 1)[1]  # Get data after comma
                else:
                    img_mime_type = "image/png"
                    img_data = base64_data  # Assume it's pure base64

                img_bytes = base64.b64decode(img_data)
                ext = img_mime_type.split("/")[-1]
                # Handle potential complex mime types like 'svg+xml'
                ext = ext.split("+")[0]
                filename = f"{image_id}.{ext}"

                image = Image(
                    id=image_id,
                    content=img_bytes,  # Store as bytes
                    mime_type=img_mime_type,
                    filename=filename,
                )
                images.append(image)

                # Replace the *first* available placeholder in the markdown
                placeholder = "![image](/image/placeholder)"
                if placeholder in content_for_image_replacement:
                    img_ref = f"![{image_id}]({filename})"
                    content_for_image_replacement = content_for_image_replacement.replace(
                        placeholder, img_ref, 1
                    )
                else:
                    msg = "Found image data for %s but no placeholder left in markdown."
                    self.logger.warning(msg, image_id)
        modified_markdown = content_for_image_replacement
        return Document(
            content=modified_markdown,
            images=images,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
            page_count=max_page,
            metadata=result.get("metadata", {}),
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.INFO)  # Add basic logging for testing

    pdf_path = "src/docler/resources/pdf_sample.pdf"  # Adjust path if needed
    converter = UpstageConverter()
    result_doc = anyenv.run_sync(converter.convert_file(pdf_path))
    print("--- Converted Markdown ---")
    print(result_doc.content)
    print("\n--- Extracted Images ---")
    print(result_doc.images)
    print(f"\nPage Count: {result_doc.page_count}")
