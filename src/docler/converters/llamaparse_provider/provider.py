"""Document converter using LlamaParse."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar

from docler.converters.base import DocumentConverter
from docler.models import Document, Image


if TYPE_CHECKING:
    from docler.common_types import StrPath, SupportedLanguage


logger = logging.getLogger(__name__)


class LlamaParseConverter(DocumentConverter):
    """Document converter using LlamaParse."""

    NAME = "llamaparse"
    REQUIRED_PACKAGES: ClassVar = {"llama-parse"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/html",
        "text/markdown",
    }

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        api_key: str | None = None,
    ):
        """Initialize the LlamaParse converter.

        Args:
            languages: List of supported languages
            api_key: LlamaParse API key, defaults to LLAMAPARSE_API_KEY env var
        """
        super().__init__(languages=languages)
        self.api_key = api_key or os.getenv("LLAMAPARSE_API_KEY")
        if not self.api_key:
            msg = "LLAMAPARSE_API_KEY environment variable not set"
            raise ValueError(msg)

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a document using LlamaParse."""
        from llama_parse import LlamaParse, ResultType
        import upath

        path = upath.UPath(file_path)

        # Initialize parser with markdown output
        assert self.api_key, "API key not set"
        parser = LlamaParse(api_key=self.api_key, result_type=ResultType.MD)

        # Get structured result including images
        result = parser.get_json_result(str(path))
        print(result)
        pages = result[0]["pages"]  # First document's pages

        # Collect content and images across pages
        content_parts: list[str] = []
        images: list[Image] = []

        for page in pages:
            # Add markdown content
            if page.get("md"):
                content_parts.append(page["md"])

            # Process images in this page
            for img in page.get("images", []):
                # Create standardized image ID
                image_count = len(images)
                image_id = f"img-{image_count}"

                # Get image data and metadata
                img_data = img["data"]  # Base64 encoded image
                img_type = img.get("type", "png")
                filename = f"{image_id}.{img_type}"

                # Create image object
                image = Image(
                    id=image_id,
                    content=img_data,  # Already base64 encoded
                    mime_type=f"image/{img_type}",
                    filename=filename,
                )
                images.append(image)

        return Document(
            content="\n\n".join(content_parts),
            images=images,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
            page_count=len(pages),
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    converter = LlamaParseConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
