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
        self.api_key = api_key or os.getenv("LLAMAPARSE_API_KEY") or ""
        if not self.api_key:
            msg = "LLAMAPARSE_API_KEY environment variable not set"
            raise ValueError(msg)

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a document using LlamaParse."""
        import base64

        from llama_parse import LlamaParse, ResultType
        import requests
        import upath

        path = upath.UPath(file_path)
        parser = LlamaParse(api_key=self.api_key, result_type=ResultType.MD)
        result = parser.get_json_result(str(path))

        pages = result[0]["pages"]  # First document's pages
        job_id = result[0]["job_id"]
        content_parts: list[str] = []
        images: list[Image] = []

        for page in pages:
            if page.get("md"):
                content_parts.append(page["md"])

            # Process images directly from the page
            for img in page.get("images", []):
                image_count = len(images)
                id_ = f"img-{image_count}"

                # Get image data directly from API
                asset_name = img["name"]
                asset_url = f"{parser.base_url}/api/parsing/job/{job_id}/result/image/{asset_name}"  # noqa: E501

                response = requests.get(
                    asset_url, headers={"Authorization": f"Bearer {self.api_key}"}
                )
                response.raise_for_status()
                img_data = base64.b64encode(response.content).decode("utf-8")

                # Determine image type or default to png
                img_type = "png"  # Default image type
                if "." in asset_name:
                    extension = asset_name.split(".")[-1].lower()
                    if extension in ["jpg", "jpeg", "png", "gif", "webp", "svg"]:
                        img_type = extension

                filename = f"{id_}.{img_type}"
                mime = f"image/{img_type}"

                image = Image(id=id_, content=img_data, mime_type=mime, filename=filename)
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
    import anyenv

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    converter = LlamaParseConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
