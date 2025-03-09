"""Document converter using DataLab's API."""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import TYPE_CHECKING, ClassVar, Literal

import anyenv

from docler.base import DocumentConverter
from docler.models import Document, Image


if TYPE_CHECKING:
    from docler.common_types import StrPath
    from docler.lang_code import SupportedLanguage


logger = logging.getLogger(__name__)

API_BASE = "https://www.datalab.to/api/v1"
MAX_POLLS = 300
POLL_INTERVAL = 2

Mode = Literal["marker", "table_rec", "ocr", "layout"]


class DataLabConverter(DocumentConverter):
    """Document converter using DataLab's API."""

    NAME = "datalab"
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        # PDF
        "application/pdf",
        # Images
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/tiff",
        "image/jpg",
        # Office Documents
        "application/msword",  # doc
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
        # Presentations
        "application/vnd.ms-powerpoint",  # ppt
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx  # noqa: E501
    }

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        api_key: str | None = None,
        mode: Mode = "marker",
        force_ocr: bool = False,
        use_llm: bool = False,
        max_pages: int | None = None,
    ) -> None:
        """Initialize the DataLab converter.

        Args:
            api_key: DataLab API key.
            mode: API endpoint to use.
            languages: Languages to use for OCR.
            force_ocr: Whether to force OCR on every page.
            use_llm: Whether to use LLM for enhanced accuracy.
            max_pages: Maximum number of pages to process.
        """
        super().__init__(languages=languages)
        self.api_key = api_key or os.getenv("DATALAB_API_KEY")
        assert self.api_key, "API key is required"
        self.mode = mode
        self.force_ocr = force_ocr
        self.use_llm = use_llm
        self.max_pages = max_pages

    async def _convert_path_async(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a file using DataLab's API.

        Args:
            file_path: Path to the file to process.
            mime_type: MIME type of the file.

        Returns:
            Converted document.

        Raises:
            ValueError: If conversion fails.
        """
        import upath
        from upathtools import read_path

        path = upath.UPath(file_path)

        # Prepare form data
        form = {
            "output_format": "markdown",
        }
        data = await read_path(path, mode="rb")
        files = {"file": (path.name, data, mime_type)}
        if self.languages:
            form["langs"] = ",".join(self.languages)
        if self.force_ocr:
            form["force_ocr"] = "true"
        if self.use_llm:
            form["use_llm"] = "true"
        if self.max_pages:
            form["max_pages"] = str(self.max_pages)

        # Create synchronous client
        headers = {"X-Api-Key": self.api_key}
        response = await anyenv.post(
            f"{API_BASE}/{self.mode}",
            data=form,
            files=files,
            headers=headers,  # type: ignore
        )
        json_data = await response.json()
        if not json_data["success"]:
            msg = f"Failed to submit conversion: {json_data['error']}"
            raise ValueError(msg)
        # Poll for results
        check_url = json_data["request_check_url"]
        for _ in range(MAX_POLLS):
            time.sleep(POLL_INTERVAL)

            result = await anyenv.get_json(check_url, headers=headers, return_type=dict)  # type: ignore

            if result["status"] == "complete":
                break
        else:
            msg = "Conversion timed out"
            raise TimeoutError(msg)

        if not result["success"]:
            msg = f"Conversion failed: {result['error']}"
            raise ValueError(msg)

        # Convert images if present
        images: list[Image] = []
        if result.get("images"):
            for filename, img_data in result["images"].items():
                if img_data.startswith("data:"):
                    img_data = img_data.split(",", 1)[1]

                ext = filename.split(".")[-1].lower()
                image = Image(
                    id=filename,
                    content=base64.b64decode(img_data),
                    mime_type=f"image/{ext}",
                    filename=filename,
                )
                images.append(image)

        return Document(
            content=result["markdown"],
            images=images,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
            page_count=result.get("page_count"),
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"

    api_key = os.getenv("DATALAB_API_KEY")
    converter = DataLabConverter(api_key=api_key)
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
