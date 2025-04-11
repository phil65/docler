"""Document converter using DataLab's API."""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, ClassVar, Literal

import anyenv
import upath
from upathtools import read_path

from docler.configs.converter_configs import DataLabConfig
from docler.converters.base import DocumentConverter
from docler.converters.datalab_provider.utils import normalize_markdown_images
from docler.log import get_logger
from docler.models import Document, Image
from docler.utils import get_api_key


if TYPE_CHECKING:
    from docler.common_types import StrPath, SupportedLanguage


logger = get_logger(__name__)

API_BASE = "https://www.datalab.to/api/v1"
MAX_POLLS = 300
POLL_INTERVAL = 2

Mode = Literal["marker", "table_rec", "ocr", "layout"]


class DataLabConverter(DocumentConverter[DataLabConfig]):
    """Document converter using DataLab's API."""

    Config = DataLabConfig

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
        force_ocr: bool = False,
        use_llm: bool = False,
        max_pages: int | None = None,
    ):
        """Initialize the DataLab converter.

        Args:
            api_key: DataLab API key.
            languages: Languages to use for OCR.
            force_ocr: Whether to force OCR on every page.
            use_llm: Whether to use LLM for enhanced accuracy.
            max_pages: Maximum number of pages to process.
        """
        super().__init__(languages=languages)
        self.api_key = api_key or get_api_key("DATALAB_API_KEY")
        self.force_ocr = force_ocr
        self.use_llm = use_llm
        self.max_pages = max_pages

    @property
    def price_per_page(self) -> float:
        """Price per page in USD."""
        return 0.003 if self.use_llm else 0.0015

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
        path = upath.UPath(file_path)
        form = {"output_format": "markdown"}
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
        headers = {"X-Api-Key": self.api_key}
        url = f"{API_BASE}/marker"
        response = await anyenv.post(url, data=form, files=files, headers=headers)
        json_data = await response.json()
        if not json_data["success"]:
            msg = f"Failed to submit conversion: {json_data['error']}"
            raise ValueError(msg)
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

        images: list[Image] = []
        md_content = result["markdown"]
        if result.get("images"):
            image_replacements = {}
            for i, (original_name, img_data) in enumerate(result["images"].items()):
                img_id = f"img-{i}"
                ext = original_name.split(".")[-1].lower()
                fname = f"{img_id}.{ext}"
                image_replacements[original_name] = (img_id, fname)
                if img_data.startswith("data:"):
                    img_data = img_data.split(",", 1)[1]
                content = base64.b64decode(img_data)
                mime = f"image/{ext}"
                image = Image(id=img_id, content=content, mime_type=mime, filename=fname)
                images.append(image)

            md_content = normalize_markdown_images(md_content, image_replacements)

        return Document(
            content=md_content,
            images=images,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
            page_count=result.get("page_count"),
        )


if __name__ == "__main__":
    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = DataLabConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
