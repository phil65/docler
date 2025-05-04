"""Document converter using DataLab's API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import anyenv
import upath
from upathtools import read_path

from docler.configs.converter_configs import DataLabConfig
from docler.converters.base import DocumentConverter
from docler.converters.datalab_provider.utils import get_response, process_response
from docler.log import get_logger
from docler.models import Document
from docler.utils import get_api_key


if TYPE_CHECKING:
    from docler.common_types import StrPath, SupportedLanguage


logger = get_logger(__name__)

Mode = Literal["marker", "table_rec", "ocr", "layout"]

# See https://www.datalab.to/app/reference
# https://www.datalab.to/openapi.json


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
        form = {"output_format": "markdown", "paginate": True}
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
        result = await get_response(form, files, self.api_key)
        md_content, images = process_response(result)
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
