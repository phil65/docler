"""Document converter using remote Docling service."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import anyenv
import httpx
from mkdown import Document
import upath

from docler.configs.converter_configs import DoclingRemoteConfig
from docler.converters.base import DocumentConverter
from docler.converters.docling_remote_provider.utils import process_response
from docler.log import get_logger


if TYPE_CHECKING:
    from schemez import MimeType

    from docler.common_types import StrPath, SupportedLanguage

logger = get_logger(__name__)

# Default endpoint configurations
DEFAULT_API_ENDPOINT = "http://localhost:5001"
CONVERT_SOURCE = "/v1alpha/convert/source"
CONVERT_FILE = "/v1alpha/convert/file"

OCREngine = Literal["easyocr", "tesseract_cli", "tesseract", "rapidocr", "ocrmac"]
PDFBackend = Literal["pypdfium2", "dlparse_v1", "dlparse_v2", "dlparse_v4"]
TableMode = Literal["fast", "accurate"]


# https://github.com/docling-project/docling-serve


class DoclingRemoteConverter(DocumentConverter[DoclingRemoteConfig]):
    """Document converter using remote Docling service."""

    Config = DoclingRemoteConfig
    NAME = "docling_remote"
    REQUIRED_PACKAGES: ClassVar = {"httpx"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        # PDF
        "application/pdf",
        # Office Documents
        "application/msword",  # .doc
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx  # noqa: E501
        "application/vnd.ms-powerpoint",  # .ppt
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-excel",  # .xls
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        # Open/Libre Office
        "application/vnd.oasis.opendocument.text",  # .odt
        "application/vnd.oasis.opendocument.spreadsheet",  # .ods
        "application/vnd.oasis.opendocument.presentation",  # .odp
        # HTML/Text
        "text/html",
        "text/markdown",
        "text/asciidoc",
        # Images
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
        page_range: str | None = None,
        api_key: str | None = None,
        endpoint: str = DEFAULT_API_ENDPOINT,
        ocr_engine: OCREngine = "easyocr",
        pdf_backend: PDFBackend = "dlparse_v4",
        table_mode: TableMode = "fast",
        force_ocr: bool = False,
        image_scale: float = 2.0,
        do_table_structure: bool = True,
        do_code_enrichment: bool = False,
        do_formula_enrichment: bool = False,
        do_picture_classification: bool = False,
        do_picture_description: bool = False,
    ):
        """Initialize the remote Docling converter.

        Args:
            languages: List of supported languages.
            page_range: Page range to exctract (0-based)
            api_key: Optional API key for auth.
            endpoint: Base URL of the Docling service.
            ocr_engine: OCR engine to use.
            pdf_backend: PDF backend to use.
            table_mode: Table mode to use.
            force_ocr: Whether to force OCR even on digital documents.
            image_scale: Scale factor for image resolution.
            do_table_structure: Extract table structure.
            do_code_enrichment: Enable code extraction.
            do_formula_enrichment: Enable formula extraction.
            do_picture_classification: Enable picture classification.
            do_picture_description: Enable picture description.

        Raises:
            MissingConfigurationError: If endpoint cannot be used.
        """
        super().__init__(languages=languages, page_range=page_range)
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.config = {
            "from_formats": [
                "pdf",
                "docx",
                "pptx",
                "html",
                "image",
                "asciidoc",
                "md",
                "xlsx",
            ],
            "to_formats": ["md"],
            "image_export_mode": "embedded",
            "do_ocr": True,
            "force_ocr": force_ocr,
            "ocr_engine": ocr_engine,
            "pdf_backend": pdf_backend,
            "table_mode": table_mode,
            "abort_on_error": False,
            "return_as_file": False,
            "do_table_structure": do_table_structure,
            "do_code_enrichment": do_code_enrichment,
            "do_formula_enrichment": do_formula_enrichment,
            "do_picture_classification": do_picture_classification,
            "do_picture_description": do_picture_description,
            "images_scale": image_scale,
            "include_images": True,
        }

        if languages:
            self.config["ocr_lang"] = languages

    async def _convert_path_async(
        self,
        file_path: StrPath,
        mime_type: MimeType,
    ) -> Document:
        """Convert a document using remote Docling service.

        Args:
            file_path: Path to the document to convert.
            mime_type: MIME type of the file.

        Returns:
            Converted document.

        Raises:
            ValueError: If conversion fails or response is malformed.
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        path = upath.UPath(file_path)
        file_content = await anyenv.run_in_thread(path.read_bytes)
        url = f"{self.endpoint}{CONVERT_FILE}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"files": (path.name, file_content, mime_type)}
            data = {"parameters": anyenv.dump_json(self.config)}
            try:
                response = await client.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as e:
                msg = f"Docling service error: {e}"
                raise ValueError(msg) from e
            except Exception as e:
                msg = f"Failed to convert document via Docling service: {e}"
                self.logger.exception(msg)
                raise ValueError(msg) from e

        document = result["document"]
        if not document.get("md_content"):
            if result.get("errors"):
                msg = f"Conversion failed: {result['errors']}"
            else:
                msg = "No markdown content found in response"
            raise ValueError(msg)
        content, images = process_response(document)
        return Document(
            content=content,
            images=images,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
            metadata=result.get("timings", {}),  # Store timings as metadata
        )


if __name__ == "__main__":
    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = DoclingRemoteConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
