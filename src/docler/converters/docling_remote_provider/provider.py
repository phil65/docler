"""Document converter using remote Docling service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import anyenv
import httpx2

from docler.converters.base import ConverterResult, DocumentConverter
from docler.converters.docling_remote_provider.utils import process_response
from docler.log import get_logger
from docler_config.converter_configs import DoclingRemoteConfig


if TYPE_CHECKING:
    from io import BytesIO

    from schemez import MimeType

    from docler.common_types import PageRangeString, SupportedLanguage
    from docler_config.converter_configs import DoclingEngine

logger = get_logger(__name__)

# docling-serve's v1 API (checked against docling-serve v1.40.0, commit 5c101b6).
DEFAULT_API_ENDPOINT = "http://localhost:5001"
CONVERT_FILE = "/v1/convert/file"

PDFBackend = Literal["pypdfium2", "docling_parse", "dlparse_v1", "dlparse_v2", "dlparse_v4"]
TableMode = Literal["fast", "accurate"]

# docler's engine names (docling's option classes) -> docling-serve's OCR presets.
OCR_PRESETS: dict[str, str] = {
    "easy_ocr": "easyocr",
    "rapid_ocr": "rapidocr",
    "tesseract_cli_ocr": "tesseract",
    "tesseract_ocr": "tesserocr",
    "ocr_mac": "ocrmac",
}
FROM_FORMATS = ["pdf", "docx", "pptx", "html", "image", "asciidoc", "md", "xlsx"]


# https://github.com/docling-project/docling-serve (docs/usage.md: the options)


class DoclingRemoteConverter(DocumentConverter[DoclingRemoteConfig]):
    """Document converter using remote Docling service."""

    Config = DoclingRemoteConfig
    NAME = "docling_remote"
    REQUIRED_PACKAGES: ClassVar = {"httpx2"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        # PDF
        "application/pdf",
        # Office Documents
        "application/msword",  # .doc
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
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
        page_range: PageRangeString | None = None,
        api_key: str | None = None,
        endpoint: str = DEFAULT_API_ENDPOINT,
        ocr_engine: DoclingEngine = "easy_ocr",
        pdf_backend: PDFBackend = "dlparse_v4",
        table_mode: TableMode = "fast",
        force_ocr: bool = False,
        image_scale: float = 2.0,
        do_table_structure: bool = True,
        do_code_enrichment: bool = False,
        do_formula_enrichment: bool = False,
        do_picture_classification: bool = False,
        do_picture_description: bool = False,
    ) -> None:
        """Initialize the remote Docling converter.

        Args:
            languages: List of supported languages.
            page_range: Page range(s) to extract, like "1-5,7-10" (1-based; PDFs only)
            api_key: Optional API key for auth.
            endpoint: Base URL of the Docling service.
            ocr_engine: OCR engine to use (sent as docling-serve's `ocr_preset`).
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
        # One multipart form field per option (lists repeat the field), as v1 expects.
        self.config: dict[str, Any] = {
            "from_formats": FROM_FORMATS,
            "to_formats": ["md"],
            "image_export_mode": "embedded",
            "do_ocr": True,
            "force_ocr": force_ocr,
            "ocr_preset": OCR_PRESETS[ocr_engine],
            "pdf_backend": pdf_backend,
            "table_mode": table_mode,
            "abort_on_error": False,
            "do_table_structure": do_table_structure,
            "do_code_enrichment": do_code_enrichment,
            "do_formula_enrichment": do_formula_enrichment,
            "do_picture_classification": do_picture_classification,
            "do_picture_description": do_picture_description,
            "images_scale": image_scale,
            "include_images": True,
        }
        if languages:
            self.config["ocr_lang"] = list(languages)

    async def _convert_async(self, data: BytesIO, mime_type: MimeType) -> ConverterResult:
        """Convert a document using remote Docling service.

        Args:
            data: File content as BytesIO.
            mime_type: MIME type of the file.

        Returns:
            Intermediate conversion result.

        Raises:
            ValueError: If conversion fails or response is malformed.
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        file_content = data.read()
        form = _form_fields(self.config)
        if self.page_range:
            if mime_type == "application/pdf":
                from docler.converters.docling_provider.utils import to_docling_page_range

                if span := to_docling_page_range(self.page_range):
                    form["page_range"] = [str(span[0]), str(span[1])]
                else:
                    from docler.pdf_utils import extract_pdf_pages

                    # v1's `page_range` takes one contiguous range: cut the pages locally.
                    file_content = await anyenv.run_in_thread(
                        extract_pdf_pages, file_content, self.page_range
                    )
            else:
                logger.warning("page_range only applies to PDFs; converting all of %s", mime_type)
        # Use a generic filename since we don't have path info
        ext = mime_type.split("/")[-1]
        filename = f"document.{ext}"
        url = f"{self.endpoint}{CONVERT_FILE}"
        async with httpx2.AsyncClient(timeout=60.0) as client:
            files = {"files": (filename, file_content, mime_type)}
            try:
                response = await client.post(url, headers=headers, files=files, data=form)
                response.raise_for_status()
                result = response.json()
            except httpx2.HTTPError as e:
                msg = f"Docling service error: {e}"
                raise ValueError(msg) from e
            except Exception as e:
                msg = f"Failed to convert document via Docling service: {e}"
                self.logger.exception(msg)
                raise ValueError(msg) from e

        document = result.get("document") or {}
        if not document.get("md_content"):
            if result.get("errors"):
                msg = f"Conversion failed: {result['errors']}"
            else:
                msg = "No markdown content found in response"
            raise ValueError(msg)
        content, images = process_response(document)
        return ConverterResult(
            content=content,
            images=images,
            metadata=result.get("timings", {}),
        )


def _form_fields(options: dict[str, Any]) -> dict[str, str | list[str]]:
    """Encode options as multipart form values (booleans as `true` / `false`)."""

    def encode(value: Any) -> str:
        return str(value).lower() if isinstance(value, bool) else str(value)

    return {
        key: [encode(v) for v in value] if isinstance(value, list) else encode(value)
        for key, value in options.items()
    }


if __name__ == "__main__":
    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = DoclingRemoteConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
