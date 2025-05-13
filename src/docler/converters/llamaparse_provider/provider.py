"""Document converter using LlamaParse."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import anyenv
from mkdown import Document
import upath

from docler.configs.converter_configs import LlamaParseConfig, LlamaParseMode
from docler.converters.base import DocumentConverter
from docler.converters.llamaparse_provider.utils import process_response
from docler.log import get_logger
from docler.utils import get_api_key


if TYPE_CHECKING:
    from schemez import MimeType

    from docler.common_types import StrPath, SupportedLanguage


logger = get_logger(__name__)


class LlamaParseConverter(DocumentConverter[LlamaParseConfig]):
    """Document converter using LlamaParse."""

    Config = LlamaParseConfig

    NAME = "llamaparse"
    REQUIRED_PACKAGES: ClassVar = {"llama-parse"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        # PDF
        "application/pdf",
        # Office Documents
        "application/msword",  # .doc
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx  # noqa: E501
        "application/vnd.ms-word.document.macroEnabled.12",  # .docm
        "application/vnd.ms-powerpoint",  # .ppt
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx # noqa: E501
        "application/vnd.ms-powerpoint.presentation.macroEnabled.12",  # .pptm
        "application/vnd.ms-excel",  # .xls
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        "application/vnd.ms-excel.sheet.macroEnabled.12",  # .xlsm
        "application/vnd.ms-excel.sheet.binary.macroEnabled.12",  # .xlsb
        # Open/Libre Office
        "application/vnd.oasis.opendocument.text",  # .odt
        "application/vnd.oasis.opendocument.spreadsheet",  # .ods
        "application/vnd.oasis.opendocument.presentation",  # .odp
        # Text formats
        "text/html",
        "text/markdown",
        "text/plain",  # .txt
        "text/rtf",  # .rtf
        "text/csv",  # .csv
        "text/tab-separated-values",  # .tsv
        "application/xml",  # .xml
        "application/epub+zip",  # .epub
        # Images
        "image/jpeg",  # .jpg, .jpeg
        "image/png",  # .png
        "image/gif",  # .gif
        "image/bmp",  # .bmp
        "image/svg+xml",  # .svg
        "image/tiff",  # .tiff
        "image/webp",  # .webp
        # Audio
        "audio/mpeg",  # .mp3
        "audio/mp4",  # .mp4 audio
        "audio/wav",  # .wav
        "audio/webm",  # .webm audio
        "audio/m4a",  # .m4a
    }

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        page_range: str | None = None,
        api_key: str | None = None,
        adaptive_long_table: bool = True,
        parse_mode: LlamaParseMode = "parse_page_with_llm",
        skip_diagonal_text: bool = False,
        disable_ocr: bool = False,
        continuous_mode: bool = True,
        html_tables: bool = False,
    ):
        """Initialize the LlamaParse converter.

        Args:
            languages: List of supported languages
            page_range: Page range(s) to extract, like "1-5,7-10" (1-based)
            api_key: LlamaParse API key, defaults to LLAMAPARSE_API_KEY env var
            adaptive_long_table: Whether to use adaptive long table
            parse_mode: Parse mode, defaults to "parse_page_with_llm"
            skip_diagonal_text: Whether to skip diagonal text
            disable_ocr: Whether to disable OCR for images
            continuous_mode: Whether to use continuous mode
            html_tables: Whether to output HTML tables instead of markdown
        """
        super().__init__(languages=languages, page_range=page_range)
        self.api_key = api_key or get_api_key("LLAMAPARSE_API_KEY")
        self.language = self.languages[0] if self.languages else None
        self.adaptive_long_table = adaptive_long_table
        self.parse_mode = parse_mode
        self.skip_diagonal_text = skip_diagonal_text
        self.disable_ocr = disable_ocr
        # self.bounding_box = None
        self.continuous_mode = continuous_mode
        self.html_tables = html_tables

    @property
    def price_per_page(self) -> float:
        """Price per page in USD."""
        return 0.0045

    def _convert_path_sync(self, file_path: StrPath, mime_type: MimeType) -> Document:
        """Convert a document using LlamaParse."""
        from llama_parse import LlamaParse, ResultType

        path = upath.UPath(file_path)
        parser = LlamaParse(
            api_key=self.api_key,
            result_type=ResultType.MD,
            language=self.language,
            adaptive_long_table=self.adaptive_long_table,
            parse_mode=self.parse_mode,
            target_pages=self.page_range,
            skip_diagonal_text=self.skip_diagonal_text,
            disable_ocr=self.disable_ocr,
            continuous_mode=self.continuous_mode,
            output_tables_as_HTML=self.html_tables,
            # take_screenshot=True,
            # we are doing page separators manually right now
            # page_separator=r'<!-- docler:page_break {"next_page":{pageNumber}} -->',
        )
        result = parser.get_json_result(str(path))
        content_parts, images = process_response(result, self.api_key)
        return Document(
            content="\n\n".join(content_parts),
            images=images,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
        )


if __name__ == "__main__":
    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = LlamaParseConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
