from __future__ import annotations

from typing import TYPE_CHECKING, Final


if TYPE_CHECKING:
    from collections.abc import Mapping


HTML_MIME_TYPE: Final = "text/html"
MARKDOWN_MIME_TYPE: Final = "text/markdown"
PDF_MIME_TYPE: Final = "application/pdf"
PLAIN_TEXT_MIME_TYPE: Final = "text/plain"
POWER_POINT_MIME_TYPE: Final = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)
DOCX_MIME_TYPE: Final = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
# Excel formats
EXCEL_MIME_TYPE: Final = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
EXCEL_BINARY_MIME_TYPE: Final = "application/vnd.ms-excel"
EXCEL_MACRO_MIME_TYPE: Final = "application/vnd.ms-excel.sheet.macroEnabled.12"
EXCEL_BINARY_2007_MIME_TYPE: Final = (
    "application/vnd.ms-excel.sheet.binary.macroEnabled.12"
)
EXCEL_ADDON_MIME_TYPE: Final = "application/vnd.ms-excel.addin.macroEnabled.12"
EXCEL_TEMPLATE_MIME_TYPE: Final = "application/vnd.ms-excel.template.macroEnabled.12"

# OpenDocument spreadsheet format
OPENDOC_SPREADSHEET_MIME_TYPE: Final = (
    "application/vnd.oasis.opendocument.spreadsheet"  # ods
)
PLAIN_TEXT_MIME_TYPES: Final[set[str]] = {PLAIN_TEXT_MIME_TYPE, MARKDOWN_MIME_TYPE}

AUDIO_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/webm",
    "audio/x-wav",
    "audio/ogg",
    "audio/flac",
    "audio/m4a",
    "video/mp4",
}

IMAGE_MIME_TYPES: Final[set[str]] = {
    "image/bmp",
    "image/gif",
    "image/jp2",
    "image/jpeg",
    "image/jpm",
    "image/jpx",
    "image/mj2",
    "image/pjpeg",
    "image/png",
    "image/tiff",
    "image/webp",
    "image/x-bmp",
    "image/x-ms-bmp",
    "image/x-portable-anymap",
    "image/x-portable-bitmap",
    "image/x-portable-graymap",
    "image/x-portable-pixmap",
    "image/x-tiff",
}
IMAGE_MIME_TYPE_EXT_MAP: Final[Mapping[str, str]] = {
    "image/bmp": "bmp",
    "image/x-bmp": "bmp",
    "image/x-ms-bmp": "bmp",
    "image/gif": "gif",
    "image/jpeg": "jpg",
    "image/pjpeg": "jpg",
    "image/png": "png",
    "image/tiff": "tiff",
    "image/x-tiff": "tiff",
    "image/jp2": "jp2",
    "image/jpx": "jpx",
    "image/jpm": "jpm",
    "image/mj2": "mj2",
    "image/webp": "webp",
    "image/x-portable-anymap": "pnm",
    "image/x-portable-bitmap": "pbm",
    "image/x-portable-graymap": "pgm",
    "image/x-portable-pixmap": "ppm",
}
PANDOC_SUPPORTED_MIME_TYPES: Final[set[str]] = {
    "application/csl+json",
    "application/docbook+xml",
    "application/epub+zip",
    "application/rtf",
    "application/vnd.oasis.opendocument.text",
    DOCX_MIME_TYPE,
    "application/x-biblatex",
    "application/x-bibtex",
    "application/x-endnote+xml",
    "application/x-fictionbook+xml",
    "application/x-ipynb+json",
    "application/x-jats+xml",
    "application/x-latex",
    "application/x-opml+xml",
    "application/x-research-info-systems",
    "application/x-typst",
    "text/csv",
    "text/tab-separated-values",
    "text/troff",
    "text/x-commonmark",
    "text/x-dokuwiki",
    "text/x-gfm",
    "text/x-markdown",
    "text/x-markdown-extra",
    "text/x-mdoc",
    "text/x-multimarkdown",
    "text/x-org",
    "text/x-pod",
    "text/x-rst",
}

SPREADSHEET_MIME_TYPES: Final[set[str]] = {
    EXCEL_MIME_TYPE,
    EXCEL_BINARY_MIME_TYPE,
    EXCEL_MACRO_MIME_TYPE,
    EXCEL_BINARY_2007_MIME_TYPE,
    EXCEL_ADDON_MIME_TYPE,
    EXCEL_TEMPLATE_MIME_TYPE,
    OPENDOC_SPREADSHEET_MIME_TYPE,
}

EXT_TO_MIME_TYPE: Final[Mapping[str, str]] = {
    ".txt": PLAIN_TEXT_MIME_TYPE,
    ".md": MARKDOWN_MIME_TYPE,
    ".pdf": PDF_MIME_TYPE,
    ".html": HTML_MIME_TYPE,
    ".htm": HTML_MIME_TYPE,
    ".xlsx": EXCEL_MIME_TYPE,
    ".xls": EXCEL_BINARY_MIME_TYPE,
    ".xlsm": EXCEL_MACRO_MIME_TYPE,
    ".xlsb": EXCEL_BINARY_2007_MIME_TYPE,
    ".xlam": EXCEL_ADDON_MIME_TYPE,
    ".xla": EXCEL_TEMPLATE_MIME_TYPE,
    ".ods": OPENDOC_SPREADSHEET_MIME_TYPE,
    ".pptx": POWER_POINT_MIME_TYPE,
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
    ".jp2": "image/jp2",
    ".jpx": "image/jpx",
    ".jpm": "image/jpm",
    ".mj2": "image/mj2",
    ".pnm": "image/x-portable-anymap",
    ".pbm": "image/x-portable-bitmap",
    ".pgm": "image/x-portable-graymap",
    ".ppm": "image/x-portable-pixmap",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".rst": "text/x-rst",
    ".org": "text/x-org",
    ".epub": "application/epub+zip",
    ".rtf": "application/rtf",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".docx": DOCX_MIME_TYPE,
    ".doc": "application/msword",
    ".bib": "application/x-bibtex",
    ".ipynb": "application/x-ipynb+json",
    ".tex": "application/x-latex",
}

SUPPORTED_MIME_TYPES: Final[set[str]] = (
    PLAIN_TEXT_MIME_TYPES
    | IMAGE_MIME_TYPES
    | PANDOC_SUPPORTED_MIME_TYPES
    | SPREADSHEET_MIME_TYPES
    | {PDF_MIME_TYPE, POWER_POINT_MIME_TYPE, HTML_MIME_TYPE}
)
