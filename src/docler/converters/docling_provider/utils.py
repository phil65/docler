"""Language code handling for OCR backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docler.common_types import (
    MAC_CODES,
    RAPID_CODES,
    TESSERACT_CODES,
)


if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrMacOptions,
        RapidOcrOptions,
        TesseractCliOcrOptions,
        TesseractOcrOptions,
    )

    from docler.common_types import PageRangeString, SupportedLanguage


def convert_languages(
    languages: list[SupportedLanguage],
    backend_type: type[
        EasyOcrOptions
        | TesseractCliOcrOptions
        | TesseractOcrOptions
        | OcrMacOptions
        | RapidOcrOptions
    ],
) -> list[str]:
    """Convert language codes for specific backend.

    Args:
        languages: List of language codes to convert
        backend_type: OCR backend class to convert for

    Returns:
        List of language codes in the format expected by the backend
    """
    from docling.datamodel.pipeline_options import (
        OcrMacOptions,
        RapidOcrOptions,
        TesseractCliOcrOptions,
        TesseractOcrOptions,
    )

    if backend_type in (TesseractCliOcrOptions, TesseractOcrOptions):
        return [TESSERACT_CODES[lang] for lang in languages]
    if backend_type == OcrMacOptions:
        return [MAC_CODES[lang] for lang in languages]
    if backend_type == RapidOcrOptions:
        return [RAPID_CODES[lang] for lang in languages]
    # EasyOCR uses standard 2-letter codes
    return list(languages)


def to_docling_page_range(page_range: PageRangeString) -> tuple[int, int] | None:
    """Convert a page range string to Docling's (start, end) range (1-based, inclusive).

    Args:
        page_range: String like "1-5,7,9-11". 1-based page numbers.

    Returns:
        (first, last) page if the pages are contiguous, None if they aren't (Docling
        takes a single range: extract the pages first).

    Raises:
        ValueError: If the page range format is invalid.
    """
    from docler.pdf_utils import parse_page_range

    pages = sorted(parse_page_range(page_range))
    if not pages:
        msg = f"Invalid page range format: {page_range}"
        raise ValueError(msg)
    if pages != list(range(pages[0], pages[-1] + 1)):
        return None
    return (pages[0], pages[-1])
