"""Language code handling for OCR backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docler.common_types import (
    MAC_CODES,
    RAPID_CODES,
    TESSERACT_CODES,
    SupportedLanguage,
)


if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrMacOptions,
        RapidOcrOptions,
        TesseractCliOcrOptions,
        TesseractOcrOptions,
    )


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


def _parse_page_range(page_range: str | None) -> tuple[int, int] | None:
    """Convert a page range string to a tuple of (start, end) page numbers.

    Args:
        page_range: String like "1-5" or None. 1-based page numbers.

    Returns:
        Tuple of (start, end) page numbers (0-based) or None if no range specified.

    Raises:
        ValueError: If the page range format is invalid.
    """
    if not page_range:
        return None
    try:
        # Handle only first range for now (ignore possible additional ranges after comma)
        first_range = page_range.split(",")[0]
        start, end = map(int, first_range.split("-"))
        # Convert to 0-based indexing
        return (start - 1, end - 1)
    except ValueError as e:
        msg = f"Invalid page range format: {page_range}. Expected format: '1-5'"
        raise ValueError(msg) from e
