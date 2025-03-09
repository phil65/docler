"""Language code handling for OCR backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from docling.datamodel.pipeline_options import (
    OcrMacOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)


if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
    )


SupportedLanguage = Literal["en", "de", "fr", "es", "zh"]

# Mapping tables for different backends
TESSERACT_CODES: dict[SupportedLanguage, str] = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "zh": "chi",
}

MAC_CODES: dict[SupportedLanguage, str] = {
    "en": "en-US",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "zh": "zh-CN",
}

RAPID_CODES: dict[SupportedLanguage, str] = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
}


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
    if backend_type in (TesseractCliOcrOptions, TesseractOcrOptions):
        return [TESSERACT_CODES[lang] for lang in languages]
    if backend_type == OcrMacOptions:
        return [MAC_CODES[lang] for lang in languages]
    if backend_type == RapidOcrOptions:
        return [RAPID_CODES[lang] for lang in languages]
    # EasyOCR uses standard 2-letter codes
    return list(languages)
