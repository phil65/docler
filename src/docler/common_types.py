"""Common types used in the Docler library."""

from __future__ import annotations

import os
from typing import Literal


StrPath = str | os.PathLike[str]
VectorId = str | int
SupportedLanguage = Literal["en", "de", "fr", "es", "zh"]

DEFAULT_CHUNKER_MODEL = "openrouter:openai/o3-mini"  # google/gemini-2.0-flash-lite-001
DEFAULT_CONVERTER_MODEL = "google-gla:gemini-2.0-flash"
DEFAULT_ANNOTATOR_MODEL = "google-gla:gemini-2.0-flash"
DEFAULT_IMAGE_ANNOTATOR_MODEL = "google-gla:gemini-2.0-flash"
DEFAULT_PROOF_READER_MODEL = "google-gla:gemini-2.0-flash"

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

# https://github.com/Goldziher/kreuzberg/blob/main/kreuzberg/_ocr/_paddleocr.py
PADDLE_OCR_CODES: dict[SupportedLanguage, str] = {
    "en": "en",
    "de": "german",
    "fr": "french",
    "zh": "ch",
}
