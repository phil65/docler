"""Common types used in the Docler library."""

from __future__ import annotations

import os
from typing import Literal


StrPath = str | os.PathLike[str]

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
