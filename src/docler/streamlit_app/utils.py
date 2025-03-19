"""Utility functions for the Streamlit app."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docler.common_types import SupportedLanguage


# Language options
LANGUAGES: list[SupportedLanguage] = ["en", "de", "fr", "es", "zh"]


def format_image_content(data: bytes | str, mime_type: str) -> str:
    """Convert image content to base64 data URL.

    Args:
        data: Raw bytes or base64 string of image data
        mime_type: MIME type of the image

    Returns:
        Data URL format of the image for embedding in HTML/Markdown
    """
    if isinstance(data, bytes):
        b64_content = base64.b64encode(data).decode()
    else:
        b64_content = data.split(",")[-1] if "," in data else data
    return f"data:{mime_type};base64,{b64_content}"
