"""Utility functions for the Streamlit app."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docler.converters.base import SupportedLanguage


# Language options
LANGUAGES: list[SupportedLanguage] = ["en", "de", "fr", "es", "zh"]


def format_image_content(image_content: bytes | str, mime_type: str) -> str:
    """Convert image content to base64 data URL.

    Args:
        image_content: Raw bytes or base64 string of image data
        mime_type: MIME type of the image

    Returns:
        Data URL format of the image for embedding in HTML/Markdown
    """
    if isinstance(image_content, bytes):
        # Convert bytes to base64
        b64_content = base64.b64encode(image_content).decode()
    else:
        # Already base64 string - ensure no data URL prefix
        b64_content = (
            image_content.split(",")[-1] if "," in image_content else image_content
        )

    return f"data:{mime_type};base64,{b64_content}"
