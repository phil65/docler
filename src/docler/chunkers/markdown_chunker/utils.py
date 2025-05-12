"""Base markdown chunking implementation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    from mkdown import Image


def assign_images(content: str, all_images: list[Image]) -> tuple[str, list[Image]]:
    """Find images referenced in content and assign them to chunk.

    Returns:
        Tuple of (content, chunk_images)
    """
    chunk_images: list[Image] = []
    image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

    for match in re.finditer(image_pattern, content):
        image_path = match.group(2)
        for image in all_images:
            if image.filename == image_path:
                chunk_images.append(image)
                break

    return content, chunk_images


def split_by_headers(text: str) -> Iterator[tuple[str, str, int]]:
    """Split text by markdown headers.

    Returns:
        Iterator of (header, content, level) tuples
    """
    # Matches markdown headers (# to ######)
    header_pattern = r"^(#{1,6})\s+(.+)$"
    current_header = ""
    current_level = 0
    current_content: list[str] = []

    for line in text.splitlines():
        if match := re.match(header_pattern, line):
            # Yield previous section if exists
            if current_content:
                yield current_header, "\n".join(current_content), current_level
                current_content = []

            current_level = len(match.group(1))
            current_header = match.group(2)
        else:
            current_content.append(line)
    if current_content:
        yield current_header, "\n".join(current_content), current_level
