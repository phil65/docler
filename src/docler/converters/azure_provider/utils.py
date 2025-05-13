from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from mkdown import Image, create_image_reference, create_page_break


if TYPE_CHECKING:
    from collections.abc import Iterator

    from azure.ai.documentintelligence.models import AnalyzeResult


def to_image(response: Iterator[bytes], i: int) -> Image:
    content = b"".join(response)
    image_id = f"img-{i}"
    filename = f"{image_id}.png"
    return Image(id=image_id, content=content, mime_type="image/png", filename=filename)


def update_content(content: str, images: list[Image]) -> str:
    figure_pattern = r"<figure>(.*?)</figure>"
    figure_blocks = re.findall(figure_pattern, content, re.DOTALL)
    for i, block in enumerate(figure_blocks):
        if i < len(images):
            image = images[i]
            img_ref = create_image_reference(image.id, image.filename or "")
            content = content.replace(f"<figure>{block}</figure>", img_ref, 1)
    return content


def get_metadata(result: AnalyzeResult) -> dict[str, Any]:
    metadata = {}
    if result.documents:
        doc = result.documents[0]  # Get first document
        if doc.fields:
            metadata = {
                name: field.get("valueString") or field.get("content", "")
                for name, field in doc.fields.items()
            }
    return metadata


def replace_page_breaks(content: str) -> str:
    azure_marker = r"<!--\s*PageBreak\s*-->"
    page_num = 1

    def replace_marker(match: re.Match[str]) -> str:
        nonlocal page_num
        page_num += 1
        return create_page_break(next_page=page_num, newline_separators=1)

    processed_content = re.sub(azure_marker, replace_marker, content)
    first_page_marker = create_page_break(next_page=1, newline_separators=1).lstrip()
    return first_page_marker + processed_content
