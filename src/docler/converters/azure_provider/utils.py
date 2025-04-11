from __future__ import annotations

import re
from typing import TYPE_CHECKING

from docler.models import Image


if TYPE_CHECKING:
    from collections.abc import Iterator


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
            img_ref = f"\n\n![{image.id}]({image.filename})\n\n"
            content = content.replace(f"<figure>{block}</figure>", img_ref, 1)
    return content
