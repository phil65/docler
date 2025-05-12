"""Utils for docling-remote provider."""

from __future__ import annotations

import base64
import re
from typing import Any

from mkdown import Image, create_image_reference


def process_response(document: dict[str, Any]) -> tuple[str, list[Image]]:
    content = document["md_content"]
    images: list[Image] = []

    if "![" in content:
        img_pattern = r"!\[([^]]*)\]\(data:image/([^;]+);base64,([^)]+)\)"
        image_counter = 0

        def replace_image(match) -> str:
            nonlocal image_counter
            alt_text = match.group(1)
            img_type = match.group(2)
            img_data = match.group(3)

            image_id = f"img-{image_counter}"
            filename = f"{image_id}.{img_type}"
            image_counter += 1
            content = base64.b64decode(img_data)
            mime = f"image/{img_type}"
            image = Image(id=image_id, content=content, mime_type=mime, filename=filename)
            images.append(image)
            return create_image_reference(alt_text or image_id, filename)

        content = re.sub(img_pattern, replace_image, content)
    return content, images
