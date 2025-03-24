"""Utilities."""

from __future__ import annotations

import base64
import io
from io import BytesIO
import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import PIL.Image


def pil_to_bytes(image: PIL.Image.Image) -> bytes:
    """Convert PIL image to bytes in its native format."""
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=image.format or "JPEG")
    return img_byte_arr.getvalue()


def get_mime_from_pil(image: PIL.Image.Image) -> str:
    """Get MIME type from PIL image format."""
    format_ = image.format or "JPEG"
    return f"image/{format_.lower()}"


def decode_base64_to_image(encoded_string, image_format="PNG"):
    from PIL import Image

    try:
        image_data = base64.b64decode(encoded_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:  # noqa: BLE001
        msg = f"Failed to decode image: {e!s}"
        raise ValueError(msg)  # noqa: B904


def encode_image_to_base64(image, image_format="WEBP", quality=20):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def png_to_webp(content: str) -> str:
    pattern = re.compile(r"!\[Image\]\(data:image/png;base64,([^)]*)\)")
    matches = pattern.findall(content)

    for match in matches:
        try:
            png_image = decode_base64_to_image(match, "PNG")

            if png_image.format != "PNG":
                continue

            if png_image.width > 1080:  # noqa: PLR2004
                webp_image = png_image.resize((
                    1080,
                    int(1080 * png_image.height / png_image.width),
                ))
            else:
                webp_image = png_image

            webp_encoded_string = encode_image_to_base64(webp_image, "WEBP", quality=20)
            content = content.replace(
                f"data:image/png;base64,{match}",
                f"data:image/webp;base64,{webp_encoded_string}",
            )

        except Exception:  # noqa: BLE001
            continue

    return content


def get_api_key(env_var: str) -> str:
    """Get API key from environment variable."""
    import os

    key = os.getenv(env_var)
    if not key:
        msg = f"Required environment variable {env_var} not set"
        raise ValueError(msg)
    return key
