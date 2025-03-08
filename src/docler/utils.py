from __future__ import annotations

from io import BytesIO
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
