"""Utilities."""

from __future__ import annotations

import base64
import io
from io import BytesIO
import re
from typing import TYPE_CHECKING, Literal, overload


if TYPE_CHECKING:
    import PIL.Image
    from PIL.ImageFile import ImageFile

    from docler.common_types import StrPath


def shift_page_range(page_range: str, shift: int = 0) -> str:
    """Shift page numbers in a page range string by the specified amount.

    Args:
        page_range: Page range string like "1-5,7,9-11"
        shift: Amount to shift page numbers by (e.g., -1 to convert 1-based to 0-based)

    Returns:
        Shifted page range string

    Raises:
        ValueError: If the page range format is invalid
    """
    parts = []
    try:
        for part in page_range.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                if start + shift < 0 or end + shift < 0:
                    msg = f"Invalid shift {shift} for page range {page_range}"
                    raise ValueError(msg)  # noqa: TRY301
                parts.append(f"{start + shift}-{end + shift}")
            else:
                page = int(part)
                if page + shift < 0:
                    msg = f"Invalid shift {shift} for page {page}"
                    raise ValueError(msg)  # noqa: TRY301
                parts.append(str(page + shift))
    except ValueError as e:
        if "Invalid shift" in str(e):
            raise
        msg = f"Invalid page range format: {page_range}. Expected format: '1-5,7,9-11'"
        raise ValueError(msg) from e

    return ",".join(parts)


def pil_to_bytes(image: PIL.Image.Image) -> bytes:
    """Convert PIL image to bytes in its native format."""
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=image.format or "JPEG")
    return img_byte_arr.getvalue()


def get_mime_from_pil(image: PIL.Image.Image, fallback: str = "JPEG") -> str:
    """Get MIME type from PIL image format."""
    fmt = image.format or fallback
    return f"image/{fmt.lower()}"


@overload
def check_mime(
    path: StrPath,
    *,
    allowed_mime_types: set[str] | None = None,
    raise_if_none_found: Literal[True],
) -> str: ...


@overload
def check_mime(
    path: StrPath,
    *,
    allowed_mime_types: set[str] | None = None,
    raise_if_none_found: Literal[False] = False,
) -> str | None: ...


def check_mime(
    path: StrPath,
    *,
    allowed_mime_types: set[str] | None = None,
    raise_if_none_found: bool = False,
) -> str | None:
    """Return and optionally check MIME type for a path.

    Determines the MIME type of a file or extension and raises an error
    if the type cannot be determined or if it is not allowed.

    Args:
        path: Path to the file (like "test.jpg", or just the extension starting with ".")
        allowed_mime_types: Set of allowed MIME types.
        raise_if_none_found: Whether to raise an error if the type cant be determined.
    """
    import mimetypes

    path_str = str(path)
    if path_str.startswith("."):
        path_str = f"file{path_str}"
    mime = mimetypes.guess_type(path_str)[0]
    if mime is None and raise_if_none_found:
        msg = f"Could not determine MIME type for {path}"
        raise ValueError(msg)
    if allowed_mime_types and mime not in allowed_mime_types:
        msg = f"Invalid MIME type: {mime}. Allowed types: {allowed_mime_types}"
        raise ValueError(msg)
    return mime


def decode_base64_to_image(encoded_string, image_format="PNG") -> ImageFile:
    """Decode a base64 string to an image."""
    from PIL import Image

    try:
        image_data = base64.b64decode(encoded_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:  # noqa: BLE001
        msg = f"Failed to decode image: {e!s}"
        raise ValueError(msg)  # noqa: B904


def encode_image_to_base64(image, image_format="WEBP", quality=20):
    """Encode an image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def png_to_webp(content: str) -> str:
    """Convert PNG images to WebP format in Markdown content."""
    pattern = re.compile(r"!\[Image\]\(data:image/png;base64,([^)]*)\)")
    matches = pattern.findall(content)

    for match in matches:
        try:
            png_image = decode_base64_to_image(match, "PNG")

            if png_image.format != "PNG":
                continue

            if png_image.width > 1080:  # noqa: PLR2004
                ratio = png_image.height / png_image.width
                webp_image = png_image.resize((1080, int(1080 * ratio)))
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
    """Get environment variable, throw if not set."""
    import os

    key = os.getenv(env_var)
    if not key:
        msg = f"Required environment variable {env_var} not set"
        raise ValueError(msg)
    return key


def add_line_numbers(text: str) -> str:
    """Add line numbers to text."""
    lines = text.splitlines()
    return "\n".join(f"{i + 1:5d} | {line}" for i, line in enumerate(lines))
