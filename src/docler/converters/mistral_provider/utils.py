from __future__ import annotations

from mkdown import Image


def convert_image(img) -> Image:
    img_data = img.image_base64
    if img_data.startswith("data:image/"):
        img_data = img_data.split(",", 1)[1]
    ext = img.id.split(".")[-1].lower() if "." in img.id else "jpeg"
    mime = f"image/{ext}"
    return Image(id=img.id, content=img_data, mime_type=mime, filename=img.id)


def _parse_page_range(page_range: str | None) -> list[int] | None:
    """Convert a page range string to a list of page numbers.

    Args:
        page_range: String like "1-5,7,9-11" or None. 1-based page numbers.

    Returns:
        List of page numbers (1-based) or None if no range specified.
        Mistral API expects 1-based page numbers.

    Raises:
        ValueError: If the page range format is invalid.
    """
    if not page_range:
        return None
    pages: set[int] = set()
    try:
        for part in page_range.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part))
    except ValueError as e:
        msg = f"Invalid page range format: {page_range}. Expected format: '1-5,7,9-11'"
        raise ValueError(msg) from e
    return sorted(pages)
