from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from mkdown import Image


if TYPE_CHECKING:
    from mistralai import OCRResponse

    from docler.common_types import PageRangeString


def convert_image(img) -> Image:
    img_data = img.image_base64
    if img_data.startswith("data:image/"):
        img_data = img_data.split(",", 1)[1]
    ext = img.id.split(".")[-1].lower() if "." in img.id else "jpeg"
    mime = f"image/{ext}"
    return Image(id=img.id, content=img_data, mime_type=mime, filename=img.id)


def _parse_page_range(page_range: PageRangeString | None) -> list[int] | None:
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


def get_images(response: OCRResponse) -> list[Image]:
    imgs = [i for page in response.pages for i in page.images if i.id and i.image_base64]
    image_count = 1  # Start after the main image
    images = []
    for img in imgs:
        extracted_img_data_b64 = img.image_base64
        assert extracted_img_data_b64
        header, extracted_img_data_b64 = extracted_img_data_b64.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        extracted_ext = mime.split("/")[-1]
        img_data = base64.b64decode(extracted_img_data_b64)
        img_id = f"extracted-img-{image_count}"
        filename = f"{img_id}.{extracted_ext}"
        image_count += 1
        obj = Image(id=img_id, content=img_data, mime_type=mime, filename=filename)
        images.append(obj)
    return images
