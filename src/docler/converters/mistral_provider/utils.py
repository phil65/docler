from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from mkdown import Image


if TYPE_CHECKING:
    from mistralai import OCRImageObject, OCRResponse


def convert_image(img: OCRImageObject) -> Image:
    img_data = img.image_base64
    assert img_data
    if img_data.startswith("data:image/"):
        header, img_data = img_data.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        ext = mime.split("/")[-1]
    else:
        mime = "image/jpeg"
        ext = "jpeg"
    img_bytes = base64.b64decode(img_data)
    filename = img.id if img.id.endswith(f".{ext}") else f"{img.id}.{ext}"
    return Image(id=img.id, content=img_bytes, mime_type=mime, filename=filename)


def get_images(response: OCRResponse) -> list[Image]:
    imgs = [i for page in response.pages for i in page.images if i.id and i.image_base64]
    images = []
    for idx, img in enumerate(imgs, start=1):
        extracted_img_data_b64 = img.image_base64
        assert extracted_img_data_b64
        header, extracted_img_data_b64 = extracted_img_data_b64.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        extracted_ext = mime.split("/")[-1]
        img_data = base64.b64decode(extracted_img_data_b64)
        img_id = f"extracted-img-{idx}"
        filename = f"{img_id}.{extracted_ext}"
        obj = Image(id=img_id, content=img_data, mime_type=mime, filename=filename)
        images.append(obj)
    return images
