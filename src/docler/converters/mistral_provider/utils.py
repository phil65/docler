from __future__ import annotations

from docler.models import Image


def convert_image(img) -> Image:
    img_data = img.image_base64
    if img_data.startswith("data:image/"):
        img_data = img_data.split(",", 1)[1]
    ext = img.id.split(".")[-1].lower() if "." in img.id else "jpeg"
    mime = f"image/{ext}"
    return Image(id=img.id, content=img_data, mime_type=mime, filename=img.id)
