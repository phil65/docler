import base64
from typing import Any

import anyenv
from mkdown import Image, create_page_break


def process_response(result: list[Any], api_key: str) -> tuple[list[str], list[Image]]:
    from llama_index.core.constants import DEFAULT_BASE_URL

    content_parts: list[str] = []
    images: list[Image] = []
    pages = result[0]["pages"]
    job_id = result[0]["job_id"]
    for page_num, page in enumerate(pages, start=1):
        if page.get("md"):
            comment = create_page_break(next_page=page_num)
            content_parts.append(comment)
            content_parts.append(page["md"])
        for img in page.get("images", []):
            image_count = len(images)
            id_ = f"img-{image_count}"
            asset_name = img["name"]
            asset_url = (
                f"{DEFAULT_BASE_URL}/api/parsing/job/{job_id}/result/image/{asset_name}"
            )
            headers = {"Authorization": f"Bearer {api_key}"}
            response = anyenv.get_bytes_sync(asset_url, headers=headers)
            img_data = base64.b64encode(response).decode("utf-8")
            img_type = "png"
            if "." in asset_name:
                extension = asset_name.split(".")[-1].lower()
                if extension in ["jpg", "jpeg", "png", "gif", "webp", "svg"]:
                    img_type = extension

            filename = f"{id_}.{img_type}"
            mime = f"image/{img_type}"
            image = Image(id=id_, content=img_data, mime_type=mime, filename=filename)
            images.append(image)
    return content_parts, images
