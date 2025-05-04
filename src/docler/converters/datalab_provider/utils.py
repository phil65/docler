"""Document converter using DataLab's API."""

from __future__ import annotations

import re
import time
from typing import Any

import anyenv


API_BASE = "https://www.datalab.to/api/v1"
MAX_POLLS = 300
POLL_INTERVAL = 2


def normalize_markdown_images(
    content: str, image_replacements: dict[str, tuple[str, str]]
) -> str:
    """Normalize image references in markdown content.

    Args:
        content: Original markdown content with image references
        image_replacements: Map of original file names to (image_id, filename) tuples

    Returns:
        Markdown with normalized image references
    """
    # First replace file paths in markdown links
    result = content
    for original_name, (_, filename) in image_replacements.items():
        result = result.replace(f"]({original_name})", f"]({filename})")

    # Then fix image alt texts with proper IDs
    def replace_image_alt(match):
        """Replace image alt text with appropriate image ID."""
        filename = match.group(2)
        # Get the correct image ID for this filename
        for orig_name, (img_id, new_filename) in image_replacements.items():
            if filename in (new_filename, orig_name):
                return f"![{img_id}]({filename})"
        # If no match found, keep the alt text
        return f"![{match.group(1)}]({filename})"

    # Replace in all image patterns
    result = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_image_alt, result)

    # Replace any remaining empty image refs with proper IDs
    for img_id, filename in image_replacements.values():
        result = result.replace(f"![]({filename})", f"![{img_id}]({filename})")

    return result


# def convert(markdown: str, image_dict: dict[str, str]):
#     import base64

#     from docler.models import Image

#     images: list[Image] = []
#     md_content = markdown
#     if images:
#         image_replacements = {}
#         for i, (original_name, img_data) in enumerate(image_dict.items()):
#             img_id = f"img-{i}"
#             ext = original_name.split(".")[-1].lower()
#             fname = f"{img_id}.{ext}"
#             image_replacements[original_name] = (img_id, fname)
#             if img_data.startswith("data:"):
#                 img_data = img_data.split(",", 1)[1]
#             content = base64.b64decode(img_data)
#             mime = f"image/{ext}"
#             image = Image(id=img_id, content=content, mime_type=mime, filename=fname)
#             images.append(image)

#         md_content = normalize_markdown_images(md_content, image_replacements)
#     return md_content, images


async def get_result(
    form: dict[str, Any],
    files: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    headers = {"X-Api-Key": api_key}
    url = f"{API_BASE}/marker"
    response = await anyenv.post(url, data=form, files=files, headers=headers)
    json_data = await response.json()
    if not json_data["success"]:
        msg = f"Failed to submit conversion: {json_data['error']}"
        raise ValueError(msg)
    check_url = json_data["request_check_url"]
    for _ in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        result = await anyenv.get_json(check_url, headers=headers, return_type=dict)  # type: ignore
        if result["status"] == "complete":
            break
    else:
        msg = "Conversion timed out"
        raise TimeoutError(msg)

    if not result["success"]:
        msg = f"Conversion failed: {result['error']}"
        raise ValueError(msg)
    return result
