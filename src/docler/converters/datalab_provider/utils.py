"""Document converter using DataLab's API."""

from __future__ import annotations

import base64
import re
import time
from typing import Any

import anyenv
from mkdown import Image, create_image_reference, create_page_break

from docler.log import get_logger
from docler.utils import pil_to_bytes


logger = get_logger(__name__)

API_BASE = "https://www.datalab.to/api/v1"
MAX_POLLS = 300
POLL_INTERVAL = 2


def _normalize_markdown_images(
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
                return create_image_reference(img_id, filename)
        # If no match found, keep the alt text
        return create_image_reference(match.group(1), filename)

    # Replace in all image patterns
    result = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_image_alt, result)

    # Replace any remaining empty image refs with proper IDs
    for img_id, filename in image_replacements.values():
        result = result.replace(f"![]({filename})", f"![{img_id}]({filename})")

    return result


async def get_response(
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


def process_response(result: dict[str, Any]) -> tuple[str, list[Image]]:
    images: list[Image] = []
    md_content = result["markdown"]

    # Convert DataLab page break format to standard format
    # DataLab uses a format like "{2}--------------" when paginate=True
    # Make pattern flexible to handle variations in spacing and dash count
    page_break_pattern = r"(?:^|\n\n)\s*\{(\d+)\}\s*-+\s*\n\n"

    def replace_page_break(match):
        try:
            page_num = int(match.group(1))
            return create_page_break(next_page=page_num + 1, newline_separators=2)
        except (ValueError, IndexError) as e:
            logger.warning("Failed to parse page number from page break marker: %s", e)
            # Return the original match if we can't parse it
            return match.group(0)

    # Count matches before replacement for logging
    original_count = len(re.findall(page_break_pattern, md_content))
    md_content = re.sub(page_break_pattern, replace_page_break, md_content)
    if original_count > 0:
        msg = "Converted %d DataLab page breaks to standard format"
        logger.debug(msg, original_count)
    from PIL.Image import Image as PILImage

    if result.get("images"):
        image_replacements = {}
        for i, (original_name, img_data) in enumerate(result["images"].items()):
            img_id = f"img-{i}"
            ext = original_name.split(".")[-1].lower()
            fname = f"{img_id}.{ext}"
            image_replacements[original_name] = (img_id, fname)
            if isinstance(img_data, PILImage):
                content = pil_to_bytes(img_data)
            else:
                if img_data.startswith("data:"):
                    img_data = img_data.split(",", 1)[1]
                content = base64.b64decode(img_data)
            mime = f"image/{ext}"
            image = Image(id=img_id, content=content, mime_type=mime, filename=fname)
            images.append(image)

        md_content = _normalize_markdown_images(md_content, image_replacements)
    return md_content, images
