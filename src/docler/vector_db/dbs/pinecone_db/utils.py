"""Pinecone vector store backend implementation."""

from __future__ import annotations

import base64
from typing import Any


def convert_filters(filters: dict[str, Any]) -> dict:
    """Convert standard filters to Pinecone filter format.

    Args:
        filters: Dictionary of filters

    Returns:
        Pinecone-compatible filter object
    """
    if not filters:
        return {}

    pinecone_filter = {}
    for key, value in filters.items():
        pinecone_filter[key] = {"$in": value} if isinstance(value, list) else value
    return pinecone_filter


def prepare_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Prepare metadata for Pinecone storage."""
    import anyenv

    prepared = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool):
            prepared[key] = value
        elif isinstance(value, list | dict):
            # Convert complex types to JSON strings
            try:
                # First try to store it directly if simple enough
                if isinstance(value, list) and all(
                    isinstance(x, str | int | float | bool) for x in value
                ):
                    prepared[key] = value  # type: ignore
                else:
                    prepared[f"{key}_json"] = anyenv.dump_json(value)
            except (TypeError, ValueError):
                dumped = anyenv.dump_json(str(value)).encode()
                prepared[f"{key}_b64"] = base64.b64encode(dumped).decode()
        elif value is not None:
            prepared[key] = str(value)

    # Ensure text field is preserved
    if "text" in metadata:
        prepared["text"] = str(metadata["text"])

    return prepared


def restore_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Restore encoded metadata fields."""
    import anyenv

    restored = metadata.copy()

    for key in list(restored.keys()):
        if key.endswith("_json") and key[:-5] not in restored:
            try:
                base_key = key[:-5]
                restored[base_key] = anyenv.load_json(restored[key])
                del restored[key]
            except Exception:  # noqa: BLE001
                pass

        elif key.endswith("_b64") and key[:-4] not in restored:
            try:
                base_key = key[:-4]
                json_str = base64.b64decode(restored[key]).decode()
                restored[base_key] = anyenv.load_json(json_str)
                del restored[key]
            except Exception:  # noqa: BLE001
                pass

    return restored
