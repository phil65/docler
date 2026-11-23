"""Utilities for handling Markdown with embedded metadata comments."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import anyenv


if TYPE_CHECKING:
    from collections.abc import Generator


# Default prefix for metadata comments
DEFAULT_PREFIX = "docler"

# Standard metadata types
PAGE_BREAK_TYPE = "page_break"
CHUNK_BOUNDARY_TYPE = "chunk_boundary"


def create_image_reference(
    label: str,
    path: str,
    newline_separators: int = 0,
) -> str:
    seps = "\n" * newline_separators
    return f"{seps}![{label}]({path}){seps}"


def create_page_break(
    next_page: int,
    newline_separators: int = 0,
    metadata: dict[str, Any] | None = None,
) -> str:
    seps = "\n" * newline_separators
    meta = metadata or {}
    meta["next_page"] = next_page
    comment = create_metadata_comment(PAGE_BREAK_TYPE, meta)
    return f"{seps}{comment}{seps}"


def create_metadata_comment(data_type: str, data: dict[str, Any]) -> str:
    """Creates a formatted XML comment containing JSON metadata.

    Args:
        data_type: The specific type of metadata (e.g., 'chunk_boundary', 'image').
        data: A dictionary containing the metadata payload (JSON-serializable).

    Returns:
        A string formatted as <!-- docler:data_type {compact_json_payload} -->.

    Raises:
        TypeError: If the data dictionary contains non-JSON-serializable types.
        ValueError: If data_type is empty.
    """
    if not data_type:
        msg = "Metadata comment data_type cannot be empty."
        raise ValueError(msg)

    try:
        json_payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    except TypeError as e:
        err_msg = f"Data for {DEFAULT_PREFIX}:{data_type} is not JSON serializable"
        raise TypeError(err_msg) from e

    comment_content = f"{DEFAULT_PREFIX}:{data_type} {json_payload}"
    return f"<!-- {comment_content} -->"


def parse_metadata_comments(
    content: str,
    data_type: str,
) -> Generator[dict[str, Any]]:
    """Finds and parses specific metadata comments in content.

    Args:
        content: The Markdown string to search within.
        data_type: The specific type of metadata comment to find.

    Yields:
        Dictionaries representing the parsed JSON payload of each found comment.

    Raises:
        ValueError: If prefix or data_type are empty.
        anyenv.JsonLoadError: If the payload within a matched comment is invalid JSON.
    """
    if not data_type:
        msg = "Metadata comment data_type cannot be empty."
        raise ValueError(msg)

    # Pattern: <!-- prefix:data_type {JSON_PAYLOAD} -->
    # - \s* handles optional whitespace around the payload.
    # - (.*?) captures the JSON payload non-greedily.
    pattern_str = rf"<!--\s*{DEFAULT_PREFIX}:{re.escape(data_type)}\s+(.*?)\s*-->"
    pattern = re.compile(pattern_str)

    for match in pattern.finditer(content):
        json_payload = match.group(1)
        yield anyenv.load_json(json_payload)


def create_chunk_boundary(
    chunk_id: int,
    keywords: list[str] | None = None,
    extra_data: dict[str, Any] | None = None,
) -> str:
    """Create a chunk boundary comment with metadata.

    This is a convenience wrapper around create_metadata_comment specifically
    for marking chunk boundaries in markdown documents.

    Args:
        chunk_id: Unique identifier for this chunk
        keywords: List of keywords or key concepts in this chunk
        extra_data: Additional metadata to include in the comment

    Returns:
        A formatted comment string marking a chunk boundary
    """
    data: dict[str, Any] = {"chunk_id": chunk_id}

    if keywords:
        data["keywords"] = keywords
    if extra_data:
        data.update(extra_data)

    return create_metadata_comment(CHUNK_BOUNDARY_TYPE, data)


def get_chunk_boundaries(content: str) -> Generator[dict[str, Any]]:
    """Extract all chunk boundary metadata from markdown content.

    This is a convenience wrapper around parse_metadata_comments specifically
    for finding chunk boundaries in markdown documents.

    Args:
        content: Markdown content to parse
        prefix: Namespace prefix for the comments

    Yields:
        Dictionaries containing chunk boundary metadata
    """
    yield from parse_metadata_comments(content, CHUNK_BOUNDARY_TYPE)


def split_markdown_by_page(content: str) -> list[str]:
    """Splits Markdown content into pages based on page break comments.

    Args:
        content: The Markdown string to split.

    Returns:
        A list of strings, where each string is the content of a page.
        The page break comments themselves are not included in the output strings.
    """
    # Pattern to match the entire page break comment for splitting
    # Matches the comment structure but doesn't need to capture the payload
    pattern_str = rf"<!--\s*{DEFAULT_PREFIX}:{PAGE_BREAK_TYPE}\s+.*?\s*-->\n?"
    # Include optional trailing newline (\n?) in the delimiter to avoid leading
    # newlines in subsequent pages.
    pattern = re.compile(pattern_str)
    # re.split might leave an empty string at the beginning if content starts
    # with the delimiter, or at the end. Filter these out if they are truly empty.
    # However, pages themselves can be empty, so only filter if it's the first/last
    # and completely empty due to splitting artifacts.
    # A simpler approach is often just to return the direct result unless
    # specific cleanup is strictly required. Let's keep it simple for now.
    # If the first page is empty because the doc starts with a page break, keep it.
    return pattern.split(content)


def split_markdown_by_chunks(content: str) -> list[tuple[dict[str, Any], str]]:
    """Splits Markdown content into chunks based on chunk boundary comments.

    Args:
        content: The Markdown string to split.

    Returns:
        A list of tuples where each tuple contains:
        - The chunk metadata dictionary
        - The chunk content string
    """
    # First extract all chunk metadata with positions
    boundaries = []
    pattern_str = rf"<!--\s*{DEFAULT_PREFIX}:{CHUNK_BOUNDARY_TYPE}\s+(.*?)\s*-->"
    pattern = re.compile(pattern_str)

    for match in pattern.finditer(content):
        try:
            metadata = anyenv.load_json(match.group(1), return_type=dict)
            boundaries.append((match.start(), match.end(), metadata))
        except anyenv.JsonLoadError:
            # Skip invalid JSON
            continue

    if not boundaries:
        return []

    # Sort by start position to ensure correct order
    boundaries.sort(key=lambda x: x[0])

    # Extract chunks by slicing between boundaries
    result = []
    for i, (_start_pos, end_pos, metadata) in enumerate(boundaries):
        # For the last boundary, content goes to the end of the string
        next_start = len(content) if i == len(boundaries) - 1 else boundaries[i + 1][0]

        # Extract chunk content (skip the comment itself)
        chunk_content = content[end_pos:next_start].strip()
        result.append((metadata, chunk_content))

    return result
