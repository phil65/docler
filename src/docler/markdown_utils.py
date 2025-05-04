"""Utilities for handling Markdown with embedded metadata comments."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Generator


# Default prefix for metadata comments
DEFAULT_PREFIX = "docler"

# Standard metadata types
PAGE_BREAK_TYPE = "page_break"
PAGE_META_TYPE = "page_meta"
CHUNK_BOUNDARY_TYPE = "chunk_boundary"


def create_metadata_comment(
    data_type: str,
    data: dict[str, Any],
    prefix: str = DEFAULT_PREFIX,
) -> str:
    """Creates a formatted XML comment containing JSON metadata.

    Args:
        data_type: The specific type of metadata (e.g., 'page_meta', 'image').
        data: A dictionary containing the metadata payload (JSON-serializable).
        prefix: The namespace prefix for the comment.

    Returns:
        A string formatted as <!-- prefix:data_type {compact_json_payload} -->.

    Raises:
        TypeError: If the data dictionary contains non-JSON-serializable types.
        ValueError: If prefix or data_type are empty.
    """
    if not prefix:
        msg = "Metadata comment prefix cannot be empty."
        raise ValueError(msg)
    if not data_type:
        msg = "Metadata comment data_type cannot be empty."
        raise ValueError(msg)

    try:
        # Use compact separators and sort keys for consistency
        json_payload = json.dumps(data, separators=(",", ":"), sort_keys=True)
    except TypeError as e:
        err_msg = f"Data for {prefix}:{data_type} is not JSON serializable"
        raise TypeError(err_msg) from e

    comment_content = f"{prefix}:{data_type} {json_payload}"
    return f"<!-- {comment_content} -->"


def parse_metadata_comments(
    content: str,
    data_type: str,
    prefix: str = DEFAULT_PREFIX,
) -> Generator[dict[str, Any], None, None]:
    """Finds and parses specific metadata comments in content.

    Args:
        content: The Markdown string to search within.
        data_type: The specific type of metadata comment to find.
        prefix: The namespace prefix used in the comments.

    Yields:
        Dictionaries representing the parsed JSON payload of each found comment.

    Raises:
        ValueError: If prefix or data_type are empty.
        json.JSONDecodeError: If the payload within a matched comment is invalid JSON.
    """
    if not prefix:
        msg = "Metadata comment prefix cannot be empty."
        raise ValueError(msg)
    if not data_type:
        msg = "Metadata comment data_type cannot be empty."
        raise ValueError(msg)

    # Pattern: <!-- prefix:data_type {JSON_PAYLOAD} -->
    # - \s* handles optional whitespace around the payload.
    # - (.*?) captures the JSON payload non-greedily.
    pattern_str = rf"<!--\s*{re.escape(prefix)}:{re.escape(data_type)}\s+(.*?)\s*-->"
    pattern = re.compile(pattern_str)

    for match in pattern.finditer(content):
        json_payload = match.group(1)
        try:
            yield json.loads(json_payload)
        except json.JSONDecodeError as e:
            # Add context to the error
            line_num = content.count("\n", 0, match.start()) + 1
            err_msg = (
                f"Invalid JSON in {prefix}:{data_type} comment "
                f"near line {line_num}: '{json_payload}'"
            )
            raise json.JSONDecodeError(err_msg, e.doc, e.pos) from e


def create_chunk_boundary(
    chunk_id: int,
    start_line: int | None = None,
    end_line: int | None = None,
    keywords: list[str] | None = None,
    token_count: int | None = None,
    extra_data: dict[str, Any] | None = None,
    prefix: str = DEFAULT_PREFIX,
) -> str:
    """Create a chunk boundary comment with metadata.

    This is a convenience wrapper around create_metadata_comment specifically
    for marking chunk boundaries in markdown documents.

    Args:
        chunk_id: Unique identifier for this chunk
        start_line: Starting line number for this chunk (1-based)
        end_line: Ending line number for this chunk (1-based)
        keywords: List of keywords or key concepts in this chunk
        token_count: Number of tokens in this chunk
        extra_data: Additional metadata to include in the comment
        prefix: Namespace prefix for the comment

    Returns:
        A formatted comment string marking a chunk boundary
    """
    data: dict[str, Any] = {"chunk_id": chunk_id}

    if start_line is not None:
        data["start_line"] = start_line
    if end_line is not None:
        data["end_line"] = end_line
    if keywords:
        data["keywords"] = keywords
    if token_count is not None:
        data["token_count"] = token_count
    if extra_data:
        data.update(extra_data)

    return create_metadata_comment(CHUNK_BOUNDARY_TYPE, data, prefix)


def get_chunk_boundaries(
    content: str, prefix: str = DEFAULT_PREFIX
) -> Generator[dict[str, Any], None, None]:
    """Extract all chunk boundary metadata from markdown content.

    This is a convenience wrapper around parse_metadata_comments specifically
    for finding chunk boundaries in markdown documents.

    Args:
        content: Markdown content to parse
        prefix: Namespace prefix for the comments

    Yields:
        Dictionaries containing chunk boundary metadata
    """
    yield from parse_metadata_comments(content, CHUNK_BOUNDARY_TYPE, prefix)


def split_markdown_by_page(
    content: str,
    page_break_type: str = PAGE_BREAK_TYPE,
    prefix: str = DEFAULT_PREFIX,
) -> list[str]:
    """Splits Markdown content into pages based on page break comments.

    Args:
        content: The Markdown string to split.
        page_break_type: The data_type used for page break comments.
        prefix: The namespace prefix used in the comments.

    Returns:
        A list of strings, where each string is the content of a page.
        The page break comments themselves are not included in the output strings.
    """
    if not prefix:
        msg = "Metadata comment prefix cannot be empty."
        raise ValueError(msg)
    if not page_break_type:
        msg = "Page break comment data_type cannot be empty."
        raise ValueError(msg)

    # Pattern to match the entire page break comment for splitting
    # Matches the comment structure but doesn't need to capture the payload
    pattern_str = (
        rf"<!--\s*{re.escape(prefix)}:{re.escape(page_break_type)}\s+.*?\s*-->\n?"
    )
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


def split_markdown_by_chunks(
    content: str,
    chunk_boundary_type: str = CHUNK_BOUNDARY_TYPE,
    prefix: str = DEFAULT_PREFIX,
) -> list[tuple[dict[str, Any], str]]:
    """Splits Markdown content into chunks based on chunk boundary comments.

    Args:
        content: The Markdown string to split.
        chunk_boundary_type: The data_type used for chunk boundary comments.
        prefix: The namespace prefix used in the comments.

    Returns:
        A list of tuples where each tuple contains:
        - The chunk metadata dictionary
        - The chunk content string
    """
    if not prefix:
        msg = "Metadata comment prefix cannot be empty."
        raise ValueError(msg)
    if not chunk_boundary_type:
        msg = "Chunk boundary comment data_type cannot be empty."
        raise ValueError(msg)

    # First extract all chunk metadata with positions
    boundaries = []
    pattern_str = (
        rf"<!--\s*{re.escape(prefix)}:{re.escape(chunk_boundary_type)}\s+(.*?)\s*-->"
    )
    pattern = re.compile(pattern_str)

    for match in pattern.finditer(content):
        try:
            metadata = json.loads(match.group(1))
            boundaries.append((match.start(), match.end(), metadata))
        except json.JSONDecodeError:
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


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing create_metadata_comment ---")
    pb_data = {"next_page": 2}
    pb_comment = create_metadata_comment("page_break", pb_data)
    print(f"Page Break Comment: {pb_comment}")

    pm_data = {"page_num": 1, "width": 600}
    pm_comment = create_metadata_comment("page_meta", pm_data)
    print(f"Page Meta Comment: {pm_comment}")

    print("\n--- Testing parse_metadata_comments ---")
    test_content = f"""
    Some text on page 1.
    {pm_comment}
    More text.
    <!-- docler:page_meta {{"page_num": 1, "confidence": 0.95}} -->
    {pb_comment}
    Text on page 2.
    <!-- docler:page_meta {{"page_num": 2}} -->
    <!-- docler:other_type {{"value": true}} -->
    Invalid comment: <!-- docler:page_meta {"page_num": 3} -->
    """

    print("Parsing 'page_meta':")
    meta_comments = list(parse_metadata_comments(test_content, "page_meta"))
    print(meta_comments)

    print("\nParsing 'page_break':")
    break_comments = list(parse_metadata_comments(test_content, "page_break"))
    print(break_comments)

    print("\nTesting invalid JSON parsing:")
    try:
        list(
            parse_metadata_comments(
                test_content + "<!-- docler:page_meta {invalid json -->", "page_meta"
            )
        )
    except json.JSONDecodeError as e:
        print(f"Caught expected JSON error: {e}")

    print("\n--- Testing split_markdown_by_page ---")
    split_content = f"""Page 1 content.
{pm_comment}
{pb_comment}
Page 2 content.
<!-- docler:page_meta {{"page_num": 2}} -->
{create_metadata_comment("page_break", {"next_page": 3})}
Page 3 content which might be empty.
{create_metadata_comment("page_break", {"next_page": 4})}
"""
    pages = split_markdown_by_page(split_content)
    print(f"Split into {len(pages)} pages:")
    for i, page_content in enumerate(pages):
        print(f"--- Page {i + 1} ---")
        print(page_content.strip())  # Use strip for cleaner demo output
        print("----------------")

    print("\nTesting split with leading break:")
    leading_break_content = f"""{pb_comment}Page 1 content."""
    pages_leading = split_markdown_by_page(leading_break_content)
    print(f"Split (leading break) into {len(pages_leading)} pages:")
    # Expecting ['', 'Page 1 content.']
    print(pages_leading)

    print("\n--- Testing Chunk Boundary Utilities ---")
    chunk1 = create_chunk_boundary(
        chunk_id=1, start_line=1, end_line=10, keywords=["introduction", "overview"]
    )
    print(f"Chunk Boundary Comment: {chunk1}")

    chunk2 = create_chunk_boundary(
        chunk_id=2,
        start_line=11,
        end_line=25,
        keywords=["architecture", "components"],
        token_count=350,
    )

    chunk3 = create_chunk_boundary(
        chunk_id=3,
        start_line=26,
        end_line=40,
        keywords=["implementation", "code"],
        extra_data={"semantic_level": "section"},
    )

    chunked_content = f"""
{chunk1}
# Introduction

This is the introduction section. It provides an overview of the document.
More introduction text here.

{chunk2}
## Architecture

This section describes the system architecture and its components.
Detailed architecture information follows.

{chunk3}
## Implementation

Code examples and implementation details go here.
"""

    chunk_boundaries = list(get_chunk_boundaries(chunked_content))
    print("\nChunk boundary metadata:")
    for i, metadata in enumerate(chunk_boundaries):
        print(f"Chunk {i + 1}: {metadata}")

    chunks = split_markdown_by_chunks(chunked_content)
    print(f"\nSplit into {len(chunks)} chunks:")
    for metadata, content in chunks:
        print(f"--- Chunk {metadata['chunk_id']} ---")
        print(f"Metadata: {metadata}")
        print(f"Content snippet: {content[:50]}...")
        print("----------------")
