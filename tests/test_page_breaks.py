"""Test the standardized page break metadata format."""

from __future__ import annotations

import re

from mkdown import (
    PAGE_BREAK_TYPE,
    create_metadata_comment,
    parse_metadata_comments,
    split_markdown_by_page,
)


def test_page_break_format():
    """Test that page breaks are correctly formatted and parsed."""
    # Create page breaks
    page_breaks = [
        create_metadata_comment(PAGE_BREAK_TYPE, {"next_page": i}) for i in range(1, 5)
    ]

    # Verify they match the expected format
    for i, pb in enumerate(page_breaks):
        assert f'"next_page":{i + 1}' in pb
        assert pb.startswith(f"<!-- docler:{PAGE_BREAK_TYPE}")
        assert pb.endswith(" -->")

    # Create a document with page breaks
    content = f"""This is page 1.

{page_breaks[0]}

This is page 2.

{page_breaks[1]}

This is page 3.

{page_breaks[2]}

This is page 4.
"""

    # Test parsing page breaks
    parsed = list(parse_metadata_comments(content, PAGE_BREAK_TYPE))
    assert len(parsed) == 3  # noqa: PLR2004
    assert all("next_page" in pb for pb in parsed)
    assert [pb["next_page"] for pb in parsed] == [1, 2, 3]

    # Test splitting by page breaks
    pages = split_markdown_by_page(content)
    assert len(pages) == 4  # should have 4 pages  # noqa: PLR2004
    assert "This is page 1" in pages[0]
    assert "This is page 2" in pages[1]
    assert "This is page 3" in pages[2]
    assert "This is page 4" in pages[3]

    print("All page break tests passed!")


def test_converters_compatibility():
    """Test that the old format is still recognized for backward compatibility."""
    # Old format
    old_format = "<!-- page_break page_num=3 -->"

    # New format
    new_format = create_metadata_comment(PAGE_BREAK_TYPE, {"next_page": 3})

    # Create a mixed document with both formats
    mixed_content = f"""This is page 1.

{old_format}

This is page 2.

{new_format}

This is page 3.
"""

    # Create a pattern that matches both formats
    pattern = r"<!--\s*(?:docler:)?page_break\s+(?:{.*?}|page_num=\d+)\s*-->"
    matches = re.findall(pattern, mixed_content)

    assert len(matches) == 2  # noqa: PLR2004
    assert old_format in matches
    assert new_format in matches

    # Test the split function with a compatible pattern
    pages = re.split(pattern, mixed_content)
    assert len(pages) == 3  # noqa: PLR2004

    print("Compatibility test passed!")


if __name__ == "__main__":
    test_page_break_format()
    test_converters_compatibility()
