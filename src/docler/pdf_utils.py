"""Document converter using PyPDF2."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from PyPDF2 import PdfReader, PdfWriter

from docler.log import get_logger


if TYPE_CHECKING:
    from docler.common_types import PageRangeString


logger = get_logger(__name__)


def _parse_page_range(page_range: PageRangeString | None) -> set[int]:
    """Convert a page range string to a set of page numbers.

    Args:
        page_range: String like "1-5,7,9-11" or None. 1-based page numbers.

    Returns:
        Set of page numbers (0-based)

    Raises:
        ValueError: If the page range format is invalid.
    """
    if not page_range:
        return set()
    pages: set[int] = set()
    try:
        for part in page_range.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                # Convert to 0-based indexing
                pages.update(range(start - 1, end))
            else:
                # Convert to 0-based indexing
                pages.add(int(part) - 1)
    except ValueError as e:
        msg = f"Invalid page range format: {page_range}. Expected format: '1-5,7,9-11'"
        raise ValueError(msg) from e
    else:
        return pages


def shift_page_range(page_range: PageRangeString, shift: int = 0) -> PageRangeString:
    """Shift page numbers in a page range string by the specified amount.

    Args:
        page_range: Page range string like "1-5,7,9-11"
        shift: Amount to shift page numbers by (e.g., -1 to convert 1-based to 0-based)

    Returns:
        Shifted page range string

    Raises:
        ValueError: If the page range format is invalid
    """
    parts = []
    try:
        for part in page_range.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                if start + shift < 0 or end + shift < 0:
                    msg = f"Invalid shift {shift} for page range {page_range}"
                    raise ValueError(msg)  # noqa: TRY301
                parts.append(f"{start + shift}-{end + shift}")
            else:
                page = int(part)
                if page + shift < 0:
                    msg = f"Invalid shift {shift} for page {page}"
                    raise ValueError(msg)  # noqa: TRY301
                parts.append(str(page + shift))
    except ValueError as e:
        if "Invalid shift" in str(e):
            raise
        msg = f"Invalid page range format: {page_range}. Expected format: '1-5,7,9-11'"
        raise ValueError(msg) from e

    return ",".join(parts)


def extract_pdf_pages(data: bytes, page_range: PageRangeString | None) -> bytes:
    """Extract specific pages from a PDF file and return as new PDF.

    Args:
        data: Source PDF file content as bytes
        page_range: String like "1-5,7,9-11" or None for all pages. 1-based.

    Returns:
        New PDF containing only specified pages as bytes

    Raises:
        ValueError: If page range is invalid or PDF data cannot be processed
    """
    pages = _parse_page_range(page_range)
    with io.BytesIO(data) as pdf_io, io.BytesIO() as output:
        try:
            reader = PdfReader(pdf_io)
            writer = PdfWriter()
            page_indices = pages if pages else range(len(reader.pages))
            for i in page_indices:
                if 0 <= i < len(reader.pages):
                    writer.add_page(reader.pages[i])
            writer.write(output)
            return output.getvalue()
        except Exception as e:
            msg = f"Failed to extract pages from PDF: {e}"
            raise ValueError(msg) from e
