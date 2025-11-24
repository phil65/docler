"""Document converter using pypdf."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

# import fitz  # PyMuPDF
from pypdf import PdfReader, PdfWriter

from docler.log import get_logger
from docler.models import PageDimensions, PageMetadata


if TYPE_CHECKING:
    from docler.common_types import PageRangeString


logger = get_logger(__name__)


def parse_page_range(page_range: PageRangeString, shift: int = 0) -> set[int]:
    """Convert a page range string to a set of page numbers.

    Args:
        page_range: String like "1-5,7,9-11" or None.
        shift: Amount to shift page numbers by (e.g., -1 to convert 1-based to 0-based)

    Returns:
        Set of page numbers (shifted by specified amount)

    Raises:
        ValueError: If the page range format is invalid.
    """
    if shift:
        page_range = shift_page_range(page_range, shift)

    pages: set[int] = set()
    try:
        for part in page_range.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part))
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


def decrypt_pdf(data: bytes, password: str | None) -> bytes:
    """Decrypt password-protected PDF and return decrypted bytes.

    Args:
        data: Encrypted PDF file content as bytes
        password: Password to decrypt the PDF, or None to try empty password

    Returns:
        Decrypted PDF content as bytes

    Raises:
        ValueError: If decryption fails or password is incorrect
    """
    with io.BytesIO(data) as pdf_io, io.BytesIO() as output:
        try:
            reader = PdfReader(pdf_io)
            if not reader.is_encrypted:
                return data  # Already decrypted

            # Try empty password first if no password provided
            if password is None:
                if reader.decrypt(""):
                    # Successfully decrypted with empty password
                    writer = PdfWriter()
                    for page in reader.pages:
                        writer.add_page(page)
                    writer.write(output)
                    return output.getvalue()
                msg = "PDF is encrypted and requires a password"
                raise ValueError(msg)  # noqa: TRY301

            # Try provided password
            if not reader.decrypt(password):
                msg = "Incorrect password for encrypted PDF"
                raise ValueError(msg)  # noqa: TRY301

            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            writer.write(output)
            return output.getvalue()
        except Exception as e:
            if "Incorrect password" in str(e) or "requires a password" in str(e):
                raise
            msg = f"Failed to decrypt PDF: {e}"
            raise ValueError(msg) from e


def extract_pdf_pages(
    data: bytes, page_range: PageRangeString | None, password: str | None = None
) -> bytes:
    """Extract specific pages from a PDF file and return as new PDF.

    Args:
        data: Source PDF file content as bytes
        page_range: String like "1-5,7,9-11" or None for all pages. 1-based.
        password: Password for encrypted PDFs

    Returns:
        New PDF containing only specified pages as bytes

    Raises:
        ValueError: If page range is invalid, PDF data cannot be processed,
                    or decryption fails
    """
    with io.BytesIO(data) as pdf_io, io.BytesIO() as output:
        try:
            reader = PdfReader(pdf_io)

            # Handle encrypted PDFs
            if reader.is_encrypted:
                if password is not None:
                    # Try provided password
                    if not reader.decrypt(password):
                        msg = "Incorrect password for encrypted PDF"
                        raise ValueError(msg)  # noqa: TRY301
                # Try empty password first
                elif not reader.decrypt(""):
                    msg = "PDF is encrypted but no password provided"
                    raise ValueError(msg)  # noqa: TRY301

            pages = (
                parse_page_range(page_range, shift=-1) if page_range else range(len(reader.pages))
            )
            writer = PdfWriter()
            for i in pages:
                if 0 <= i < len(reader.pages):
                    writer.add_page(reader.pages[i])
            writer.write(output)
            return output.getvalue()
        except Exception as e:
            if "encrypted" in str(e).lower():
                raise
            msg = f"Failed to extract pages from PDF: {e}"
            raise ValueError(msg) from e


def get_pdf_info(data: bytes, password: str | None = None) -> PageMetadata:
    """Get PDF metadata including page count, dimensions, and file info.

    Args:
        data: PDF file content as bytes
        password: Password for encrypted PDFs

    Returns:
        PageMetadata model containing PDF information

    Raises:
        ValueError: If PDF data cannot be processed or decryption fails
    """
    with io.BytesIO(data) as pdf_io:
        try:
            reader = PdfReader(pdf_io)

            # Handle encrypted PDFs
            is_encrypted = False
            if reader.is_encrypted:
                if password is not None:
                    # Try provided password
                    if not reader.decrypt(password):
                        msg = "Incorrect password for encrypted PDF"
                        raise ValueError(msg)  # noqa: TRY301
                # Try empty password first (many PDFs are encrypted with empty password)
                elif reader.decrypt(""):
                    # Successfully decrypted with empty password, treat as not encrypted
                    is_encrypted = False
                else:
                    # Return basic info for truly encrypted PDF without decryption
                    return PageMetadata(
                        page_count=0,
                        file_size=len(data),
                        is_encrypted=True,
                        page_dimensions=[],
                        title="",
                        author="",
                    )

            # Basic info
            page_count = len(reader.pages)
            file_size = len(data)

            # Page dimensions (in points)
            page_dimensions = []
            for page in reader.pages:
                media_box = page.mediabox
                width = float(media_box.width)
                height = float(media_box.height)
                page_dimensions.append(PageDimensions(width=width, height=height))

            # Document metadata
            metadata = reader.metadata
            title = metadata.title or "" if metadata else ""
            author = metadata.author or "" if metadata else ""

            return PageMetadata(
                page_count=page_count,
                file_size=file_size,
                is_encrypted=is_encrypted,
                page_dimensions=page_dimensions,
                title=title,
                author=author,
            )
        except Exception as e:
            if "Incorrect password" in str(e):
                raise
            msg = f"Failed to get PDF info: {e}"
            raise ValueError(msg) from e
