"""Tests for DocumentConverter.convert_content (PDF handling) without a real backend."""

import io
from typing import TYPE_CHECKING, ClassVar

from pypdf import PdfReader, PdfWriter
import pytest

from docler.converters.base import ConverterResult, DocumentConverter


if TYPE_CHECKING:
    from pathlib import Path

    from schemez import MimeType


class RecordingConverter(DocumentConverter):
    """Records the bytes it receives."""

    NAME = "test_recording"
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}

    def __init__(self) -> None:
        super().__init__()
        self.received: list[bytes] = []

    async def _convert_async(self, data: io.BytesIO, mime_type: MimeType) -> ConverterResult:
        self.received.append(data.getvalue())
        return ConverterResult(content="ok")


@pytest.fixture
def encrypted_pdf(resources_dir: Path) -> bytes:
    writer = PdfWriter(clone_from=resources_dir / "pdf_sample.pdf")
    writer.encrypt("right")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


async def test_wrong_pdf_password_raises(encrypted_pdf: bytes):
    converter = RecordingConverter()
    with pytest.raises(ValueError, match="Incorrect"):
        await converter.convert_content(encrypted_pdf, "application/pdf", pdf_password="wrong")
    assert not converter.received


async def test_missing_pdf_password_raises(encrypted_pdf: bytes):
    converter = RecordingConverter()
    with pytest.raises(ValueError, match="password"):
        await converter.convert_content(encrypted_pdf, "application/pdf")
    assert not converter.received


async def test_right_pdf_password_decrypts(encrypted_pdf: bytes):
    converter = RecordingConverter()
    await converter.convert_content(encrypted_pdf, "application/pdf", pdf_password="right")
    [data] = converter.received
    assert not PdfReader(io.BytesIO(data)).is_encrypted


async def test_unparsable_pdf_goes_to_converter():
    converter = RecordingConverter()
    await converter.convert_content(b"not a pdf", "application/pdf")
    assert converter.received == [b"not a pdf"]
