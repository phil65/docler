"""Tests for PDF utilities with empty password handling."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from docler.pdf_utils import decrypt_pdf, get_pdf_info, parse_page_range, shift_page_range


class TestPageRangeUtils:
    """Test page range parsing and shifting utilities."""

    def test_parse_page_range_simple(self):
        """Test parsing simple page ranges."""
        assert parse_page_range("1-3") == {1, 2, 3}
        assert parse_page_range("5") == {5}
        assert parse_page_range("1,3,5") == {1, 3, 5}
        assert parse_page_range("1-3,5,7-9") == {1, 2, 3, 5, 7, 8, 9}

    def test_parse_page_range_with_shift(self):
        """Test parsing page ranges with shift (e.g., 1-based to 0-based)."""
        assert parse_page_range("1-3", shift=-1) == {0, 1, 2}
        assert parse_page_range("5", shift=-1) == {4}
        assert parse_page_range("1,3,5", shift=-1) == {0, 2, 4}

    def test_parse_page_range_invalid(self):
        """Test parsing invalid page ranges."""
        with pytest.raises(ValueError, match="Invalid page range format"):
            parse_page_range("invalid")

        with pytest.raises(ValueError, match="Invalid page range format"):
            parse_page_range("1-2-3")

    def test_shift_page_range(self):
        """Test shifting page range strings."""
        assert shift_page_range("1-3", shift=-1) == "0-2"
        assert shift_page_range("5", shift=-1) == "4"
        assert shift_page_range("1,3,5", shift=-1) == "0,2,4"
        assert shift_page_range("1-3,5,7-9", shift=2) == "3-5,7,9-11"

    def test_shift_page_range_invalid_shift(self):
        """Test shifting with invalid shift values."""
        with pytest.raises(ValueError, match="Invalid shift"):
            shift_page_range("1-3", shift=-2)


class TestDecryptPdf:
    """Test PDF decryption functionality."""

    def test_decrypt_unencrypted_pdf(self):
        """Test decrypting an unencrypted PDF returns original data."""
        mock_pdf_data = b"mock_pdf_data"

        with patch("docler.pdf_utils.PdfReader") as mock_reader_class:
            mock_reader = Mock()
            mock_reader.is_encrypted = False
            mock_reader_class.return_value = mock_reader

            result = decrypt_pdf(mock_pdf_data, "password")
            assert result == mock_pdf_data

    def test_decrypt_with_correct_password(self):
        """Test decrypting PDF with correct password."""
        mock_pdf_data = b"mock_pdf_data"

        with (
            patch("docler.pdf_utils.PdfReader") as mock_reader_class,
            patch("docler.pdf_utils.PdfWriter") as mock_writer_class,
        ):
            # Setup mock reader
            mock_reader = Mock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = True
            mock_reader.pages = [Mock(), Mock()]
            mock_reader_class.return_value = mock_reader

            # Setup mock writer
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            # Mock the output
            with patch("io.BytesIO") as mock_bytesio:
                mock_output = Mock()
                mock_output.getvalue.return_value = b"decrypted_pdf_data"
                mock_bytesio.return_value.__enter__.return_value = mock_output

                decrypt_pdf(mock_pdf_data, "correct_password")

                mock_reader.decrypt.assert_called_with("correct_password")
                assert mock_writer.add_page.call_count == 2  # noqa: PLR2004
                mock_writer.write.assert_called_once()

    def test_decrypt_with_incorrect_password(self):
        """Test decrypting PDF with incorrect password."""
        mock_pdf_data = b"mock_pdf_data"

        with patch("docler.pdf_utils.PdfReader") as mock_reader_class:
            mock_reader = Mock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = False
            mock_reader_class.return_value = mock_reader

            with pytest.raises(ValueError, match="Incorrect password"):
                decrypt_pdf(mock_pdf_data, "wrong_password")

    def test_decrypt_with_none_password_empty_works(self):
        """Test decrypting PDF with None password when empty password works."""
        mock_pdf_data = b"mock_pdf_data"

        with (
            patch("docler.pdf_utils.PdfReader") as mock_reader_class,
            patch("docler.pdf_utils.PdfWriter") as mock_writer_class,
        ):
            # Setup mock reader that succeeds with empty password
            mock_reader = Mock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = True
            mock_reader.pages = [Mock()]
            mock_reader_class.return_value = mock_reader

            # Setup mock writer
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            # Mock the output
            with patch("io.BytesIO") as mock_bytesio:
                mock_output = Mock()
                mock_output.getvalue.return_value = b"decrypted_pdf_data"
                mock_bytesio.return_value.__enter__.return_value = mock_output

                decrypt_pdf(mock_pdf_data, None)

                # Should try empty password first
                mock_reader.decrypt.assert_called_with("")
                mock_writer.add_page.assert_called_once()
                mock_writer.write.assert_called_once()

    def test_decrypt_with_none_password_empty_fails(self):
        """Test decrypting PDF with None password when empty password fails."""
        mock_pdf_data = b"mock_pdf_data"

        with patch("docler.pdf_utils.PdfReader") as mock_reader_class:
            mock_reader = Mock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = False
            mock_reader_class.return_value = mock_reader

            with pytest.raises(ValueError, match="requires a password"):
                decrypt_pdf(mock_pdf_data, None)


class TestGetPdfInfo:
    """Test PDF info extraction functionality."""

    def test_get_pdf_info_unencrypted(self):
        """Test getting info from unencrypted PDF."""
        mock_pdf_data = b"mock_pdf_data"

        with patch("docler.pdf_utils.PdfReader") as mock_reader_class:
            # Setup mock reader
            mock_reader = Mock()
            mock_reader.is_encrypted = False
            mock_reader.pages = [Mock(), Mock()]

            # Setup page dimensions
            for i, page in enumerate(mock_reader.pages):
                page.mediabox.width = 612.0 + i
                page.mediabox.height = 792.0 + i

            # Setup metadata
            mock_metadata = Mock()
            mock_metadata.title = "Test PDF"
            mock_metadata.author = "Test Author"
            mock_reader.metadata = mock_metadata

            mock_reader_class.return_value = mock_reader

            result = get_pdf_info(mock_pdf_data)

            assert result.page_count == 2  # noqa: PLR2004
            assert result.file_size == len(mock_pdf_data)
            assert not result.is_encrypted
            assert len(result.page_dimensions) == 2  # noqa: PLR2004
            assert result.page_dimensions[0].width == 612.0  # noqa: PLR2004
            assert result.page_dimensions[0].height == 792.0  # noqa: PLR2004
            assert result.page_dimensions[1].width == 613.0  # noqa: PLR2004
            assert result.page_dimensions[1].height == 793.0  # noqa: PLR2004
            assert result.title == "Test PDF"
            assert result.author == "Test Author"

    def test_get_pdf_info_encrypted_with_password(self):
        """Test getting info from encrypted PDF with correct password."""
        mock_pdf_data = b"mock_pdf_data"

        with patch("docler.pdf_utils.PdfReader") as mock_reader_class:
            mock_reader = Mock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = True
            mock_reader.pages = [Mock()]
            mock_reader.pages[0].mediabox.width = 612.0
            mock_reader.pages[0].mediabox.height = 792.0

            mock_metadata = Mock()
            mock_metadata.title = "Encrypted PDF"
            mock_metadata.author = "Test Author"
            mock_reader.metadata = mock_metadata

            mock_reader_class.return_value = mock_reader

            result = get_pdf_info(mock_pdf_data, "correct_password")

            mock_reader.decrypt.assert_called_with("correct_password")
            assert result.page_count == 1
            assert not result.is_encrypted  # Should be False after successful decryption
            assert result.title == "Encrypted PDF"

    def test_get_pdf_info_encrypted_empty_password_works(self):
        """Test getting info from PDF encrypted with empty password."""
        mock_pdf_data = b"mock_pdf_data"

        with patch("docler.pdf_utils.PdfReader") as mock_reader_class:
            mock_reader = Mock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = True
            mock_reader.pages = [Mock()]
            mock_reader.pages[0].mediabox.width = 612.0
            mock_reader.pages[0].mediabox.height = 792.0

            mock_metadata = Mock()
            mock_metadata.title = "Empty Password PDF"
            mock_metadata.author = None
            mock_reader.metadata = mock_metadata

            mock_reader_class.return_value = mock_reader

            # No password provided, should try empty password
            result = get_pdf_info(mock_pdf_data)

            # Should try empty password
            mock_reader.decrypt.assert_called_with("")
            assert result.page_count == 1
            assert (
                not result.is_encrypted
            )  # Should be False after successful empty password
            assert result.title == "Empty Password PDF"
            assert result.author == ""  # None should become empty string

    def test_get_pdf_info_encrypted_no_password_fails(self):
        """Test getting info from truly encrypted PDF without password."""
        mock_pdf_data = b"mock_pdf_data"

        with patch("docler.pdf_utils.PdfReader") as mock_reader_class:
            mock_reader = Mock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = False  # Empty password fails
            mock_reader_class.return_value = mock_reader

            result = get_pdf_info(mock_pdf_data)

            # Should try empty password and fail
            mock_reader.decrypt.assert_called_with("")
            assert result.page_count == 0
            assert result.is_encrypted
            assert result.page_dimensions == []
            assert result.title == ""
            assert result.author == ""

    def test_get_pdf_info_with_metadata_none(self):
        """Test getting info from PDF with None metadata."""
        mock_pdf_data = b"mock_pdf_data"

        with patch("docler.pdf_utils.PdfReader") as mock_reader_class:
            mock_reader = Mock()
            mock_reader.is_encrypted = False
            mock_reader.pages = [Mock()]
            mock_reader.pages[0].mediabox.width = 612.0
            mock_reader.pages[0].mediabox.height = 792.0
            mock_reader.metadata = None

            mock_reader_class.return_value = mock_reader

            result = get_pdf_info(mock_pdf_data)

            assert result.title == ""
            assert result.author == ""

    def test_get_pdf_info_invalid_pdf(self):
        """Test getting info from invalid PDF data."""
        with pytest.raises(ValueError, match="Failed to get PDF info"):
            get_pdf_info(b"invalid_pdf_data")
