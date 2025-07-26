"""Integration test for Docler API endpoints."""

from __future__ import annotations

import base64
import io
import json
import os
from typing import TYPE_CHECKING

from fastapi import UploadFile
import pytest

from docler_api.routes import convert_document


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_convert_document_with_mistral(resources_dir: Path):
    """Test API convert document with Mistral converter using a PDF file.

    This test requires MISTRAL_API_KEY environment variable to be set.
    """
    # Check if API key is available
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY environment variable not set")

    # Prepare test file
    pdf_path = resources_dir / "pdf_sample.pdf"
    assert pdf_path.exists(), f"Test PDF file not found: {pdf_path}"

    # Read file content
    with pdf_path.open("rb") as f:
        file_content = f.read()

    # Create UploadFile object with BytesIO
    file_obj = io.BytesIO(file_content)
    upload_file = UploadFile(filename="pdf_sample.pdf", file=file_obj)

    # Prepare Mistral config as JSON string (matching form data format)
    config_dict = {
        "type": "mistral",
        "languages": ["en"],
        "api_key": api_key,
    }
    config_json = json.dumps(config_dict)

    # Call the API function with form-style parameters
    result = await convert_document(
        file=upload_file,
        config=config_json,  # ← Now passing JSON string
    )

    # Validate response
    assert result is not None
    assert hasattr(result, "content")
    assert hasattr(result, "images")
    assert hasattr(result, "title")
    assert hasattr(result, "source_path")
    assert hasattr(result, "mime_type")

    # Validate content
    assert result.content is not None
    assert len(result.content) > 0
    assert isinstance(result.content, str)

    # Validate metadata
    # assert result.title == "pdf_sample"
    assert result.mime_type == "application/pdf"
    assert result.source_path
    assert result.source_path.endswith("pdf_sample.pdf")

    # Images should be a list (may be empty)
    assert result.images
    isinstance(result.images, list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_convert_document_with_page_range(resources_dir: Path):
    """Test API convert document with page range."""
    # Check if API key is available
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY environment variable not set")

    # Prepare test file
    pdf_path = resources_dir / "pdf_sample.pdf"
    assert pdf_path.exists(), f"Test PDF file not found: {pdf_path}"

    # Read file content
    with pdf_path.open("rb") as f:
        file_content = f.read()

    # Create UploadFile object
    file_obj = io.BytesIO(file_content)
    upload_file = UploadFile(filename="pdf_sample.pdf", file=file_obj)

    # Prepare config with page range
    config_dict = {
        "type": "mistral",
        "languages": ["en"],
        "api_key": api_key,
        "page_range": "1-2",  # ← Test page range functionality
    }
    config_json = json.dumps(config_dict)

    # Call the API function
    result = await convert_document(file=upload_file, config=config_json)

    # Validate response
    assert result is not None
    assert result.content is not None
    assert len(result.content) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_always_includes_images_as_base64(resources_dir: Path):
    """Test that the API always includes images as properly encoded base64 strings.

    This test validates that images are correctly converted to base64 in API responses.
    """
    # Check if API key is available
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY environment variable not set")

    # Prepare test file
    pdf_path = resources_dir / "pdf_sample.pdf"
    assert pdf_path.exists(), f"Test PDF file not found: {pdf_path}"

    # Read file content
    with pdf_path.open("rb") as f:
        file_content = f.read()

    # Create UploadFile object
    file_obj = io.BytesIO(file_content)
    upload_file = UploadFile(filename="pdf_sample.pdf", file=file_obj)

    # Prepare Mistral config
    config_dict = {
        "type": "mistral",
        "languages": ["en"],
        "api_key": api_key,
    }
    config_json = json.dumps(config_dict)

    # Call the API function (images are always included as base64)
    result = await convert_document(
        file=upload_file,
        config=config_json,
    )

    # Validate basic response structure
    assert result is not None
    assert hasattr(result, "images")
    assert isinstance(result.images, list)

    # If we have images, validate their encoding
    if result.images:
        for i, image in enumerate(result.images):
            # Images should always be base64 strings in API responses
            typ = type(image.content)
            assert isinstance(image.content, str), (
                f"Image {i} content should be string in API response, got {typ}"
            )

            # Check that content is not empty
            assert len(image.content) > 0, f"Image {i} content is empty"

            # Validate it's proper base64 by decoding it
            try:
                decoded = base64.b64decode(image.content)
                # Check if the decoded data looks like valid image data
                valid_magic_bytes = [
                    b"\xff\xd8\xff",  # JPEG
                    b"\x89PNG\r\n\x1a\n",  # PNG
                    b"GIF87a",  # GIF87a
                    b"GIF89a",  # GIF89a
                    b"RIFF",  # WEBP (starts with RIFF)
                    b"II*\x00",  # TIFF (little endian)
                    b"MM\x00*",  # TIFF (big endian)
                ]

                has_valid_magic = any(
                    decoded.startswith(magic) for magic in valid_magic_bytes
                )
                assert has_valid_magic, (
                    f"Image {i} decoded content does not start with valid magic bytes."
                    f" First 10 bytes: {decoded[:10]!r}"
                )

                # Validate MIME type consistency with decoded content
                if decoded.startswith(b"\xff\xd8\xff"):
                    assert image.mime_type.startswith("image/jpeg"), (
                        f"Image {i} has JPEG magic but wrong MIME type: {image.mime_type}"
                    )
                elif decoded.startswith(b"\x89PNG"):
                    assert image.mime_type == "image/png", (
                        f"Image {i} has PNG magic but wrong MIME type: {image.mime_type}"
                    )
                elif decoded.startswith((b"GIF87a", b"GIF89a")):
                    assert image.mime_type == "image/gif", (
                        f"Image {i} has GIF magic but wrong MIME type: {image.mime_type}"
                    )
                elif decoded.startswith(b"RIFF") and b"WEBP" in decoded[:12]:
                    assert image.mime_type == "image/webp", (
                        f"Image {i} has WEBP magic but wrong MIME type: {image.mime_type}"
                    )

            except Exception as e:  # noqa: BLE001
                pytest.fail(f"Image {i} base64 content cannot be decoded: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_converter_direct_image_handling(resources_dir: Path):
    """Test that the converter itself produces proper bytes before API processing.

    This test bypasses the API to check the raw converter output.
    """
    # Check if API key is available
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY environment variable not set")

    # Import and test the converter directly
    from docler.converters.mistral_provider import MistralConverter

    # Prepare test file
    pdf_path = resources_dir / "pdf_sample.pdf"
    assert pdf_path.exists(), f"Test PDF file not found: {pdf_path}"

    # Create converter instance
    converter = MistralConverter(api_key=api_key)

    # Convert document directly
    result = await converter.convert_file(pdf_path)

    # Validate basic response structure
    assert result is not None
    assert hasattr(result, "images")
    assert isinstance(result.images, list)

    # If we have images, validate their encoding at the converter level
    if result.images:
        for i, image in enumerate(result.images):
            # At converter level, content should always be bytes
            typ = type(image.content)
            assert isinstance(image.content, bytes), (
                f"Image {i} content should be bytes at converter level, got {typ}"
            )

            # Check that content is not empty
            assert len(image.content) > 0, f"Image {i} content is empty"

            # Validate that the content starts with valid image magic bytes
            valid_magic_bytes = [
                b"\xff\xd8\xff",  # JPEG
                b"\x89PNG\r\n\x1a\n",  # PNG
                b"GIF87a",  # GIF87a
                b"GIF89a",  # GIF89a
                b"RIFF",  # WEBP (starts with RIFF)
                b"II*\x00",  # TIFF (little endian)
                b"MM\x00*",  # TIFF (big endian)
            ]

            has_valid_magic = any(
                image.content.startswith(magic) for magic in valid_magic_bytes
            )
            content = image.content[:10]
            assert has_valid_magic, (
                f"Image {i} doesnt start with valid magic bytes."
                f" First 10 bytes: {content!r}"
            )

            # Validate MIME type consistency
            if image.content.startswith(b"\xff\xd8\xff"):
                assert image.mime_type.startswith("image/jpeg"), (
                    f"Image {i} has JPEG magic but wrong MIME type: {image.mime_type}"
                )
            elif image.content.startswith(b"\x89PNG"):
                assert image.mime_type == "image/png", (
                    f"Image {i} has PNG magic but wrong MIME type: {image.mime_type}"
                )
            elif image.content.startswith((b"GIF87a", b"GIF89a")):
                assert image.mime_type == "image/gif", (
                    f"Image {i} has GIF magic but wrong MIME type: {image.mime_type}"
                )
            elif image.content.startswith(b"RIFF") and b"WEBP" in image.content[:12]:
                assert image.mime_type == "image/webp", (
                    f"Image {i} has WEBP magic but wrong MIME type: {image.mime_type}"
                )

            # Test base64 conversion method
            try:
                base64_str = image.to_base64()
                assert isinstance(base64_str, str), "to_base64() should return a string"
                # Verify we can decode it back to the original bytes
                decoded_back = base64.b64decode(base64_str)
                assert decoded_back == image.content, (
                    f"Image {i} base64 round-trip failed"
                )
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"Image {i} to_base64() method failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "--integration"])
