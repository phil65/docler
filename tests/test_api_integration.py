"""Integration test for Docler API endpoints."""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

from fastapi import UploadFile
from pydantic import SecretStr
import pytest

from docler.configs.converter_configs import MistralConfig
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

    # Prepare Mistral config with isolated environment to avoid conflicts
    assert api_key
    config = MistralConfig(languages={"en"}, api_key=SecretStr(api_key))

    # Call the API function
    result = await convert_document(
        file=upload_file, config=config, include_images_as_base64=True
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
    assert isinstance(result.images, list)


if __name__ == "__main__":
    pytest.main([__file__, "--integration"])
