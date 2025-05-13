"""Tests for document output functionality."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import httpx
import pytest
import upath

from docler.converters.azure_provider.provider import AzureConverter
from docler.converters.datalab_provider import DataLabConverter
from docler.converters.docling_provider.provider import DoclingConverter
from docler.converters.llamaparse_provider.provider import LlamaParseConverter
from docler.converters.marker_provider.provider import MarkerConverter
from docler.converters.mistral_provider.provider import MistralConverter
from docler.converters.upstage_provider.provider import UpstageConverter


if TYPE_CHECKING:
    from mkdown import Document
    from syrupy.data import Snapshot


SAMPLE_PATH = "src/docler/resources/pdf_sample.pdf"


def is_responsive(url):
    try:
        response = httpx.get(url)
        if response.status_code == 200:  # noqa: PLR2004
            return True
    except ConnectionError:
        return False


async def _test_provider_export(provider_cls, snapshot: type[Snapshot]):
    provider = provider_cls()
    doc: Document = await provider.convert_file(SAMPLE_PATH)
    with tempfile.TemporaryDirectory() as tmpdir:
        await doc.export_to_directory(tmpdir)
        base = upath.UPath(tmpdir)
        md_files = list(base.glob("*.md"))
        assert md_files, "No markdown file exported"
        md_content = md_files[0].read_text(encoding="utf-8")
        file_list = sorted(
            str(f.relative_to(base)) for f in base.rglob("*") if f.is_file()
        )
        assert md_content == snapshot(name="markdown_content")
        assert file_list == snapshot(name="files_list")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_datalab_export(snapshot):
    """Test DataLab provider export functionality."""
    await _test_provider_export(DataLabConverter, snapshot)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_marker_export(snapshot):
    """Test Marker provider export functionality."""
    await _test_provider_export(MarkerConverter, snapshot)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_docling_export(snapshot):
    """Test Docling provider export functionality."""
    await _test_provider_export(DoclingConverter, snapshot)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upstage_export(snapshot):
    """Test Upstage provider export functionality."""
    await _test_provider_export(UpstageConverter, snapshot)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_azure_export(snapshot):
    """Test Azure provider export functionality."""
    await _test_provider_export(AzureConverter, snapshot)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llamaparse_export(snapshot):
    """Test LlamaParse provider export functionality."""
    await _test_provider_export(LlamaParseConverter, snapshot)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mistral_export(snapshot):
    """Test Mistral provider export functionality."""
    await _test_provider_export(MistralConverter, snapshot)


# @pytest.mark.integration
# @pytest.mark.asyncio
# async def test_docling_remote_export(docker_services: Services, snapshot):
#     """Test DoclingRemote provider export functionality."""
#     docker_services.wait_until_responsive(
#         timeout=30.0, pause=0.1, check=lambda: is_responsive("http://localhost:5001")
#     )
#     await _test_provider_export(DoclingRemoteConverter, snapshot)
