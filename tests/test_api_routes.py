"""Tests for the API routes that need no network."""

from __future__ import annotations

import io
import json
from typing import Any

from fastapi import HTTPException, UploadFile
import pytest

from docler.converters.base import DocumentConverter
from docler.markdown import Document
from docler_api.routes import chunk_document, convert_document


ATTACKER = "https://attacker.example/collect"

URL_CONFIGS = [
    pytest.param({"type": "upstage", "base_url": ATTACKER}, id="upstage"),
    pytest.param({"type": "azure", "endpoint": ATTACKER}, id="azure"),
    pytest.param({"type": "docling_remote", "endpoint": ATTACKER}, id="docling_remote"),
    pytest.param(
        {"type": "aggregated", "converters": [{"type": "upstage", "base_url": ATTACKER}]},
        id="aggregated",
    ),
]


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[DocumentConverter]:
    """Record converters that would send a request, instead of sending it."""
    monkeypatch.setenv("UPSTAGE_API_KEY", "server-secret")
    monkeypatch.setenv("AZURE_DOC_INTELLIGENCE_KEY", "server-secret")
    monkeypatch.setenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "https://server.example")
    monkeypatch.setenv("DOCLING_REMOTE_API_KEY", "server-secret")
    converters: list[DocumentConverter] = []

    async def fake_convert_content(self: DocumentConverter, *_: Any, **__: Any) -> Document:
        converters.append(self)
        return Document(content="ok", title="t")

    monkeypatch.setattr(DocumentConverter, "convert_content", fake_convert_content)
    return converters


def _upload() -> UploadFile:
    return UploadFile(filename="doc.pdf", file=io.BytesIO(b"%PDF-1.4"))


@pytest.mark.parametrize("config", URL_CONFIGS)
async def test_convert_rejects_client_urls(config: dict[str, Any], sent: list[Any]):
    with pytest.raises(HTTPException) as exc_info:
        await convert_document(file=_upload(), config=json.dumps(config))
    assert exc_info.value.status_code == 400  # noqa: PLR2004
    assert not sent


@pytest.mark.parametrize("config", URL_CONFIGS)
async def test_chunk_rejects_client_urls(config: dict[str, Any], sent: list[Any]):
    with pytest.raises(HTTPException) as exc_info:
        await chunk_document(file=_upload(), converter_config=json.dumps(config))
    assert exc_info.value.status_code == 400  # noqa: PLR2004
    assert not sent


async def test_convert_uses_server_url_by_default(sent: list[Any], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UPSTAGE_BASE_URL", "https://upstage.internal/parse")
    await convert_document(file=_upload(), config='{"type": "upstage"}')
    assert [str(c.base_url) for c in sent] == ["https://upstage.internal/parse"]
