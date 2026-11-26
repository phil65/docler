"""Tests for the HTTP API (no network: converters are faked)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx2
import pytest

from docler.converters.base import DocumentConverter
from docler.markdown import Document, Image, create_image_reference
from docler_api.main import create_app
from docler_api.settings import ApiSettings, check_bind


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


ATTACKER = "https://attacker.example/collect"
PDF = Path(__file__).parents[1] / "src/docler/resources/pdf_sample.pdf"
PNG = b"\x89PNG\r\n\x1a\nimage-bytes"
TOKEN = "s3cret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

URL_CONFIGS = [
    pytest.param({"type": "upstage", "base_url": ATTACKER}, id="upstage"),
    pytest.param({"type": "azure", "endpoint": ATTACKER}, id="azure"),
    pytest.param({"type": "docling_remote", "endpoint": ATTACKER}, id="docling_remote"),
    pytest.param(
        {"type": "aggregated", "converters": [{"type": "upstage", "base_url": ATTACKER}]},
        id="aggregated",
    ),
    # pydantic-settings matches field names case-insensitively
    pytest.param({"type": "upstage", "BASE_URL": ATTACKER}, id="upstage-upper"),
    pytest.param({"type": "azure", "Endpoint": ATTACKER}, id="azure-mixed-case"),
]


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[DocumentConverter]:
    """Record converters that would convert, answering with a document with one image."""
    monkeypatch.setenv("UPSTAGE_API_KEY", "server-secret")
    monkeypatch.setenv("AZURE_DOC_INTELLIGENCE_KEY", "server-secret")
    monkeypatch.setenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "https://server.example")
    monkeypatch.setenv("DOCLING_REMOTE_API_KEY", "server-secret")
    converters: list[DocumentConverter] = []

    async def fake_convert_content(self: DocumentConverter, *_: Any, **__: Any) -> Document:
        converters.append(self)
        image = Image(id="img-0", content=PNG, mime_type="image/png", filename="img-0.png")
        ref = create_image_reference("fig", "img-0.png")
        content = f"# A\n\nalpha {ref}\n\n# B\n\nbeta {ref}\n"
        return Document(content=content, images=[image], title="doc.pdf")

    monkeypatch.setattr(DocumentConverter, "convert_content", fake_convert_content)
    return converters


def make_client(host: str = "127.0.0.1", **settings: Any) -> httpx2.AsyncClient:
    app = create_app(ApiSettings(**settings))
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=f"http://{host}")


@pytest.fixture
async def client() -> AsyncIterator[httpx2.AsyncClient]:
    async with make_client(token=TOKEN) as c:
        yield c


def _files(name: str = "doc.pdf", data: bytes = b"%PDF-1.4") -> dict[str, Any]:
    return {"file": (name, data, "application/pdf")}


UPSTAGE = '{"type": "upstage"}'
MARKDOWN_CHUNKER = json.dumps({"type": "markdown", "min_chunk_size": 1, "chunk_overlap": 0})


async def test_convert_returns_document_and_images(client: httpx2.AsyncClient, sent: list[Any]):
    data = {"converter": UPSTAGE}
    response = await client.post("/v1/convert", files=_files(), data=data, headers=AUTH)
    assert response.status_code == 200, response.text  # noqa: PLR2004
    body = response.json()
    assert body["document"]["content"].startswith("# A")
    assert "images" not in body["document"]
    assert body["chunks"] is None
    assert [i["id"] for i in body["images"]] == ["img-0"]
    assert base64.b64decode(body["images"][0]["content"]) == PNG
    assert len(sent) == 1


async def test_convert_with_chunker_lists_each_image_once(
    client: httpx2.AsyncClient, sent: list[Any]
):
    data = {"converter": UPSTAGE, "chunker": MARKDOWN_CHUNKER}
    response = await client.post("/v1/convert", files=_files(), data=data, headers=AUTH)
    assert response.status_code == 200, response.text  # noqa: PLR2004
    body = response.json()
    assert len(body["chunks"]) == 2  # noqa: PLR2004
    assert all(chunk["image_ids"] == ["img-0"] for chunk in body["chunks"])
    assert [i["id"] for i in body["images"]] == ["img-0"]
    assert response.text.count(base64.b64encode(PNG).decode()) == 1


async def test_convert_images_none_keeps_ids_without_data(
    client: httpx2.AsyncClient, sent: list[Any]
):
    data = {"converter": UPSTAGE, "chunker": MARKDOWN_CHUNKER, "images": "none"}
    response = await client.post("/v1/convert", files=_files(), data=data, headers=AUTH)
    assert response.status_code == 200, response.text  # noqa: PLR2004
    body = response.json()
    assert body["images"] == [
        {
            "id": "img-0",
            "mime_type": "image/png",
            "filename": "img-0.png",
            "description": None,
            "metadata": {},
            "content": None,
        }
    ]
    assert body["chunks"][0]["image_ids"] == ["img-0"]
    assert "img-0.png" in body["document"]["content"]


async def test_convert_rejects_unknown_images_mode(client: httpx2.AsyncClient, sent: list[Any]):
    data = {"converter": UPSTAGE, "images": "files"}
    response = await client.post("/v1/convert", files=_files(), data=data, headers=AUTH)
    assert response.status_code == 422  # noqa: PLR2004
    assert not sent


@pytest.mark.parametrize("config", URL_CONFIGS)
async def test_convert_rejects_client_urls(
    config: dict[str, Any], client: httpx2.AsyncClient, sent: list[Any]
):
    data = {"converter": json.dumps(config)}
    response = await client.post("/v1/convert", files=_files(), data=data, headers=AUTH)
    assert response.status_code == 400  # noqa: PLR2004
    assert not sent


async def test_convert_rejects_client_urls_in_chunker(client: httpx2.AsyncClient, sent: list[Any]):
    chunker = json.dumps({"type": "markdown", "base_url": ATTACKER})
    data = {"converter": UPSTAGE, "chunker": chunker}
    response = await client.post("/v1/convert", files=_files(), data=data, headers=AUTH)
    assert response.status_code == 400  # noqa: PLR2004
    assert not sent


async def test_convert_uses_server_url_by_default(
    client: httpx2.AsyncClient, sent: list[Any], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("UPSTAGE_BASE_URL", "https://upstage.internal/parse")
    await client.post("/v1/convert", files=_files(), data={"converter": UPSTAGE}, headers=AUTH)
    assert [str(c.base_url) for c in sent] == ["https://upstage.internal/parse"]


async def test_pdf_metadata(client: httpx2.AsyncClient):
    files = _files("sample.pdf", PDF.read_bytes())
    response = await client.post("/v1/pdf/metadata", files=files, headers=AUTH)
    assert response.status_code == 200, response.text  # noqa: PLR2004
    assert response.json()["page_count"] == 4  # noqa: PLR2004


async def test_pdf_metadata_rejects_non_pdf(client: httpx2.AsyncClient):
    files = {"file": ("a.txt", b"hi", "text/plain")}
    response = await client.post("/v1/pdf/metadata", files=files, headers=AUTH)
    assert response.status_code == 400  # noqa: PLR2004


async def test_list_converters_and_chunkers(client: httpx2.AsyncClient, sent: list[Any]):
    converters = (await client.get("/v1/converters", headers=AUTH)).json()["converters"]
    assert any(c["name"] == "upstage" and c["config_schema"] for c in converters)
    chunkers = (await client.get("/v1/chunkers", headers=AUTH)).json()["chunkers"]
    assert any(c["name"] == "markdown" for c in chunkers)


async def test_old_endpoints_are_gone(client: httpx2.AsyncClient):
    for path in ("/api/convert", "/api/chunk", "/api/pdf/metadata"):
        response = await client.post(path, files=_files(), headers=AUTH)
        assert response.status_code == 404  # noqa: PLR2004


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {TOKEN}"}],
    ids=["none", "wrong", "not-bearer"],
)
async def test_token_required(headers: dict[str, str], client: httpx2.AsyncClient, sent: list[Any]):
    data = {"converter": UPSTAGE}
    response = await client.post("/v1/convert", files=_files(), data=data, headers=headers)
    assert response.status_code == 401  # noqa: PLR2004
    assert response.headers["www-authenticate"] == "Bearer"
    assert (await client.get("/v1/converters", headers=headers)).status_code == 401  # noqa: PLR2004
    assert not sent


async def test_health_and_ready_need_no_token(client: httpx2.AsyncClient):
    assert (await client.get("/health")).json() == {"status": "ok"}
    assert (await client.get("/ready")).json() == {"status": "ready"}


async def test_localhost_without_token_is_open():
    async with make_client() as c:
        assert (await c.get("/v1/chunkers")).status_code == 200  # noqa: PLR2004


async def test_non_loopback_without_token_is_refused():
    async with make_client(host="10.0.0.5") as c:
        assert (await c.get("/v1/chunkers")).status_code == 403  # noqa: PLR2004
        assert (await c.get("/health")).status_code == 200  # noqa: PLR2004


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_check_bind_allows_loopback_without_token(host: str):
    check_bind(host, ApiSettings())


def test_check_bind_refuses_public_host_without_token():
    with pytest.raises(RuntimeError, match="DOCLER_API_TOKEN"):
        check_bind("0.0.0.0", ApiSettings())
    check_bind("0.0.0.0", ApiSettings(token=TOKEN))


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DOCLER_API_TOKEN", TOKEN)
    monkeypatch.setenv("DOCLER_API_CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("DOCLER_API_MAX_UPLOAD_MB", "2")
    settings = ApiSettings.from_env()
    assert settings.token == TOKEN
    assert settings.cors_origins == ("https://a.example", "https://b.example")
    assert settings.max_upload_bytes == 2 * 1024 * 1024
    monkeypatch.delenv("DOCLER_API_CORS_ORIGINS")
    assert ApiSettings.from_env().cors_origins == ()


async def test_cors_is_off_by_default(client: httpx2.AsyncClient):
    headers = {**AUTH, "Origin": "https://evil.example"}
    response = await client.get("/v1/chunkers", headers=headers)
    assert "access-control-allow-origin" not in response.headers


async def test_cors_allows_configured_origins():
    async with make_client(token=TOKEN, cors_origins=("https://app.example",)) as c:
        preflight = {
            "Origin": "https://app.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization",
        }
        response = await c.options("/v1/convert", headers=preflight)
        assert response.status_code == 200  # noqa: PLR2004
        assert response.headers["access-control-allow-origin"] == "https://app.example"
        assert "access-control-allow-credentials" not in response.headers

        response = await c.get("/v1/chunkers", headers={"Origin": "https://app.example"})
        assert response.status_code == 401  # noqa: PLR2004
        assert response.headers["access-control-allow-origin"] == "https://app.example"

        other = {**AUTH, "Origin": "https://evil.example"}
        response = await c.get("/v1/chunkers", headers=other)
        assert "access-control-allow-origin" not in response.headers


async def test_upload_over_limit_is_413(sent: list[Any]):
    async with make_client(token=TOKEN, max_upload_bytes=1024) as c:
        data = {"converter": UPSTAGE}
        files = _files(data=b"%PDF" + b"x" * 4096)
        response = await c.post("/v1/convert", files=files, data=data, headers=AUTH)
        assert response.status_code == 413  # noqa: PLR2004
        response = await c.post("/v1/convert", files=_files(), data=data, headers=AUTH)
        assert response.status_code == 200  # noqa: PLR2004
    assert len(sent) == 1


async def test_streamed_upload_over_limit_is_413(sent: list[Any]):
    """A body without Content-Length is cut off once it passes the limit."""
    boundary = "b0undary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="converter"\r\n\r\n'
        f"{UPSTAGE}\r\n--{boundary}\r\nContent-Disposition: form-data; "
        f'name="file"; filename="doc.pdf"\r\nContent-Type: application/pdf\r\n\r\n'
    ).encode()

    async def body() -> AsyncIterator[bytes]:
        yield head
        for _ in range(100):
            yield b"x" * 1024
        yield f"\r\n--{boundary}--\r\n".encode()

    async with make_client(token=TOKEN, max_upload_bytes=4096) as c:
        headers = {**AUTH, "Content-Type": f"multipart/form-data; boundary={boundary}"}
        response = await c.post("/v1/convert", content=body(), headers=headers)
    assert "content-length" not in response.request.headers
    assert response.status_code == 413  # noqa: PLR2004
    assert not sent


def test_server_only_fields_cover_every_url_field():
    """A new config field holding a URL must be added to SERVER_ONLY_FIELDS."""
    from pydantic import HttpUrl
    from pydantic_settings import BaseSettings

    from docler_api.routes import SERVER_ONLY_FIELDS
    from docler_config.provider import ProviderConfig

    def subclasses(cls: type) -> list[type]:
        return [cls, *(k for sub in cls.__subclasses__() for k in subclasses(sub))]

    url_fields = {
        name
        for kls in subclasses(ProviderConfig)
        if issubclass(kls, BaseSettings)
        for name, field in kls.model_fields.items()
        if HttpUrl in getattr(field.annotation, "__args__", (field.annotation,))
        or any(part in name for part in ("url", "endpoint", "host", "api_base"))
    }
    assert url_fields <= SERVER_ONLY_FIELDS
