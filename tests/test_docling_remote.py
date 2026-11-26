"""DoclingRemoteConverter against a fake docling-serve speaking its v1 API.

The fake mimics docling-serve v1.40.0 (commit 5c101b6): `POST /v1/convert/file` takes
`files` plus one form field per option (lists repeat the field, `page_range` is two
fields), and answers `{"document": {"md_content": ...}, "status", "errors", "timings"}`.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, Header, HTTPException, UploadFile
import httpx2
from pypdf import PdfReader
import pytest

from docler.converters.docling_remote_provider import provider
from docler_config.converter_configs import DoclingRemoteConfig


PDF = Path(__file__).parents[1] / "src/docler/resources/pdf_sample.pdf"
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()


def fake_docling_serve(seen: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/convert/file")
    async def convert_file(
        files: list[UploadFile],
        from_formats: Annotated[list[str], Form()],
        to_formats: Annotated[list[str], Form()],
        image_export_mode: Annotated[str, Form()],
        do_ocr: Annotated[bool, Form()],
        force_ocr: Annotated[bool, Form()],
        ocr_preset: Annotated[str, Form()],
        pdf_backend: Annotated[str, Form()],
        table_mode: Annotated[str, Form()],
        images_scale: Annotated[float, Form()],
        include_images: Annotated[bool, Form()],
        do_table_structure: Annotated[bool, Form()],
        ocr_lang: Annotated[list[str] | None, Form()] = None,
        page_range: Annotated[list[int] | None, Form()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        if ocr_preset not in {"auto", "easyocr", "rapidocr", "tesseract", "tesserocr", "ocrmac"}:
            raise HTTPException(422, f"unknown ocr_preset {ocr_preset}")
        body = await files[0].read()
        seen.append({
            "pages": len(PdfReader(io.BytesIO(body)).pages),
            "from_formats": from_formats,
            "to_formats": to_formats,
            "image_export_mode": image_export_mode,
            "do_ocr": do_ocr,
            "force_ocr": force_ocr,
            "ocr_preset": ocr_preset,
            "ocr_lang": ocr_lang,
            "pdf_backend": pdf_backend,
            "table_mode": table_mode,
            "images_scale": images_scale,
            "page_range": page_range,
            "authorization": authorization,
        })
        md = f"# Title\n\n![Figure](data:image/png;base64,{PNG})\n\ntext"
        return {
            "document": {"filename": files[0].filename, "md_content": md},
            "status": "success",
            "errors": [],
            "processing_time": 0.1,
            "timings": {"convert": 0.1},
        }

    return app


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    transport = httpx2.ASGITransport(app=fake_docling_serve(requests))
    real_client = httpx2.AsyncClient

    def client(**kwargs: Any) -> httpx2.AsyncClient:
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(provider.httpx2, "AsyncClient", client)
    return requests


async def test_posts_v1_form_fields_and_parses_markdown_and_images(seen: list[dict[str, Any]]):
    config = DoclingRemoteConfig(
        api_key="secret", ocr_engine="rapid_ocr", force_ocr=True, table_mode="accurate"
    )
    converter = config.get_provider()
    result = await converter._convert_async(io.BytesIO(PDF.read_bytes()), "application/pdf")

    request = seen[-1]
    assert request["ocr_preset"] == "rapidocr"
    assert request["force_ocr"] is True
    assert request["table_mode"] == "accurate"
    assert request["to_formats"] == ["md"]
    assert "pdf" in request["from_formats"]
    assert request["image_export_mode"] == "embedded"
    assert request["ocr_lang"] == ["en"]
    assert request["page_range"] is None
    assert request["authorization"] == "Bearer secret"
    assert result.content.startswith("# Title")
    assert "data:image" not in result.content
    assert len(result.images) == 1
    assert result.images[0].content == base64.b64decode(PNG)
    assert result.images[0].mime_type == "image/png"
    assert result.metadata == {"convert": 0.1}


async def test_contiguous_page_range_uses_v1_page_range(seen: list[dict[str, Any]]):
    converter = provider.DoclingRemoteConverter(page_range="2-3")
    await converter._convert_async(io.BytesIO(PDF.read_bytes()), "application/pdf")
    assert seen[-1]["page_range"] == [2, 3]
    assert seen[-1]["pages"] == 4  # noqa: PLR2004


async def test_non_contiguous_page_range_cuts_pages_locally(seen: list[dict[str, Any]]):
    converter = provider.DoclingRemoteConverter(page_range="2,4")
    await converter._convert_async(io.BytesIO(PDF.read_bytes()), "application/pdf")
    assert seen[-1]["page_range"] is None
    assert seen[-1]["pages"] == 2  # noqa: PLR2004


async def test_failed_conversion_reports_errors(monkeypatch: pytest.MonkeyPatch):
    def handle(request: httpx2.Request) -> httpx2.Response:
        errors = [{"component_type": "pipeline", "module_name": "x", "error_message": "boom"}]
        body = {"document": {"md_content": None}, "status": "failure", "errors": errors}
        return httpx2.Response(200, json=body)

    real_client = httpx2.AsyncClient

    def client(**kwargs: Any) -> httpx2.AsyncClient:
        return real_client(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(provider.httpx2, "AsyncClient", client)
    converter = provider.DoclingRemoteConverter()
    with pytest.raises(ValueError, match="boom"):
        await converter._convert_async(io.BytesIO(PDF.read_bytes()), "application/pdf")
