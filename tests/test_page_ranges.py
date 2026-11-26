"""Page ranges (1-based in docler) as passed to backends."""

from __future__ import annotations

import io
import sys
import types
from typing import Any

import pytest

from docler.converters.docling_provider.utils import to_docling_page_range
from docler.converters.llamaparse_provider.provider import LlamaParseConverter


def test_llamaparse_target_pages_are_zero_based(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    class FakeLlamaParse:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def get_json_result(self, _: str) -> list[Any]:
            return []

    result_type = types.SimpleNamespace(MD="md")
    fake = types.SimpleNamespace(LlamaParse=FakeLlamaParse, ResultType=result_type)
    monkeypatch.setitem(sys.modules, "llama_parse", fake)
    converter = LlamaParseConverter(api_key="k", page_range="1-3,7")
    with pytest.raises(Exception, match="no result"):
        converter._convert_sync(io.BytesIO(b"%PDF"), "application/pdf")
    assert seen["target_pages"] == "0-2,6"


@pytest.mark.parametrize(
    ("page_range", "expected"),
    [("1-5", (1, 5)), ("5", (5, 5)), ("2,3,4", (2, 4)), ("1-3,7", None)],
)
def test_docling_page_range_is_one_based(page_range: str, expected: tuple[int, int] | None):
    assert to_docling_page_range(page_range) == expected


async def test_docling_remote_sends_only_the_page_range(monkeypatch: pytest.MonkeyPatch):
    from pathlib import Path

    import httpx2
    from pypdf import PdfReader

    from docler.converters.docling_remote_provider import provider

    uploaded: list[bytes] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = request.content
        start = body.index(b"%PDF")
        end = body.rindex(b"%%EOF") + len(b"%%EOF")
        uploaded.append(body[start:end])
        return httpx2.Response(200, json={"document": {"md_content": "text"}})

    real_client = httpx2.AsyncClient

    def client(**kwargs: Any) -> httpx2.AsyncClient:
        return real_client(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(provider.httpx2, "AsyncClient", client)
    pdf = Path(__file__).parents[1] / "src/docler/resources/pdf_sample.pdf"
    data = pdf.read_bytes()
    assert len(PdfReader(io.BytesIO(data)).pages) == 4  # noqa: PLR2004

    converter = provider.DoclingRemoteConverter(page_range="2,4")
    await converter._convert_async(io.BytesIO(data), "application/pdf")
    assert len(PdfReader(io.BytesIO(uploaded[-1])).pages) == 2  # noqa: PLR2004

    converter = provider.DoclingRemoteConverter()
    await converter._convert_async(io.BytesIO(data), "application/pdf")
    assert len(PdfReader(io.BytesIO(uploaded[-1])).pages) == 4  # noqa: PLR2004
