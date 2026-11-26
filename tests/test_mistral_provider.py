"""Mistral OCR provider with a fake client."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import mistralai.client
import pytest

from docler.converters.mistral_provider.provider import MistralConverter


class FakeMistral:
    deleted: list[str]
    fail = False

    def __init__(self, **_: Any) -> None:
        self.files = SimpleNamespace(
            upload=lambda **_: SimpleNamespace(id="file-1"),
            get_signed_url=lambda **_: SimpleNamespace(url="https://signed"),
            delete=lambda file_id: FakeMistral.deleted.append(file_id),
        )
        self.ocr = SimpleNamespace(process=self._process)

    def _process(self, pages: list[int] | None = None, **_: Any) -> Any:
        if FakeMistral.fail:
            raise RuntimeError("ocr failed")
        return SimpleNamespace(
            pages=[SimpleNamespace(index=i, markdown=f"page {i}", images=[]) for i in pages or [0]]
        )


@pytest.fixture(autouse=True)
def fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeMistral.deleted = []
    FakeMistral.fail = False
    monkeypatch.setattr(mistralai.client, "Mistral", FakeMistral)


def test_uploaded_file_is_deleted():
    MistralConverter(api_key="k")._convert_sync(io.BytesIO(b"%PDF"), "application/pdf")
    assert FakeMistral.deleted == ["file-1"]


def test_uploaded_file_is_deleted_on_error():
    FakeMistral.fail = True
    with pytest.raises(RuntimeError):
        MistralConverter(api_key="k")._convert_sync(io.BytesIO(b"%PDF"), "application/pdf")
    assert FakeMistral.deleted == ["file-1"]


def test_page_breaks_use_document_page_numbers():
    converter = MistralConverter(api_key="k", page_range="3-4")
    result = converter._convert_sync(io.BytesIO(b"%PDF"), "application/pdf")
    assert '{"next_page":3}' in result.content
    assert '{"next_page":4}' in result.content
    assert '{"next_page":1}' not in result.content
