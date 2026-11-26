"""MarkItDown converter tests with a fake markitdown module (it needs Python < 3.14)."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import sys
import types
from typing import Any, ClassVar

import pytest

from docler.converters.markitdown_provider import MarkItDownConverter


@dataclass
class FakeStreamInfo:
    mimetype: str | None = None
    extension: str | None = None


@dataclass
class FakeResult:
    text_content: str
    title: str | None = None


class FakeMarkItDown:
    calls: ClassVar[list[dict[str, Any]]] = []
    result: ClassVar[FakeResult | None] = FakeResult("# Hello")

    def convert(self, source: Any, **kwargs: Any) -> FakeResult | None:
        FakeMarkItDown.calls.append(kwargs)
        return FakeMarkItDown.result


@pytest.fixture
def fake_markitdown(monkeypatch: pytest.MonkeyPatch) -> type[FakeMarkItDown]:
    module = types.ModuleType("markitdown")
    module.MarkItDown = FakeMarkItDown  # type: ignore[attr-defined]
    module.StreamInfo = FakeStreamInfo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "markitdown", module)
    FakeMarkItDown.calls = []
    FakeMarkItDown.result = FakeResult("# Hello")
    return FakeMarkItDown


def test_passes_mime_type_and_extension(fake_markitdown: type[FakeMarkItDown]):
    converter = MarkItDownConverter()
    result = converter._convert_sync(BytesIO(b"# Hello"), "text/markdown")
    assert result.content == "# Hello"
    info = fake_markitdown.calls[0]["stream_info"]
    assert info == FakeStreamInfo(mimetype="text/markdown", extension=".md")


def test_no_result_is_a_clear_error(fake_markitdown: type[FakeMarkItDown]):
    """Old markitdown (< 0.1) returns None for streams (seen as "'NoneType' ... text_content")."""
    fake_markitdown.result = None
    converter = MarkItDownConverter()
    with pytest.raises(ValueError, match=r"markitdown returned no result"):
        converter._convert_sync(BytesIO(b"# Hello"), "text/markdown")
