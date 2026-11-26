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
