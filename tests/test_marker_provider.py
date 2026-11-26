"""Marker provider with a fake marker package."""

import io
import sys
import types
from typing import Any

import pytest

from docler.converters.marker_provider.provider import MarkerConverter


@pytest.fixture
def model_loads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    loads: list[int] = []

    def create_model_dict() -> dict[str, Any]:
        loads.append(1)
        return {}

    class PdfConverter:
        def __init__(self, **_: Any) -> None:
            pass

        def __call__(self, _: str) -> Any:
            return types.SimpleNamespace(model_dump=dict)

    modules = {
        "marker": types.ModuleType("marker"),
        "marker.models": types.SimpleNamespace(create_model_dict=create_model_dict),
        "marker.converters": types.ModuleType("marker.converters"),
        "marker.converters.pdf": types.SimpleNamespace(PdfConverter=PdfConverter),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "docler.converters.marker_provider.provider.process_response", lambda _: ("", [])
    )
    return loads


def test_models_load_once_per_converter(model_loads: list[int]):
    converter = MarkerConverter()
    for _ in range(3):
        converter._convert_sync(io.BytesIO(b"%PDF"), "application/pdf")
    assert model_loads == [1]
