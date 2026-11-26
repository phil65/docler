"""Converter configs build their converters.

A config field the converter doesn't take only fails, with TypeError, when the
converter is created.
"""

from typing import TYPE_CHECKING

from docler_config.converter_configs import DataLabConfig


if TYPE_CHECKING:
    import pytest


MAX_PAGES = 3


def test_datalab_config_with_max_pages_builds_its_converter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATALAB_API_KEY", "test")
    converter = DataLabConfig(max_pages=MAX_PAGES).get_provider()
    assert converter.max_pages == MAX_PAGES  # ty: ignore[unresolved-attribute]
