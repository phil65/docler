"""Converter configs read their own provider's env vars."""

from typing import TYPE_CHECKING

from docler_config.converter_configs import AzureConfig


if TYPE_CHECKING:
    import pytest


def test_azure_config_reads_azure_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLAMAPARSE_API_KEY", "llama-key")
    monkeypatch.setenv("LLAMAPARSE_ENDPOINT", "https://llama.example")
    monkeypatch.setenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "https://azure.example")
    monkeypatch.setenv("AZURE_DOC_INTELLIGENCE_KEY", "azure-key")
    config = AzureConfig()
    assert config.endpoint == "https://azure.example"
    # AZURE_DOC_INTELLIGENCE_KEY is the converter's own fallback, never LlamaParse's key.
    assert config.api_key is None


def test_azure_config_ignores_llamaparse_env(monkeypatch: pytest.MonkeyPatch):
    for name in ("AZURE_DOC_INTELLIGENCE_ENDPOINT", "AZURE_DOC_INTELLIGENCE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLAMAPARSE_API_KEY", "llama-key")
    monkeypatch.setenv("LLAMAPARSE_ENDPOINT", "https://llama.example")
    config = AzureConfig()
    assert config.endpoint is None
    assert config.api_key is None


def test_azure_config_explicit_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AZURE_DOC_INTELLIGENCE_KEY", "azure-key")
    config = AzureConfig(endpoint="https://e.example", api_key="explicit")  # ty: ignore[invalid-argument-type]
    fields = config.get_config_fields()
    assert fields["api_key"] == "explicit"
    assert fields["endpoint"] == "https://e.example"
