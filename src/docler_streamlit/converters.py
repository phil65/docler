"""Converters configuration for the Streamlit app."""

from __future__ import annotations

from typing import TYPE_CHECKING

# from docler.converters.llm_provider import LLMConverter
# from docler.converters.markitdown_provider import MarkItDownConverter
from docler.converters.azure_provider import AzureConverter
from docler.converters.datalab_provider import DataLabConverter
from docler.converters.docling_provider import DoclingConverter
from docler.converters.llamaparse_provider import LlamaParseConverter
from docler.converters.marker_provider import MarkerConverter
from docler.converters.mistral_provider import MistralConverter
from docler.converters.upstage_provider import UpstageConverter


if TYPE_CHECKING:
    from docler.converters.base import DocumentConverter

ls: list[type[DocumentConverter]] = [
    DataLabConverter,
    DoclingConverter,
    MarkerConverter,
    MistralConverter,
    LlamaParseConverter,
    AzureConverter,
    # MarkItDownConverter
    UpstageConverter,
]
CONVERTERS = {cls.NAME: cls for cls in ls}
