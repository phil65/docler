"""Converters configuration for the Streamlit app."""

from __future__ import annotations

from docler.converters.datalab_provider import DataLabConverter
from docler.converters.docling_provider import DoclingConverter
from docler.converters.kreuzberg_provider import KreuzbergConverter
from docler.converters.llm_provider import LLMConverter
from docler.converters.marker_provider import MarkerConverter
from docler.converters.markitdown_provider import MarkItDownConverter
from docler.converters.mistral_provider import MistralConverter


# Available converters with their configs
CONVERTERS = {
    "DataLab": DataLabConverter,
    "Docling": DoclingConverter,
    "Kreuzberg": KreuzbergConverter,
    "LLM": LLMConverter,
    "Marker": MarkerConverter,
    "MarkItDown": MarkItDownConverter,
    "Mistral": MistralConverter,
}
