"""Docler: main package.

Abstractions & Tools for OCR / document processing.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("docler")
__title__ = "Docler"

__author__ = "Philipp Temminghoff"
__author_email__ = "philipptemminghoff@googlemail.com"
__copyright__ = "Copyright (c) 2025 Philipp Temminghoff"
__license__ = "MIT"
__url__ = "https://github.com/phil65/docler"

from docler.converters.base import DocumentConverter
from docler.converters.dir_converter import Conversion, DirectoryConverter
from docler.converters.registry import ConverterRegistry

# Import providers
from docler.converters.aggregated_converter import AggregatedConverter
from docler.converters.azure_provider import AzureConverter
from docler.converters.datalab_provider import DataLabConverter
from docler.converters.docling_provider import DoclingConverter
from docler.converters.llamaparse_provider import LlamaParseConverter
from docler.converters.llm_provider import LLMConverter
from docler.converters.marker_provider import MarkerConverter
from docler.converters.markitdown_provider import MarkItDownConverter
from docler.converters.mistral_provider import MistralConverter
from docler.converters.upstage_provider import UpstageConverter

__all__ = [
    "AggregatedConverter",
    "AzureConverter",
    "Conversion",
    "ConverterRegistry",
    "DataLabConverter",
    "DirectoryConverter",
    "DoclingConverter",
    "DocumentConverter",
    "LLMConverter",
    "LlamaParseConverter",
    "MarkItDownConverter",
    "MarkerConverter",
    "MistralConverter",
    "UpstageConverter",
    "__version__",
]
