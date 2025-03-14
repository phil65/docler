"""Document conversion library supporting multiple providers."""

from __future__ import annotations

from docler.base import DocumentConverter
from docler.converters.dir_converter import Conversion, DirectoryConverter
from docler.models import Document, Image, ImageReferenceFormat
from docler.registry import ConverterRegistry

# Import providers
from docler.converters.docling_provider import DoclingConverter
from docler.converters.marker_provider import MarkerConverter
from docler.converters.mistral_provider import MistralConverter
from docler.converters.olmocr_provider import OlmConverter
from docler.converters.llm_provider import LLMConverter
from docler.converters.datalab_provider import DataLabConverter
from docler.converters.llamaparse_provider import LlamaParseConverter

__version__ = "0.1.3"

__all__ = [
    "Conversion",
    "ConverterRegistry",
    "DataLabConverter",
    "DirectoryConverter",
    "DoclingConverter",
    "Document",
    "DocumentConverter",
    "Image",
    "ImageReferenceFormat",
    "LLMConverter",
    "LlamaParseConverter",
    "MarkerConverter",
    "MistralConverter",
    "OlmConverter",
]
