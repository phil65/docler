"""Document conversion library supporting multiple providers."""

from __future__ import annotations

from docler.base import DocumentConverter
from docler.dir_converter import Conversion, DirectoryConverter
from docler.models import Document, Image, ImageReferenceFormat
from docler.registry import ConverterRegistry

# Import providers
from docler.docling_provider import DoclingConverter
from docler.marker_provider import MarkerConverter
from docler.mistral_provider import MistralConverter
from docler.olmocr_provider import OlmConverter

__version__ = "0.1.0"

__all__ = [
    "Conversion",
    # Registry
    "ConverterRegistry",
    # Directory handling
    "DirectoryConverter",
    # Providers
    "DoclingConverter",
    "Document",
    # Core classes
    "DocumentConverter",
    "Image",
    "ImageReferenceFormat",
    "MarkerConverter",
    "MistralConverter",
    "OlmConverter",
]
