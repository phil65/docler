"""Streamlit application package for Docler document processing."""

from docler_streamlit.app import main
from docler_streamlit.chunk_app import main as chunk_app
from docler_streamlit.ocr_app import main as ocr_app

__all__ = ["chunk_app", "main", "ocr_app"]
