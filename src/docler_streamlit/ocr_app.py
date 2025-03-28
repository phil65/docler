"""Streamlit app for document conversion."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import anyenv
import streamlit as st

from docler.converters.base import DocumentConverter
from docler_streamlit.utils import display_document_preview


if TYPE_CHECKING:
    from docler.common_types import SupportedLanguage


logging.basicConfig(level=logging.INFO)

LANGUAGES: list[SupportedLanguage] = ["en", "de", "fr", "es", "zh"]
ALLOWED_EXTENSIONS = ["pdf", "docx", "jpg", "png", "ppt", "pptx", "xls", "xlsx"]


def main():
    """Main Streamlit app."""
    st.title("Document Converter")
    uploaded_file = st.file_uploader("Choose a file", type=ALLOWED_EXTENSIONS)
    opts = list(DocumentConverter.registry.keys())
    selected_converters = st.multiselect("Select converters", opts, default=["datalab"])
    language = st.selectbox("Select language", options=LANGUAGES, index=0)
    if uploaded_file and selected_converters and st.button("Convert"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        converter_tabs = st.tabs(selected_converters)
        for tab, converter_name in zip(converter_tabs, selected_converters):
            with tab:
                try:
                    with st.spinner(f"Converting with {converter_name}..."):
                        converter_cls = DocumentConverter.registry[converter_name]
                        converter = converter_cls(languages=[language])
                        doc = anyenv.run_sync(converter.convert_file(temp_path))
                        display_document_preview(doc)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Conversion failed: {e!s}")
        Path(temp_path).unlink()


if __name__ == "__main__":
    from streambricks import run

    run(main)
