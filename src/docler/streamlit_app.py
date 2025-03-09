"""Streamlit app for document conversion."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import anyenv
import streamlit as st

from docler.datalab_provider import DataLabConverter
from docler.docling_provider import DoclingConverter
from docler.kreuzberg_provider import KreuzbergConverter
from docler.litellm_provider import LiteLLMConverter
from docler.marker_provider import MarkerConverter
from docler.markitdown_provider import MarkItDownConverter
from docler.mistral_provider import MistralConverter


if TYPE_CHECKING:
    from docler.base import DocumentConverter
    from docler.lang_code import SupportedLanguage


# Setup logging
logging.basicConfig(level=logging.INFO)

# Available converters with their configs
CONVERTERS: dict[str, type[DocumentConverter]] = {
    "DataLab": DataLabConverter,
    "Docling": DoclingConverter,
    "Kreuzberg": KreuzbergConverter,
    "LiteLLM": LiteLLMConverter,
    "Marker": MarkerConverter,
    "MarkItDown": MarkItDownConverter,
    "Mistral": MistralConverter,
    # "OLM": OlmConverter,
}

# Language options
LANGUAGES: list[SupportedLanguage] = ["en", "de", "fr", "es", "zh"]


def convert_file(
    file_path: str | Path,
    converter_cls: type[DocumentConverter],
    language: SupportedLanguage,
) -> str:
    """Convert a single file using the specified converter."""
    converter = converter_cls(languages=[language])
    doc = anyenv.run_sync(converter.convert_file(file_path))
    return doc.content


def main():
    """Main Streamlit app."""
    st.title("Document Converter")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Converter selection
    selected_converters = st.multiselect(
        "Select converters",
        options=list(CONVERTERS.keys()),
        default=["MarkItDown"],
    )

    # Language selection
    language = st.selectbox(
        "Select primary language",
        options=LANGUAGES,
        index=0,
    )

    if uploaded_file and selected_converters and st.button("Convert"):
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        # Create tabs for results
        tabs = st.tabs(selected_converters)

        # Convert with each selected converter
        for tab, converter_name in zip(tabs, selected_converters):
            with tab:
                try:
                    with st.spinner(f"Converting with {converter_name}..."):
                        converter_cls = CONVERTERS[converter_name]
                        content = convert_file(temp_path, converter_cls, language)
                        st.markdown(f"```markdown\n{content}\n```")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Conversion failed: {e!s}")

        # Cleanup
        Path(temp_path).unlink()


if __name__ == "__main__":
    main()
