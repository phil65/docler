"""Streamlit app for document conversion."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import anyenv
import streamlit as st

from docler.datalab_provider import DataLabConverter
from docler.docling_provider import DoclingConverter
from docler.kreuzberg_provider import KreuzbergConverter
from docler.litellm_provider import LLMConverter
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
    "LiteLLM": LLMConverter,
    "Marker": MarkerConverter,
    "MarkItDown": MarkItDownConverter,
    "Mistral": MistralConverter,
    # "OLM": OlmConverter,
}

# Language options
LANGUAGES: list[SupportedLanguage] = ["en", "de", "fr", "es", "zh"]


def format_image_content(image_content: bytes | str, mime_type: str) -> str:
    """Convert image content to base64 data URL."""
    if isinstance(image_content, bytes):
        # Convert bytes to base64
        b64_content = base64.b64encode(image_content).decode()
    else:
        # Already base64 string - ensure no data URL prefix
        b64_content = (
            image_content.split(",")[-1] if "," in image_content else image_content
        )

    return f"data:{mime_type};base64,{b64_content}"


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

        # Create tabs for converters
        converter_tabs = st.tabs(selected_converters)

        # Convert with each selected converter
        for tab, converter_name in zip(converter_tabs, selected_converters):
            with tab:
                try:
                    with st.spinner(f"Converting with {converter_name}..."):
                        converter_cls = CONVERTERS[converter_name]
                        converter = converter_cls(languages=[language])
                        doc = anyenv.run_sync(converter.convert_file(temp_path))

                        # Create nested tabs
                        raw_tab, rendered_tab, images_tab = st.tabs([
                            "Raw Markdown",
                            "Rendered",
                            "Images",
                        ])

                        # Show raw markdown
                        with raw_tab:
                            st.markdown(f"```markdown\n{doc.content}\n```")

                        # Show rendered markdown
                        with rendered_tab:
                            st.markdown(doc.content)

                        # Show images
                        with images_tab:
                            if not doc.images:
                                st.info("No images extracted")
                            else:
                                for image in doc.images:
                                    # Create image data URL
                                    data_url = format_image_content(
                                        image.content,
                                        image.mime_type,
                                    )

                                    # Show image details and preview
                                    st.markdown(f"**ID:** {image.id}")
                                    if image.filename:
                                        st.markdown(f"**Filename:** {image.filename}")
                                    st.markdown(f"**MIME Type:** {image.mime_type}")
                                    st.image(data_url)
                                    st.divider()

                except Exception as e:  # noqa: BLE001
                    st.error(f"Conversion failed: {e!s}")

        # Cleanup
        Path(temp_path).unlink()


if __name__ == "__main__":
    main()
