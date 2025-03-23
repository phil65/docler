"""Streamlit app for document conversion."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import anyenv
import streamlit as st

from docler.converters.azure_provider import AzureConverter
from docler.converters.datalab_provider import DataLabConverter
from docler.converters.docling_provider import DoclingConverter
from docler.converters.kreuzberg_provider import KreuzbergConverter
from docler.converters.llamaparse_provider import LlamaParseConverter
from docler.converters.llm_provider import LLMConverter
from docler.converters.marker_provider import MarkerConverter
from docler.converters.markitdown_provider import MarkItDownConverter
from docler.converters.mistral_provider import MistralConverter
from docler.converters.upstage_provider import UpstageConverter


if TYPE_CHECKING:
    from docler.common_types import SupportedLanguage
    from docler.converters.base import DocumentConverter


logging.basicConfig(level=logging.INFO)

# Available converters with their configs
CONVERTERS: dict[str, type[DocumentConverter]] = {
    "DataLab": DataLabConverter,
    "Docling": DoclingConverter,
    "Kreuzberg": KreuzbergConverter,
    "LLM": LLMConverter,
    "Marker": MarkerConverter,
    "MarkItDown": MarkItDownConverter,
    "Mistral": MistralConverter,
    "LlamaParse": LlamaParseConverter,
    "azure": AzureConverter,
    "upstage": UpstageConverter,
}

# Language options
LANGUAGES: list[SupportedLanguage] = ["en", "de", "fr", "es", "zh"]
ALLOWED_EXTENSIONS = ["pdf", "docx", "jpg", "png", "ppt", "pptx", "xls", "xlsx"]


def format_image_content(data: bytes | str, mime_type: str) -> str:
    """Convert image content to base64 data URL."""
    if isinstance(data, bytes):
        b64_content = base64.b64encode(data).decode()
    else:
        b64_content = data.split(",")[-1] if "," in data else data
    return f"data:{mime_type};base64,{b64_content}"


def main():
    """Main Streamlit app."""
    st.title("Document Converter")
    uploaded_file = st.file_uploader("Choose a file", type=ALLOWED_EXTENSIONS)
    selected_converters = st.multiselect(
        "Select converters",
        options=list(CONVERTERS.keys()),
        default=["DataLab"],
    )
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
                        converter_cls = CONVERTERS[converter_name]
                        converter = converter_cls(languages=[language])
                        doc = anyenv.run_sync(converter.convert_file(temp_path))
                        tabs = ["Raw Markdown", "Rendered", "Images"]
                        raw_tab, rendered_tab, images_tab = st.tabs(tabs)
                        with raw_tab:
                            st.markdown(f"```markdown\n{doc.content}\n```")
                        with rendered_tab:
                            st.markdown(doc.content)
                        with images_tab:
                            if not doc.images:
                                st.info("No images extracted")
                            else:
                                for image in doc.images:
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

        Path(temp_path).unlink()


if __name__ == "__main__":
    from streambricks import run

    run(main)
