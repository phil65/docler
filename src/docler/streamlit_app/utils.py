"""Utility functions for the Streamlit app."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import streamlit as st


if TYPE_CHECKING:
    from docler.common_types import SupportedLanguage
    from docler.models import Document, TextChunk


LANGUAGES: list[SupportedLanguage] = ["en", "de", "fr", "es", "zh"]


def format_image_content(data: bytes | str, mime_type: str) -> str:
    """Convert image content to base64 data URL.

    Args:
        data: Raw bytes or base64 string of image data
        mime_type: MIME type of the image

    Returns:
        Data URL format of the image for embedding in HTML/Markdown
    """
    if isinstance(data, bytes):
        b64_content = base64.b64encode(data).decode()
    else:
        b64_content = data.split(",")[-1] if "," in data else data
    return f"data:{mime_type};base64,{b64_content}"


def display_chunk_preview(chunk: TextChunk, expanded: bool = False) -> None:
    """Display a tabbed preview of a text chunk.

    Args:
        chunk: TextChunk to display
        expanded: Whether the expander should be initially expanded
    """
    header_text = f"Chunk {chunk.chunk_index + 1}"
    if chunk.metadata.get("header"):
        header_text += f" - {chunk.metadata['header']}"
    header_text += f" ({len(chunk.text)} chars)"

    with st.expander(header_text, expanded=expanded):
        tabs = ["Raw", "Rendered", "Debug Info", "Images"]
        raw_tab, rendered_tab, debug_tab, images_tab = st.tabs(tabs)

        with raw_tab:
            st.code(chunk.text, language="markdown")

        with rendered_tab:
            st.markdown(chunk.text)

        with debug_tab:
            debug_info = {
                "Chunk Index": chunk.chunk_index,
                "Source": chunk.source_doc_id,
                "Images": len(chunk.images),
                **chunk.metadata,
            }
            st.json(debug_info)

        with images_tab:
            if not chunk.images:
                st.info("No images in this chunk")
            else:
                for image in chunk.images:
                    data_url = format_image_content(image.content, image.mime_type)
                    st.markdown(f"**ID:** {image.id}")
                    if image.filename:
                        st.markdown(f"**Filename:** {image.filename}")
                    st.markdown(f"**MIME Type:** {image.mime_type}")
                    st.image(data_url)
                    st.divider()


def display_document_preview(doc: Document) -> None:
    """Display a tabbed preview of a document.

    Args:
        doc: Document to display
    """
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
                data_url = format_image_content(image.content, image.mime_type)
                st.markdown(f"**ID:** {image.id}")
                if image.filename:
                    st.markdown(f"**Filename:** {image.filename}")
                st.markdown(f"**MIME Type:** {image.mime_type}")
                st.image(data_url)
                st.divider()
