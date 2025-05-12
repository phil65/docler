"""Utility functions for the Streamlit app."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st


if TYPE_CHECKING:
    from mkdown import Document, TextChunk

    from docler.common_types import SupportedLanguage


LANGUAGES: list[SupportedLanguage] = ["en", "de", "fr", "es", "zh"]


def display_chunk_preview(chunk: TextChunk, expanded: bool = False):
    """Display a tabbed preview of a text chunk.

    Args:
        chunk: TextChunk to display
        expanded: Whether the expander should be initially expanded
    """
    header_text = f"Chunk {chunk.chunk_index + 1}"
    if chunk.metadata.get("header"):
        header_text += f" - {chunk.metadata['header']}"
    header_text += f" ({len(chunk.content)} chars)"
    with st.expander(header_text, expanded=expanded):
        tabs = ["Raw", "Rendered", "Debug Info", "Images"]
        raw_tab, rendered_tab, debug_tab, images_tab = st.tabs(tabs)
        with raw_tab:
            st.code(chunk.content, language="markdown")
        with rendered_tab:
            st.markdown(chunk.content)
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
                    data_url = image.to_base64_url()
                    st.markdown(f"**ID:** {image.id}")
                    if image.filename:
                        st.markdown(f"**Filename:** {image.filename}")
                    st.markdown(f"**MIME Type:** {image.mime_type}")
                    st.image(data_url)
                    st.divider()


def display_document_preview(doc: Document):
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
                data_url = image.to_base64_url()
                st.markdown(f"**ID:** {image.id}")
                if image.filename:
                    st.markdown(f"**Filename:** {image.filename}")
                st.markdown(f"**MIME Type:** {image.mime_type}")
                st.image(data_url)
                st.divider()
