"""Main Streamlit application for document processing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import streamlit as st

from docler.streamlit_app.state import SessionState
from docler.streamlit_app.step1_conversion import show_step_1
from docler.streamlit_app.step2_chunking import show_step_2
from docler.streamlit_app.step3_vectorstore import show_step_3


if TYPE_CHECKING:
    from docler.models import Document


logging.basicConfig(level=logging.INFO)


def main():
    """Main Streamlit app."""
    st.title("Document Processing Pipeline")
    state = SessionState.get()
    with st.sidebar:
        st.title("Navigation")
        st.button("Reset App", on_click=state.reset)
        st.write(f"Current step: {state.step}")
        if state.uploaded_file_name:
            st.write(f"File: {state.uploaded_file_name}")
        if state.document:
            doc: Document = state.document
            st.write("Document Info:")
            st.write(f"- Title: {doc.title or 'Untitled'}")
            st.write(f"- Images: {len(doc.images)}")
            st.write(f"- Length: {len(doc.content)} chars")
        if state.chunks:
            st.write(f"- Chunks: {len(state.chunks)}")
            if state.vector_store_id:
                st.write(f"- Vector Store: {state.vector_store_id}")

    if state.step == 1:
        show_step_1()
    elif state.step == 2:  # noqa: PLR2004
        show_step_2()
    elif state.step == 3:  # noqa: PLR2004
        show_step_3()


if __name__ == "__main__":
    from streambricks import run

    run(main)
