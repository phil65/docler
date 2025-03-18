"""Main Streamlit application for document processing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import streamlit as st

from docler.streamlit_app.state import init_session_state, reset_app
from docler.streamlit_app.step1_conversion import show_step_1
from docler.streamlit_app.step2_chunking import show_step_2


if TYPE_CHECKING:
    from docler.models import Document


# Setup logging
logging.basicConfig(level=logging.INFO)


def main():
    """Main Streamlit app."""
    st.title("Document Processing Pipeline")

    # Initialize session state
    init_session_state()

    # Add reset button to sidebar
    with st.sidebar:
        st.title("Navigation")
        st.button("Reset App", on_click=reset_app)

        # Display current step
        st.write(f"Current step: {st.session_state.step}")

        # Display uploaded file if available
        if st.session_state.uploaded_file_name:
            st.write(f"File: {st.session_state.uploaded_file_name}")

        # Display document info
        if st.session_state.document:
            doc: Document = st.session_state.document
            st.write("Document Info:")
            st.write(f"- Title: {doc.title or 'Untitled'}")
            st.write(f"- Images: {len(doc.images)}")
            st.write(f"- Length: {len(doc.content)} chars")

    # Show the appropriate step
    if st.session_state.step == 1:
        show_step_1()
    elif st.session_state.step == 2:  # noqa: PLR2004
        show_step_2()


if __name__ == "__main__":
    main()
