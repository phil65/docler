"""Session state management for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def init_session_state():
    """Initialize session state variables if they don't exist."""
    # General app state
    if "step" not in st.session_state:
        st.session_state.step = 1

    # Document-related states
    if "document" not in st.session_state:
        st.session_state.document = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None

    # Chunking-related states
    if "chunks" not in st.session_state:
        st.session_state.chunks = None


def next_step():
    """Move to the next step in the workflow."""
    st.session_state.step += 1


def prev_step():
    """Move to the previous step in the workflow."""
    st.session_state.step -= 1


def reset_app():
    """Reset the app to its initial state."""
    # Keep certain settings like selected converter/chunker
    selected_converter = st.session_state.get("selected_converter")
    selected_chunker = st.session_state.get("selected_chunker")

    # Reset state
    for key in list(st.session_state.keys()):
        if key not in ["selected_converter", "selected_chunker"]:
            del st.session_state[key]

    # Restore settings
    if selected_converter:
        st.session_state.selected_converter = selected_converter
    if selected_chunker:
        st.session_state.selected_chunker = selected_chunker

    # Initialize fresh state
    init_session_state()
