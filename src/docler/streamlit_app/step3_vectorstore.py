"""Step 3: Vector Store uploading interface."""

from __future__ import annotations

import logging
import os
from typing import cast

import anyenv
import streamlit as st

from docler.chunkers.base import TextChunk
from docler.streamlit_app.state import prev_step
from docler.vector_db.dbs.openai_db.manager import OpenAIVectorManager


logger = logging.getLogger(__name__)


def show_step_3():
    """Show vector store upload screen (step 3)."""
    st.header("Step 3: Upload to Vector Store")
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("← Back", on_click=prev_step)

    if not st.session_state.chunks:
        st.warning("No chunks to upload. Please go back and chunk a document first.")
        return

    chunks = cast(list[TextChunk], st.session_state.chunks)
    st.subheader("Vector Store Configuration")
    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY environment variable not set. "
            "Please set this environment variable before proceeding."
        )
        return
    option = st.radio(
        "Vector Store Action", ["Create new store", "Use existing store"], index=0
    )

    vector_store_id = None
    if option == "Create new store":
        store_name = st.text_input(
            "New Vector Store Name",
            value="docler-store",
            help="Name for the new vector store",
        )

        if st.button("Create Vector Store"):
            with st.spinner("Creating vector store..."):
                try:
                    manager = OpenAIVectorManager()
                    vector_db = anyenv.run_sync(manager.create_vector_store(store_name))
                    st.session_state.vector_store_id = vector_db.vector_store_id
                    st.success(f"Vector store {store_name!r} created successfully!")
                    vector_store_id = vector_db.vector_store_id
                except Exception as e:
                    st.error(f"Failed to create vector store: {e}")
                    logger.exception("Vector store creation failed")
    else:
        existing_id = st.text_input(
            "Existing Vector Store ID", help="ID of an existing OpenAI vector store"
        )

        if st.button("Connect to Vector Store"):
            with st.spinner("Connecting to vector store..."):
                try:
                    if not existing_id:
                        st.error("Vector Store ID is required")
                    else:
                        manager = OpenAIVectorManager()
                        vector_db = anyenv.run_sync(manager.get_vector_store(existing_id))
                        st.session_state.vector_store_id = vector_db.vector_store_id
                        st.success(
                            f"Connected to vector store {existing_id!r} successfully!"
                        )
                        vector_store_id = vector_db.vector_store_id
                except Exception as e:
                    st.error(f"Failed to connect to vector store: {e}")
                    logger.exception("Vector store connection failed")

    if vector_store_id or st.session_state.get("vector_store_id"):
        vector_store_id = vector_store_id or st.session_state.vector_store_id
        st.subheader("Upload Chunks")
        st.write(f"Found {len(chunks)} chunks to upload")
        if st.button("Upload Chunks to Vector Store"):
            with st.spinner("Uploading chunks..."):
                try:
                    manager = OpenAIVectorManager()
                    vector_db = anyenv.run_sync(manager.get_vector_store(vector_store_id))
                    chunk_ids = anyenv.run_sync(vector_db.add_chunks(chunks))
                    st.session_state.uploaded_chunks = len(chunk_ids)
                    st.success(
                        f"Successfully uploaded {len(chunk_ids)} chunks to vector store!"
                    )

                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    logger.exception("Chunk upload failed")

        if st.session_state.get("uploaded_chunks"):
            num = st.session_state.uploaded_chunks
            st.success(f"{num} chunks uploaded to vector store {vector_store_id}")
            st.subheader("Test Your Vector Store")
            query = st.text_input("Enter a query to test your vector store:")
            if query:
                with st.spinner("Searching..."):
                    manager = OpenAIVectorManager()
                    vector_db = anyenv.run_sync(manager.get_vector_store(vector_store_id))
                    results = anyenv.run_sync(vector_db.similar_chunks(query, k=3))
                    if results:
                        st.write(f"Found {len(results)} relevant chunks:")
                        for i, (chunk, score) in enumerate(results):
                            with st.expander(f"Result {i + 1} - Score: {score:.4f}"):
                                st.markdown(chunk.text)
                    else:
                        st.info("No results found.")
