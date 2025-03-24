"""Step 3: Vector Store uploading interface."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import anyenv
import streamlit as st

from docler.models import ChunkedDocument
from docler.streamlit_app.state import SessionState
from docler.vector_db.dbs import chroma_db, openai_db, pinecone_db


if TYPE_CHECKING:
    from docler.vector_db.base import VectorDB


logger = logging.getLogger(__name__)

VECTOR_STORES = {
    "OpenAI": openai_db.OpenAIVectorManager,
    "Pinecone": pinecone_db.PineconeVectorManager,
    # "Qdrant": qdrant_db.QdrantVectorManager,
    "Chroma": chroma_db.ChromaVectorManager,
}


def show_provider_config(provider: str) -> dict:  # noqa: PLR0911
    """Show configuration options for the selected provider.

    Args:
        provider: Name of the vector store provider

    Returns:
        Dictionary of configuration options
    """
    if provider == "OpenAI":
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            options=["auto", "static"],
            help="How OpenAI should chunk documents",
        )
        return {"chunking_strategy": chunking_strategy}

    if provider == "Pinecone":
        cols = st.columns(2)
        with cols[0]:
            cloud = st.selectbox("Cloud Provider", options=["aws", "gcp", "azure"])
        with cols[1]:
            region = st.text_input("Region", value="us-west-2")

        return {"cloud": cloud, "region": region}

    if provider == "Qdrant":
        cols = st.columns(2)
        with cols[0]:
            location_type = st.radio(
                "Location Type", ["Memory", "Local Path", "Server URL"]
            )

        if location_type == "Memory":
            return {"location": None}
        if location_type == "Local Path":
            path = st.text_input("Local Storage Path")
            return {"location": path}
        # Server URL
        url = st.text_input("Qdrant Server URL", value="http://localhost:6333")
        return {"url": url}

    if provider == "Chroma":
        persist_dir = st.text_input(
            "Persistence Directory (optional)", help="Leave empty for in-memory storage"
        )
        return {"persist_directory": persist_dir or None}

    return {}


def show_step_3():
    """Show vector store upload screen (step 3)."""
    state = SessionState.get()
    st.header("Step 3: Upload to Vector Store")
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("← Back", on_click=state.prev_step)

    if not state.chunked_doc:
        st.warning("No chunks to upload. Please go back and chunk a document first.")
        return

    chunked_doc = cast(ChunkedDocument, state.chunked_doc)
    chunks = chunked_doc.chunks

    # Vector DB Provider Selection
    st.subheader("Vector Store Configuration")
    provider = st.selectbox(
        "Select Vector Store Provider",
        options=list(VECTOR_STORES.keys()),
        key="selected_vector_provider",
    )

    config_options = show_provider_config(provider)
    st.divider()
    opts = ["Create new store", "Use existing store"]
    action = st.radio("Vector Store Action", opts, index=0)
    vector_db: VectorDB | None = None
    if action == "Create new store":
        store_name = st.text_input(
            "New Vector Store Name",
            value="docler-store",
            help=f"Name for the new {provider} vector store",
        )

        if st.button("Create Vector Store"):
            with st.spinner(f"Creating {provider} vector store..."):
                try:
                    manager = VECTOR_STORES[provider]()
                    vector_db = anyenv.run_sync(
                        manager.create_vector_store(store_name, **config_options)
                    )
                    assert vector_db is not None, "Vector store creation failed"
                    state.vector_store_id = vector_db.vector_store_id
                    state.vector_provider = provider
                    st.success(
                        f"{provider} vector store '{store_name}' created successfully!"
                    )
                except Exception as e:
                    st.error(f"Failed to create vector store: {e}")
                    logger.exception("Vector store creation failed")
    else:
        try:
            manager = VECTOR_STORES[provider]()
            stores = anyenv.run_sync(manager.list_vector_stores())

            if not stores:
                st.info(f"No existing {provider} vector stores found.")
                store_id = st.text_input(
                    f"{provider} Vector Store ID",
                    help=f"ID of an existing {provider} vector store",
                )
            else:
                store_options = {f"{s['name']} ({s['id']})": s["id"] for s in stores}
                store_display = st.selectbox(
                    "Select Vector Store",
                    options=list(store_options.keys()),
                    help=f"Available {provider} vector stores",
                )
                store_id = store_options.get(store_display, "")

            if store_id and st.button("Connect to Vector Store"):
                with st.spinner(f"Connecting to {provider} vector store..."):
                    try:
                        manager = VECTOR_STORES[provider]()
                        vector_db = anyenv.run_sync(
                            manager.get_vector_store(store_id, **config_options)
                        )
                        assert vector_db is not None, "Vector store creation failed"
                        state.vector_store_id = vector_db.vector_store_id
                        state.vector_provider = provider
                        msg = f"Connected to {provider} vector store {store_id!r}!"
                        st.success(msg)
                    except Exception as e:
                        st.error(f"Failed to connect to vector store: {e}")
                        logger.exception("Vector store connection failed")
        except Exception as e:  # noqa: BLE001
            st.error(f"Error listing vector stores: {e}")
            store_id = st.text_input(
                f"{provider} Vector Store ID",
                help=f"ID of an existing {provider} vector store",
            )
            if store_id and st.button("Connect to Vector Store"):
                with st.spinner(f"Connecting to {provider} vector store..."):
                    try:
                        manager = VECTOR_STORES[provider]()
                        vector_db = anyenv.run_sync(
                            manager.get_vector_store(store_id, **config_options)
                        )
                        assert vector_db is not None
                        state.vector_store_id = vector_db.vector_store_id
                        state.vector_provider = provider
                        msg = f"Connected to {provider} vector store {store_id!r}!"
                        st.success(msg)
                    except Exception as e:
                        st.error(f"Failed to connect to vector store: {e}")
                        logger.exception("Vector store connection failed")

    # Upload chunks if we have a connected vector store
    if store_id := state.vector_store_id:
        st.divider()
        st.subheader("Upload Chunks")
        st.write(f"Found {len(chunks)} chunks to upload")

        if st.button("Upload Chunks to Vector Store"):
            with st.spinner("Uploading chunks..."):
                try:
                    provider = state.vector_provider or provider
                    manager = VECTOR_STORES[provider]()
                    vector_db = anyenv.run_sync(
                        manager.get_vector_store(store_id, **config_options)
                    )
                    assert vector_db is not None, "Vector store not found"
                    chunk_ids = anyenv.run_sync(vector_db.add_chunks(chunks))
                    state.uploaded_chunks = len(chunk_ids)
                    msg = f"Uploaded {len(chunk_ids)} chunks to the vector store!"
                    st.success(msg)
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    logger.exception("Chunk upload failed")

        # Test vector search if chunks have been uploaded
        if state.uploaded_chunks:
            num = state.uploaded_chunks
            provider = state.vector_provider or provider
            st.success(f"{num} chunks uploaded to {provider} vector store '{store_id}'")

            st.divider()
            st.subheader("Test Your Vector Store")
            query = st.text_input("Enter a query to test your vector store:")
            if query:
                with st.spinner("Searching..."):
                    try:
                        manager = VECTOR_STORES[provider]()
                        vector_db = anyenv.run_sync(
                            manager.get_vector_store(store_id, **config_options)
                        )
                        assert vector_db is not None, "Vector store not found"
                        results = anyenv.run_sync(vector_db.similar_chunks(query, k=3))

                        if results:
                            st.write(f"Found {len(results)} relevant chunks:")
                            for i, (chunk, score) in enumerate(results):
                                with st.expander(f"Result {i + 1} - Score: {score:.4f}"):
                                    st.markdown(chunk.text)
                        else:
                            st.info("No results found.")
                    except Exception as e:
                        st.error(f"Search failed: {e}")
                        logger.exception("Vector search failed")


if __name__ == "__main__":
    from streambricks import run

    from docler.models import ChunkedDocument, TextChunk

    state = SessionState.get()
    chunk = TextChunk(text="Sample chunk content", source_doc_id="test", chunk_index=0)
    state.chunked_doc = ChunkedDocument(content="Sample content", chunks=[chunk])
    state.uploaded_file_name = "sample.txt"
    run(show_step_3)
