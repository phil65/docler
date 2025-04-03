# """Integration tests for vector managers."""

# from __future__ import annotations

# import os
# from typing import TYPE_CHECKING
# import uuid

# import pytest

# # from docler.vector_db.dbs.chroma_db import ChromaVectorManager
# from docler.vector_db.dbs.pinecone_db import PineconeVectorManager


# if TYPE_CHECKING:
#     from docler.vector_db.base_manager import VectorManagerBase

# managers = [
#     # ChromaVectorManager(persist_directory="chroma"),
#     PineconeVectorManager(),
# ]


# @pytest.mark.integration
# @pytest.mark.parametrize("manager", managers)
# @pytest.mark.skipif(os.environ.get("CI", False), reason="Skip integration in CI")
# async def test_vector_manager_lifecycle(manager: VectorManagerBase):
#     """Test basic vector store lifecycle (create, list, delete)."""
#     store_name = f"test-{uuid.uuid4().hex[:8]}"

#     try:
#         store = await manager.create_vector_store(store_name)
#         assert store is not None
#         assert store.vector_store_id is not None
#         stores = await manager.list_vector_stores()
#         store_ids = {s.db_id for s in stores}
#         assert store.vector_store_id in store_ids
#         success = await manager.delete_vector_store(store_name)
#         assert success is True
#         stores = await manager.list_vector_stores()
#         store_ids = {s.db_id for s in stores}
#         assert store.vector_store_id not in store_ids

#     finally:
#         # Cleanup
#         await manager.close()


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
