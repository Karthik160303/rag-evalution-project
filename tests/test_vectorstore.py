
from vectorstore.qdrant_store import QdrantVectorStore


def test_create_collection():

    store = QdrantVectorStore()

    try:
        store.create_collection()

        assert store.collection_exists()

    finally:
        store.close()
