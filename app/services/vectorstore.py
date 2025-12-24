# app/services/vectorstore.py
from langchain_chroma import Chroma
from app.services.embeddings import EmbeddingsService
from typing import List

class VectorStoreService:
    def __init__(self, collection_name: str = "documents"):
        self.embeddings = EmbeddingsService()
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings.client,
            persist_directory="data/chroma"
        )

    def add_texts(self, texts: List[str], metadatas: List[dict], ids: List[str] | None = None):
        self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query: str, k: int = 5):
        return self.store.similarity_search(query, k=k)
