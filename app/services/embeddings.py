# app/services/embeddings.py
from langchain_openai import OpenAIEmbeddings  # ou autre provider
import os

class EmbeddingsService:
    def __init__(self):
        self.client = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)
