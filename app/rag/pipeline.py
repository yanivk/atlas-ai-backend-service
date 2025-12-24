# app/rag/pipeline.py
from typing import List
from app.services.vectorstore import VectorStoreService
from app.services.llm import LLMService

class RagPipeline:
    def __init__(self):
        self.vectorstore = VectorStoreService()
        self.llm = LLMService()

    def index_document(self, doc_id: str, text: str):
        # Simple chunking naïf pour commencer
        chunk_size = 800
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        self.vectorstore.add_texts(chunks, metadatas, ids)

        return {
            "doc_id": doc_id,
            "chunks_indexed": len(chunks)
        }

    def answer_question(self, doc_id: str, question: str, k: int = 5):
        # Récupérer les chunks pertinents
        results = self.vectorstore.similarity_search(question, k=k)

        # Filtrer sur le bon doc_id
        filtered = [d for d in results if d.metadata.get("doc_id") == doc_id]
        if not filtered:
            filtered = results  # fallback si rien ne matche

        context = "\n\n".join([d.page_content for d in filtered])

        prompt = f"""
Tu es un assistant IA. Tu réponds à la question en te basant UNIQUEMENT sur le contexte suivant.

Contexte:
{context}

Question:
{question}

Réponse (en français, claire, concise, structurée):
"""

        answer = self.llm.generate(prompt)

        sources = [
            {
                "doc_id": d.metadata.get("doc_id"),
                "chunk_index": d.metadata.get("chunk_index"),
                "content_preview": d.page_content[:200]
            } for d in filtered
        ]

        return {
            "answer": answer,
            "sources": sources
        }
