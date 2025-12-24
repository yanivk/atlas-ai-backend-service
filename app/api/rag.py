# app/api/rag.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.rag.pipeline import RagPipeline

router = APIRouter(prefix="/rag", tags=["rag"])
pipeline = RagPipeline()

class IndexRequest(BaseModel):
    doc_id: str
    text: str

class QueryRequest(BaseModel):
    doc_id: str
    question: str
    k: int = 5

@router.post("/index")
async def index_document(body: IndexRequest):
    result = pipeline.index_document(doc_id=body.doc_id, text=body.text)
    return result

@router.post("/query")
async def query_rag(body: QueryRequest):
    result = pipeline.answer_question(
        doc_id=body.doc_id,
        question=body.question,
        k=body.k
    )
    return result
