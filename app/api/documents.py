from fastapi import APIRouter, UploadFile, File
from pypdf import PdfReader
from io import BytesIO

router = APIRouter()

@router.post("/documents/parse")
async def parse_document(file: UploadFile = File(...)):
    content = await file.read()

    reader = PdfReader(BytesIO(content))
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return {
        "filename": file.filename,
        "text": text
    }
