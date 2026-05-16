import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CRAG RAG API")

DOCUMENTS_FOLDER = "./Documents"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question to ask")
    max_iter: int = Field(default=3, ge=1, le=10, description="Max retries between 1 and 10")


def success_response(message: str, data: dict = None):
    res = {"success": True, "message": message}
    if data is not None:
        res["data"] = data
    return res


def error_response(message: str):
    return {"success": False, "message": message}


@app.get("/health")
def health():
    return success_response("Server is running")


@app.post("/upload-pdf", status_code=201)
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail=error_response("Only PDF files are allowed"))

    if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail=error_response("File size exceeds 10MB limit"))

    try:
        os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
        save_path = os.path.join(DOCUMENTS_FOLDER, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return success_response("File uploaded successfully", {"filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=error_response(f"Failed to save file: {str(e)}"))


@app.get("/documents")
def list_documents():
    try:
        if not os.path.exists(DOCUMENTS_FOLDER):
            return success_response("No documents found", {"documents": [], "total": 0})

        files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith(".pdf")]

        if not files:
            return success_response("No documents found", {"documents": [], "total": 0})

        documents = []
        for f in files:
            file_path = os.path.join(DOCUMENTS_FOLDER, f)
            size_kb = round(os.path.getsize(file_path) / 1024, 2)
            documents.append({"filename": f, "size_kb": size_kb})

        return success_response("Documents fetched successfully", {"documents": documents, "total": len(documents)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=error_response(f"Failed to list documents: {str(e)}"))


@app.post("/ask")
def ask(req: AskRequest):
    try:
        # workflow will be wired here later
        return success_response("Answer generated", {"question": req.question, "answer": "not implemented yet"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=error_response(f"Something went wrong: {str(e)}"))
