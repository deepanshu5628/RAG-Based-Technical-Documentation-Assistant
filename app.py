import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from rag.graph import workflow
from rag.loader import add_documents_from_file, add_documents_from_url

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


@app.post("/ingest", status_code=201)
def ingest(file: Optional[UploadFile] = File(default=None), url: Optional[str] = Form(default=None)):
    # must provide either a file or a url
    if not file and not url:
        raise HTTPException(status_code=400, detail=error_response("Provide either a file or a url"))

    if file and url:
        raise HTTPException(status_code=400, detail=error_response("Provide either a file or a url, not both"))

    try:
        if file:
            allowed = (".pdf", ".txt", ".md", ".html")
            if not any(file.filename.endswith(ext) for ext in allowed):
                raise HTTPException(status_code=400, detail=error_response(f"Unsupported file type. Allowed: {list(allowed)}"))
            if file.size and file.size > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=error_response("File size exceeds 10MB limit"))

            os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
            save_path = os.path.join(DOCUMENTS_FOLDER, file.filename)
            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # rebuild the vector store with the new file
            add_documents_from_file(save_path)
            return success_response("File ingested successfully", {"filename": file.filename})

        if url:
            add_documents_from_url(url)
            return success_response("URL ingested successfully", {"url": url})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=error_response(f"Failed to ingest: {str(e)}"))


@app.get("/documents")
def list_documents():
    try:
        if not os.path.exists(DOCUMENTS_FOLDER):
            return success_response("No documents found", {"documents": [], "total": 0})

        files = [f for f in os.listdir(DOCUMENTS_FOLDER) if any(f.endswith(ext) for ext in (".pdf", ".txt", ".md", ".html"))]

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
        result = workflow.invoke({
            "user_question": req.question,
            "curr_iter": 0,
            "max_iter": req.max_iter,
        })
        return success_response("Answer generated", {
            "question": req.question,
            "answer": result["answer"],
            "answergiven": result["answergiven"],
            "sources": result.get("sources", []),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=error_response(f"Something went wrong: {str(e)}"))
