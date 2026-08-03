"""FastAPI backend for the Basic RAG demo.

Endpoints:
  GET  /health          liveness check
  GET  /                service info (model names, mock mode)
  GET  /api/docs        list ingested documents
  POST /api/docs        ingest an uploaded .txt/.md file
  POST /api/docs/sample ingest the bundled sample documents
  DELETE /api/docs/{name}  remove a document
  POST /api/query       ask a question against the documents
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import sample_docs
from .config import settings
from .rag import RAGPipeline

app = FastAPI(title="Basic RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


# ----------------------------------------------------------------------
# Info
# ----------------------------------------------------------------------
@app.get("/")
def info():
    return {
        "name": "Basic RAG API",
        "version": app.version,
        "mock_mode": settings.mock_mode,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------------------------------------------------
# Documents (ingest)
# ----------------------------------------------------------------------
@app.get("/api/docs")
def list_docs():
    return {"documents": pipeline.store.list_docs()}


@app.post("/api/docs")
async def ingest_upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in TEXT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or 'none'}'. Use .txt, .md or .markdown.",
        )
    text = (await file.read()).decode("utf-8", errors="replace")
    return pipeline.ingest(Path(file.filename).stem, text)


@app.post("/api/docs/sample")
def ingest_samples():
    results = [pipeline.ingest(name, text) for name, text in sample_docs.DOCS.items()]
    return {
        "ingested": results,
        "total_chunks": sum(r["chunks"] for r in results),
    }


@app.delete("/api/docs/{doc_name}")
def delete_doc(doc_name: str):
    pipeline.store.delete_doc(doc_name)
    return {"deleted": doc_name}


# ----------------------------------------------------------------------
# Query (RAG)
# ----------------------------------------------------------------------
@app.post("/api/query")
def query(req: QueryRequest):
    result = pipeline.query(req.question, top_k=req.top_k)
    result["status"] = "ok" if result["sources"] else "empty"
    return result
