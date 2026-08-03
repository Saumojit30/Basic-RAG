import pytest
from fastapi.testclient import TestClient

from app.chunker import chunk_text
from app.llm import LLMClient
from app.main import app
from app.rag import RAGPipeline
from app.vectorstore import VectorStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_db():
    yield
    for doc in client.get("/api/docs").json()["documents"]:
        client.delete(f"/api/docs/{doc['name']}")


# ----------------------------------------------------------------------
# Health & info
# ----------------------------------------------------------------------
def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_mock_mode_active():
    assert client.get("/").json()["mock_mode"] is True


# ----------------------------------------------------------------------
# Full pipeline round-trip
# ----------------------------------------------------------------------
def test_ingest_samples_then_query():
    resp = client.post("/api/docs/sample")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_chunks"] > 0
    assert len(body["ingested"]) == 5

    q = client.post(
        "/api/query",
        json={"question": "What is retrieval-augmented generation?"},
    )
    assert q.status_code == 200
    data = q.json()
    assert data["status"] == "ok"
    assert data["sources"], "query should return sources"
    assert "retrieval" in data["answer"].lower()

    assert data["sources"][0]["doc_name"] == "what-is-rag"


def test_query_semantic_ranking():
    client.post("/api/docs/sample")
    q = client.post(
        "/api/query",
        json={"question": "How should I split long documents into pieces?"},
    )
    data = q.json()
    assert data["sources"][0]["doc_name"] == "chunking-strategies"


def test_query_top_k():
    client.post("/api/docs/sample")
    q = client.post(
        "/api/query",
        json={"question": "What is RAG?", "top_k": 2},
    )
    assert len(q.json()["sources"]) == 2


def test_query_without_docs():
    resp = client.post("/api/query", json={"question": "anything"})
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


# ----------------------------------------------------------------------
# Uploads
# ----------------------------------------------------------------------
def test_upload_txt_file():
    resp = client.post(
        "/api/docs",
        files={
            "file": (
                "hello.txt",
                b"RAG stands for Retrieval-Augmented Generation. "
                b"It combines retrieval with generation.",
                "text/plain",
            )
        },
    )
    assert resp.status_code == 200
    assert resp.json()["chunks"] == 1


def test_upload_rejects_non_text():
    resp = client.post(
        "/api/docs", files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert resp.status_code == 400


def test_delete_doc():
    client.post(
        "/api/docs",
        files={"file": ("temp.txt", b"temporary document to delete", "text/plain")},
    )
    assert client.get("/api/docs").json()["documents"][0]["name"] == "temp"
    resp = client.delete("/api/docs/temp")
    assert resp.status_code == 200
    assert client.get("/api/docs").json()["documents"] == []


# ----------------------------------------------------------------------
# Chunker
# ----------------------------------------------------------------------
def test_chunker_respects_size_and_overlap():
    chunks = chunk_text("word " * 400, chunk_size=100, overlap=10)
    assert len(chunks) >= 3
    assert all(len(c) <= 120 for c in chunks)


def test_chunker_keeps_paragraphs():
    text = "Paragraph one about cats.\n\nParagraph two about dogs."
    chunks = chunk_text(text, chunk_size=30, overlap=0)
    assert chunks == ["Paragraph one about cats.", "Paragraph two about dogs."]


def test_chunker_empty_input():
    assert chunk_text("   \n\n ") == []


# ----------------------------------------------------------------------
# Embedding isolation (vectors from different models must never mix)
# ----------------------------------------------------------------------
def test_docs_report_embedding_name():
    client.post("/api/docs/sample")
    doc = client.get("/api/docs").json()["documents"][0]
    assert doc["embedding_name"] == "mock-hash"


def test_search_filters_by_embedding_name(tmp_path):
    store = VectorStore(str(tmp_path / "iso.db"))
    llm = LLMClient(api_key="")  # mock mode -> embedding_name "mock-hash"
    rag = RAGPipeline(llm=llm, store=store)

    rag.ingest("doc", "Cats are small domestic animals.")
    vec = llm.embed_texts(["cats"])[0]

    assert store.search(vec, 4, embedding_name="mock-hash"), "same-name chunks match"
    assert store.search(vec, 4, embedding_name="local:all-MiniLM-L6-v2") == [], (
        "chunks from another embedding model are never returned"
    )


def test_llm_client_overrides():
    llm = LLMClient(
        api_key="sk-fake",
        base_url="https://api.groq.com/openai/v1",
        chat_model="llama-3.3-70b-versatile",
        embedding_model="nomic-embed-text-v1_5",
    )
    assert llm.mock is False
    assert llm.chat_model == "llama-3.3-70b-versatile"
    assert llm.embedding_name == "nomic-embed-text-v1_5@api.groq.com"
