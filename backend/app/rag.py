"""The RAG pipeline: retrieve relevant chunks, then generate an answer.

Per question:
  1. embed the question into a vector,
  2. find the top-k most similar stored chunks (cosine similarity),
  3. stuff those chunks into a prompt,
  4. ask the LLM to answer using ONLY those chunks, with citations.
"""

from .chunker import chunk_text
from .config import settings
from .llm import LLMClient
from .vectorstore import VectorStore

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the context below.
Rules:
- Base your answer strictly on the provided context.
- If the context does not contain the answer, say you don't know. Never invent facts.
- Cite sources with [1], [2], ... matching the numbers shown in the context.
- Keep the answer clear, concise, and in the same language as the question."""


class RAGPipeline:
    def __init__(
        self,
        llm: LLMClient | None = None,
        store: VectorStore | None = None,
        embedder=None,
    ) -> None:
        """embedder produces vectors; llm generates answers.

        By default the LLM client embeds too. Pass a custom embedder (e.g. a
        local sentence-transformers model) to decouple the two.
        """
        self.llm = llm or LLMClient()
        self.store = store or VectorStore()
        self.embedder = embedder or self.llm

    # ------------------------- INGEST -----------------------------
    def ingest(self, doc_name: str, text: str) -> dict:
        """Chunk -> embed -> store. Re-ingesting replaces the old copy."""
        chunks = chunk_text(text)
        if not chunks:
            return {"doc_name": doc_name, "chunks": 0, "message": "empty document"}
        embeddings = self.embedder.embed_texts(chunks)
        self.store.delete_doc(doc_name)
        self.store.add_chunks(
            doc_name, chunks, embeddings, embedding_name=self.embedder.embedding_name
        )
        return {"doc_name": doc_name, "chunks": len(chunks)}

    # ------------------------- QUERY ------------------------------
    def query(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or settings.top_k
        query_vec = self.embedder.embed_texts([question])[0]
        sources = self.store.search(
            query_vec, top_k, embedding_name=self.embedder.embedding_name
        )

        if not sources:
            return {
                "answer": (
                    "I have no documents to answer from. Ingest some text first "
                    "(sample docs or an upload), then ask again."
                ),
                "sources": [],
                "model": settings.chat_model,
                "mock": self.llm.mock,
            }

        context = "\n\n".join(
            f"[{i + 1}] ({r['doc_name']}, chunk {r['chunk_index']})\n{r['text']}"
            for i, r in enumerate(sources)
        )
        answer = self.llm.generate(
            SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}"
        )
        return {
            "answer": answer,
            "sources": sources,
            "model": settings.chat_model,
            "mock": self.llm.mock,
        }
