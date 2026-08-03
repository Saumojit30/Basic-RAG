"""Basic RAG - Streamlit frontend.

A second, self-contained UI for the same RAG pipeline used by the Next.js
app (backend/app/*). It is configured from the sidebar and defaults to Groq
(https://console.groq.com) for blazing-fast LLM inference.

How to run:
    cd streamlit_app
    pip install -r requirements.txt
    streamlit run app.py

Embeddings come from either:
  - a LOCAL model (sentence-transformers all-MiniLM-L6-v2) - free, private,
    no API key, works offline after the one-time model download; or
  - the SAME API as the chat model (Groq's nomic-embed-text-v1_5, OpenAI's
    text-embedding-3-small, ...) - requires an API key with an embeddings
    endpoint.

IMPORTANT: every chunk is tagged with the embedding model that produced it,
and retrieval only searches chunks with the SAME tag. Switching embedding
methods invalidates previously ingested chunks - re-ingest after switching
(the app warns you when this is needed).
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import streamlit as st  # noqa: E402

from app.llm import LLMClient  # noqa: E402
from app.rag import RAGPipeline  # noqa: E402
from app.sample_docs import DOCS  # noqa: E402
from app.vectorstore import VectorStore  # noqa: E402

st.set_page_config(page_title="Basic RAG - Streamlit", layout="wide")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMB_LOCAL = f"Local model ({LOCAL_EMBEDDING_MODEL}) - free, private"
EMB_API = "API embeddings (same key/endpoint)"

SUGGESTED_QUESTIONS = [
    "What is retrieval-augmented generation?",
    "How do embeddings work?",
    "What is a good chunk size?",
    "Why do we need a vector database?",
]


def local_embedder_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)}) - retrieved by embedding similarity"):
        for i, s in enumerate(sources, 1):
            st.markdown(
                f"[{i}] **{s['doc_name']}** / chunk {s['chunk_index']} "
                f"(similarity {s['score']:.3f})"
            )
            st.progress(max(0.0, min(1.0, s["score"])), text="similarity to question")
            st.caption(s["text"])


class LocalEmbedder:
    """Embeddings from a small model that runs on your machine."""

    MODEL_NAME = LOCAL_EMBEDDING_MODEL

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(self.MODEL_NAME)

    @property
    def embedding_name(self) -> str:
        return f"local:{self.MODEL_NAME}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            vec.tolist()
            for vec in self._model.encode(texts, normalize_embeddings=True)
        ]


@st.cache_resource(show_spinner=False)
def build_pipeline(
    api_key: str,
    base_url: str,
    chat_model: str,
    embedding_model: str,
    embedding_choice: str,
) -> RAGPipeline:
    llm = LLMClient(
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        embedding_model=embedding_model,
    )
    embedder = llm if embedding_choice == EMB_API else LocalEmbedder()
    return RAGPipeline(llm=llm, embedder=embedder)


# ----------------------------------------------------------------------
# Sidebar: provider & embeddings configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("Basic RAG")

    st.subheader("1. LLM (Groq by default)")
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_key = ""
    groq_key = st.text_input(
        "API key",
        type="password",
        value=default_key,
        help="Get one at https://console.groq.com/keys . Leave empty for MOCK MODE "
        "(the pipeline runs with fake embeddings and echo answers).",
    )
    base_url = st.text_input("API base URL (OpenAI-compatible)", value=GROQ_BASE_URL)
    chat_model = st.text_input(
        "Chat model",
        value="llama-3.3-70b-versatile",
        help="Groq models: llama-3.3-70b-versatile, llama-3.1-8b-instant, "
        "llama-4-scout-17b-16e-instruct, ... OpenAI: gpt-4o-mini.",
    )

    st.subheader("2. Embeddings")
    has_local = local_embedder_available()
    if has_local:
        embedding_choice = st.radio(
            "Embedding source",
            [EMB_LOCAL, EMB_API],
            help="Local = free, private, no API call. API = same key/endpoint "
            "(Groq offers nomic-embed-text-v1_5).",
        )
    else:
        embedding_choice = EMB_API
        st.warning(
            "Local embeddings unavailable - install with: pip install sentence-transformers"
        )
    embedding_model = st.text_input(
        "Embedding model (API choice)",
        value="nomic-embed-text-v1_5",
        help="Groq: nomic-embed-text-v1_5. OpenAI: text-embedding-3-small.",
    )
    top_k = st.slider("top_k (chunks retrieved per question)", 1, 10, 4)

    st.divider()
    st.caption("Pipeline: chunk -> embed -> store -> retrieve top-k -> generate with citations.")

pipeline = build_pipeline(
    groq_key, base_url, chat_model, embedding_model, embedding_choice
)

with st.sidebar:
    if st.button("Load 5 sample docs about RAG", use_container_width=True):
        with st.spinner("Chunking + embedding..."):
            total = sum(pipeline.ingest(name, text)["chunks"] for name, text in DOCS.items())
            st.success(f"Ingested {len(DOCS)} sample docs ({total} chunks).")
    uploaded = st.file_uploader("Or upload your own .txt / .md", type=["txt", "md", "markdown"])
    if uploaded is not None:
        if st.button("Ingest uploaded file", use_container_width=True):
            with st.spinner("Chunking + embedding..."):
                result = pipeline.ingest(Path(uploaded.name).stem, uploaded.read().decode("utf-8", "replace"))
                st.success(f"Ingested '{result['doc_name']}' ({result['chunks']} chunks).")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

# ----------------------------------------------------------------------
# Knowledge base state (with embedding-consistency guard)
# ----------------------------------------------------------------------
docs = pipeline.store.list_docs()
current_embedding = pipeline.embedder.embedding_name
stale = [d for d in docs if d["embedding_name"] != current_embedding]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_embedding" not in st.session_state:
    st.session_state.last_embedding = current_embedding

# ----------------------------------------------------------------------
# Main column: chat
# ----------------------------------------------------------------------
col_chat, col_docs = st.columns([3, 1], gap="large")

with col_chat:
    st.header("Ask your documents")
    if pipeline.llm.mock:
        st.info(
            "MOCK MODE: no API key set - the pipeline runs with hashed bag-of-words "
            "embeddings and the 'answer' is the retrieved context echoed back. Add a "
            "Groq key in the sidebar for real answers."
        )
    if stale:
        st.warning(
            f"Embedding method changed ({current_embedding}). Chunks below were "
            "embedded with a different model and cannot be searched. Switch back, or "
            "re-ingest your documents (sample docs are one click in the sidebar)."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    question = st.chat_input("Ask a question about the ingested documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving + generating..."):
                result = pipeline.query(question, top_k=top_k)
            if result["sources"]:
                st.markdown(result["answer"])
                render_sources(result["sources"])
            else:
                st.info(result["answer"])
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )

with col_docs:
    st.header("Knowledge base")
    if not docs:
        st.caption("No documents yet. Load the sample docs in the sidebar.")
    for d in docs:
        emb = d["embedding_name"] or "(legacy)"
        match = "[OK]" if d["embedding_name"] == current_embedding else "[MISMATCH]"
        st.markdown(
            f"**{d['name']}**  \n{d['chunks']} chunks / {(d['chars'] / 1000):.1f}k chars  \n"
            f"`{emb}` {match}"
        )
        if st.button("remove", key=f"rm-{d['name']}", type="secondary"):
            pipeline.store.delete_doc(d["name"])
            st.rerun()
    st.caption(f"Embedding in use: `{current_embedding}`")
    if docs:
        st.caption("Suggested questions: " + ", ".join(f'"{q}"' for q in SUGGESTED_QUESTIONS[:2]))
