# Basic RAG — a minimal, fully explainable RAG app

> **Project 01 in a multi-project RAG repository.** See `../README.md` for the repo roadmap and future project tracking.

A tiny Retrieval-Augmented Generation application built to *be explained*:
every piece of the pipeline lives in this repo, small enough to read in one
sitting. Ingest documents, then ask questions — the app retrieves the most
relevant passages and generates an answer with citations.

- **Backend** — Python / FastAPI (deployed on **Render**): chunking, embeddings,
  vector search, and the LLM call.
- **Frontend (web)** — Next.js / TypeScript (deployed on **Vercel**): chat UI,
  upload, and document management.
- **Frontend (Python)** — Streamlit app (`streamlit_app/`): a second, fully
  self-contained UI for the same pipeline, configured for **Groq** out of the
  box.

> **Architecture guide:** see [`ARCHITECTURE.md`](ARCHITECTURE.md) for a
> line-by-line trace of the whole system — how data flows from every UI
> through chunking → embedding → storage → top-k retrieval → LLM generation,
> with `file:line` anchors and diagrams.

---

## 1. What is RAG? (the 2-minute version)

LLMs are trained on a fixed snapshot of public data. They don't know your
documents, they can be outdated, and when they don't know something they
*hallucinate* — confidently making things up.

**RAG fixes that by retrieving before generating.** Instead of asking the
model to pull the answer from memory, you:

1. **Ingest** your documents: split into chunks → turn each chunk into a
   *vector* (a list of numbers capturing its meaning) → store the vectors.
2. **Query**: convert the question into a vector the same way → find the
   stored chunks whose vectors point in the same "direction" (cosine
   similarity) → hand those chunks to the LLM with the instruction:
   *"answer using ONLY this context, and cite [1], [2]…"*.

```
 INGEST (once per document)                    QUERY (per question)
 ──────────────────────────                    ─────────────────────
 document ──► chunk ──► embed ──► store        question ──► embed ──┐
                 ▲                               vector store search │
          split into 300-800                   (cosine similarity) ─┘
          char pieces with                        │
          ~10-20% overlap                        top-k chunks
                                                  │
                                            prompt = context + question
                                                  │
                                                  ▼
                                        LLM ──► grounded answer + citations
```

The answer is built on *real retrieved text*, so it's accurate, traceable,
and always up to date — adding new documents requires no retraining.

---

## 2. Project layout

```
BasicRAG/
├── backend/                  # FastAPI service (Render)
│   ├── app/
│   │   ├── main.py           # HTTP API + CORS
│   │   ├── config.py         # env-var configuration
│   │   ├── llm.py            # OpenAI-compatible client (+ MOCK MODE)
│   │   ├── chunker.py        # splits documents into overlapping chunks
│   │   ├── vectorstore.py    # SQLite + cosine similarity search
│   │   ├── rag.py            # the RAG pipeline (retrieve → generate)
│   │   └── sample_docs.py    # bundled docs that explain RAG itself
│   ├── tests/                # pytest suite (runs in mock mode, no API key)
│   └── requirements.txt
├── streamlit_app/            # Streamlit UI (defaults to Groq)
│   ├── app.py                # chat + upload + sources, all configurable
│   └── requirements.txt
├── frontend/                 # Next.js UI (Vercel)
│   ├── app/page.tsx          # chat + upload + sources UI
│   └── package.json
└── render.yaml               # Render blueprint (one-click backend deploy)
```

**Read the pipeline in this order:** `chunker.py` → `vectorstore.py` →
`rag.py` → `main.py`. Each file has comments explaining the *why*.

### Why mock mode?

If no API key is set (`GROQ_API_KEY` / `OPENAI_API_KEY`), the backend runs
in **mock mode**: embeddings are hashed bag-of-words vectors (similar text
→ similar vectors), and the "answer" is the retrieved context echoed back.
You can demo and test the *entire* pipeline — chunking, semantic retrieval,
ranking — for free. Set a Groq key to see real generated answers.

### Embedding consistency (important)

Every chunk is tagged with the embedding model that produced it, and
retrieval only searches chunks with the **same tag** — vectors from different
models live in different spaces and must never be compared. If you switch
embedding models, re-ingest your documents (the Streamlit app warns you
automatically).

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness check |
| GET | `/` | service info (models, mock mode) |
| GET | `/api/docs` | list ingested documents |
| POST | `/api/docs` | upload a `.txt`/`.md` file (multipart `file`) |
| POST | `/api/docs/sample` | ingest the 5 bundled sample docs |
| DELETE | `/api/docs/{name}` | remove a document |
| POST | `/api/query` | `{"question": "...", "top_k": 4}` → answer + sources |

---

## 3. Run locally

### Backend (port 8000)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Optional: `copy .env.example .env` and set `GROQ_API_KEY=gsk_...` (free at
https://console.groq.com/keys) — or leave it empty to use mock mode. The
backend defaults to Groq (`llama-3.3-70b-versatile` + `nomic-embed-text-v1_5`).

### Frontend (port 3000)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000, click **Load sample docs**, then ask
*"What is retrieval-augmented generation?"*.

### Streamlit app (Groq-first)

```bash
cd streamlit_app
python -m venv .venv
.venv\Scripts\activate              # Windows PowerShell  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt     # downloads sentence-transformers + torch on first run
streamlit run app.py
```

Open http://localhost:8501. In the sidebar:

1. Paste your **Groq API key** (get one free at https://console.groq.com/keys).
   The base URL defaults to `https://api.groq.com/openai/v1` and the chat
   model to `llama-3.3-70b-versatile`. Leave the key empty for mock mode.
2. Choose embeddings: **local model** (default — free, private, no API call;
   Groq's inference API is super fast for *generation* but you may prefer not
   to spend tokens on *embeddings*) or **API embeddings**
   (`nomic-embed-text-v1_5` on Groq).
3. **Load sample docs**, then ask a question. Every answer shows its
   retrieved sources with similarity scores.

> The Streamlit app reuses the exact same pipeline modules as the API
> backend and shares the same SQLite knowledge base — start either UI and
> both see the same documents.
>
> Both UIs and the FastAPI backend talk to Groq by default (OpenAI-compatible
> endpoint): the key goes in `GROQ_API_KEY`, the URL defaults to
> `https://api.groq.com/openai/v1`, chat is `llama-3.3-70b-versatile`,
> embeddings are `nomic-embed-text-v1_5`.

### Tests

```bash
cd backend
.venv\Scripts\python -m pytest tests -q
```

---

## 4. Deploy

### Backend → Render (free)

1. Push this repo to GitHub.
2. Render dashboard → **New** → **Blueprint** → connect the repo.
   `render.yaml` defines the service automatically.
3. After it deploys, open the service → **Environment** → add
   `GROQ_API_KEY` (the blueprint already sets Groq's URL, chat and
   embedding models). It re-deploys automatically.
4. Note the URL: `https://your-app-name.onrender.com` — hit
   `/health` to verify.

> Free-tier Render instances sleep after ~15 min of inactivity; the first
> request may take ~30 s to wake up. The SQLite store is ephemeral, so
> re-ingest sample docs after an instance restart.

### Frontend → Vercel (free)

1. In the Vercel dashboard: **Add New → Project** → import the repo, set the
   **Root Directory** to `frontend`.
2. Add environment variable `NEXT_PUBLIC_API_URL` = your Render URL
   (e.g. `https://your-app-name.onrender.com`).
3. Deploy. Next.js is auto-detected — no config needed.

### Streamlit app → Streamlit Community Cloud (free)

1. Push the repo to GitHub, then go to https://share.streamlit.io → **Create app**.
2. Select the repo, branch, main file `streamlit_app/app.py`, and set the
   **Python version** to match your local one.
3. In **Advanced settings → Secrets** add `GROQ_API_KEY=gsk_...` (format is
   TOML, see `streamlit_app/.streamlit/secrets.example.toml`).
4. Deploy — sentence-transformers downloads the local embedding model on the
   first run, so the first cold start takes a couple of minutes.

---

## 5. Honest limitations & how to grow it

This demo intentionally trades scale for clarity:

| Current | Production swap |
|---|---|
| SQLite + full scan per query | a real vector DB: Qdrant, pgvector, Upstash, Pinecone |
| Single-stage embedding search | add a reranker (e.g. Cohere rerank) |
| Plain text chunking | structure-aware chunking (headers, sections, PDFs) |
| No auth / multi-user | namespaces per user, auth proxy |

**Next learning steps:** add streaming responses (SSE), hybrid keyword +
semantic search, metadata filtering, or an eval set with hit-rate metrics to
tune `CHUNK_SIZE`/`TOP_K`.
