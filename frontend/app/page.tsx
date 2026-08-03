"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Source = { score: number; doc_name: string; chunk_index: number; text: string };
type Answer = { answer: string; sources: Source[]; model: string; mock: boolean };
type DocInfo = { name: string; chunks: number; chars: number };
type ServiceInfo = {
  mock_mode: boolean;
  embedding_model: string;
  chat_model: string;
  chunk_size: number;
  chunk_overlap: number;
};

const SUGGESTED_QUESTIONS = [
  "What is retrieval-augmented generation?",
  "How do embeddings work?",
  "What is a good chunk size?",
  "Why do we need a vector database?",
];

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export default function Home() {
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [service, setService] = useState<ServiceInfo | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [notice, setNotice] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  const refreshDocs = useCallback(async () => {
    try {
      const data = await jsonFetch<{ documents: DocInfo[] }>(`${API_URL}/api/docs`);
      setDocs(data.documents);
    } catch {
      setDocs([]);
    }
  }, []);

  useEffect(() => {
    jsonFetch<ServiceInfo>(`${API_URL}/`)
      .then(setService)
      .catch(() => setError(`Cannot reach the backend at ${API_URL}. Start it with: uvicorn app.main:app --reload`));
    refreshDocs();
  }, [refreshDocs]);

  const ask = useCallback(
    async (q: string) => {
      const text = q.trim();
      if (!text || busy) return;
      setBusy(true);
      setError("");
      setAnswer(null);
      try {
        const data = await jsonFetch<Answer>(`${API_URL}/api/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text, top_k: 4 }),
        });
        setAnswer(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went wrong");
      } finally {
        setBusy(false);
      }
    },
    [busy]
  );

  const loadSamples = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const data = await jsonFetch<{ total_chunks: number }>(`${API_URL}/api/docs/sample`, {
        method: "POST",
      });
      setNotice(`Ingested 5 sample documents (${data.total_chunks} chunks). Try a question!`);
      refreshDocs();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load samples");
    } finally {
      setBusy(false);
    }
  }, [refreshDocs]);

  const upload = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await jsonFetch<{ doc_name: string; chunks: number }>(`${API_URL}/api/docs`, {
        method: "POST",
        body: form,
      });
      setNotice(`Ingested "${data.doc_name}" (${data.chunks} chunks).`);
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
      refreshDocs();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }, [file, refreshDocs]);

  const removeDoc = useCallback(
    async (name: string) => {
      try {
        await jsonFetch(`${API_URL}/api/docs/${encodeURIComponent(name)}`, { method: "DELETE" });
        refreshDocs();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Delete failed");
      }
    },
    [refreshDocs]
  );

  return (
    <div className="app">
      <aside>
        <header>
          <h1>Basic RAG</h1>
          <p>
            A minimal, fully explainable Retrieval-Augmented Generation app:
            ingest documents, then ask questions with citations.
          </p>
          <div className="pipeline-steps">
            <span className="step"><b>1</b> ingest &amp; chunk</span>
            <span className="step"><b>2</b> embed</span>
            <span className="step"><b>3</b> store</span>
            <span className="step"><b>4</b> retrieve top-k</span>
            <span className="step"><b>5</b> generate with sources</span>
          </div>
        </header>

        <div className="card">
          <h2>Service</h2>
          {service && (
            <div className="meta">
              <div className="status-line">
                <span className={service.mock_mode ? "dot mock" : "dot ok"} />
                {service.mock_mode ? "MOCK MODE - no API key" : "Live LLM"}
              </div>
              <div>embedding: {service.embedding_model}</div>
              <div>chat: {service.chat_model}</div>
              <div>
                chunk: {service.chunk_size} chars, overlap {service.chunk_overlap}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <h2>Knowledge base</h2>
          {docs.length === 0 && (
            <p className="empty">No documents yet. Load the samples below, or upload your own.</p>
          )}
          {docs.map((d) => (
            <div className="doc-item" key={d.name}>
              <div>
                <div className="doc-name">{d.name}</div>
                <div className="doc-meta">{d.chunks} chunks / {(d.chars / 1000).toFixed(1)}k chars</div>
              </div>
              <button className="danger" onClick={() => removeDoc(d.name)}>
                remove
              </button>
            </div>
          ))}
          <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 8 }}>
            <button onClick={loadSamples} disabled={busy}>
              Load sample docs (about RAG)
            </button>
            <div className="file-row">
              <input
                ref={fileInput}
                type="file"
                accept=".txt,.md,.markdown"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <button className="secondary" onClick={upload} disabled={busy || !file}>
                Upload .txt / .md
              </button>
            </div>
          </div>
        </div>
      </aside>

      <main>
        <div className="card">
          <form
            className="chat-form"
            onSubmit={(e) => {
              e.preventDefault();
              ask(question);
            }}
          >
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about the ingested documents..."
            />
            <button type="submit" disabled={busy || !question.trim()}>
              {busy ? "..." : "Ask"}
            </button>
          </form>
          <div className="suggestions" style={{ marginTop: 10 }}>
            {SUGGESTED_QUESTIONS.map((q) => (
              <button key={q} className="suggestion" onClick={() => ask(q)} disabled={busy}>
                {q}
              </button>
            ))}
          </div>
        </div>

        {notice && <div className="meta">{notice}</div>}
        {error && <div className="error">{error}</div>}

        {answer && (
          <>
            <div className="card">
              <div className={`answer-box ${answer.mock ? "mocked" : ""}`}>{answer.answer}</div>
            </div>
            <div className="card">
              <h2 style={{ marginBottom: 10 }}>
                Sources ({answer.sources.length}) - retrieved by embedding similarity
              </h2>
              <div className="sources">
                {answer.sources.map((s, i) => (
                  <div className="source" key={i}>
                    <div className="source-head">
                      <span>
                        [{i + 1}] <b>{s.doc_name}</b> / chunk {s.chunk_index}
                      </span>
                      <span className="score-bar">
                        <div style={{ width: `${Math.round(s.score * 100)}%` }} />
                      </span>
                    </div>
                    <div className="source-text">{s.text}</div>
                  </div>
                ))}
              </div>
              <p className="meta" style={{ marginTop: 10 }}>
                model: {answer.model} - the answer was generated from these passages, with
                citations [1..n].
              </p>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
