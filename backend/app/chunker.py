"""Turn a document into small, overlapping chunks.

Why chunk at all?
  - embedding models accept a limited number of tokens per request,
  - long passages dilute the meaning when averaged into one vector,
  - retrieval works best with short, self-contained passages,
  - the LLM prompt stays small, so answers stay fast and cheap.

Chunks are cut on paragraph boundaries first, then on sentences, and
neighbouring chunks overlap a little so context is never lost at seams.
"""

import re

from .config import settings


def chunk_text(
    text: str, chunk_size: int | None = None, overlap: int | None = None
) -> list[str]:
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()
    ]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            chunks.append(current)
        if len(para) > chunk_size:
            chunks.extend(_split_long_paragraph(para, chunk_size, overlap))
            current = ""
        else:
            current = f"{_tail(current, overlap)}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


def _split_long_paragraph(para: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", para) if s.strip()]
    if not sentences:
        sentences = [para]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
        if len(sentence) > chunk_size:
            current = ""
            for i in range(0, len(sentence), chunk_size):
                chunks.append(sentence[i : i + chunk_size])
        else:
            current = f"{_tail(current, overlap)} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _tail(text: str, n: int) -> str:
    return text[-n:].strip() if n and text else ""
