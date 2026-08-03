"""Thin wrapper around any OpenAI-compatible embeddings + chat API.

Configure it with a plain OpenAI key, a Groq key (base_url
https://api.groq.com/openai/v1), or any other OpenAI-compatible endpoint.

Without an API key the client falls back to MOCK MODE:
  - embeddings -> hashed bag-of-words vectors (similar text gets similar vectors)
  - chat       -> echoes the retrieved context back

This lets the entire pipeline be demoed and tested without spending a cent.
"""

import hashlib
import math
import re
from urllib.parse import urlparse

from openai import OpenAI

from .config import settings


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """All arguments fall back to environment settings.

        Passing them explicitly (e.g. from a Streamlit sidebar) lets one
        process talk to several providers without restarting.
        """
        self.api_key = (api_key or settings.api_key).strip()
        self.base_url = base_url or settings.base_url
        self.chat_model = chat_model or settings.chat_model
        self.embedding_model = embedding_model or settings.embedding_model
        self.mock = not self.api_key
        self._client = (
            None
            if self.mock
            else OpenAI(api_key=self.api_key, base_url=self.base_url)
        )

    @property
    def embedding_name(self) -> str:
        """Stable id of this embedding setup; stored next to every chunk.

        Retrieval only compares vectors produced by the SAME embedding model,
        so chunks are tagged and searches are filtered on this id.
        """
        if self.mock:
            return "mock-hash"
        host = urlparse(self.base_url).netloc or self.base_url
        return f"{self.embedding_model}@{host}"

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector of floats per input text."""
        if self.mock:
            return [self._mock_embed(t) for t in texts]
        response = self._client.embeddings.create(
            model=self.embedding_model, input=texts
        )
        return [d.embedding for d in response.data]

    @staticmethod
    def _mock_embed(text: str) -> list[float]:
        """Deterministic, dependency-free 'embeddings' for mock mode.

        Hash each word (and character n-gram) to a vector dimension and count.
        Texts sharing words land close together in vector space.
        """
        vec = [0.0] * settings.embedding_dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        grams: set[str] = set()
        for token in tokens:
            grams.add(token)
            for n in (2, 3):
                grams.update(token[i : i + n] for i in range(len(token) - n + 1))
        for gram in grams:
            idx = int(hashlib.md5(gram.encode()).hexdigest(), 16) % settings.embedding_dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def generate(self, system: str, user: str) -> str:
        if self.mock:
            return (
                "MOCK MODE - no API key configured, so this 'answer' is just the "
                "retrieved context echoed back. Set a key (sidebar or backend/.env) "
                "to get real answers.\n\n" + user
            )
        response = self._client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
