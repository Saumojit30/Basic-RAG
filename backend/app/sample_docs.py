"""Bundled sample documents so the demo works out of the box.

The docs explain RAG itself - ask the app questions about RAG and watch
it retrieve from these files and answer with citations.
"""

DOCS: dict[str, str] = {
    "what-is-rag": """
What is Retrieval-Augmented Generation (RAG)?

Large language models are trained on a fixed snapshot of public internet data.
They are incredibly good at language, but they do not know about your private
documents, they can be wrong about recent facts, and they confidently invent
things when they do not know (this is called hallucination).

RAG fixes exactly this. Instead of asking the model to pull the answer from
its memory, we first RETRIEVE relevant passages from our own knowledge base,
and then GENERATE an answer grounded in those passages. The model is told:
"here is the context, answer ONLY using it, and cite where each fact comes
from." Because the answer is built on real retrieved text, it is accurate,
traceable, and always up to date - you can simply add new documents to the
knowledge base without retraining anything.

The pipeline has two phases. The INGEST phase happens once per document:
the text is split into chunks, each chunk is converted into a vector
(a list of numbers describing its meaning), and the vectors are stored in a
vector database. The QUERY phase happens per question: the question is
converted into a vector the same way, the database returns the most similar
chunks, and those chunks are put into a prompt together with the question.
The LLM then writes the final answer with citations.

RAG is popular because it combines the strengths of both sides: retrieval
gives you ground truth and freshness, generation gives you fluent, natural
language answers.
""".strip(),
    "how-embeddings-work": """
How do embeddings work?

An embedding is a list of numbers - a vector - that captures the MEANING of a
piece of text. A good embedding model maps text to vectors such that texts
with similar meaning end up close together in vector space, while unrelated
texts end up far apart. The words "cat" and "kitten" will sit near each
other; "cat" and "quantum physics" will not.

How is meaning turned into numbers? Modern models are transformers trained
on huge amounts of text. They learn to compress each token into a vector
based on its context: what words surround it. After training, the final
vector for a sentence is a blend of its tokens' learned representations.
You do not need to understand the math to use embeddings - you call an API
that turns text in, vectors out.

Embeddings are the foundation of retrieval: the question is embedded, and
the system finds stored chunks whose vectors point in the same direction.
The standard measure of "same direction" is cosine similarity, which is the
dot product of two vectors divided by their lengths - a value from -1 to 1,
where 1 means perfectly aligned.

Real embeddings are expensive to produce (a large model runs on every text),
so in production you embed a chunk once at ingest time and reuse the stored
vector forever. Only the short question needs to be embedded at query time.
""".strip(),
    "chunking-strategies": """
Chunking: how to split documents for retrieval.

Before a document can be retrieved from, it must be split into chunks, and
the chunk size has a big impact on answer quality.

Small chunks (around 100-300 characters) are very precise: a question about
one specific detail will match tightly. The downside is missing context -
the sentence that answers the question may sit in the chunk before or after
the best match.

Large chunks (1000+ characters) carry full context but dilute the vector
with unrelated content, so the similarity score becomes noisy, and a single
chunk may contain several topics, making citations less precise.

A good default is 300-800 characters per chunk with a 10-20% overlap between
neighboring chunks. The overlap ensures a fact that straddles a chunk
boundary is still fully contained in at least one chunk.

Chunk boundaries matter too: split on paragraph or sentence boundaries
rather than arbitrary character counts, so each chunk starts and ends with
complete thoughts. Structured documents (HTML, markdown, PDFs with headings)
are often chunked by their structure - one chunk per section - and richer
systems use a recursive splitter that tries longer separators first and
falls back to shorter ones when a chunk is still too big.

There is no single perfect size: tune it for your documents and evaluate
retrieval quality on a set of test questions.
""".strip(),
    "retrieval-and-ranking": """
Retrieval and ranking: how the system finds the right chunks.

At query time the user's question is embedded, and the vector store returns
the chunks with the highest similarity scores. Two things decide quality:
whether the right passages exist in the store, and whether the ranking puts
them on top.

The number of chunks returned is called top_k. Returning more chunks gives
the model more material (and can improve recall), but it also grows the
prompt, which costs more tokens and can distract the model with noise.
Typical values are 3 to 10.

Because embeddings only see meaning, not exact words, a query like "how do
I fix a leaking tap" can match a chunk about "repairing a faucet" - that is
the power of semantic search over keyword search. But embeddings can also
miss exact matches, and their ranking is not perfect. Production systems
add a second-stage RERANKER: a smaller model that re-scores the top few
dozen candidates and orders them properly for the LLM.

Retrieval quality is measured on test sets with metrics like hit rate (was
the right chunk in the top k?) and mean reciprocal rank (how high did it
rank?). If retrieval is bad, no prompt engineering will save the answers,
so teams often start by testing retrieval in isolation before tuning the
generation step.
""".strip(),
    "vector-databases": """
Vector databases: where embeddings live.

The vector store needs to answer one question fast: given this vector,
which stored vectors are closest? For a hundred chunks you can simply scan
everything and compare, like this demo does. But production knowledge bases
hold millions of chunks, each 1000+ numbers, and a brute-force scan would
take seconds.

Vector databases (Qdrant, Pinecone, Weaviate, Milvus, pgvector, Chroma,
Upstash Vector, and many more) solve this with approximate nearest neighbor
(ANN) indexes such as HNSW or IVF. These indexes trade a tiny amount of
accuracy for enormous speed - finding the top 10 matches out of 10 million
in milliseconds.

A vector database is a database: it persists vectors, supports CRUD, handles
concurrency, and often stores the original chunk text alongside the vector
so the application gets both in one call. Many also support metadata
filtering ("only search documents from 2026", "only the engineering docs"),
hybrid search (combining keyword and semantic search), and namespace
isolation between tenants.

Choosing one comes down to your scale, budget, and where you already host
data. Postgres users often pick pgvector because there is nothing new to
run. Everyone else typically picks a purpose-built store. For this demo a
SQLite file is more than enough - and it makes the whole pipeline visible
in one small file.
""".strip(),
}
