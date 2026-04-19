"""
RAG v2 (2024): Better Retrieval
=================================
Improvements over v1:
  - Hybrid search: dense (OpenAI embeddings) + sparse (BM25 keyword)
  - Reranking with a cross-encoder prompt (GPT-4o as reranker)
  - Metadata filtering: restrict retrieval by date, category, author, etc.

Requirements: openai, numpy, python-dotenv, rank_bm25
"""

import os
import math
import numpy as np
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# 1. CORPUS WITH METADATA
# ─────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_1",
        "text": (
            "Python is a high-level, interpreted programming language. "
            "Created by Guido van Rossum in 1991, it supports procedural, "
            "object-oriented, and functional paradigms."
        ),
        "metadata": {"category": "programming", "year": 2023, "author": "Alice"},
    },
    {
        "id": "doc_2",
        "text": (
            "Machine learning enables computers to learn from data. "
            "Key techniques: linear regression, decision trees, neural networks. "
            "Three main types: supervised, unsupervised, reinforcement learning."
        ),
        "metadata": {"category": "ai", "year": 2023, "author": "Bob"},
    },
    {
        "id": "doc_3",
        "text": (
            "The Transformer (2017) uses self-attention to process sequences in parallel. "
            "BERT, GPT, and T5 all build on this architecture. "
            "It replaced recurrent models like LSTMs in NLP."
        ),
        "metadata": {"category": "ai", "year": 2024, "author": "Alice"},
    },
    {
        "id": "doc_4",
        "text": (
            "Vector databases store high-dimensional embeddings for similarity search. "
            "Examples: Pinecone, Weaviate, Chroma, FAISS. "
            "ANN algorithms like HNSW make retrieval fast at scale."
        ),
        "metadata": {"category": "databases", "year": 2024, "author": "Charlie"},
    },
    {
        "id": "doc_5",
        "text": (
            "RAG (Retrieval-Augmented Generation) pairs a retriever with a generator. "
            "The retriever fetches relevant documents; the LLM produces a grounded answer. "
            "RAG reduces hallucinations and avoids expensive fine-tuning."
        ),
        "metadata": {"category": "ai", "year": 2024, "author": "Bob"},
    },
    {
        "id": "doc_6",
        "text": (
            "Hybrid search combines dense vector search with sparse keyword search (BM25). "
            "Dense search captures semantic meaning; sparse search excels at exact keyword matches. "
            "Combining both via reciprocal rank fusion (RRF) outperforms either alone."
        ),
        "metadata": {"category": "search", "year": 2024, "author": "Charlie"},
    },
    {
        "id": "doc_7",
        "text": (
            "Cross-encoder rerankers score query-document pairs jointly. "
            "Unlike bi-encoders that embed independently, cross-encoders attend across both inputs. "
            "They are slower but significantly more accurate for relevance ranking."
        ),
        "metadata": {"category": "search", "year": 2024, "author": "Alice"},
    },
    {
        "id": "doc_8",
        "text": (
            "Metadata filtering restricts retrieval to a subset of the corpus before vector search. "
            "Filters like date ranges, category, or author reduce noise and improve precision. "
            "Pre-filtering is more efficient than post-filtering for large corpora."
        ),
        "metadata": {"category": "search", "year": 2024, "author": "Bob"},
    },
]


# ─────────────────────────────────────────────
# 2. DENSE RETRIEVAL (OpenAI embeddings)
# ─────────────────────────────────────────────
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


def build_dense_index(documents: list[dict]) -> list[dict]:
    """Embed every document and store embedding in-memory."""
    print("Building dense index...")
    indexed = []
    for doc in documents:
        emb = get_embedding(doc["text"])
        indexed.append({**doc, "embedding": emb})
    return indexed


def dense_search(
    query: str, index: list[dict], k: int = 5, metadata_filter: dict | None = None
) -> list[dict]:
    """Return top-K documents by cosine similarity, with optional metadata pre-filter."""
    query_emb = get_embedding(query)
    pool = index
    if metadata_filter:
        pool = [d for d in index if _matches_filter(d["metadata"], metadata_filter)]
    scored = [
        {**d, "dense_score": cosine_similarity(query_emb, d["embedding"])}
        for d in pool
    ]
    scored.sort(key=lambda x: x["dense_score"], reverse=True)
    return scored[:k]


def _matches_filter(metadata: dict, filter_: dict) -> bool:
    """Simple AND filter: all filter keys must match the document metadata."""
    for key, value in filter_.items():
        if isinstance(value, dict):
            # Support operators: {"$gte": 2024}
            doc_val = metadata.get(key)
            if "$gte" in value and doc_val < value["$gte"]:
                return False
            if "$lte" in value and doc_val > value["$lte"]:
                return False
            if "$eq" in value and doc_val != value["$eq"]:
                return False
        else:
            if metadata.get(key) != value:
                return False
    return True


# ─────────────────────────────────────────────
# 3. SPARSE RETRIEVAL (BM25 from scratch)
# ─────────────────────────────────────────────
class BM25:
    """Minimal BM25 implementation — no external library needed."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [doc.lower().split() for doc in documents]
        self.N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / self.N
        self.df = self._compute_df()
        self.idf = self._compute_idf()

    def _compute_df(self) -> dict[str, int]:
        df: dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        return df

    def _compute_idf(self) -> dict[str, float]:
        idf = {}
        for term, df in self.df.items():
            idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        return idf

    def score(self, query_terms: list[str], doc_index: int) -> float:
        doc = self.corpus[doc_index]
        doc_len = len(doc)
        tf_map = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf[term] * numerator / denominator
        return score

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        query_terms = query.lower().split()
        scores = [(i, self.score(query_terms, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ─────────────────────────────────────────────
# 4. HYBRID SEARCH — Reciprocal Rank Fusion
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[tuple[int, float]],
    documents: list[dict],
    rrf_k: int = 60,
    k: int = 5,
) -> list[dict]:
    """Fuse dense and sparse ranked lists via RRF."""
    rrf_scores: dict[str, float] = {}

    # Dense ranks
    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

    # Sparse ranks (map index → doc id)
    id_list = [d["id"] for d in documents]
    for rank, (doc_idx, _bm25_score) in enumerate(sparse_results):
        doc_id = id_list[doc_idx]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

    # Sort by combined RRF score
    doc_map = {d["id"]: d for d in documents}
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [
        {**doc_map[doc_id], "rrf_score": score}
        for doc_id, score in fused
        if doc_id in doc_map
    ]


def hybrid_search(
    query: str,
    dense_index: list[dict],
    bm25: BM25,
    documents: list[dict],
    k: int = 5,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Run dense + sparse search, then fuse with RRF."""
    dense_results = dense_search(query, dense_index, k=k * 2, metadata_filter=metadata_filter)
    sparse_results = bm25.search(query, k=k * 2)
    return reciprocal_rank_fusion(dense_results, sparse_results, documents, k=k)


# ─────────────────────────────────────────────
# 5. RERANKING WITH GPT AS CROSS-ENCODER
# ─────────────────────────────────────────────
def rerank_with_gpt(
    query: str, candidates: list[dict], top_n: int = 3
) -> list[dict]:
    """Use GPT to score each candidate's relevance to the query (0-10)."""
    print(f"\n[Reranking {len(candidates)} candidates with GPT cross-encoder...]")
    scored = []
    for cand in candidates:
        prompt = f"""Rate the relevance of the following document to the query on a scale from 0 to 10.
Respond with ONLY a single integer.

Query: {query}

Document: {cand['text']}

Relevance score (0-10):"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        raw = response.choices[0].message.content.strip()
        try:
            score = float(raw.split()[0])
        except (ValueError, IndexError):
            score = 0.0
        scored.append({**cand, "rerank_score": score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_n]


# ─────────────────────────────────────────────
# 6. GENERATE ANSWER
# ─────────────────────────────────────────────
def generate_answer(query: str, context_docs: list[dict]) -> str:
    context = "\n\n".join(
        [f"[{i+1}] {d['text']}" for i, d in enumerate(context_docs)]
    )
    system = (
        "You are a precise assistant. Answer using only the provided context. "
        "If uncertain, say so. Cite source numbers like [1] or [2]."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 7. FULL RAG v2 PIPELINE
# ─────────────────────────────────────────────
def rag_v2_pipeline(
    query: str,
    dense_index: list[dict],
    bm25: BM25,
    documents: list[dict],
    metadata_filter: dict | None = None,
    initial_k: int = 6,
    final_k: int = 3,
) -> dict:
    print(f"\n{'='*60}")
    print(f"RAG v2 | Query: {query}")
    if metadata_filter:
        print(f"         Filter: {metadata_filter}")
    print("="*60)

    # Step 1: Hybrid search (dense + sparse + RRF)
    candidates = hybrid_search(
        query, dense_index, bm25, documents, k=initial_k, metadata_filter=metadata_filter
    )
    print(f"\n[Hybrid search returned {len(candidates)} candidates]")
    for c in candidates:
        print(f"  doc={c['id']}  rrf={c.get('rrf_score', 0):.4f}  text={c['text'][:60]}...")

    # Step 2: Rerank with GPT cross-encoder
    reranked = rerank_with_gpt(query, candidates, top_n=final_k)
    print(f"\n[Top {final_k} after reranking]")
    for r in reranked:
        print(f"  doc={r['id']}  rerank_score={r['rerank_score']:.1f}")

    # Step 3: Generate answer
    answer = generate_answer(query, reranked)
    print(f"\n[Answer]\n{answer}")

    return {
        "query": query,
        "metadata_filter": metadata_filter,
        "candidates": candidates,
        "reranked": reranked,
        "answer": answer,
    }


# ─────────────────────────────────────────────
# 8. DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Build indexes once
    dense_index = build_dense_index(DOCUMENTS)
    bm25 = BM25([d["text"] for d in DOCUMENTS])

    # --- Demo 1: No filter ---
    rag_v2_pipeline(
        query="How does hybrid search improve retrieval?",
        dense_index=dense_index,
        bm25=bm25,
        documents=DOCUMENTS,
    )

    # --- Demo 2: Metadata filter: only 2024 AI docs ---
    rag_v2_pipeline(
        query="What is RAG and how does it work?",
        dense_index=dense_index,
        bm25=bm25,
        documents=DOCUMENTS,
        metadata_filter={"category": "ai", "year": {"$gte": 2024}},
    )

    # --- Demo 3: Filter by author ---
    rag_v2_pipeline(
        query="What are cross-encoder rerankers?",
        dense_index=dense_index,
        bm25=bm25,
        documents=DOCUMENTS,
        metadata_filter={"author": "Alice"},
    )
