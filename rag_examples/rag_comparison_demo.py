"""
RAG Evolution: Side-by-Side Comparison
========================================
Runs the same query through all 4 RAG versions to show the difference
in retrieval quality, context richness, and answer accuracy.

Run: python rag_comparison_demo.py
"""

import os
import json
import time
import math
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────────
# SHARED CORPUS
# ─────────────────────────────────────────────
CORPUS = [
    {"id": "d1", "text": "RAG (Retrieval-Augmented Generation) combines retrieval with generation to reduce hallucinations.", "meta": {"year": 2020, "topic": "rag"}},
    {"id": "d2", "text": "RAG was introduced by Meta AI using Dense Passage Retrieval (DPR) and a BART generator.", "meta": {"year": 2020, "topic": "rag"}},
    {"id": "d3", "text": "Hybrid search combines dense vector search with sparse BM25 keyword search using Reciprocal Rank Fusion.", "meta": {"year": 2023, "topic": "search"}},
    {"id": "d4", "text": "Cross-encoder rerankers jointly score query-document pairs and outperform bi-encoders for relevance ranking.", "meta": {"year": 2023, "topic": "search"}},
    {"id": "d5", "text": "Agentic RAG adds query decomposition, iterative retrieval, and self-correction to standard RAG pipelines.", "meta": {"year": 2024, "topic": "rag"}},
    {"id": "d6", "text": "Graph RAG uses knowledge graphs to capture entity relationships for richer, multi-hop retrieval.", "meta": {"year": 2025, "topic": "rag"}},
    {"id": "d7", "text": "Meta AI created both RAG and PyTorch. Meta AI Research is based in Menlo Park, California.", "meta": {"year": 2021, "topic": "organizations"}},
    {"id": "d8", "text": "Google created BERT, TensorFlow, and the Transformer architecture used by most modern LLMs.", "meta": {"year": 2022, "topic": "organizations"}},
]

TEST_QUERY = "How has RAG evolved from basic retrieval to agentic and graph-based approaches?"


# ─────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    r = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return r.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


def answer_with_gpt(context: str, query: str, model: str = "gpt-4o-mini") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer using only the provided context. Be comprehensive."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# RAG v1: Basic
# ─────────────────────────────────────────────
def run_rag_v1(query: str, docs: list[dict]) -> dict:
    start = time.time()
    indexed = [{**d, "embedding": get_embedding(d["text"])} for d in docs]
    q_emb = get_embedding(query)
    scored = sorted(indexed, key=lambda x: cosine_similarity(q_emb, x["embedding"]), reverse=True)
    top3 = scored[:3]
    context = "\n".join([d["text"] for d in top3])
    answer = answer_with_gpt(context, query)
    return {
        "version": "v1 Basic",
        "retrieved": [d["id"] for d in top3],
        "context_length": len(context),
        "answer": answer,
        "latency_s": round(time.time() - start, 2),
        "features": ["cosine_similarity", "top_k"],
    }


# ─────────────────────────────────────────────
# RAG v2: Hybrid + Rerank
# ─────────────────────────────────────────────
class MiniBM25:
    def __init__(self, texts: list[str]):
        self.corpus = [t.lower().split() for t in texts]
        N = len(self.corpus)
        avgdl = sum(len(d) for d in self.corpus) / N
        df = {}
        for doc in self.corpus:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((N - v + 0.5) / (v + 0.5) + 1) for t, v in df.items()}
        self.corpus_raw = self.corpus
        self.avgdl = avgdl
        self.k1, self.b = 1.5, 0.75

    def score(self, query: str, idx: int) -> float:
        doc = self.corpus_raw[idx]
        tf_map = Counter(doc)
        dl = len(doc)
        s = 0.0
        for t in query.lower().split():
            if t not in self.idf:
                continue
            tf = tf_map.get(t, 0)
            s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def run_rag_v2(query: str, docs: list[dict]) -> dict:
    start = time.time()
    texts = [d["text"] for d in docs]
    indexed = [{**d, "embedding": get_embedding(d["text"])} for d in docs]
    bm25 = MiniBM25(texts)
    q_emb = get_embedding(query)

    # Dense scores
    for d in indexed:
        d["dense"] = cosine_similarity(q_emb, d["embedding"])

    # BM25 scores
    bm25_scores = {d["id"]: bm25.score(query, i) for i, d in enumerate(indexed)}
    max_bm25 = max(bm25_scores.values()) or 1

    # RRF
    dense_rank = {d["id"]: r for r, d in enumerate(sorted(indexed, key=lambda x: x["dense"], reverse=True))}
    sparse_rank = {d["id"]: r for r, d in enumerate(sorted(docs, key=lambda x: bm25_scores[x["id"]], reverse=True))}
    rrf = {d["id"]: 1/(60+dense_rank[d["id"]]) + 1/(60+sparse_rank[d["id"]]) for d in docs}
    top4 = sorted(docs, key=lambda x: rrf[x["id"]], reverse=True)[:4]

    # GPT rerank
    def gpt_score(doc):
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Rate relevance 0-10. Query: {query}\nDoc: {doc['text']}\nScore:"}],
            temperature=0, max_tokens=3,
        )
        try:
            return float(r.choices[0].message.content.strip())
        except:
            return 0.0

    reranked = sorted(top4, key=gpt_score, reverse=True)[:3]
    context = "\n".join([d["text"] for d in reranked])
    answer = answer_with_gpt(context, query)
    return {
        "version": "v2 Hybrid+Rerank",
        "retrieved": [d["id"] for d in reranked],
        "context_length": len(context),
        "answer": answer,
        "latency_s": round(time.time() - start, 2),
        "features": ["dense_search", "bm25_sparse", "rrf_fusion", "gpt_reranker"],
    }


# ─────────────────────────────────────────────
# RAG v3: Agentic (simplified)
# ─────────────────────────────────────────────
def run_rag_v3(query: str, docs: list[dict]) -> dict:
    start = time.time()
    indexed = [{**d, "embedding": get_embedding(d["text"])} for d in docs]
    q_emb = get_embedding(query)

    def retrieve(q: str, k: int = 3) -> list[dict]:
        emb = get_embedding(q)
        return sorted(indexed, key=lambda x: cosine_similarity(emb, x["embedding"]), reverse=True)[:k]

    # Step 1: Decompose
    decomp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f'Break into 2-3 sub-questions. Return JSON: {{"questions": ["..."]}}\nQuestion: {query}'
        }],
        response_format={"type": "json_object"},
        temperature=0,
    )
    sub_qs = json.loads(decomp.choices[0].message.content).get("questions", [query])

    # Step 2: Multi-hop retrieval
    all_docs = {}
    for sq in sub_qs:
        for d in retrieve(sq, k=2):
            all_docs[d["id"]] = d

    # Step 3: Self-critique and refine
    draft_context = "\n".join([d["text"] for d in list(all_docs.values())[:5]])
    draft = answer_with_gpt(draft_context, query)
    critique = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f'Is this answer complete? JSON: {{"complete": true/false, "missing": "..."}}\nQ: {query}\nA: {draft}'
        }],
        response_format={"type": "json_object"},
        temperature=0,
    )
    crit_data = json.loads(critique.choices[0].message.content)
    if not crit_data.get("complete", True):
        extra = retrieve(crit_data.get("missing", query), k=2)
        for d in extra:
            all_docs[d["id"]] = d

    final_context = "\n".join([d["text"] for d in list(all_docs.values())[:6]])
    answer = answer_with_gpt(final_context, query)

    return {
        "version": "v3 Agentic",
        "sub_questions": sub_qs,
        "retrieved": list(all_docs.keys()),
        "context_length": len(final_context),
        "answer": answer,
        "latency_s": round(time.time() - start, 2),
        "features": ["query_decomposition", "multi_hop", "self_correction"],
    }


# ─────────────────────────────────────────────
# RAG v4: Graph RAG + Long Context
# ─────────────────────────────────────────────
def run_rag_v4(query: str, docs: list[dict]) -> dict:
    start = time.time()
    indexed = [{**d, "embedding": get_embedding(d["text"])} for d in docs]

    # Extract mini knowledge graph
    kg_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f'Extract entity relationships from these texts as triples.\n'
                f'Return JSON: {{"triples": [{{"s": "...", "r": "...", "o": "..."}}]}}\n\n'
                + "\n".join([d["text"] for d in docs])
            )
        }],
        response_format={"type": "json_object"},
        temperature=0,
    )
    triples = json.loads(kg_resp.choices[0].message.content).get("triples", [])
    graph_context = "\n".join([f"  {t.get('s','')} --[{t.get('r','')}]--> {t.get('o','')}" for t in triples[:15]])

    # Long context: pack all relevant docs
    q_emb = get_embedding(query)
    all_sorted = sorted(indexed, key=lambda x: cosine_similarity(q_emb, x["embedding"]), reverse=True)
    long_context = "\n".join([d["text"] for d in all_sorted])

    # Combined context: graph + passages
    combined = f"Knowledge Graph:\n{graph_context}\n\nAll Passages:\n{long_context}"

    answer = answer_with_gpt(combined, query, model="gpt-4o")

    return {
        "version": "v4 Graph+LongCtx",
        "graph_triples": len(triples),
        "retrieved": [d["id"] for d in all_sorted],
        "context_length": len(combined),
        "graph_context": graph_context,
        "answer": answer,
        "latency_s": round(time.time() - start, 2),
        "features": ["knowledge_graph", "long_context", "graph_traversal", "hybrid_orchestration"],
    }


# ─────────────────────────────────────────────
# MAIN COMPARISON
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("RAG EVOLUTION: Side-by-Side Comparison")
    print("="*70)
    print(f"\nQuery: {TEST_QUERY}\n")

    runners = [run_rag_v1, run_rag_v2, run_rag_v3, run_rag_v4]
    results = []

    for runner in runners:
        try:
            result = runner(TEST_QUERY, CORPUS)
            results.append(result)
            print(f"\n{'─'*60}")
            print(f"  {result['version']}")
            print(f"  Features:    {', '.join(result['features'])}")
            print(f"  Retrieved:   {result['retrieved']}")
            print(f"  Ctx length:  {result['context_length']} chars")
            print(f"  Latency:     {result['latency_s']}s")
            print(f"\n  Answer:\n  {result['answer'][:300]}...")
        except Exception as e:
            print(f"\n[ERROR in {runner.__name__}]: {e}")

    # Print comparison table
    print(f"\n\n{'='*70}")
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Version':<22} {'Ctx Len':>8} {'Latency':>9} {'Features'}")
    print("─"*70)
    for r in results:
        print(f"{r['version']:<22} {r['context_length']:>8} {r['latency_s']:>8}s  {', '.join(r['features'][:2])}...")

    print("\nKey Takeaways:")
    print("  v1: Simple, fast, limited semantic understanding")
    print("  v2: Better recall via hybrid + precision via reranking")
    print("  v3: Handles complex queries via decomposition + self-correction")
    print("  v4: Captures relationships, global synthesis via graph + long context")
