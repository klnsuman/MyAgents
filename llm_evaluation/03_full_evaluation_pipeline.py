"""
LLM Evaluation — Part 3: Full Evaluation Pipeline
===================================================
Combines ALL metrics from Part 1 and Part 2 into one unified pipeline.
Evaluates a real mini-RAG system end-to-end and produces a diagnostic report.

Flow:
  1. Build a small RAG system (embed corpus, vector store)
  2. For each test query: retrieve → generate answer
  3. Evaluate with ALL metrics:
       IR Metrics:  Precision@K, Recall@K, NDCG@K, MRR
       RAG Metrics: Context Recall, Context Precision, Faithfulness, Answer Relevance
  4. Print a full diagnostic report with improvement suggestions

Requirements: openai, numpy, python-dotenv
"""

import os
import json
import math
import numpy as np
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────────────────
# CORPUS AND GROUND TRUTH
# ─────────────────────────────────────────────────────────
CORPUS = [
    {"id": "c01", "text": "RAG (Retrieval-Augmented Generation) was introduced by Meta AI in 2020. It combines a dense retriever with a seq2seq generator like BART."},
    {"id": "c02", "text": "RAG reduces hallucinations by grounding the LLM in retrieved external documents rather than relying on parametric memory."},
    {"id": "c03", "text": "BM25 is a classic sparse retrieval algorithm based on term frequency and inverse document frequency. It excels at exact keyword matching."},
    {"id": "c04", "text": "Dense retrieval uses neural embeddings to capture semantic meaning. It finds documents similar in meaning even without keyword overlap."},
    {"id": "c05", "text": "Hybrid search combines dense (embeddings) and sparse (BM25) retrieval. Results are merged using Reciprocal Rank Fusion (RRF)."},
    {"id": "c06", "text": "The Transformer architecture uses self-attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. It was introduced in 2017."},
    {"id": "c07", "text": "BERT is a bidirectional Transformer encoder pre-trained on masked language modeling and next sentence prediction tasks."},
    {"id": "c08", "text": "GPT models are autoregressive Transformer decoders. GPT generates text left-to-right using causal (masked) self-attention."},
    {"id": "c09", "text": "PyTorch uses dynamic computation graphs (define-by-run) making it popular for research. It was developed by Meta AI."},
    {"id": "c10", "text": "TensorFlow was developed by Google Brain. TensorFlow 2.x uses Keras as its high-level API with eager execution by default."},
    {"id": "c11", "text": "Vector databases store high-dimensional embeddings and support approximate nearest neighbor (ANN) search using algorithms like HNSW."},
    {"id": "c12", "text": "Cosine similarity measures the angle between two vectors. It is scale-invariant and widely used for embedding similarity in NLP."},
    {"id": "c13", "text": "Cross-encoder rerankers score query-document pairs jointly. They are slower than bi-encoders but significantly more accurate."},
    {"id": "c14", "text": "Context window limits how much text an LLM can process at once. GPT-4 has a 128K token context window."},
    {"id": "c15", "text": "Hallucination in LLMs refers to generating plausible-sounding but factually incorrect information not supported by the input."},
]

GROUND_TRUTH_ANSWERS = {
    "What is RAG and how does it work?": (
        "RAG (Retrieval-Augmented Generation) was introduced by Meta AI in 2020. "
        "It combines a dense retriever with a generator like BART. "
        "The retriever fetches relevant documents from a knowledge base. "
        "The LLM uses those documents as context to generate grounded answers. "
        "RAG reduces hallucinations by grounding the model in retrieved facts."
    ),
    "What is the difference between dense and sparse retrieval?": (
        "Dense retrieval uses neural embeddings to find semantically similar documents. "
        "Sparse retrieval (BM25) uses keyword matching via TF-IDF. "
        "Dense handles synonyms; sparse handles exact keyword matches. "
        "Hybrid search combines both via Reciprocal Rank Fusion."
    ),
    "How does the Transformer attention mechanism work?": (
        "Transformer attention uses Query, Key, Value matrices. "
        "The formula is Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. "
        "It was introduced in 2017. BERT uses encoder with bidirectional attention. "
        "GPT uses decoder with causal masked attention."
    ),
}

RELEVANT_DOCS = {
    "What is RAG and how does it work?":                    {"c01", "c02", "c15"},
    "What is the difference between dense and sparse retrieval?": {"c03", "c04", "c05"},
    "How does the Transformer attention mechanism work?":    {"c06", "c07", "c08"},
}


# ─────────────────────────────────────────────────────────
# MINI RAG SYSTEM
# ─────────────────────────────────────────────────────────
_vector_store: list[dict] = []


def get_embedding(text: str) -> list[float]:
    r = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return r.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


def build_index(corpus: list[dict]) -> None:
    global _vector_store
    print("  Indexing corpus...")
    _vector_store = [{**doc, "embedding": get_embedding(doc["text"])} for doc in corpus]


def retrieve(query: str, k: int = 5) -> list[dict]:
    q_emb = get_embedding(query)
    scored = sorted(_vector_store, key=lambda d: cosine_similarity(q_emb, d["embedding"]), reverse=True)
    return [{"id": d["id"], "text": d["text"]} for d in scored[:k]]


def generate_answer(query: str, context_docs: list[dict]) -> str:
    context = "\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(context_docs)])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the provided context. Be comprehensive."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────
# IR METRICS (from Part 1)
# ─────────────────────────────────────────────────────────
def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    dcg = sum(
        1 / math.log2(i + 2)
        for i, doc in enumerate(retrieved[:k])
        if doc in relevant
    )
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def mrr(retrieved: list, relevant: set) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


# ─────────────────────────────────────────────────────────
# RAG METRICS (from Part 2) — GPT-as-Judge
# ─────────────────────────────────────────────────────────
def gpt_judge(prompt: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict evaluator. Respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def eval_context_recall(query: str, ground_truth: str, ctx_docs: list[dict]) -> float:
    context = "\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(ctx_docs)])
    result = gpt_judge(f"""Does the context contain all information needed to answer the question?
Break the ground truth into statements and check each against the context.

Question: {query}
Ground Truth: {ground_truth}
Context: {context}

Return JSON: {{"statements": [{{"text": "...", "in_context": true/false}}], "score": 0.0}}""")
    return float(result.get("score", 0.0))


def eval_context_precision(query: str, ctx_docs: list[dict]) -> float:
    chunks = "\n".join([f"Chunk {i+1}: {d['text']}" for i, d in enumerate(ctx_docs)])
    result = gpt_judge(f"""For each chunk, is it relevant to answering the question?

Question: {query}
{chunks}

Return JSON: {{"chunks": [{{"index": 1, "relevant": true/false}}]}}""")

    flags = [c["relevant"] for c in result.get("chunks", [])]
    weighted_sum = 0.0
    rel_count = 0
    for k, is_rel in enumerate(flags, start=1):
        if is_rel:
            rel_count += 1
            weighted_sum += rel_count / k
    return round(weighted_sum / rel_count, 4) if rel_count > 0 else 0.0


def eval_faithfulness(query: str, answer: str, ctx_docs: list[dict]) -> float:
    context = "\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(ctx_docs)])
    result = gpt_judge(f"""Extract claims from the answer and check if each is supported by the context.

Question: {query}
Answer: {answer}
Context: {context}

Return JSON: {{"claims": [{{"text": "...", "supported": true/false}}], "score": 0.0}}""")
    return float(result.get("score", 0.0))


def eval_answer_relevance(query: str, answer: str) -> float:
    result = gpt_judge(f"""Generate 3 questions that this answer is trying to answer.

Answer: {answer}

Return JSON: {{"questions": ["q1", "q2", "q3"]}}""")
    gen_qs = result.get("questions", [])
    if not gen_qs:
        return 0.0
    orig_emb = get_embedding(query)
    sims = [cosine_similarity(orig_emb, get_embedding(gq)) for gq in gen_qs]
    return round(float(np.mean(sims)), 4)


# ─────────────────────────────────────────────────────────
# DIAGNOSTIC REPORT
# ─────────────────────────────────────────────────────────
def print_diagnostic_report(all_results: list[dict]) -> None:
    metrics = {
        "precision_5": np.mean([r["precision@5"] for r in all_results]),
        "recall_5":    np.mean([r["recall@5"] for r in all_results]),
        "ndcg_5":      np.mean([r["ndcg@5"] for r in all_results]),
        "mrr":         np.mean([r["mrr"] for r in all_results]),
        "ctx_recall":  np.mean([r["context_recall"] for r in all_results]),
        "ctx_prec":    np.mean([r["context_precision"] for r in all_results]),
        "faithfulness":np.mean([r["faithfulness"] for r in all_results]),
        "ans_rel":     np.mean([r["answer_relevance"] for r in all_results]),
    }

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                 FULL RAG EVALUATION REPORT                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  RETRIEVAL METRICS (how well the retriever finds docs)              ║
║  ─────────────────────────────────────────────────────              ║
║  Precision@5    = {metrics['precision_5']:.3f}   {'✅' if metrics['precision_5'] >= 0.6 else '⚠'} (of 5 fetched, how many relevant)   ║
║  Recall@5       = {metrics['recall_5']:.3f}   {'✅' if metrics['recall_5'] >= 0.7 else '⚠'} (of all relevant, how many found)  ║
║  NDCG@5         = {metrics['ndcg_5']:.3f}   {'✅' if metrics['ndcg_5'] >= 0.7 else '⚠'} (best docs ranked at top?)        ║
║  MRR            = {metrics['mrr']:.3f}   {'✅' if metrics['mrr'] >= 0.7 else '⚠'} (first relevant doc found early?) ║
║                                                                      ║
║  RAG GENERATION METRICS (quality of full pipeline output)           ║
║  ──────────────────────────────────────────────────────             ║
║  Context Recall    = {metrics['ctx_recall']:.3f}   {'✅' if metrics['ctx_recall'] >= 0.7 else '⚠'} (context had all needed info?)   ║
║  Context Precision = {metrics['ctx_prec']:.3f}   {'✅' if metrics['ctx_prec'] >= 0.7 else '⚠'} (context had low noise?)          ║
║  Faithfulness      = {metrics['faithfulness']:.3f}   {'✅' if metrics['faithfulness'] >= 0.8 else '⚠'} (no hallucinations?)            ║
║  Answer Relevance  = {metrics['ans_rel']:.3f}   {'✅' if metrics['ans_rel'] >= 0.7 else '⚠'} (answer on-topic?)               ║
╠══════════════════════════════════════════════════════════════════════╣
║  DIAGNOSIS & FIXES                                                  ║""")

    issues = []
    if metrics["precision_5"] < 0.6:
        issues.append("║  ⚠ Low Precision  → Add reranking or reduce K                       ║")
    if metrics["recall_5"] < 0.7:
        issues.append("║  ⚠ Low Recall     → Add BM25 hybrid search, increase K              ║")
    if metrics["ndcg_5"] < 0.7:
        issues.append("║  ⚠ Low NDCG       → Reranker needed (best docs not ranked first)    ║")
    if metrics["ctx_recall"] < 0.7:
        issues.append("║  ⚠ Low Ctx Recall → Retriever missing docs (check chunking/indexing)║")
    if metrics["ctx_prec"] < 0.7:
        issues.append("║  ⚠ Low Ctx Prec   → Too much noise → use reranking/metadata filter  ║")
    if metrics["faithfulness"] < 0.8:
        issues.append("║  ⚠ Low Faithful   → LLM hallucinating → stricter system prompt      ║")
    if metrics["ans_rel"] < 0.7:
        issues.append("║  ⚠ Low Ans Rel    → Off-topic answers → better query understanding  ║")

    if not issues:
        print("║  ✅ All metrics look good! RAG pipeline is performing well.         ║")
    else:
        for issue in issues:
            print(issue)

    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Metric decision tree
    print("""
  HOW TO USE THESE METRICS TO FIX YOUR RAG:

  Step 1: Check Context Recall first
    Low?  → Your retriever doesn't find relevant docs
    Fix:    Use hybrid search (BM25 + dense), increase K, fix chunking

  Step 2: Check Context Precision
    Low?  → Too much noise in retrieved context
    Fix:    Add reranker, metadata filtering, decrease K

  Step 3: Check Faithfulness
    Low?  → LLM is adding info not in context (hallucinating)
    Fix:    Prompt: "only use provided context", lower temperature

  Step 4: Check Answer Relevance
    Low?  → LLM answers something else or is too vague
    Fix:    Better system prompt, query rewriting

  Key insight:
    Context Recall/Precision = retriever problem
    Faithfulness/Answer Relevance = generator (LLM) problem
""")


# ─────────────────────────────────────────────────────────
# MAIN: END-TO-END EVALUATION
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  FULL RAG EVALUATION PIPELINE")
    print("=" * 70)

    # Phase 1: Build the RAG system
    print("\nPhase 1: Building RAG index...")
    build_index(CORPUS)

    # Phase 2: Run RAG + Evaluate each query
    print("\nPhase 2: Running RAG + Evaluation...")
    all_results = []
    K = 5

    for query, ground_truth in GROUND_TRUTH_ANSWERS.items():
        print(f"\n  Query: {query[:55]}...")

        # RAG: Retrieve
        retrieved = retrieve(query, k=K)
        retrieved_ids = [d["id"] for d in retrieved]
        relevant = RELEVANT_DOCS[query]

        # RAG: Generate
        answer = generate_answer(query, retrieved)

        # IR Metrics
        p5  = precision_at_k(relevant, retrieved_ids, K)
        r5  = recall_at_k(relevant, retrieved_ids, K)
        n5  = ndcg_at_k(retrieved_ids, relevant, K)
        rr  = mrr(retrieved_ids, relevant)

        print(f"    IR:  P@5={p5:.3f}  R@5={r5:.3f}  NDCG@5={n5:.3f}  MRR={rr:.3f}")

        # RAG Metrics
        print("    Evaluating with GPT-as-Judge...")
        cr = eval_context_recall(query, ground_truth, retrieved)
        cp = eval_context_precision(query, retrieved)
        fa = eval_faithfulness(query, answer, retrieved)
        ar = eval_answer_relevance(query, answer)

        print(f"    RAG: CtxRecall={cr:.3f}  CtxPrec={cp:.3f}  Faith={fa:.3f}  AnsRel={ar:.3f}")

        all_results.append({
            "query":            query,
            "retrieved_ids":    retrieved_ids,
            "answer":           answer,
            "precision@5":      p5,
            "recall@5":         r5,
            "ndcg@5":           n5,
            "mrr":              rr,
            "context_recall":   cr,
            "context_precision":cp,
            "faithfulness":     fa,
            "answer_relevance": ar,
        })

    # Phase 3: Print per-query table
    print(f"\n\n{'='*75}")
    print(f"  {'Query':<30} {'P@5':>5} {'R@5':>5} {'NDCG':>5} {'CR':>5} {'CP':>5} {'FA':>5} {'AR':>5}")
    print(f"{'─'*75}")
    for r in all_results:
        q = r["query"][:28]
        print(
            f"  {q:<30} {r['precision@5']:>5.2f} {r['recall@5']:>5.2f} "
            f"{r['ndcg@5']:>5.2f} {r['context_recall']:>5.2f} "
            f"{r['context_precision']:>5.2f} {r['faithfulness']:>5.2f} "
            f"{r['answer_relevance']:>5.2f}"
        )

    # Phase 4: Diagnostic report
    print_diagnostic_report(all_results)
