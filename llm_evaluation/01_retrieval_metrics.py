"""
LLM Evaluation — Part 1: Retrieval Metrics
============================================
Traditional Information Retrieval (IR) metrics.
These measure HOW WELL your retriever finds relevant documents.
They do NOT care about the LLM answer — only what was fetched.

Metrics covered:
  1. Precision@K   — Of K retrieved docs, how many are relevant?
  2. Recall@K      — Of all relevant docs, how many did we fetch in top-K?
  3. F1@K          — Balance between Precision and Recall
  4. NDCG@K        — Position-aware: relevant docs ranked higher score more
  5. MRR           — Mean Reciprocal Rank: where is the FIRST relevant doc?

Visual summary:
  ┌─────────────────────────────────────────────────┐
  │  Ground truth relevant: {D1, D3, D5}            │
  │  Retrieved top-5:       [D1, D2, D3, D4, D6]   │
  │                                                 │
  │  Precision@5 = 2/5 = 0.40  (D1,D3 relevant)   │
  │  Recall@5    = 2/3 = 0.67  (missed D5)         │
  │  F1@5        = 0.50                             │
  │  NDCG@5      = rewards D1 at rank-1 more       │
  │  MRR         = 1/1 = 1.0   (D1 at rank-1)     │
  └─────────────────────────────────────────────────┘

Requirements: openai, numpy, python-dotenv
"""

import os
import math
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────────────────
# SAMPLE DATA: Simulated retrieval results
# ─────────────────────────────────────────────────────────
#
# In a real system:
#   - relevant_docs  = labeled by humans (ground truth)
#   - retrieved_docs = what your RAG retriever returned
#
EVALUATION_CASES = [
    {
        "query": "What is RAG and how does it reduce hallucinations?",
        "relevant_docs": ["d1", "d2", "d5"],        # ground truth (human labeled)
        "retrieved_docs": ["d1", "d3", "d2", "d6", "d5"],  # what retriever returned
        "relevance_scores": {                        # graded relevance (0=none, 1=partial, 2=perfect)
            "d1": 2, "d2": 2, "d3": 0, "d5": 1, "d6": 0
        },
    },
    {
        "query": "How does BM25 work?",
        "relevant_docs": ["d7", "d8"],
        "retrieved_docs": ["d3", "d7", "d1", "d8", "d9"],
        "relevance_scores": {
            "d3": 0, "d7": 2, "d1": 0, "d8": 1, "d9": 0
        },
    },
    {
        "query": "What is the Transformer architecture?",
        "relevant_docs": ["d10", "d11", "d12"],
        "retrieved_docs": ["d10", "d11", "d4", "d12", "d2"],
        "relevance_scores": {
            "d10": 2, "d11": 2, "d4": 0, "d12": 2, "d2": 0
        },
    },
]


# ─────────────────────────────────────────────────────────
# 1. PRECISION@K
# ─────────────────────────────────────────────────────────
"""
WHAT IT ANSWERS: "Of the K documents I fetched, how many were actually relevant?"

Formula: Precision@K = |{Relevant} ∩ {Retrieved_top_K}| / K

Example:
  Relevant = {D1, D3, D5}
  Retrieved = [D1, D2, D3, D4, D6]   (K=5)
  Relevant in top-5 = {D1, D3} → 2 docs

  Precision@5 = 2/5 = 0.40

WHY IT MATTERS:
  High precision = your retriever is not noisy
  Low precision  = retriever fetches many irrelevant docs (wasting LLM context)

TRADE-OFF:
  Precision@1 is very strict (only checks top-1 result)
  Precision@10 is more lenient (larger retrieved set)
"""
def precision_at_k(relevant_docs: set, retrieved_docs: list, k: int) -> float:
    retrieved_k = retrieved_docs[:k]
    relevant_in_k = sum(1 for doc in retrieved_k if doc in relevant_docs)
    return relevant_in_k / k if k > 0 else 0.0


# ─────────────────────────────────────────────────────────
# 2. RECALL@K
# ─────────────────────────────────────────────────────────
"""
WHAT IT ANSWERS: "Of ALL relevant documents that exist, how many did I find?"

Formula: Recall@K = |{Relevant} ∩ {Retrieved_top_K}| / |{Relevant}|

Example:
  Relevant = {D1, D3, D5}           → 3 relevant docs exist
  Retrieved = [D1, D2, D3, D4, D6]  (K=5)
  Found = {D1, D3}                  → found 2, missed D5

  Recall@5 = 2/3 = 0.67

WHY IT MATTERS:
  High recall = you found most of the relevant docs
  Low recall  = relevant docs are being missed → LLM won't have needed info

TRADE-OFF with Precision:
  Increase K → recall goes up, precision goes down (more noise)
  Decrease K → precision goes up, recall goes down (miss relevant docs)
"""
def recall_at_k(relevant_docs: set, retrieved_docs: list, k: int) -> float:
    if not relevant_docs:
        return 0.0
    retrieved_k = set(retrieved_docs[:k])
    relevant_found = len(relevant_docs & retrieved_k)
    return relevant_found / len(relevant_docs)


# ─────────────────────────────────────────────────────────
# 3. F1@K
# ─────────────────────────────────────────────────────────
"""
WHAT IT ANSWERS: "What is the harmonic balance between Precision and Recall?"

Formula: F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

WHY HARMONIC MEAN (not average)?
  If Precision=1.0 and Recall=0.0:
    Average = 0.5  (misleadingly optimistic)
    Harmonic = 0.0 (correctly shows failure)

  Harmonic mean punishes extreme imbalances.
"""
def f1_at_k(relevant_docs: set, retrieved_docs: list, k: int) -> float:
    p = precision_at_k(relevant_docs, retrieved_docs, k)
    r = recall_at_k(relevant_docs, retrieved_docs, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ─────────────────────────────────────────────────────────
# 4. NDCG@K (Normalized Discounted Cumulative Gain)
# ─────────────────────────────────────────────────────────
"""
WHAT IT ANSWERS: "Are the most relevant documents ranked at the TOP?"

Problem with Precision@K / Recall@K:
  They treat all positions equally.
  But rank-1 is FAR more important than rank-5 for the LLM context.

NDCG rewards:
  - Highly relevant docs appearing early (rank 1, 2, 3)
  - Penalizes highly relevant docs appearing late

Formula:
  DCG@K = Σ (rel_i / log2(i+1))  for i = 1 to K
  IDCG@K = DCG of the ideal (perfect) ranking
  NDCG@K = DCG@K / IDCG@K

Example with graded relevance (0=not relevant, 1=partial, 2=highly relevant):

  Retrieved: [D1(rel=2), D2(rel=0), D3(rel=2), D4(rel=0), D5(rel=1)]

  DCG@5 = 2/log2(2) + 0/log2(3) + 2/log2(4) + 0/log2(5) + 1/log2(6)
        = 2/1 + 0 + 2/2 + 0 + 1/2.58
        = 2 + 0 + 1 + 0 + 0.387 = 3.387

  Ideal ranking = [D1(2), D3(2), D5(1), D2(0), D4(0)]
  IDCG@5 = 2/1 + 2/1.58 + 1/2 + 0 + 0 = 2 + 1.26 + 0.5 = 3.76

  NDCG@5 = 3.387 / 3.76 = 0.90
"""
def dcg_at_k(relevance_scores: list, k: int) -> float:
    """Compute Discounted Cumulative Gain for a ranked list of relevance scores."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0, ranks start at 1
    return dcg


def ndcg_at_k(retrieved_docs: list, relevance_map: dict, k: int) -> float:
    """
    retrieved_docs: ordered list of doc IDs from retriever
    relevance_map:  {doc_id: relevance_score} (0, 1, or 2)
    """
    # Actual DCG
    actual_rels = [relevance_map.get(doc, 0) for doc in retrieved_docs[:k]]
    actual_dcg = dcg_at_k(actual_rels, k)

    # Ideal DCG (sort all known docs by relevance descending)
    ideal_rels = sorted(relevance_map.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_rels, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ─────────────────────────────────────────────────────────
# 5. MRR (Mean Reciprocal Rank)
# ─────────────────────────────────────────────────────────
"""
WHAT IT ANSWERS: "Where does the FIRST relevant document appear in the ranked list?"

Formula:
  RR (Reciprocal Rank) = 1 / rank_of_first_relevant_doc
  MRR = average RR across all queries

Example:
  Query 1: Retrieved = [D2, D1, D3]  → D1 is relevant, found at rank 2 → RR = 1/2 = 0.5
  Query 2: Retrieved = [D7, D8, D5]  → D7 is relevant, found at rank 1 → RR = 1/1 = 1.0
  Query 3: Retrieved = [D4, D2, D9]  → none relevant in top-3         → RR = 0

  MRR = (0.5 + 1.0 + 0) / 3 = 0.5

WHY IT MATTERS:
  Useful when you only care about finding AT LEAST ONE relevant doc quickly.
  Common in Q&A: you just need the first correct document.
"""
def reciprocal_rank(relevant_docs: set, retrieved_docs: list) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(cases: list[dict]) -> float:
    rrs = []
    for case in cases:
        rr = reciprocal_rank(set(case["relevant_docs"]), case["retrieved_docs"])
        rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


# ─────────────────────────────────────────────────────────
# 6. EVALUATION RUNNER
# ─────────────────────────────────────────────────────────
def evaluate_retrieval(cases: list[dict], k_values: list[int] = [1, 3, 5]) -> dict:
    """Run all retrieval metrics across all evaluation cases."""
    results = {}

    for k in k_values:
        precisions, recalls, f1s, ndcgs = [], [], [], []

        for case in cases:
            relevant = set(case["relevant_docs"])
            retrieved = case["retrieved_docs"]
            rel_map = case["relevance_scores"]

            precisions.append(precision_at_k(relevant, retrieved, k))
            recalls.append(recall_at_k(relevant, retrieved, k))
            f1s.append(f1_at_k(relevant, retrieved, k))
            ndcgs.append(ndcg_at_k(retrieved, rel_map, k))

        results[k] = {
            f"Precision@{k}": round(np.mean(precisions), 4),
            f"Recall@{k}":    round(np.mean(recalls), 4),
            f"F1@{k}":        round(np.mean(f1s), 4),
            f"NDCG@{k}":      round(np.mean(ndcgs), 4),
        }

    results["MRR"] = round(mean_reciprocal_rank(cases), 4)
    return results


# ─────────────────────────────────────────────────────────
# 7. PER-QUERY BREAKDOWN
# ─────────────────────────────────────────────────────────
def per_query_breakdown(cases: list[dict], k: int = 5) -> None:
    print(f"\n{'─'*70}")
    print(f"  Per-Query Breakdown @ K={k}")
    print(f"{'─'*70}")
    print(f"  {'Query':<40} {'P@K':>6} {'R@K':>6} {'F1@K':>6} {'NDCG@K':>8} {'RR':>6}")
    print(f"{'─'*70}")

    for case in cases:
        relevant = set(case["relevant_docs"])
        retrieved = case["retrieved_docs"]
        rel_map = case["relevance_scores"]
        q = case["query"][:38]

        p = precision_at_k(relevant, retrieved, k)
        r = recall_at_k(relevant, retrieved, k)
        f = f1_at_k(relevant, retrieved, k)
        n = ndcg_at_k(retrieved, rel_map, k)
        rr = reciprocal_rank(relevant, retrieved)

        print(f"  {q:<40} {p:>6.3f} {r:>6.3f} {f:>6.3f} {n:>8.3f} {rr:>6.3f}")


# ─────────────────────────────────────────────────────────
# 8. VISUAL EXPLANATION
# ─────────────────────────────────────────────────────────
def print_visual_explanation() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           RETRIEVAL METRICS CHEAT SHEET                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Setup:                                                          ║
║    Relevant docs (ground truth) = {D1, D3, D5}  (3 exist)      ║
║    Retrieved top-5              = [D1, D2, D3, D4, D6]          ║
║                                   ✅  ❌  ✅  ❌  ❌            ║
║    Graded relevance: D1=2, D3=2, D5=1 (highly/partially rel)   ║
║                                                                  ║
║  Precision@5 = 2/5 = 0.40                                       ║
║  "40% of what I fetched was useful"                             ║
║                                                                  ║
║  Recall@5    = 2/3 = 0.67                                       ║
║  "I found 67% of all relevant docs (missed D5)"                 ║
║                                                                  ║
║  F1@5        = 2*(0.40*0.67)/(0.40+0.67) = 0.50                ║
║  "Balanced score — not great on either dimension"               ║
║                                                                  ║
║  NDCG@5      = ~0.90  (D1 at rank-1 saves the score)           ║
║  "Position-aware: good because best doc is ranked first"        ║
║                                                                  ║
║  MRR         = 1/1 = 1.0  (first hit at rank-1)               ║
║  "Great — found A relevant doc immediately"                     ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  When to use which:                                              ║
║                                                                  ║
║  Precision@K → you care about noise (context window cost)       ║
║  Recall@K    → you care about missing info (LLM completeness)   ║
║  NDCG@K      → you care about ranking quality                   ║
║  MRR         → you need at least 1 good doc quickly             ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_visual_explanation()

    print("=" * 70)
    print("  RETRIEVAL METRICS EVALUATION")
    print("=" * 70)

    results = evaluate_retrieval(EVALUATION_CASES, k_values=[1, 3, 5])

    print(f"\n  Averaged across {len(EVALUATION_CASES)} queries:\n")
    for k, metrics in results.items():
        if k == "MRR":
            print(f"  MRR = {results['MRR']}")
            continue
        print(f"  K={k}:")
        for metric, value in metrics.items():
            bar = "█" * int(value * 20)
            print(f"    {metric:<15} {value:.4f}  {bar}")

    per_query_breakdown(EVALUATION_CASES, k=5)

    print(f"""
  Interpretation:
    Precision@5 = {results[5]['Precision@5']}  → of every 5 fetched docs, ~{results[5]['Precision@5']*5:.1f} are relevant
    Recall@5    = {results[5]['Recall@5']}  → we find ~{results[5]['Recall@5']*100:.0f}% of all relevant docs
    NDCG@5      = {results[5]['NDCG@5']}  → ranking quality (1.0 = perfect ordering)
    MRR         = {results['MRR']}  → on average, first relevant doc at rank ~{1/results['MRR']:.1f}
""")
