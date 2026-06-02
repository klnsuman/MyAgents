"""
LLM Evaluation — Part 2: RAG-Specific Metrics (RAGAS-style)
=============================================================
These metrics evaluate the FULL RAG pipeline — both retrieval AND generation.
They use GPT-4o-mini as a judge (LLM-as-a-Judge pattern).

Metrics covered:
  1. Context Recall      — Did the retrieved context contain ALL needed info?
  2. Context Precision   — Was the retrieved context mostly relevant (no noise)?
  3. Faithfulness        — Is the answer grounded in context? (no hallucinations)
  4. Answer Relevance    — Does the answer actually address the question?

The Pipeline being evaluated:
  ┌─────────┐    ┌────────────────┐    ┌──────────┐    ┌────────┐
  │  Query  │───▶│   Retriever    │───▶│ Context  │───▶│  LLM  │───▶ Answer
  └─────────┘    └────────────────┘    └──────────┘    └────────┘
                       ↑                    ↑                ↑
                 Precision/Recall      Context        Faithfulness /
                 @K (Part 1)          Precision/      Answer Relevance
                                       Recall

Requirements: openai, numpy, python-dotenv
"""

import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────────────────
# EVALUATION DATASET
# Each case has:
#   query          : user question
#   ground_truth   : the correct answer (human written)
#   retrieved_ctx  : what the RAG retriever returned
#   generated_ans  : what the LLM generated
# ─────────────────────────────────────────────────────────
EVAL_DATASET = [
    {
        "query": "What is RAG and why does it reduce hallucinations?",
        "ground_truth": (
            "RAG (Retrieval-Augmented Generation) combines a retrieval system "
            "with a language model. The retriever finds relevant documents from a "
            "knowledge base. The LLM uses those documents as context to generate answers. "
            "RAG reduces hallucinations because the LLM is grounded in retrieved facts "
            "rather than relying solely on parametric memory."
        ),
        "retrieved_ctx": [
            "RAG stands for Retrieval-Augmented Generation. It was introduced by Meta AI in 2020.",
            "RAG combines a dense retriever with a seq2seq generator like BART.",
            "The weather in Paris today is sunny with 22 degrees Celsius.",   # ← irrelevant noise
            "RAG reduces hallucinations by grounding the LLM in retrieved external documents.",
            "The LLM uses retrieved context instead of relying on parametric memory.",
        ],
        "generated_ans": (
            "RAG (Retrieval-Augmented Generation) is a technique that combines a retriever "
            "with a language model. The retriever fetches relevant documents, and the LLM "
            "uses them as context. This reduces hallucinations because the model is grounded "
            "in factual retrieved content rather than just its training data."
        ),
    },
    {
        "query": "How does the Transformer architecture work?",
        "ground_truth": (
            "The Transformer uses self-attention to process sequences in parallel. "
            "It has an encoder and decoder, each with multi-head attention layers. "
            "Self-attention computes Q, K, V matrices. BERT uses the encoder only; "
            "GPT uses the decoder only."
        ),
        "retrieved_ctx": [
            "The Transformer architecture was introduced in 'Attention is All You Need' (2017).",
            "Self-attention uses Query, Key, Value matrices: Attention = softmax(QK^T/sqrt(d_k)) * V",
            "BERT uses only the Transformer encoder. GPT uses only the decoder.",
        ],
        "generated_ans": (
            "The Transformer uses self-attention mechanisms. It was introduced in 2017. "
            "The attention formula is softmax(QK^T/sqrt(d_k)) * V. "
            "BERT uses the encoder while GPT uses the decoder. "
            "Additionally, the Transformer can process text in 47 languages simultaneously."  # ← hallucination
        ),
    },
    {
        "query": "What is the difference between PyTorch and TensorFlow?",
        "ground_truth": (
            "PyTorch uses dynamic computation graphs (define-by-run), making it "
            "more flexible for research. TensorFlow originally used static graphs but "
            "TF2 introduced eager execution. PyTorch was created by Meta AI; "
            "TensorFlow by Google Brain."
        ),
        "retrieved_ctx": [
            "PyTorch was developed by Meta AI and uses dynamic computation graphs.",
            "TensorFlow was developed by Google Brain. TF2 uses eager execution by default.",
            "Dynamic graphs (PyTorch) are more flexible; static graphs (TF1) are faster to deploy.",
        ],
        "generated_ans": (
            "PyTorch, made by Meta AI, uses dynamic computation graphs which are flexible for research. "
            "TensorFlow, made by Google Brain, originally used static graphs but TF2 added eager execution. "
            "PyTorch is generally preferred for research; TensorFlow for production deployment."
        ),
    },
]


# ─────────────────────────────────────────────────────────
# UTILITY: GPT Judge call
# ─────────────────────────────────────────────────────────
def gpt_judge(prompt: str, response_format: str = "json") -> dict:
    """Call GPT-4o-mini as a judge. Returns parsed JSON."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, objective evaluator. "
                    "Always respond with valid JSON only. No extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


# ─────────────────────────────────────────────────────────
# 1. CONTEXT RECALL
# ─────────────────────────────────────────────────────────
"""
WHAT IT ANSWERS:
  "Did the retrieved context contain ALL the information needed to answer?"

HOW IT WORKS:
  1. Take the ground truth answer (human-written correct answer)
  2. Break it into individual statements/sentences
  3. For each statement, ask GPT: "Can this be found in the retrieved context?"
  4. Score = statements_attributable_to_context / total_statements

FORMULA:
  Context Recall = |statements in ground truth attributable to context|
                   ─────────────────────────────────────────────────────
                          |total statements in ground truth|

EXAMPLE:
  Ground truth: "RAG retrieves docs. It reduces hallucinations."
  Context: "RAG retrieves relevant documents from knowledge base."

  Statement 1: "RAG retrieves docs"          → found in context ✅
  Statement 2: "It reduces hallucinations"   → NOT in context  ❌

  Context Recall = 1/2 = 0.50

WHY IT MATTERS:
  Low Context Recall = your retriever is MISSING critical information
  The LLM can't answer what it doesn't have in context
"""
def context_recall(query: str, ground_truth: str, retrieved_ctx: list[str]) -> dict:
    context_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(retrieved_ctx)])

    prompt = f"""Evaluate if each statement in the ground truth answer can be inferred from the retrieved context.

Question: {query}

Ground Truth Answer: {ground_truth}

Retrieved Context:
{context_str}

For each statement in the ground truth, determine if it can be attributed to the retrieved context.

Return JSON:
{{
  "statements": [
    {{"statement": "...", "attributable": true/false, "reason": "..."}}
  ],
  "score": <float 0-1>,
  "explanation": "..."
}}"""

    result = gpt_judge(prompt)
    return result


# ─────────────────────────────────────────────────────────
# 2. CONTEXT PRECISION
# ─────────────────────────────────────────────────────────
"""
WHAT IT ANSWERS:
  "Was each retrieved chunk actually needed to answer the question?"
  (signal-to-noise ratio of your retrieved context)

HOW IT WORKS:
  1. For each retrieved chunk (in order), ask GPT:
     "Is this chunk relevant to answering the question?"
  2. Give higher weight to relevant chunks that appear EARLY (position matters)
  3. Score = weighted precision across all positions

FORMULA (weighted by position):
  Context Precision@K = Σ(precision_at_k * relevance_at_k) / |relevant chunks|

  Where precision_at_k = proportion of relevant chunks in top-k

EXAMPLE:
  Retrieved: [C1(relevant), C2(noise), C3(relevant), C4(noise), C5(noise)]

  Precision@1 = 1/1 = 1.0  → C1 relevant → contributes 1.0 * 1 = 1.0
  Precision@2 = 1/2 = 0.5  → C2 not relevant → no contribution
  Precision@3 = 2/3 = 0.67 → C3 relevant → contributes 0.67 * 1 = 0.67
  Precision@4 = 2/4 = 0.5  → C4 not relevant → no contribution
  Precision@5 = 2/5 = 0.4  → C5 not relevant → no contribution

  Context Precision = (1.0 + 0.67) / 2 = 0.835

WHY IT MATTERS:
  Low Context Precision = context is full of noise
  Noise wastes tokens and can confuse the LLM
"""
def context_precision(query: str, retrieved_ctx: list[str]) -> dict:
    # Step 1: Get relevance for each chunk
    chunks_str = "\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(retrieved_ctx)])

    prompt = f"""For each retrieved chunk, determine if it is relevant to answering the question.

Question: {query}

Retrieved Chunks:
{chunks_str}

Return JSON:
{{
  "chunks": [
    {{"chunk_index": 1, "relevant": true/false, "reason": "..."}}
  ]
}}"""

    result = gpt_judge(prompt)
    relevance_flags = [c["relevant"] for c in result.get("chunks", [])]

    # Step 2: Compute weighted precision (position-aware)
    weighted_sum = 0.0
    relevant_count = 0

    for k, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            relevant_count += 1
            precision_k = relevant_count / k
            weighted_sum += precision_k

    score = weighted_sum / relevant_count if relevant_count > 0 else 0.0

    result["score"] = round(score, 4)
    result["relevant_count"] = relevant_count
    result["total_chunks"] = len(relevance_flags)
    result["relevance_flags"] = relevance_flags
    return result


# ─────────────────────────────────────────────────────────
# 3. FAITHFULNESS
# ─────────────────────────────────────────────────────────
"""
WHAT IT ANSWERS:
  "Is every claim in the answer supported by the retrieved context?"
  (hallucination detection)

HOW IT WORKS:
  1. Extract all atomic claims/statements from the generated answer
  2. For each claim, check if the retrieved context supports it
  3. Score = supported_claims / total_claims

FORMULA:
  Faithfulness = |claims supported by context|
                 ──────────────────────────────
                      |total claims in answer|

EXAMPLE:
  Answer: "RAG retrieves docs. It reduces hallucinations. It also reads minds."
  Context: "RAG retrieves docs. It reduces hallucinations."

  Claim 1: "RAG retrieves docs"          → supported ✅
  Claim 2: "reduces hallucinations"      → supported ✅
  Claim 3: "reads minds"                 → NOT in context → HALLUCINATION ❌

  Faithfulness = 2/3 = 0.67

WHY IT MATTERS:
  Low Faithfulness = your LLM is hallucinating beyond the given context
  This is the most critical metric for trustworthy RAG systems
"""
def faithfulness(query: str, generated_ans: str, retrieved_ctx: list[str]) -> dict:
    context_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(retrieved_ctx)])

    # Step 1: Extract claims
    claims_prompt = f"""Extract all individual factual claims from this answer.
Each claim should be atomic (one fact per claim).

Answer: {generated_ans}

Return JSON: {{"claims": ["claim1", "claim2", ...]}}"""

    claims_result = gpt_judge(claims_prompt)
    claims = claims_result.get("claims", [])

    if not claims:
        return {"score": 0.0, "claims": [], "explanation": "No claims extracted"}

    # Step 2: Verify each claim against context
    verify_prompt = f"""For each claim, determine if it is supported by the retrieved context.

Question: {query}

Retrieved Context:
{context_str}

Claims to verify:
{json.dumps(claims, indent=2)}

Return JSON:
{{
  "verified_claims": [
    {{"claim": "...", "supported": true/false, "reason": "..."}}
  ]
}}"""

    verify_result = gpt_judge(verify_prompt)
    verified = verify_result.get("verified_claims", [])

    supported = sum(1 for c in verified if c.get("supported", False))
    score = supported / len(verified) if verified else 0.0

    return {
        "score": round(score, 4),
        "total_claims": len(verified),
        "supported_claims": supported,
        "hallucinated_claims": len(verified) - supported,
        "verified_claims": verified,
    }


# ─────────────────────────────────────────────────────────
# 4. ANSWER RELEVANCE
# ─────────────────────────────────────────────────────────
"""
WHAT IT ANSWERS:
  "Does the generated answer actually address the question asked?"

HOW IT WORKS (reverse question generation):
  1. Generate N questions from the answer (what question does this answer?)
  2. Embed each generated question
  3. Compute cosine similarity to the original question
  4. Score = average cosine similarity

INTUITION:
  If an answer is relevant, questions generated from it should match the original.
  A vague/off-topic answer generates questions that diverge from the original.

FORMULA:
  Answer Relevance = (1/N) * Σ cosine_similarity(original_question, generated_question_i)

WHY IT MATTERS:
  LLM might give a correct but off-topic answer
  E.g., Q: "What is BERT?" A: "GPT is a decoder-only model" → not relevant!
"""
def get_embedding(text: str) -> list[float]:
    r = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return r.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


def answer_relevance(query: str, generated_ans: str, n_questions: int = 3) -> dict:
    # Step 1: Generate N questions from the answer
    gen_prompt = f"""Given this answer, generate {n_questions} different questions that this answer addresses.
The questions should capture what the answer is fundamentally about.

Answer: {generated_ans}

Return JSON: {{"questions": ["q1", "q2", "q3"]}}"""

    gen_result = gpt_judge(gen_prompt)
    generated_questions = gen_result.get("questions", [])

    if not generated_questions:
        return {"score": 0.0, "generated_questions": []}

    # Step 2: Embed original question and generated questions
    original_emb = get_embedding(query)
    similarities = []
    for gq in generated_questions:
        gq_emb = get_embedding(gq)
        sim = cosine_similarity(original_emb, gq_emb)
        similarities.append({"question": gq, "similarity": round(sim, 4)})

    score = np.mean([s["similarity"] for s in similarities])

    return {
        "score": round(float(score), 4),
        "original_question": query,
        "generated_questions": similarities,
        "explanation": (
            f"Score {score:.3f}: "
            + ("High relevance — answer addresses the question." if score > 0.8
               else "Medium relevance — partially addresses the question." if score > 0.6
               else "Low relevance — answer diverges from the question.")
        ),
    }


# ─────────────────────────────────────────────────────────
# 5. FULL RAG EVALUATION RUNNER
# ─────────────────────────────────────────────────────────
def evaluate_rag_case(case: dict) -> dict:
    """Run all 4 RAG metrics on a single evaluation case."""
    print(f"\n  Query: {case['query'][:60]}...")

    print("    [1/4] Context Recall...")
    cr = context_recall(case["query"], case["ground_truth"], case["retrieved_ctx"])

    print("    [2/4] Context Precision...")
    cp = context_precision(case["query"], case["retrieved_ctx"])

    print("    [3/4] Faithfulness...")
    faith = faithfulness(case["query"], case["generated_ans"], case["retrieved_ctx"])

    print("    [4/4] Answer Relevance...")
    ar = answer_relevance(case["query"], case["generated_ans"])

    return {
        "query": case["query"],
        "context_recall":    cr.get("score", 0.0),
        "context_precision": cp.get("score", 0.0),
        "faithfulness":      faith.get("score", 0.0),
        "answer_relevance":  ar.get("score", 0.0),
        "details": {
            "context_recall":    cr,
            "context_precision": cp,
            "faithfulness":      faith,
            "answer_relevance":  ar,
        },
    }


def print_results_table(all_results: list[dict]) -> None:
    print(f"\n{'='*75}")
    print("  RAG EVALUATION RESULTS")
    print(f"{'='*75}")
    print(f"  {'Query':<38} {'C.Recall':>9} {'C.Prec':>8} {'Faith':>7} {'A.Rel':>7}")
    print(f"{'─'*75}")

    cr_all, cp_all, fa_all, ar_all = [], [], [], []
    for r in all_results:
        q = r["query"][:36]
        cr = r["context_recall"]
        cp = r["context_precision"]
        fa = r["faithfulness"]
        ar = r["answer_relevance"]
        print(f"  {q:<38} {cr:>9.3f} {cp:>8.3f} {fa:>7.3f} {ar:>7.3f}")
        cr_all.append(cr); cp_all.append(cp); fa_all.append(fa); ar_all.append(ar)

    print(f"{'─'*75}")
    avg_cr = np.mean(cr_all)
    avg_cp = np.mean(cp_all)
    avg_fa = np.mean(fa_all)
    avg_ar = np.mean(ar_all)
    print(f"  {'AVERAGE':<38} {avg_cr:>9.3f} {avg_cp:>8.3f} {avg_fa:>7.3f} {avg_ar:>7.3f}")

    print(f"\n  Diagnosis:")
    if avg_cr < 0.7:
        print(f"  ⚠ Context Recall={avg_cr:.2f}  → Retriever is MISSING relevant docs")
    else:
        print(f"  ✓ Context Recall={avg_cr:.2f}  → Retriever covers needed information")

    if avg_cp < 0.7:
        print(f"  ⚠ Context Precision={avg_cp:.2f} → Too much noise in retrieved context")
    else:
        print(f"  ✓ Context Precision={avg_cp:.2f} → Retrieved context is clean and focused")

    if avg_fa < 0.8:
        print(f"  ⚠ Faithfulness={avg_fa:.2f}     → LLM is HALLUCINATING beyond context")
    else:
        print(f"  ✓ Faithfulness={avg_fa:.2f}     → Answers are grounded in context")

    if avg_ar < 0.7:
        print(f"  ⚠ Answer Relevance={avg_ar:.2f} → Answers are off-topic")
    else:
        print(f"  ✓ Answer Relevance={avg_ar:.2f} → Answers address the questions")


def print_metric_summary() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         RAG METRICS CHEAT SHEET (LLM-as-Judge)                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Context Recall                                                      ║
║  ├─ What: Did retriever find ALL info needed to answer?             ║
║  ├─ Low score means: Missing relevant documents                     ║
║  └─ Fix: Improve retriever (hybrid search, better chunking)         ║
║                                                                      ║
║  Context Precision                                                   ║
║  ├─ What: Is retrieved context mostly relevant (no noise)?          ║
║  ├─ Low score means: Retriever fetches too many irrelevant docs     ║
║  └─ Fix: Reranking, metadata filtering, smaller K                   ║
║                                                                      ║
║  Faithfulness                                                        ║
║  ├─ What: Is every answer claim supported by context?               ║
║  ├─ Low score means: LLM is hallucinating                           ║
║  └─ Fix: Stricter system prompt, lower temperature, citation req.   ║
║                                                                      ║
║  Answer Relevance                                                    ║
║  ├─ What: Does the answer address the actual question?              ║
║  ├─ Low score means: Answer is off-topic or too generic             ║
║  └─ Fix: Better prompt engineering, query rewriting                 ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Metric        Evaluates         Low Score Fix                      ║
║  ─────────     ───────────       ────────────────                   ║
║  C.Recall      Retriever         Better retrieval                   ║
║  C.Precision   Retriever         Reranking / filtering              ║
║  Faithfulness  Generator (LLM)   Prompt engineering                 ║
║  A.Relevance   Generator (LLM)   Query understanding                ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_metric_summary()

    print("=" * 70)
    print("  RAG EVALUATION — Running all 4 metrics on 3 test cases")
    print("=" * 70)

    all_results = []
    for case in EVAL_DATASET:
        result = evaluate_rag_case(case)
        all_results.append(result)

    print_results_table(all_results)

    # Detailed breakdown for first case
    print(f"\n\n  DETAILED BREAKDOWN — Case 1")
    print(f"  Query: {EVAL_DATASET[0]['query']}")
    print(f"  {'─'*60}")

    r = all_results[0]["details"]

    print(f"\n  Context Recall (score={all_results[0]['context_recall']:.3f}):")
    for s in r["context_recall"].get("statements", []):
        icon = "✅" if s["attributable"] else "❌"
        print(f"    {icon} {s['statement'][:60]}")

    print(f"\n  Context Precision (score={all_results[0]['context_precision']:.3f}):")
    for c in r["context_precision"].get("chunks", []):
        icon = "✅" if c["relevant"] else "❌"
        print(f"    {icon} Chunk {c['chunk_index']}: {c['reason'][:55]}")

    print(f"\n  Faithfulness (score={all_results[0]['faithfulness']:.3f}):")
    for c in r["faithfulness"].get("verified_claims", []):
        icon = "✅" if c["supported"] else "🚨"
        print(f"    {icon} {c['claim'][:60]}")

    print(f"\n  Answer Relevance (score={all_results[0]['answer_relevance']:.3f}):")
    for gq in r["answer_relevance"].get("generated_questions", []):
        print(f"    sim={gq['similarity']:.3f}  {gq['question'][:55]}")
