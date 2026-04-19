"""
RAG v3 (2025): Agentic RAG
============================
Improvements over v2:
  - Query decomposition: break complex queries into sub-questions
  - Multi-hop retrieval: iteratively retrieve based on partial answers
  - Self-correction: the agent critiques its answer and re-retrieves if needed
  - Tool use: GPT function-calling to orchestrate retrieval, search, and calculation

The agent loop:
  1. Decompose query → sub-questions
  2. For each sub-question → retrieve → partial answer
  3. Synthesize all partial answers → draft answer
  4. Self-critique → if insufficient → re-retrieve → final answer

Requirements: openai, numpy, python-dotenv
"""

import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {"id": "kb_1", "text": "Python was created by Guido van Rossum and released in 1991. It is dynamically typed and uses indentation for code blocks.", "topic": "python"},
    {"id": "kb_2", "text": "Python 3 was released in 2008. It is not backward-compatible with Python 2. Python 2 reached end-of-life in January 2020.", "topic": "python"},
    {"id": "kb_3", "text": "The GIL (Global Interpreter Lock) in CPython prevents true multi-threading for CPU-bound tasks. For parallelism, Python uses multiprocessing or async I/O.", "topic": "python"},
    {"id": "kb_4", "text": "PyTorch is an open-source machine learning framework developed by Meta AI. It uses dynamic computation graphs and is favored for research.", "topic": "ml_frameworks"},
    {"id": "kb_5", "text": "TensorFlow was developed by Google Brain. TensorFlow 2.x uses Keras as its high-level API and supports eager execution by default.", "topic": "ml_frameworks"},
    {"id": "kb_6", "text": "Transformers use self-attention: Q (query), K (key), V (value) matrices. Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V.", "topic": "transformers"},
    {"id": "kb_7", "text": "BERT is a bidirectional Transformer encoder pre-trained on masked language modeling (MLM) and next sentence prediction (NSP).", "topic": "transformers"},
    {"id": "kb_8", "text": "GPT models are autoregressive Transformer decoders. GPT-4 is a multimodal model released by OpenAI in March 2023.", "topic": "transformers"},
    {"id": "kb_9", "text": "RAG (Retrieval-Augmented Generation) was introduced by Lewis et al. in 2020. It uses a dense retriever (DPR) with a seq2seq generator.", "topic": "rag"},
    {"id": "kb_10", "text": "Agentic RAG adds reasoning loops: query decomposition, iterative retrieval, self-reflection, and tool use. It handles complex multi-hop questions.", "topic": "rag"},
    {"id": "kb_11", "text": "Vector embeddings map text to dense numeric vectors. OpenAI's text-embedding-3-large produces 3072-dimensional vectors.", "topic": "embeddings"},
    {"id": "kb_12", "text": "Cosine similarity and dot product are common metrics for embedding similarity. FAISS supports both L2 and inner-product indexes.", "topic": "embeddings"},
    {"id": "kb_13", "text": "LangChain is a framework for building LLM applications with chains, agents, and tools. It supports many vector stores and LLM providers.", "topic": "frameworks"},
    {"id": "kb_14", "text": "LlamaIndex (formerly GPT Index) specializes in connecting LLMs to external data sources via indexes and query engines.", "topic": "frameworks"},
]


# ─────────────────────────────────────────────
# 2. SIMPLE VECTOR STORE (reuse v1 approach)
# ─────────────────────────────────────────────
_vector_store: list[dict] = []


def get_embedding(text: str) -> list[float]:
    r = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return r.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


def build_vector_store(docs: list[dict]) -> None:
    global _vector_store
    print("Indexing knowledge base...")
    _vector_store = [{**d, "embedding": get_embedding(d["text"])} for d in docs]
    print(f"  {len(_vector_store)} documents indexed.")


def retrieve(query: str, k: int = 3, topic_filter: str | None = None) -> list[dict]:
    """Retrieve top-K chunks, with optional topic filter."""
    q_emb = get_embedding(query)
    pool = _vector_store
    if topic_filter:
        pool = [d for d in pool if d.get("topic") == topic_filter]
    scored = sorted(pool, key=lambda d: cosine_similarity(q_emb, d["embedding"]), reverse=True)
    return [{"id": d["id"], "text": d["text"], "topic": d["topic"]} for d in scored[:k]]


# ─────────────────────────────────────────────
# 3. TOOLS (OpenAI function-calling schema)
# ─────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "Search the knowledge base for documents relevant to a query. "
                "Optionally filter by topic (python, ml_frameworks, transformers, rag, embeddings, frameworks)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "description": "Number of documents to retrieve (default 3)"},
                    "topic_filter": {
                        "type": "string",
                        "enum": ["python", "ml_frameworks", "transformers", "rag", "embeddings", "frameworks"],
                        "description": "Optional topic filter",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decompose_query",
            "description": "Break a complex question into simpler sub-questions for multi-hop retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The complex question to decompose"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression (e.g. '1991 + 32' or '3072 * 4').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Python arithmetic expression"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "self_critique",
            "description": (
                "Evaluate whether a draft answer is complete and accurate. "
                "Returns a critique and a boolean 'needs_more_retrieval'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "draft_answer": {"type": "string"},
                },
                "required": ["question", "draft_answer"],
            },
        },
    },
]


# ─────────────────────────────────────────────
# 4. TOOL EXECUTORS
# ─────────────────────────────────────────────
def execute_tool(name: str, args: dict) -> str:
    """Route tool calls to their implementations."""

    if name == "retrieve_documents":
        docs = retrieve(
            query=args["query"],
            k=args.get("k", 3),
            topic_filter=args.get("topic_filter"),
        )
        return json.dumps(docs, ensure_ascii=False)

    elif name == "decompose_query":
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Break the given question into 2-4 simpler, independent sub-questions "
                        "that together answer the original. Return a JSON array of strings."
                    ),
                },
                {"role": "user", "content": args["question"]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = json.loads(resp.choices[0].message.content)
        # Handle both {"questions": [...]} and [...] directly
        questions = raw.get("questions", raw.get("sub_questions", list(raw.values())[0]))
        return json.dumps(questions)

    elif name == "calculate":
        try:
            # Safe eval: only allow arithmetic
            result = eval(args["expression"], {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    elif name == "self_critique":
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict quality reviewer. "
                        "Assess whether the draft answer fully and accurately answers the question. "
                        "Return JSON: {\"critique\": \"...\", \"needs_more_retrieval\": true/false, \"missing_aspects\": [...]}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {args['question']}\n\nDraft Answer: {args['draft_answer']}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return resp.choices[0].message.content

    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────
# 5. AGENTIC RAG LOOP
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert research assistant with access to a knowledge base.

For complex questions:
1. DECOMPOSE: Use decompose_query to split into sub-questions
2. RETRIEVE: Use retrieve_documents for each sub-question
3. REASON: Synthesize retrieved information into a draft answer
4. CRITIQUE: Use self_critique to check completeness
5. REFINE: If critique says needs_more_retrieval=true, retrieve more and improve the answer
6. FINAL: Provide a comprehensive, well-cited answer

Always prefer specific topic_filter when you know the topic.
"""


def agentic_rag(question: str, max_iterations: int = 8) -> str:
    """Run the Agentic RAG loop with tool use, multi-hop retrieval, and self-correction."""
    print(f"\n{'='*60}")
    print(f"RAG v3 Agentic | Question: {question}")
    print("="*60)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    iteration = 0
    retrieved_context: list[str] = []

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=1024,
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))

        # No more tool calls → we have the final answer
        if not msg.tool_calls:
            final_answer = msg.content or "(no answer)"
            print(f"\n[Final Answer]\n{final_answer}")
            return final_answer

        # Process all tool calls in this turn
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            print(f"  Tool: {tool_name}({json.dumps(tool_args)[:120]})")

            result = execute_tool(tool_name, tool_args)

            # Collect retrieved context for logging
            if tool_name == "retrieve_documents":
                docs = json.loads(result)
                for d in docs:
                    retrieved_context.append(d["text"])

            print(f"  Result: {result[:200]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Fallback if max iterations reached
    print("\n[Max iterations reached — generating best-effort answer]")
    context_str = "\n".join(retrieved_context[-9:])
    fallback = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_str}\n\n"
                    f"Answer this question as best you can: {question}"
                ),
            }
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return fallback.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 6. DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    build_vector_store(KNOWLEDGE_BASE)

    questions = [
        # Simple — tests basic retrieval + answer
        "What is RAG and when was it introduced?",

        # Multi-hop — needs Python history + GIL info
        "How does Python handle parallelism, and why was this design choice made?",

        # Complex — needs transformer knowledge + framework knowledge
        "Compare PyTorch and TensorFlow, and explain which is better for Transformer-based models like BERT.",
    ]

    for q in questions:
        answer = agentic_rag(q)
        print(f"\n{'*'*60}")
        print(f"QUESTION: {q}")
        print(f"ANSWER:   {answer[:400]}...")
        print("*"*60)
