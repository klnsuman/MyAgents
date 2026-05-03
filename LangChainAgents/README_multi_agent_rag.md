# Multi-Agent RAG with Langfuse Telemetry

## What this notebook does

Builds a production-style question-answering pipeline where three specialized AI agents collaborate in sequence, backed by an advanced retrieval system and full observability via Langfuse. Designed to demonstrate enterprise-grade AI engineering practices including traceability, cost monitoring, and audit logging.

---

## Why this matters in enterprise AI

In regulated industries like pharma and life sciences, every AI decision must be explainable and auditable. This system addresses that directly:

- **Full trace per run** — every prompt sent, every response received, every retrieval decision is logged
- **Cost visibility** — token usage and API cost per agent call tracked in Langfuse
- **Dual persistence** — Langfuse cloud for real-time monitoring + SQLite locally for audit trail
- **Session-level grouping** — each pipeline run has a unique session ID so you can replay exactly what happened

---

## Architecture overview

```
User Task
   │
   ▼
[Retriever] ──► HNSW search → MMR diversity filter → Dense+Sparse reranking
   │
   ▼
[Researcher Agent] ──► reads retrieved docs, summarizes findings  (gpt-4o-mini)
   │
   ▼
[Analyzer Agent]   ──► critically examines the research            (gpt-4o-mini)
   │
   ▼
[Writer Agent]     ──► produces final structured answer            (gpt-4o-mini)
   │
   ▼
SQLite (chat history + checkpoints) + Langfuse (full LLM traces + costs)
```

---

## Key design decisions and why (interview talking points)

### 1. Why multi-agent instead of a single LLM call?
Single LLM calls collapse research, reasoning, and writing into one step — errors compound and are hard to debug. Separating into Researcher → Analyzer → Writer means each agent has a focused role, outputs are inspectable at each stage, and you can swap or tune one agent without touching the others. In pharma, this separation also supports review workflows where different teams validate different stages.

### 2. Why HNSW for retrieval?
HNSW (Hierarchical Navigable Small World) is an approximate nearest-neighbor algorithm that scales to millions of vectors with sub-millisecond query times. Unlike brute-force cosine search, HNSW uses a graph structure that narrows candidates logarithmically. Critical for production where document stores are large and latency matters.

### 3. Why MMR (Maximal Marginal Relevance)?
Pure similarity search returns redundant results — you get 5 documents that all say the same thing. MMR balances relevance (how similar to the query) vs. diversity (how different from already-selected docs). Controlled by `lambda_mult`: 0 = max diversity, 1 = max relevance. In a clinical document search scenario this means you surface different perspectives, not just the top-ranked duplicate chunks.

### 4. Why Dense + Sparse reranking?
Dense retrieval (embeddings) captures semantic meaning but misses exact keyword matches. Sparse retrieval (BM25) is great for exact terms like drug names or gene IDs but misses synonyms. Combining both with weighted scores (70% dense + 30% sparse) gets the best of both — semantic understanding plus lexical precision. Critical in pharma where domain terms must match exactly.

### 5. Why LangGraph with checkpoints?
LangGraph provides a stateful graph execution model. `MemorySaver` checkpoints the agent state after every node, enabling:
- Multi-turn conversations that remember prior context
- Resuming a failed pipeline from the last checkpoint
- Full state inspection at any step for debugging

### 6. Why Langfuse for observability?
Langfuse captures the full LLM interaction — prompt, completion, token counts, latency, and cost — per agent call. In enterprise settings this answers: "What exactly did the AI see? What did it decide? How much did it cost?" The `@observe()` decorator wraps each function automatically, and `CallbackHandler` hooks into LangChain's execution to capture LLM-level details without manual instrumentation.

---

## Section-by-section breakdown

### Section 1 — Environment Setup & Langfuse
- Loads `.env` for API keys
- Initializes `langfuse_client` (singleton, thread-safe)
- Initializes `CallbackHandler` — this is passed to every LLM call to capture token usage and cost

### Section 2 — SQLite Database
Creates three tables in `rag_system.db`:
| Table | Purpose |
|---|---|
| `documents` | Stores ingested docs with embeddings |
| `chat_history` | Logs every agent message per session |
| `agent_checkpoints` | Saves full agent state at each step |

### Section 3 — AdvancedRetriever (HNSW + MMR)
- Embeddings are computed once at `add_documents()` time and cached — no repeated API calls at query time
- `retrieve_hnsw()` — fast ANN search, returns top-k candidates
- `retrieve_mmr()` — selects diverse subset from candidates using cached embeddings (pure numpy, no extra API calls)

### Section 4 — DenseSpareReranker
- BM25 fitted at `fit()` time; corpus embeddings pre-computed once
- `rerank()` — scores each candidate with `0.7 × dense + 0.3 × sparse`, returns top-k
- Fallback: if a doc isn't in corpus, computes embedding on the fly

### Section 5 — Multi-Agent Nodes
- Each node is a pure function on `AgentState` — easy to test in isolation
- `@observe()` creates a Langfuse span per node automatically
- `CallbackHandler` passed to `chain.invoke()` captures the full LLM prompt + completion + tokens

### Section 6 — LangGraph
- Linear graph: `START → researcher → analyzer → writer → END`
- `MemorySaver` checkpoints state after each node keyed by `thread_id = session_id`

### Section 7 — TelemetryLogger
- `log_agent_message()` — writes to SQLite `chat_history`; `@observe()` also creates a Langfuse span
- `log_checkpoint()` — serializes full `AgentState` to JSON and stores in SQLite

### Section 8 — Setup
- Instantiates retriever and reranker
- Calls `add_documents()` and `fit()` to pre-compute and cache all embeddings

### Section 9 — Pipeline runs (3 tasks)
- Each run gets a fresh `session_id` and `TelemetryLogger`
- `set_current_trace_io()` sets the trace input/output visible in Langfuse dashboard
- `flush()` ensures all buffered spans are sent before the cell completes

### Section 10 — View history and checkpoints
- Queries SQLite for chat history and checkpoints of the last session
- Cross-references with Langfuse trace ID for end-to-end audit

---

## What you can see in Langfuse after a run

| In the dashboard | What it tells you |
|---|---|
| Span tree under `multi-agent-rag-pipeline` | Full execution flow for one run |
| `researcher_node` input/output | Exact context the researcher received and its summary |
| `analyzer_node` input/output | What was analyzed and the critique produced |
| `writer_node` input/output | Final answer generated |
| Token counts per LLM call | Input + output tokens per agent |
| Cost per call | USD cost of each GPT-4o-mini call |
| Latency per span | Time taken at each stage |
| Session filter | All spans from a single run grouped together |

---

## Key classes and functions

| Name | Where | What it does |
|---|---|---|
| `AdvancedRetriever` | Section 3 | HNSW index + MMR retrieval with embedding cache |
| `DenseSpaceReranker` | Section 4 | BM25 + embedding reranking with corpus cache |
| `AgentState` | Section 5 | Shared TypedDict state passed between all agents |
| `researcher/analyzer/writer_node` | Section 5 | Agent functions with real gpt-4o-mini calls |
| `agent_graph` | Section 6 | Compiled LangGraph pipeline with checkpointing |
| `TelemetryLogger` | Section 7 | SQLite + Langfuse dual logging per session |
| `run_rag_pipeline()` | Section 9 | End-to-end orchestrator, named trace in Langfuse |
| `langfuse_handler` | Section 1 | LangChain callback capturing LLM tokens + cost |

---

## Environment variables needed (`.env`)

```
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

---

## How to find your trace in Langfuse

1. Go to **https://us.cloud.langfuse.com** and log in
2. Click **Traces** in the left menu
3. Look for name **`multi-agent-rag-pipeline`**
4. Click a trace → see full span tree with inputs, outputs, tokens, and cost
5. Use the **Sessions** tab with the printed `session_id` to filter one specific run

---

## Production next steps

- Replace toy sample docs with real document corpus (clinical trial reports, research papers)
- Add tool use to agents — PubMed search, internal knowledge base lookup
- Add evaluation scores via `langfuse_client.score_current_trace()` for automated quality checks
- Wire up alerts on cost spikes or latency outliers via Langfuse dashboards
