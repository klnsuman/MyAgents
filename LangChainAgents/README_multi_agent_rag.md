# Multi-Agent RAG with Langfuse Telemetry

## What this notebook does

Builds a complete question-answering pipeline where three AI agents work in sequence, backed by a smart retrieval system and full observability via Langfuse.

---

## Architecture overview

```
User Task
   │
   ▼
[Retriever] ──► HNSW search → MMR diversity filter → Dense+Sparse reranking
   │
   ▼
[Researcher Agent] ──► reads retrieved docs, summarizes findings
   │
   ▼
[Analyzer Agent] ──► critically examines the research
   │
   ▼
[Writer Agent] ──► produces final structured output
   │
   ▼
SQLite (chat history + checkpoints) + Langfuse (traces)
```

---

## Section-by-section breakdown

### Section 1 — Environment Setup & Langfuse
- Loads `.env` file for API keys
- Initializes `langfuse_client` for tracing
- Required `.env` keys: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

### Section 2 — SQLite Database
Creates three tables in `rag_system.db`:
| Table | Purpose |
|---|---|
| `documents` | Stores ingested docs with embeddings |
| `chat_history` | Logs every agent message per session |
| `agent_checkpoints` | Saves agent state at each step |

### Section 3 — AdvancedRetriever (HNSW + MMR)
Two retrieval strategies:

**HNSW** (`retrieve_hnsw`): Fast approximate nearest-neighbor search using `hnswlib`. Good for large-scale similarity lookup.

**MMR** (`retrieve_mmr`): Maximal Marginal Relevance — balances relevance vs. diversity. Controlled by `lambda_mult` (0=max diversity, 1=max relevance).

### Section 4 — DenseSpareReranker
Takes HNSW candidates and re-scores them using two signals combined:
- **Dense (70%)**: semantic similarity via OpenAI embeddings
- **Sparse (30%)**: lexical match via BM25 (keyword overlap)

Combined score = `0.7 × dense_score + 0.3 × sparse_score`

### Section 5 — Multi-Agent Nodes
Three agents defined as LangGraph nodes, each operating on shared `AgentState`:
- `researcher_node` — retrieves and summarizes
- `analyzer_node` — critiques the research
- `writer_node` — produces the final answer

All decorated with `@observe()` so every call is traced in Langfuse.

### Section 6 — LangGraph with Checkpoints
Wires the agents into a linear graph: `START → researcher → analyzer → writer → END`

Uses `MemorySaver` for in-memory state checkpoints (keyed by `thread_id = session_id`), enabling multi-turn conversation continuity.

### Section 7 — TelemetryLogger
Dual-writes every agent event to:
- SQLite `chat_history` table (permanent)
- Langfuse as a named event (for the trace dashboard)

Also saves full agent state snapshots to `agent_checkpoints`.

### Section 8 — Sample Documents
Adds three toy documents to bootstrap the retriever and reranker. Replace with real docs for production use.

### Section 9 — `run_rag_pipeline()`
The main entry point. Steps:
1. HNSW retrieval (top 10)
2. MMR filtering (top 5 diverse)
3. Dense+Sparse reranking (top 3)
4. Feed top 3 into `AgentState.context`
5. Run LangGraph (researcher → analyzer → writer)
6. Save checkpoint to SQLite

### Section 10 — View History & Checkpoints
`view_session_history(session_id)` — prints all agent messages for a session in order.
`view_checkpoints(session_id)` — prints all saved state checkpoints with timestamps.

### Section 11 — How to find your trace in Langfuse
1. Go to https://cloud.langfuse.com
2. Click **Traces** in the left menu
3. Paste `telemetry.trace_id` into the search box
4. Each `@observe()`-decorated function appears as a span with inputs, outputs, and duration

---

## Key classes & functions at a glance

| Name | Where | What it does |
|---|---|---|
| `AdvancedRetriever` | Section 3 | HNSW index + MMR retrieval |
| `DenseSpaceReranker` | Section 4 | BM25 + embedding reranking |
| `AgentState` | Section 5 | Shared state dict passed between agents |
| `researcher/analyzer/writer_node` | Section 5 | The three agent functions |
| `agent_graph` | Section 6 | Compiled LangGraph pipeline |
| `TelemetryLogger` | Section 7 | SQLite + Langfuse dual logging |
| `run_rag_pipeline()` | Section 9 | End-to-end orchestrator |

---

## Environment variables needed (`.env`)

```
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Known limitations / next steps noted in the notebook

- Agent nodes return placeholder text — wire in actual LLM calls (`llm.invoke(prompt)`) to make them functional
- `MemorySaver` is in-memory only — checkpoints are lost on kernel restart (manual SQLite save partially compensates)
- Scale `AdvancedRetriever` to real document corpus by calling `retriever.add_documents(your_docs)`
