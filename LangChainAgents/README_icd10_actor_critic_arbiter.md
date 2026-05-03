# ICD-10 Autonomous Coding: Actor-Critic-Arbiter with Reasoning Bank

## What this system does

Automates medical ICD-10 coding by simulating the review process that human coders follow. Instead of a single LLM making one pass, three specialized agents debate the correct codes — the same way a coder, auditor, and chief coding officer would interact in a real hospital coding department.

---

## Why this matters in healthcare AI

Medical coding is a $20B+ industry in the US. A single wrong ICD-10 code can:
- Result in claim denial or underpayment
- Trigger compliance audits
- Affect hospital quality metrics and reimbursement rates

Automating this with a single LLM is risky — there is no review step. This system introduces **structured debate** between agents so errors are caught before a code is finalized, and every decision is fully auditable.

---

## Architecture overview

```
Medical Text
     │
     ├──► Reasoning Bank (ChromaDB) — retrieves similar past cases as few-shot examples
     ├──► ICD-10 DB (ChromaDB)      — retrieves valid candidate codes
     │
     ▼
┌─────────────────────────────────────────────────┐
│              DEBATE LOOP (default 3 rounds)     │
│                                                 │
│  [Actor]  ──── proposes codes ────────────►     │
│                                    [Critic]     │
│  [Actor]  ◄─── feedback / revise ──────────     │
│                                                 │
│  Repeats until Critic agrees OR max rounds hit  │
└─────────────────────────────────────────────────┘
     │
     ▼
[Arbiter] ──► Final ICD-10 Codes + Confidence Score
     │
     ▼
Langfuse Traces (full audit: prompts, tokens, cost, latency)
```

---

## Key design decisions and why (interview talking points)

### 1. Why Actor-Critic-Arbiter instead of a single LLM?
A single LLM call has no self-correction mechanism. Errors go undetected. The Actor-Critic pattern forces structured review — the Critic challenges every code the Actor proposes, and both must justify their positions with references to the ICD-10 database and past examples. The Arbiter adds a final layer that reconciles disagreements, mirroring the escalation path in real coding departments (coder → auditor → chief coding officer).

### 2. Why a Reasoning Bank?
ICD-10 coding has thousands of nuanced rules — combination codes, sequencing guidelines, specificity requirements. Encoding all of this in a prompt is impossible. The Reasoning Bank stores past successfully coded cases. When a new medical text arrives, the system retrieves the top-3 most similar past cases and presents them to both Actor and Critic as few-shot examples. This is grounded, evidence-based reasoning rather than pure LLM generation. The bank grows over time as more cases are validated.

### 3. Why configurable debate rounds?
Different clinical scenarios need different levels of scrutiny. A straightforward UTI might be resolved in one round. A complex multi-comorbidity admission with sequencing ambiguity might need 4-5 rounds. Making rounds configurable lets you tune precision vs. speed per use case — or set higher rounds for high-value DRG cases where coding accuracy directly impacts reimbursement.

### 4. Why ChromaDB for both the ICD-10 DB and Reasoning Bank?
Both stores require semantic similarity search — "find codes similar to this clinical description" and "find past cases similar to this patient presentation." ChromaDB provides fast vector similarity search over embeddings. Using two separate collections keeps concerns separated: one for code validation (ICD-10 DB) and one for reasoning patterns (Reasoning Bank), each retrieved independently and combined in the agent prompts.

### 5. Why does the Critic also access the Reasoning Bank?
The Critic uses past cases not just to check if codes are valid, but to check if the *coding pattern* is consistent with past decisions. For example: should COPD exacerbation and pneumonia be coded separately or is there a combination code? The Reasoning Bank has examples of how similar ambiguities were resolved before, making the Critic's feedback consistent and defensible.

### 6. Why does the Arbiter see the full debate history?
The Arbiter doesn't just see the final position of Actor and Critic — it sees every round of the debate in a structured JSON transcript. This allows the Arbiter to identify which points were conceded, which remained contested, and what the core disagreement was. It can then make a nuanced final decision: side with Actor, side with Critic, or propose a reconciled set of codes that incorporates valid points from both.

### 7. Why JSON-structured responses from each agent?
Each agent returns a strict JSON schema (`proposed_codes`, `rationale`, `critic_agrees`, `verdict`, etc.). This makes the system deterministic and parseable — the debate state machine relies on structured fields like `critic_agrees: true/false` to route the LangGraph. Unstructured text responses would make routing fragile.

### 8. Why Langfuse for observability?
Every agent call is traced in Langfuse with the exact prompt sent, the full LLM response, token counts, and API cost per call. In a healthcare setting this is critical for:
- **Compliance audits** — prove exactly what the AI saw and decided
- **Cost control** — identify which case types consume the most tokens
- **Error analysis** — when a code is wrong, replay the full debate trace to find where it went wrong

---

## How the Reasoning Bank works — step by step

```
1. Hospital coders validate a complex case manually
2. That case is stored in the Reasoning Bank:
   {medical_text, final_codes, actor_rationale, critic_feedback, confidence}

3. Next time a similar case arrives:
   Actor retrieves top-3 similar past cases via semantic search
   → uses them as few-shot examples: "In a similar case, codes X, Y were used because..."

4. Critic retrieves the same past cases
   → checks if the Actor's proposal matches established patterns
   → flags if the Actor is deviating from a proven coding pattern

5. Over time the bank grows → system gets better without retraining the LLM
```

This is **Retrieval-Augmented Reasoning** — not just retrieving documents, but retrieving past *reasoning chains* and injecting them as context.

---

## Debate loop flow

```
Round 1:
  Actor  → proposes ["I21.4", "I10", "E11.9"]
  Critic → "I21.4 correct. But E11.9 should be E11.65 given elevated HbA1c. REVISE."

Round 2:
  Actor  → revises to ["I21.4", "I10", "E11.65"]
            "Agreed on E11.65 — hyperglycemia evident from HbA1c 10.2%"
  Critic → "Accepted. All codes valid and specific. ACCEPT."

→ Critic agrees at Round 2 → Arbiter receives debate transcript

Arbiter → Final: ["I21.4", "I10", "E11.65"] | Confidence: high | Sided with: Reconciled
```

---

## Section-by-section breakdown

### Section 1 — Environment Setup
- Langfuse client + `CallbackHandler` for LLM cost tracking
- All API keys loaded from `.env`

### Section 2 — ICD-10 ChromaDB
- 50 sample ICD-10 codes across 10 clinical categories (Cardiovascular, Respiratory, Endocrine, Infection, Neurological, Mental Health, Musculoskeletal, Injury, GI, Renal, Oncology)
- Each code stored with description and category metadata
- Retrieved by semantic similarity at query time (top-8 candidates)

### Section 3 — Reasoning Bank ChromaDB
- 8 validated past coding cases covering: STEMI, T2DM neuropathy, COPD+pneumonia, MDD, E.coli septic shock, knee replacement, hip fracture, cirrhosis with GI bleed
- Each entry stores: medical text, actor rationale, critic feedback, final codes, confidence
- Retrieved by semantic similarity (top-3 most similar past cases)

### Section 4 — ICD10CodingState
Shared state passed through all LangGraph nodes:
| Field | Purpose |
|---|---|
| `medical_text` | Input clinical note |
| `max_rounds` | Configurable debate limit |
| `round_number` | Current round tracker |
| `actor_codes` | Latest Actor proposal |
| `critic_agrees` | Routing signal for LangGraph |
| `debate_history` | Full transcript appended each round |
| `final_codes` | Arbiter's decision |
| `arbiter_confidence` | high/medium/low for audit |

### Section 5 — Actor Agent
- Retrieves top-3 past cases from Reasoning Bank
- Retrieves top-8 ICD-10 candidates from ChromaDB
- Constructs prompt with few-shot examples + candidates
- On revision rounds, explicitly addresses Critic's feedback
- Returns structured JSON: `{proposed_codes, rationale, response_to_critic}`

### Section 6 — Critic Agent
- Retrieves from Reasoning Bank to validate coding patterns
- Reviews Actor's codes against ICD-10 descriptions
- On final round: forced to make definitive ACCEPT/REVISE
- Returns structured JSON: `{agrees, feedback, suggested_changes, verdict}`

### Section 7 — Arbiter Agent
- Receives full `debate_history` as JSON transcript
- Makes final decision independent of Actor/Critic
- Returns: `{final_codes, rationale, sided_with, confidence, audit_note}`

### Section 8 — LangGraph
- Conditional routing via `should_continue()`:
  - `critic_agrees == True` → Arbiter
  - `round_number >= max_rounds` → Arbiter
  - Otherwise → Actor (next round)

### Section 9 — Test Cases
Three clinical scenarios:
1. **Cardiac** — Chest pressure, new LBBB, elevated troponin, HTN, T2DM
2. **Sepsis** — Altered mental status, fever, positive UA, hypotension requiring vasopressors
3. **Psychiatric** — 6-week depression, PHQ-9 18, passive SI

---

## What you can see in Langfuse per run

| Trace | What it shows |
|---|---|
| `icd10-coding-pipeline` | Full pipeline for one case |
| `actor-agent` span | Exact prompt with few-shot examples, proposed codes |
| `critic-agent` span | Review prompt, verdict, suggested changes |
| `arbiter-agent` span | Full debate transcript, final decision |
| Token counts | Input + output tokens per agent per round |
| API cost | USD cost of entire coding pipeline |
| Latency | Time per agent, total pipeline time |

---

## Key classes and functions

| Name | Purpose |
|---|---|
| `icd10_vectorstore` | ChromaDB with 50 ICD-10 code descriptions |
| `reasoning_vectorstore` | ChromaDB with 8 past coding examples |
| `ICD10CodingState` | Shared LangGraph state |
| `actor_node` | Proposes codes using RAG + few-shot reasoning |
| `critic_node` | Reviews and debates codes |
| `arbiter_node` | Finalizes codes after debate |
| `should_continue()` | LangGraph routing — debate vs. arbiter |
| `run_coding_pipeline()` | Entry point, configurable `max_rounds` |

---

## Environment variables (`.env`)

```
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

---

## Production next steps

- Load full ICD-10-CM 2024 code set (~70,000 codes) into ChromaDB
- Seed Reasoning Bank from historical coded claims with human-validated codes
- Add DRG grouper integration — map final codes to DRG for reimbursement validation
- Add a human-in-the-loop step for low-confidence cases (`arbiter_confidence == "low"`)
- Track coding accuracy metrics via Langfuse evaluations against gold-standard codes
- Add CPT code support alongside ICD-10 for procedure coding
