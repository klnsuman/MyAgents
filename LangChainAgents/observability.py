"""
Reasoning LLM Observability Stack
==================================
Three trackers for production agentic systems with reasoning models:

  1. CostTracker        - per-agent / per-model cost, including reasoning tokens
  2. MemoryTracker      - state evolution, retrieval hits, context-window pressure
  3. EvaluationTracker  - precision/recall/F1, judge scores, ground-truth deltas

Designed for systems like:
  - Actor-Critic-Arbiter pipelines
  - Multi-turn debate agents
  - RAG-backed reasoning agents
  - Long-context state-tracking agents

These are the three operational pillars NPCI / pharma / banking compliance
would expect to see before approving production deployment of reasoning LLMs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import json
import statistics


# ============================================================================
# 1. COST TRACKER (with reasoning token support)
# ============================================================================

PRICING = {
    # Reasoning models — output rate applies to ALL output incl. reasoning
    "o3":                {"input":  2.00, "output":  8.00},
    "o3-mini":           {"input":  1.10, "output":  4.40},
    "o4-mini":           {"input":  1.10, "output":  4.40},
    "claude-opus-4-7":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input":  3.00, "output": 15.00},
    "gemini-2.5-pro":    {"input":  1.25, "output": 10.00},

    # Standard models
    "gpt-5":             {"input":  5.00, "output": 15.00},
    "gpt-5-mini":        {"input":  0.30, "output":  1.20},
    "gpt-4o-mini":       {"input":  0.15, "output":  0.60},
    "gemini-2.5-flash":  {"input":  0.075, "output": 0.30},
}


class BudgetExceededError(Exception):
    pass


@dataclass
class CallRecord:
    timestamp: str
    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int        # visible answer tokens
    reasoning_tokens: int = 0     # hidden thinking tokens (billed at output rate)
    cost_usd: float = 0.0
    latency_ms: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def total_output_tokens(self) -> int:
        return self.completion_tokens + self.reasoning_tokens


class CostTracker:
    def __init__(self, budget_usd: Optional[float] = None, run_id: str = ""):
        self.calls: list[CallRecord] = []
        self.budget_usd = budget_usd
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def record(self, agent, model, prompt_tokens, completion_tokens,
               reasoning_tokens=0, latency_ms=0, metadata=None):
        cost = self._compute(model, prompt_tokens, completion_tokens, reasoning_tokens)
        rec = CallRecord(
            timestamp=datetime.now().isoformat(),
            agent=agent, model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost, latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self.calls.append(rec)
        if self.budget_usd and self.total_cost > self.budget_usd:
            raise BudgetExceededError(
                f"{self.run_id}: ${self.total_cost:.4f} > budget ${self.budget_usd:.4f}"
            )
        return rec

    def _compute(self, model, prompt, completion, reasoning):
        if model not in PRICING:
            return 0.0
        p = PRICING[model]
        # Reasoning tokens bill at output rate
        return (prompt / 1e6) * p["input"] + ((completion + reasoning) / 1e6) * p["output"]

    @property
    def total_cost(self):
        return sum(c.cost_usd for c in self.calls)

    def reasoning_efficiency(self) -> dict:
        """Per-agent: what fraction of output cost is reasoning vs visible answer?
        High reasoning share + cheap problem = waste."""
        out = {}
        for c in self.calls:
            a = out.setdefault(c.agent, {"reasoning": 0, "visible": 0, "cost": 0.0})
            a["reasoning"] += c.reasoning_tokens
            a["visible"] += c.completion_tokens
            a["cost"] += c.cost_usd
        for agent, s in out.items():
            total = s["reasoning"] + s["visible"]
            s["reasoning_share"] = (s["reasoning"] / total) if total else 0
        return out

    def per_agent(self) -> dict:
        out = {}
        for c in self.calls:
            a = out.setdefault(c.agent, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "reasoning_tokens": 0, "cost_usd": 0.0, "p95_latency_ms": []
            })
            a["calls"] += 1
            a["prompt_tokens"] += c.prompt_tokens
            a["completion_tokens"] += c.completion_tokens
            a["reasoning_tokens"] += c.reasoning_tokens
            a["cost_usd"] += c.cost_usd
            a["p95_latency_ms"].append(c.latency_ms)
        for stats in out.values():
            lat = sorted(stats["p95_latency_ms"])
            stats["p95_latency_ms"] = lat[int(0.95 * len(lat))] if lat else 0
        return out


# ============================================================================
# 2. MEMORY TRACKER
# ============================================================================
# For agents with state (LangGraph state, RAG retrieval, conversation history).
# Tracks: how state grows, retrieval quality, context pressure.

@dataclass
class MemoryEvent:
    timestamp: str
    event_type: str         # 'state_update' | 'retrieval' | 'context_overflow'
    agent: str
    detail: dict


class MemoryTracker:
    def __init__(self, context_window_limit: int = 200_000):
        self.events: list[MemoryEvent] = []
        self.context_window_limit = context_window_limit
        self.current_context_tokens = 0

    def state_update(self, agent: str, fields_changed: list[str],
                     state_size_tokens: int):
        """Record when an agent modifies graph state."""
        self.current_context_tokens = state_size_tokens
        pressure = state_size_tokens / self.context_window_limit
        self.events.append(MemoryEvent(
            timestamp=datetime.now().isoformat(),
            event_type="state_update",
            agent=agent,
            detail={
                "fields_changed": fields_changed,
                "state_size_tokens": state_size_tokens,
                "context_pressure": round(pressure, 3),
            }
        ))
        if pressure > 0.85:
            print(f"[memory] WARNING: context at {pressure:.0%} of limit "
                  f"after {agent} update")

    def retrieval(self, agent: str, query: str, hits: int,
                  top_score: float, retrieved_tokens: int):
        """Record a RAG / vector-store retrieval call."""
        self.events.append(MemoryEvent(
            timestamp=datetime.now().isoformat(),
            event_type="retrieval",
            agent=agent,
            detail={
                "query_preview": query[:60],
                "hits": hits,
                "top_score": round(top_score, 3),
                "retrieved_tokens": retrieved_tokens,
            }
        ))

    def retrieval_quality(self) -> dict:
        """Aggregate retrieval performance — low scores = poor retrieval."""
        retrievals = [e for e in self.events if e.event_type == "retrieval"]
        if not retrievals:
            return {}
        scores = [r.detail["top_score"] for r in retrievals]
        return {
            "n_retrievals": len(retrievals),
            "mean_top_score": round(statistics.mean(scores), 3),
            "median_top_score": round(statistics.median(scores), 3),
            "low_quality_retrievals": sum(1 for s in scores if s < 0.5),
            "zero_hit_retrievals": sum(1 for r in retrievals if r.detail["hits"] == 0),
        }

    def context_pressure_timeline(self) -> list[dict]:
        """How context size evolved across the run."""
        return [
            {"agent": e.agent, "tokens": e.detail["state_size_tokens"],
             "pressure": e.detail["context_pressure"]}
            for e in self.events if e.event_type == "state_update"
        ]


# ============================================================================
# 3. EVALUATION TRACKER
# ============================================================================
# Combines deterministic metrics + LLM-as-judge scoring per case.
# Aggregates into per-batch reports for offline / online eval pipelines.

@dataclass
class EvalCase:
    case_id: str
    predicted: Any
    ground_truth: Any
    metrics: dict          # {'precision': 0.8, 'recall': 0.9, 'f1': 0.85}
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class EvaluationTracker:
    def __init__(self, run_id: str = ""):
        self.cases: list[EvalCase] = []
        self.run_id = run_id or datetime.now().strftime("eval_%Y%m%d_%H%M%S")

    def add_case(self, case_id, predicted, ground_truth,
                 judge_score=None, judge_reasoning=None, metadata=None):
        metrics = self._compute_metrics(predicted, ground_truth)
        case = EvalCase(
            case_id=case_id, predicted=predicted, ground_truth=ground_truth,
            metrics=metrics, judge_score=judge_score,
            judge_reasoning=judge_reasoning, metadata=metadata or {},
        )
        self.cases.append(case)
        return case

    def _compute_metrics(self, predicted, ground_truth):
        """Set-based precision/recall/F1. Adapt to your domain."""
        pred = set(predicted) if predicted else set()
        gold = set(ground_truth) if ground_truth else set()
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    def aggregate(self) -> dict:
        if not self.cases:
            return {}
        ps = [c.metrics["precision"] for c in self.cases]
        rs = [c.metrics["recall"] for c in self.cases]
        fs = [c.metrics["f1"] for c in self.cases]
        judges = [c.judge_score for c in self.cases if c.judge_score is not None]
        return {
            "n_cases": len(self.cases),
            "macro_precision": round(statistics.mean(ps), 4),
            "macro_recall": round(statistics.mean(rs), 4),
            "macro_f1": round(statistics.mean(fs), 4),
            "macro_judge_score": round(statistics.mean(judges), 3) if judges else None,
            "perfect_cases": sum(1 for c in self.cases if c.metrics["f1"] == 1.0),
            "zero_cases": sum(1 for c in self.cases if c.metrics["f1"] == 0.0),
        }

    def regressions_vs(self, baseline_f1_by_case: dict) -> list[dict]:
        """Find cases where current run is WORSE than baseline.
        Critical for catching prompt-change regressions."""
        regressions = []
        for c in self.cases:
            base = baseline_f1_by_case.get(c.case_id)
            if base is None:
                continue
            delta = c.metrics["f1"] - base
            if delta < -0.05:
                regressions.append({
                    "case_id": c.case_id,
                    "current_f1": c.metrics["f1"],
                    "baseline_f1": base,
                    "delta": round(delta, 3),
                })
        return regressions


# ============================================================================
# 4. UNIFIED RUN: How they fit together
# ============================================================================

def demo_full_run():
    """Simulate one chart processed through Actor-Critic-Arbiter with
    reasoning models, showing how all three trackers light up together."""

    cost = CostTracker(budget_usd=2.00, run_id="chart_42")
    memory = MemoryTracker(context_window_limit=200_000)
    evals = EvaluationTracker(run_id="batch_2026_04_28")

    # ---- Actor uses reasoning model for complex case ----
    cost.record(
        agent="actor", model="o3",
        prompt_tokens=3500, completion_tokens=420, reasoning_tokens=12_000,
        latency_ms=18_400,
        metadata={"chart_id": 42, "iteration": 0, "complexity": "high"},
    )
    memory.state_update(
        agent="actor",
        fields_changed=["proposed_codes", "actor_reasoning"],
        state_size_tokens=4_200,
    )

    # ---- Critic uses cheap model — validation doesn't need reasoning ----
    cost.record(
        agent="critic", model="gpt-5-mini",
        prompt_tokens=4200, completion_tokens=180,
        latency_ms=1_200,
        metadata={"chart_id": 42, "iteration": 0},
    )
    memory.retrieval(
        agent="critic",
        query="ICD-10 I50.23 inclusion criteria",
        hits=4, top_score=0.87, retrieved_tokens=850,
    )
    memory.state_update(
        agent="critic",
        fields_changed=["verdict", "critique"],
        state_size_tokens=5_100,
    )

    # ---- Actor revises ----
    cost.record(
        agent="actor", model="o3",
        prompt_tokens=5500, completion_tokens=320, reasoning_tokens=8_000,
        latency_ms=14_200,
        metadata={"chart_id": 42, "iteration": 1},
    )
    memory.state_update(
        agent="actor",
        fields_changed=["proposed_codes"],
        state_size_tokens=5_900,
    )

    # ---- Critic accepts ----
    cost.record(
        agent="critic", model="gpt-5-mini",
        prompt_tokens=5900, completion_tokens=80,
        latency_ms=900,
        metadata={"chart_id": 42, "iteration": 1},
    )

    # ---- Arbiter finalizes with reasoning model for hard adjudication ----
    cost.record(
        agent="arbiter", model="o3-mini",
        prompt_tokens=6200, completion_tokens=500, reasoning_tokens=4_000,
        latency_ms=8_100,
        metadata={"chart_id": 42},
    )
    memory.state_update(
        agent="arbiter",
        fields_changed=["arbiter_decision"],
        state_size_tokens=6_800,
    )

    # ---- Evaluation: compare against ground truth ----
    evals.add_case(
        case_id="chart_42",
        predicted=["I50.23", "N17.9", "N18.30"],
        ground_truth=["I50.23", "N17.9", "N18.30", "E11.9"],  # missed diabetes
        judge_score=7.5,
        judge_reasoning="Captured CHF + AKI + CKD correctly. Missed diabetes.",
        metadata={"chart_id": 42, "complexity": "high"},
    )

    # ============ REPORTS ============
    print("=" * 70)
    print("COST TRACKER")
    print("=" * 70)
    print(f"Total cost for chart: ${cost.total_cost:.4f}")
    print(f"Calls: {len(cost.calls)}\n")
    print("Per-agent (with p95 latency):")
    for agent, s in cost.per_agent().items():
        print(f"  {agent:8s}  calls={s['calls']}  "
              f"reasoning_tok={s['reasoning_tokens']:>6,}  "
              f"cost=${s['cost_usd']:.4f}  "
              f"p95_lat={s['p95_latency_ms']}ms")
    print("\nReasoning efficiency (% of output tokens spent thinking):")
    for agent, s in cost.reasoning_efficiency().items():
        print(f"  {agent:8s}  reasoning_share={s['reasoning_share']:.0%}  "
              f"cost=${s['cost']:.4f}")

    print()
    print("=" * 70)
    print("MEMORY TRACKER")
    print("=" * 70)
    print("Context pressure timeline:")
    for snap in memory.context_pressure_timeline():
        bar = "█" * int(snap["pressure"] * 40)
        print(f"  after {snap['agent']:8s}  {snap['tokens']:>6,} tok  "
              f"|{bar:<40s}| {snap['pressure']:.1%}")
    print("\nRetrieval quality:")
    for k, v in memory.retrieval_quality().items():
        print(f"  {k}: {v}")

    print()
    print("=" * 70)
    print("EVALUATION TRACKER")
    print("=" * 70)
    for k, v in evals.aggregate().items():
        print(f"  {k}: {v}")
    print("\nPer-case (showing first):")
    c = evals.cases[0]
    print(f"  case={c.case_id}  predicted={c.predicted}")
    print(f"                  gold={c.ground_truth}")
    print(f"                  metrics={c.metrics}")
    print(f"                  judge={c.judge_score}/10  ({c.judge_reasoning})")

    # Cost projection
    print()
    print("=" * 70)
    print("UNIT ECONOMICS PROJECTION")
    print("=" * 70)
    print(f"Cost per chart:       ${cost.total_cost:.4f}")
    print(f"Cost per 1M charts:   ${cost.total_cost * 1_000_000:>10,.0f}")
    print(f"If we doubled charts: ${cost.total_cost * 2_000_000:>10,.0f}")
    print(f"\nBudget guard:  ${cost.budget_usd:.2f}  "
          f"({100 * cost.total_cost / cost.budget_usd:.1f}% used)")


if __name__ == "__main__":
    demo_full_run()