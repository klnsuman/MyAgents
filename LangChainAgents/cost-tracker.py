"""
Cost Tracking for Multi-Agent LLM Pipelines
============================================
Tracks token usage and cost per LLM call, per agent, per run, per batch.
Provides budget guards to fail-fast when cost exceeds threshold.

Why this matters:
  - Actor-Critic-Arbiter does 5-10x more LLM calls than single-shot
  - Loops can blow up cost if termination guards fail
  - Production needs per-chart unit economics for ROI calculation
  - Different agents may use different models (cheap Critic, expensive Arbiter)

Usage:
    tracker = CostTracker(budget_usd=0.50)
    
    # Wrap each LLM call:
    with tracker.track(agent="actor", model="gpt-5"):
        response = llm.invoke(prompt)
    
    # Or track manually if you have usage info from response:
    tracker.record(
        agent="actor", model="gpt-5",
        prompt_tokens=1250, completion_tokens=320,
    )
    
    # Get reports:
    print(tracker.summary())
    print(tracker.per_agent_breakdown())
"""

from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from typing import Optional
import json


# ---------------------------------------------------------------------------
# 1. Pricing table (USD per 1M tokens)
#    Update these as providers change pricing. Source: provider docs.
# ---------------------------------------------------------------------------

PRICING = {
    # OpenAI
    "gpt-5":            {"input":  5.00, "output": 15.00},
    "gpt-5-mini":       {"input":  0.30, "output":  1.20},
    "gpt-5-nano":       {"input":  0.05, "output":  0.40},
    "gpt-4o":           {"input":  2.50, "output": 10.00},
    "gpt-4o-mini":      {"input":  0.15, "output":  0.60},

    # Anthropic
    "claude-opus-4":    {"input": 15.00, "output": 75.00},
    "claude-sonnet-4":  {"input":  3.00, "output": 15.00},
    "claude-haiku-4":   {"input":  0.80, "output":  4.00},

    # Google
    "gemini-2.5-pro":   {"input":  1.25, "output":  5.00},
    "gemini-2.5-flash": {"input":  0.075, "output": 0.30},

    # Azure OpenAI (often matches OpenAI pricing — confirm with your tenant)
    "azure-gpt-5":      {"input":  5.00, "output": 15.00},
}


class BudgetExceededError(Exception):
    """Raised when accumulated cost crosses the configured budget."""


# ---------------------------------------------------------------------------
# 2. The unit of cost: one LLM call
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    timestamp: str
    agent: str                  # 'actor', 'critic', 'arbiter', etc.
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    metadata: dict = field(default_factory=dict)  # chart_id, run_id, anything

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# 3. The tracker
# ---------------------------------------------------------------------------

class CostTracker:
    def __init__(self, budget_usd: Optional[float] = None, run_id: str = ""):
        self.calls: list[CallRecord] = []
        self.budget_usd = budget_usd
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    # --------- Recording ---------

    def record(
        self,
        agent: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Optional[dict] = None,
    ) -> CallRecord:
        """Record one LLM call. Computes cost, raises if budget exceeded."""
        cost = self._compute_cost(model, prompt_tokens, completion_tokens)
        record = CallRecord(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            metadata=metadata or {},
        )
        self.calls.append(record)
        self._check_budget()
        return record

    @contextmanager
    def track(self, agent: str, model: str, metadata: Optional[dict] = None):
        """Context manager. Currently a placeholder - real implementations would
        intercept the LLM client to capture usage. Most providers return token
        counts in the response; capture those and call record() instead."""
        # In real use: monkey-patch llm.invoke() to capture response.usage_metadata
        # For now, this is a documentation hook.
        yield self
        # User responsible for calling self.record(...) inside the block

    def _compute_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        if model not in PRICING:
            # Don't crash - log unknown and return 0. Add to PRICING table later.
            print(f"[cost] WARNING: unknown model {model!r}, cost recorded as 0")
            return 0.0
        price = PRICING[model]
        return (
            (prompt_tokens     / 1_000_000) * price["input"]
            + (completion_tokens / 1_000_000) * price["output"]
        )

    def _check_budget(self):
        if self.budget_usd is None:
            return
        if self.total_cost > self.budget_usd:
            raise BudgetExceededError(
                f"Run {self.run_id}: cost ${self.total_cost:.4f} "
                f"exceeded budget ${self.budget_usd:.4f} after {len(self.calls)} calls"
            )

    # --------- Reports ---------

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    def per_agent_breakdown(self) -> dict:
        """Cost grouped by agent. Useful to see which agent is the cost driver."""
        agg: dict[str, dict] = {}
        for c in self.calls:
            a = agg.setdefault(c.agent, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0
            })
            a["calls"] += 1
            a["prompt_tokens"] += c.prompt_tokens
            a["completion_tokens"] += c.completion_tokens
            a["cost_usd"] += c.cost_usd
        return agg

    def per_model_breakdown(self) -> dict:
        """Cost grouped by model. Useful when mixing models across agents."""
        agg: dict[str, dict] = {}
        for c in self.calls:
            m = agg.setdefault(c.model, {
                "calls": 0, "tokens": 0, "cost_usd": 0.0
            })
            m["calls"] += 1
            m["tokens"] += c.total_tokens
            m["cost_usd"] += c.cost_usd
        return agg

    def summary(self) -> str:
        budget_line = (
            f"Budget:        ${self.budget_usd:.4f}  "
            f"({100 * self.total_cost / self.budget_usd:.1f}% used)"
            if self.budget_usd else "Budget:        (unlimited)"
        )
        return (
            f"== Cost Summary [{self.run_id}] ==\n"
            f"Total calls:   {len(self.calls)}\n"
            f"Total tokens:  {self.total_tokens:,}\n"
            f"Total cost:    ${self.total_cost:.4f}\n"
            f"{budget_line}"
        )

    def to_jsonl(self, path: str):
        """Persist all calls for offline analysis."""
        with open(path, "w") as f:
            for c in self.calls:
                f.write(json.dumps({
                    **c.__dict__,
                    "run_id": self.run_id,
                }) + "\n")


# ---------------------------------------------------------------------------
# 4. Demo: simulating an Actor-Critic-Arbiter run
# ---------------------------------------------------------------------------

def demo():
    # Per-chart budget cap of 50 cents
    tracker = CostTracker(budget_usd=0.50, run_id="chart_12345")

    # Round 1: Actor proposes
    tracker.record(
        agent="actor", model="gpt-5",
        prompt_tokens=2400, completion_tokens=180,
        metadata={"chart_id": "12345", "iteration": 0},
    )
    # Round 1: Critic evaluates (cheaper model)
    tracker.record(
        agent="critic", model="gpt-5-mini",
        prompt_tokens=2600, completion_tokens=120,
        metadata={"chart_id": "12345", "iteration": 0},
    )
    # Round 2: Actor revises
    tracker.record(
        agent="actor", model="gpt-5",
        prompt_tokens=2700, completion_tokens=200,
        metadata={"chart_id": "12345", "iteration": 1},
    )
    # Round 2: Critic accepts
    tracker.record(
        agent="critic", model="gpt-5-mini",
        prompt_tokens=2900, completion_tokens=80,
        metadata={"chart_id": "12345", "iteration": 1},
    )
    # Arbiter finalizes (cheap model, structured output)
    tracker.record(
        agent="arbiter", model="gpt-5-mini",
        prompt_tokens=3100, completion_tokens=400,
        metadata={"chart_id": "12345"},
    )

    print(tracker.summary())
    print()

    print("Per-agent breakdown:")
    for agent, stats in tracker.per_agent_breakdown().items():
        print(f"  {agent:10s}  calls={stats['calls']}  "
              f"tokens={stats['prompt_tokens'] + stats['completion_tokens']:>6,}  "
              f"cost=${stats['cost_usd']:.4f}")

    print("\nPer-model breakdown:")
    for model, stats in tracker.per_model_breakdown().items():
        print(f"  {model:15s}  calls={stats['calls']}  "
              f"tokens={stats['tokens']:>6,}  "
              f"cost=${stats['cost_usd']:.4f}")

    # Project to 1M charts
    cost_per_chart = tracker.total_cost
    print(f"\n--- Projection ---")
    print(f"Cost per chart:      ${cost_per_chart:.4f}")
    print(f"Cost per 1K charts:  ${cost_per_chart * 1_000:.2f}")
    print(f"Cost per 1M charts:  ${cost_per_chart * 1_000_000:,.0f}")


if __name__ == "__main__":
    demo()