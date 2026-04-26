"""
Minimal Actor-Critic-Arbiter Demo
==================================
A toy LangGraph showing the three-agent pattern:
  - Actor    : proposes ICD-10 codes from clinical text
  - Critic   : evaluates the proposal, accepts or sends back critique
  - Arbiter  : final say (runs after critic accepts OR after max iterations)

Uses fake / simulated agent responses so it runs without API keys.
Replace the simulate_* functions with real LLM calls to make it production.
"""

from typing import Annotated, Literal, Optional, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. Sample clinical note (the input the agents will reason over)
# ---------------------------------------------------------------------------

CLINICAL_NOTE = """
Patient is a 67-year-old male admitted with shortness of breath and bilateral
leg swelling for 5 days. History of CHF and CKD stage 3. Labs show creatinine
elevated from baseline 1.4 to 2.8. BNP 1850. Echo shows EF 30%.
Diagnosed with acute decompensated CHF and acute kidney injury on CKD.
"""


# ---------------------------------------------------------------------------
# 2. Shared graph state
# ---------------------------------------------------------------------------

class DebateState(TypedDict):
    note: str                       # the clinical note (input)
    proposed_codes: list[str]       # actor's latest proposal
    critique: Optional[str]         # critic's latest feedback (None = accepted)
    verdict: Optional[str]          # 'accept' or 'revise'
    iterations: Annotated[int, add] # incremented each round (reducer pattern)
    final_codes: list[str]          # arbiter's final decision
    history: Annotated[list[str], add]  # transcript of the debate


MAX_ITERATIONS = 5


# ---------------------------------------------------------------------------
# 3. Simulated agent logic (swap for real LLM calls in production)
# ---------------------------------------------------------------------------

def simulate_actor(note: str, critique: Optional[str], iteration: int) -> list[str]:
    """Pretend to be an LLM proposing ICD-10 codes.
    Each iteration, the actor 'improves' its proposal in response to critique."""
    proposals = [
        ["I50.9"],                                      # iter 0: too vague
        ["I50.9", "N17.9"],                             # iter 1: added AKI but unspecified
        ["I50.23", "N17.9"],                            # iter 2: specified CHF type
        ["I50.23", "N17.9", "N18.30"],                  # iter 3: added CKD stage 3
        ["I50.23", "N17.9", "N18.30", "I50.9"],         # iter 4: oops, redundant
        ["I50.23", "N17.9", "N18.30"],                  # iter 5: cleaned up
    ]
    return proposals[min(iteration, len(proposals) - 1)]


def simulate_critic(codes: list[str], iteration: int) -> tuple[str, Optional[str]]:
    """Pretend to be an LLM critiquing the proposal.
    Returns (verdict, critique). verdict='accept' means we're done."""
    critiques = [
        ("revise", "I50.9 is unspecified CHF. Note says 'acute decompensated' and EF 30% - "
                   "use I50.23 (acute on chronic systolic). AKI on CKD missing."),
        ("revise", "Good, added AKI. But N17.9 needs CKD context - patient has CKD stage 3, "
                   "so add N18.30 (CKD stage 3 unspecified)."),
        ("revise", "CHF code looks right now. Still missing the CKD code."),
        ("revise", "Almost there - check for redundant codes before finalizing."),
        ("revise", "I50.9 and I50.23 are redundant. Drop the unspecified one."),
        ("accept", None),
    ]
    return critiques[min(iteration, len(critiques) - 1)]


def simulate_arbiter(codes: list[str], history: list[str]) -> list[str]:
    """Pretend to be an LLM doing final review.
    Could override if it sees a problem; here it just confirms."""
    # In real life: review the full debate, apply business rules,
    # check against coding guidelines, etc.
    return codes


# ---------------------------------------------------------------------------
# 4. Graph nodes
# ---------------------------------------------------------------------------

def actor_node(state: DebateState) -> dict:
    iteration = state.get("iterations", 0)
    codes = simulate_actor(state["note"], state.get("critique"), iteration)
    log = f"[Actor    iter={iteration}] proposed: {codes}"
    print(log)
    return {
        "proposed_codes": codes,
        "history": [log],
    }


def critic_node(state: DebateState) -> dict:
    iteration = state.get("iterations", 0)
    verdict, critique = simulate_critic(state["proposed_codes"], iteration)
    log = f"[Critic   iter={iteration}] verdict={verdict}  critique={critique!r}"
    print(log)
    return {
        "verdict": verdict,
        "critique": critique,
        "iterations": 1,           # reducer adds this to the running total
        "history": [log],
    }


def arbiter_node(state: DebateState) -> dict:
    final = simulate_arbiter(state["proposed_codes"], state["history"])
    iters = state.get("iterations", 0)
    reason = "critic accepted" if state.get("verdict") == "accept" else f"max {MAX_ITERATIONS} iterations hit"
    log = f"[Arbiter] final codes: {final}  (reason: {reason})"
    print(log)
    return {
        "final_codes": final,
        "history": [log],
    }


# ---------------------------------------------------------------------------
# 5. Conditional edge: keep debating, or move to arbiter
# ---------------------------------------------------------------------------

def should_continue(state: DebateState) -> Literal["actor", "arbiter"]:
    """After critic runs, decide: another round, or hand off to arbiter?"""
    if state.get("verdict") == "accept":
        return "arbiter"
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "arbiter"
    return "actor"


# ---------------------------------------------------------------------------
# 6. Build the graph
# ---------------------------------------------------------------------------

graph = StateGraph(DebateState)
graph.add_node("actor", actor_node)
graph.add_node("critic", critic_node)
graph.add_node("arbiter", arbiter_node)

graph.add_edge(START, "actor")
graph.add_edge("actor", "critic")
graph.add_conditional_edges("critic", should_continue, {
    "actor": "actor",
    "arbiter": "arbiter",
})
graph.add_edge("arbiter", END)

app = graph.compile()


# ---------------------------------------------------------------------------
# 7. Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("CLINICAL NOTE:")
    print(CLINICAL_NOTE.strip())
    print("=" * 70)
    print()

    initial_state = {
        "note": CLINICAL_NOTE,
        "proposed_codes": [],
        "critique": None,
        "verdict": None,
        "final_codes": [],
        "history": [],
    }

    result = app.invoke(initial_state)

    print()
    print("=" * 70)
    print(f"DONE in {result['iterations']} iterations")
    print(f"FINAL CODES: {result['final_codes']}")
    print("=" * 70)