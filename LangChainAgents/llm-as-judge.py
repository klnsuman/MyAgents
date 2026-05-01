"""
LLM-as-Judge for ICD-10 Coding Decisions
=========================================
The missing piece from observability_stack.py: actually computing judge_score.

Why a judge (when we already have F1):
  - F1 says "did you get the right codes?"
  - Judge says "did you get them for the right reasons?"
  - For regulatory work, both matter equally.

This module shows:
  1. The judge prompt structure (what makes a good rubric)
  2. The Pydantic schema enforcing structured judge output
  3. A simulated judge for testing (no API key needed)
  4. The real-LLM judge stub (uncomment when wiring up Anthropic/OpenAI)
  5. Integration with EvaluationTracker
"""

from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# 1. Judge output schema — forces structured, parseable evaluation
# ============================================================================

class JudgeRubric(BaseModel):
    """Per-dimension scores. Each 1-10.
    Dimensions chosen for ICD-10 medical coding evaluation.
    Adapt the dimensions to your domain (fraud, customer support, etc.)."""

    correctness: int = Field(
        ge=1, le=10,
        description="Did predicted codes match clinical evidence in the note?"
    )
    specificity: int = Field(
        ge=1, le=10,
        description="Are the chosen codes appropriately specific (not too vague, "
                    "not over-specific without evidence)?"
    )
    evidence_grounding: int = Field(
        ge=1, le=10,
        description="Is each predicted code supported by quotable evidence in the note?"
    )
    reasoning_soundness: int = Field(
        ge=1, le=10,
        description="Is the system's stated reasoning logically valid? "
                    "Does it apply ICD-10 coding guidelines correctly?"
    )
    hallucination_freedom: int = Field(
        ge=1, le=10,
        description="Free of invented diagnoses or facts not in the note? "
                    "(10 = no hallucinations, 1 = severe hallucination)"
    )
    auditor_defensibility: int = Field(
        ge=1, le=10,
        description="Could you defend each code to a CMS auditor with the "
                    "given evidence and reasoning?"
    )


class JudgeVerdict(BaseModel):
    """Complete structured output from the judge."""
    rubric: JudgeRubric
    overall_score: float = Field(
        ge=0.0, le=10.0,
        description="Weighted average. Auditor_defensibility weighted 2x, others 1x."
    )
    pass_fail: str = Field(
        description="'pass' if overall >= 7.0, 'fail' otherwise. "
                    "'borderline' if 5.0 <= overall < 7.0."
    )
    strengths: list[str] = Field(description="What the system did well (2-4 items)")
    weaknesses: list[str] = Field(description="Concrete failures with examples (1-4 items)")
    suggested_improvements: list[str] = Field(
        description="Actionable fixes for prompt or pipeline (1-3 items)"
    )
    summary: str = Field(description="One-sentence overall assessment")


# ============================================================================
# 2. The judge prompt — the rubric expressed for the LLM
# ============================================================================

JUDGE_SYSTEM_PROMPT = """
You are a senior medical coding auditor evaluating an AI ICD-10 coding system.

You will be shown:
  - A clinical discharge note
  - The codes the AI predicted, with reasoning
  - The ground-truth codes (what expert human coders assigned)

Your job is NOT to redo the coding. Your job is to evaluate the AI's
work across six dimensions, each scored 1-10:

1. Correctness:           Did predicted codes match clinical evidence?
2. Specificity:           Are codes appropriately specific?
3. Evidence-grounding:    Is each code supported by quotable text?
4. Reasoning soundness:   Is the reasoning logically valid?
5. Hallucination freedom: Free of invented facts?
6. Auditor defensibility: Could you defend this to CMS?

Be a HARSH but FAIR auditor. A system that gets the codes right with
poor reasoning should score lower than one with sound reasoning.

Return JSON matching the JudgeVerdict schema. No prose outside JSON.
"""


def build_judge_prompt(note: str, predicted_codes: list, predicted_reasoning: str,
                       ground_truth_codes: list) -> str:
    return f"""
CLINICAL NOTE:
{note}

----------------------------------------
AI SYSTEM PREDICTIONS:
Codes:     {predicted_codes}
Reasoning: {predicted_reasoning}

----------------------------------------
GROUND TRUTH (expert human coders):
Codes: {ground_truth_codes}

----------------------------------------
Score this work using the JudgeVerdict schema.
"""


# ============================================================================
# 3. Simulated judge (deterministic, for testing without API)
# ============================================================================

def simulate_judge(note: str, predicted: list, predicted_reasoning: str,
                   ground_truth: list) -> JudgeVerdict:
    """Fake judge that mimics how a real LLM judge would score.
    Logic: starts from F1, adjusts based on simple heuristics."""

    pred_set = set(predicted)
    gold_set = set(ground_truth)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    # Heuristic scoring
    correctness = max(1, min(10, int(10 * tp / max(len(gold_set), 1))))
    specificity = 8 if not any(c.endswith(".9") for c in predicted) else 6
    has_evidence_quotes = any(q in predicted_reasoning.lower()
                              for q in ["documented", "echo", "labs", "history"])
    evidence_grounding = 9 if has_evidence_quotes else 5
    reasoning_soundness = 8 if len(predicted_reasoning) > 50 else 4
    hallucination_freedom = 10 - fp  # each spurious code = -1
    auditor_defensibility = (correctness + evidence_grounding) // 2

    rubric = JudgeRubric(
        correctness=correctness,
        specificity=specificity,
        evidence_grounding=evidence_grounding,
        reasoning_soundness=reasoning_soundness,
        hallucination_freedom=max(1, hallucination_freedom),
        auditor_defensibility=auditor_defensibility,
    )

    # Weighted average — auditor_defensibility weighted 2x
    overall = (
        rubric.correctness + rubric.specificity + rubric.evidence_grounding
        + rubric.reasoning_soundness + rubric.hallucination_freedom
        + 2 * rubric.auditor_defensibility
    ) / 7

    if overall >= 7.0:
        verdict = "pass"
    elif overall >= 5.0:
        verdict = "borderline"
    else:
        verdict = "fail"

    strengths, weaknesses, suggestions = [], [], []
    if tp > 0:
        strengths.append(f"Correctly identified {tp} of {len(gold_set)} codes")
    if has_evidence_quotes:
        strengths.append("Reasoning cites concrete clinical evidence")
    if fn > 0:
        missed = list(gold_set - pred_set)
        weaknesses.append(f"Missed {fn} required codes: {missed}")
        suggestions.append(f"Add prompt examples covering missed code patterns")
    if fp > 0:
        spurious = list(pred_set - gold_set)
        weaknesses.append(f"Predicted {fp} codes not in ground truth: {spurious}")
        suggestions.append("Strengthen Critic's specificity rules")
    if not has_evidence_quotes:
        weaknesses.append("Reasoning lacks quoted evidence from the note")
        suggestions.append("Require evidence_spans in Actor output schema")

    return JudgeVerdict(
        rubric=rubric,
        overall_score=round(overall, 2),
        pass_fail=verdict,
        strengths=strengths or ["No notable strengths"],
        weaknesses=weaknesses or ["No notable weaknesses"],
        suggested_improvements=suggestions or ["No specific improvements identified"],
        summary=f"Pred {len(predicted)} codes, "
                f"matched {tp}/{len(gold_set)} truth, "
                f"verdict={verdict}.",
    )


# ============================================================================
# 4. Real-LLM judge stub (uncomment + configure to use)
# ============================================================================

def real_llm_judge(note: str, predicted: list, predicted_reasoning: str,
                   ground_truth: list, llm=None) -> JudgeVerdict:
    """Uses a real LLM with structured output.
    Pass an LLM client that supports .with_structured_output (LangChain pattern)."""

    if llm is None:
        raise ValueError("Provide an LLM client. Example:\n"
                         "    from langchain_anthropic import ChatAnthropic\n"
                         "    llm = ChatAnthropic(model='claude-opus-4-7', temperature=0)\n"
                         "    real_llm_judge(..., llm=llm)")

    # Bind the schema — forces JSON matching JudgeVerdict
    judge_chain = llm.with_structured_output(JudgeVerdict)

    user_msg = build_judge_prompt(note, predicted, predicted_reasoning, ground_truth)
    full_prompt = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    return judge_chain.invoke(full_prompt)


# ============================================================================
# 5. Integration with EvaluationTracker
# ============================================================================

@dataclass
class EvalCaseWithJudge:
    case_id: str
    predicted: list
    ground_truth: list
    metrics: dict
    judge_verdict: Optional[JudgeVerdict] = None
    metadata: dict = field(default_factory=dict)


def evaluate_with_judge(case_id: str, note: str, predicted: list,
                        predicted_reasoning: str, ground_truth: list,
                        use_real_llm: bool = False, llm=None):
    """End-to-end: compute metrics + judge verdict for one case."""
    pred_set = set(predicted)
    gold_set = set(ground_truth)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    metrics = {"precision": round(p, 3), "recall": round(r, 3),
               "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}

    if use_real_llm:
        verdict = real_llm_judge(note, predicted, predicted_reasoning,
                                 ground_truth, llm=llm)
    else:
        verdict = simulate_judge(note, predicted, predicted_reasoning,
                                 ground_truth)

    return EvalCaseWithJudge(
        case_id=case_id,
        predicted=predicted,
        ground_truth=ground_truth,
        metrics=metrics,
        judge_verdict=verdict,
    )


# ============================================================================
# 6. Demo
# ============================================================================

def demo():
    note = (
        "67M with SOB and bilateral leg swelling. History of CHF, CKD stage 3, "
        "Type 2 DM. Labs: creatinine elevated 1.4 to 2.8. BNP 1850. "
        "Echo: EF 30%. Diagnosed acute decompensated CHF and AKI on CKD."
    )
    predicted = ["I50.23", "N17.9", "N18.30"]
    predicted_reasoning = (
        "Acute decompensated CHF documented with EF 30% supports I50.23. "
        "Creatinine doubling labs documented AKI N17.9. "
        "History of CKD stage 3 supports N18.30."
    )
    ground_truth = ["I50.23", "N17.9", "N18.30", "E11.9"]  # missed diabetes

    case = evaluate_with_judge(
        case_id="chart_42",
        note=note,
        predicted=predicted,
        predicted_reasoning=predicted_reasoning,
        ground_truth=ground_truth,
        use_real_llm=False,  # set True with real LLM client
    )

    print("=" * 70)
    print(f"CASE: {case.case_id}")
    print("=" * 70)
    print(f"Predicted:    {case.predicted}")
    print(f"Ground truth: {case.ground_truth}")
    print(f"Metrics:      {case.metrics}")
    print()

    v = case.judge_verdict
    print("=" * 70)
    print(f"JUDGE VERDICT: {v.pass_fail.upper()}  (overall {v.overall_score}/10)")
    print("=" * 70)
    print("\nRubric breakdown:")
    print(f"  Correctness:           {v.rubric.correctness}/10")
    print(f"  Specificity:           {v.rubric.specificity}/10")
    print(f"  Evidence grounding:    {v.rubric.evidence_grounding}/10")
    print(f"  Reasoning soundness:   {v.rubric.reasoning_soundness}/10")
    print(f"  Hallucination freedom: {v.rubric.hallucination_freedom}/10")
    print(f"  Auditor defensibility: {v.rubric.auditor_defensibility}/10")

    print(f"\nSummary: {v.summary}")
    print(f"\nStrengths:")
    for s in v.strengths:
        print(f"  + {s}")
    print(f"\nWeaknesses:")
    for w in v.weaknesses:
        print(f"  - {w}")
    print(f"\nSuggested improvements:")
    for s in v.suggested_improvements:
        print(f"  → {s}")


if __name__ == "__main__":
    demo()