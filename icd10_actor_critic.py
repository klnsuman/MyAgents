"""
ICD10 Code Detection App — Actor-Critic Debate + Arbiter + Reasoning Bank

Architecture:
  Actor        → Proposes ICD10 codes; revises them after each Critic response
  Critic       → Reviews codes, agrees/disagrees per code with feedback
  Debate Loop  → Repeats Actor↔Critic up to MAX_DEBATE_ROUNDS until full consensus
  Arbiter      → Sees the full debate history and produces the definitive code list
  ReasoningBank→ Curated rules and clinical patterns injected into every agent prompt
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional
import anthropic

MAX_DEBATE_ROUNDS = 5


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class ICD10Proposal:
    code: str
    description: str
    rationale: str
    confidence: float        # 0–1


@dataclass
class CriticVerdict:
    code: str
    agreed: bool             # True = Critic agrees with Actor on this code
    score: float             # 0–1
    feedback: str
    suggested_replacement: Optional[str] = None


@dataclass
class DebateRound:
    round_num: int
    proposals: list[ICD10Proposal]
    verdicts: list[CriticVerdict]
    consensus: bool          # True when Critic agreed on ALL codes this round


@dataclass
class FinalCode:
    code: str
    description: str
    confidence: float
    source: str              # "debate-consensus" | "actor-revised" | "arbiter-added"
    rationale: str


@dataclass
class DiagnosisResult:
    debate_rounds: list[DebateRound]
    final_codes: list[FinalCode]
    arbiter_notes: str
    rounds_to_consensus: int   # 0 = no consensus reached within limit


# ─────────────────────────────────────────────
# Reasoning Bank
# ─────────────────────────────────────────────

class ReasoningBank:
    """
    Curated ICD-10 coding rules and clinical patterns injected into every
    agent prompt. In production this would be a vector DB / retrieval layer.
    """

    RULES = [
        {
            "id": "R01", "category": "specificity",
            "rule": (
                "Always prefer the most specific ICD10 code available. "
                "Use unspecified codes (e.g. J18.9) only when the causative organism "
                "is truly unknown; otherwise use organism-specific codes (e.g. J15.0)."
            ),
        },
        {
            "id": "R02", "category": "sequencing",
            "rule": (
                "The principal diagnosis (reason for admission after study) is coded "
                "first. Comorbidities that affect treatment or LOS are coded additionally."
            ),
        },
        {
            "id": "R03", "category": "combination_codes",
            "rule": (
                "Use combination codes when available. E.g. E11.65 (T2DM with "
                "hyperglycaemia) rather than coding DM and hyperglycaemia separately."
            ),
        },
        {
            "id": "R04", "category": "excludes",
            "rule": (
                "Check Excludes1 and Excludes2 notes. Excludes1 = cannot code together; "
                "Excludes2 = may code together."
            ),
        },
        {
            "id": "R05", "category": "signs_symptoms",
            "rule": (
                "Do not separately code signs/symptoms that are integral to an established "
                "diagnosis unless instructed otherwise."
            ),
        },
        {
            "id": "R06", "category": "laterality",
            "rule": (
                "Specify laterality wherever the code set requires it "
                "(e.g. M79.621 right carpal tunnel vs M79.622 left)."
            ),
        },
        {
            "id": "R07", "category": "chronic_acute",
            "rule": (
                "When both acute and chronic forms coexist, code the acute form first "
                "unless guidelines specify otherwise."
            ),
        },
        {
            "id": "R08", "category": "completeness",
            "rule": (
                "Review for secondary diagnoses: complications, anaemia, malnutrition, "
                "pressure ulcers, DVT prophylaxis indications, substance use disorders "
                "that affect care."
            ),
        },
    ]

    COMMON_PATTERNS = [
        {
            "pattern": "chest_pain_workup",
            "keywords": ["chest pain", "troponin", "ekg", "rule out mi"],
            "typical_codes": ["R07.9", "I25.10", "Z87.891"],
            "notes": "If MI ruled out, code the symptom. If confirmed, code the MI type.",
        },
        {
            "pattern": "sepsis",
            "keywords": ["sepsis", "sirs", "bacteremia", "septic shock"],
            "typical_codes": ["A41.9", "R65.20", "R65.21"],
            "notes": "Code the underlying infection first, then sepsis. Septic shock needs R65.21.",
        },
        {
            "pattern": "diabetes_complications",
            "keywords": ["diabetes", "dm", "hyperglycemia", "neuropathy", "retinopathy"],
            "typical_codes": ["E11.65", "E11.40", "E11.311"],
            "notes": "Use E10/E11 combination codes to capture DM + complication together.",
        },
        {
            "pattern": "copd_exacerbation",
            "keywords": ["copd", "exacerbation", "acute bronchitis", "emphysema"],
            "typical_codes": ["J44.1", "J44.0"],
            "notes": "J44.1 = acute exacerbation; J44.0 = acute lower respiratory infection.",
        },
        {
            "pattern": "heart_failure",
            "keywords": ["heart failure", "chf", "ef", "bnp", "edema"],
            "typical_codes": ["I50.9", "I50.30", "I50.32", "I50.22"],
            "notes": "Specify systolic/diastolic and acute/chronic/acute-on-chronic.",
        },
    ]

    def get_relevant_patterns(self, chart_text: str) -> list[dict]:
        chart_lower = chart_text.lower()
        return [p for p in self.COMMON_PATTERNS
                if any(kw in chart_lower for kw in p["keywords"])]

    def format_for_prompt(self, chart_text: str) -> str:
        lines = ["=== REASONING BANK ===", "", "--- ICD-10 Coding Rules ---"]
        for r in self.RULES:
            lines.append(f"[{r['id']}] ({r['category'].upper()}): {r['rule']}")

        patterns = self.get_relevant_patterns(chart_text)
        if patterns:
            lines += ["", "--- Matched Clinical Patterns ---"]
            for p in patterns:
                lines.append(f"Pattern '{p['pattern']}': codes={p['typical_codes']}  |  {p['notes']}")

        lines.append("=== END REASONING BANK ===")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _extract_text(response) -> str:
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def _parse_json_array(raw: str, agent: str) -> list:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        print(f"[{agent}] WARNING: could not parse JSON array.\n{raw[:400]}")
        return []


def _parse_json_object(raw: str, agent: str) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        print(f"[{agent}] WARNING: could not parse JSON object.\n{raw[:400]}")
        return {}


def _proposals_to_json(proposals: list[ICD10Proposal]) -> str:
    return json.dumps(
        [{"code": p.code, "description": p.description,
          "rationale": p.rationale, "confidence": p.confidence}
         for p in proposals],
        indent=2,
    )


def _debate_history_to_text(rounds: list[DebateRound]) -> str:
    lines = []
    for r in rounds:
        lines.append(f"--- Round {r.round_num} ---")
        lines.append("Actor proposed:")
        for p in r.proposals:
            lines.append(f"  {p.code}: {p.description}  (conf {p.confidence:.2f})")
            lines.append(f"    Rationale: {p.rationale}")
        lines.append("Critic verdicts:")
        for v in r.verdicts:
            status = "AGREED" if v.agreed else "DISAGREED"
            lines.append(f"  [{status}] {v.code} score={v.score:.2f}: {v.feedback}")
            if v.suggested_replacement:
                lines.append(f"    → Critic suggests: {v.suggested_replacement}")
        lines.append(f"Consensus reached this round: {r.consensus}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Actor Agent
# ─────────────────────────────────────────────

ACTOR_SYSTEM = (
    "You are a board-certified medical coder (RHIA/CCS) specialising in ICD-10-CM/PCS. "
    "Your job is to propose ICD-10 codes for a medical chart. When the Critic disagrees "
    "with a code, you must reconsider and either defend it with stronger evidence or revise "
    "it to a more accurate code. Return ONLY valid JSON, no markdown fences."
)


class ActorAgent:

    def __init__(self, client: anthropic.Anthropic, bank: ReasoningBank):
        self.client = client
        self.bank = bank

    def propose(self, chart: str) -> list[ICD10Proposal]:
        """Initial proposal — no prior debate history."""
        bank_ctx = self.bank.format_for_prompt(chart)
        prompt = f"""{bank_ctx}

=== MEDICAL CHART ===
{chart}
=== END CHART ===

Analyse the chart and return a JSON array of ALL clinically relevant ICD-10 codes.
Each element:
{{
  "code": "<ICD-10 code>",
  "description": "<official ICD-10 description>",
  "rationale": "<why this code is supported by the chart>",
  "confidence": <float 0.0–1.0>
}}

Include: principal diagnosis, secondary diagnoses, comorbidities, complications.
Apply the reasoning bank rules above.
"""
        print("\n[ACTOR] Round 1 — initial proposal …")
        raw = _extract_text(self.client.messages.create(
            model="claude-opus-4-6", max_tokens=4096,
            thinking={"type": "adaptive"}, system=ACTOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ))
        proposals = self._parse(raw)
        print(f"[ACTOR] Proposed {len(proposals)} code(s).")
        return proposals

    def revise(
        self,
        chart: str,
        current_proposals: list[ICD10Proposal],
        verdicts: list[CriticVerdict],
        round_num: int,
    ) -> list[ICD10Proposal]:
        """Revise proposals after Critic feedback."""
        bank_ctx = self.bank.format_for_prompt(chart)
        disagreements = [v for v in verdicts if not v.agreed]
        verdict_json = json.dumps(
            [{"code": v.code, "agreed": v.agreed, "score": v.score,
              "feedback": v.feedback, "suggested_replacement": v.suggested_replacement}
             for v in verdicts],
            indent=2,
        )
        prompt = f"""{bank_ctx}

=== MEDICAL CHART ===
{chart}
=== END CHART ===

=== YOUR PREVIOUS PROPOSALS (Round {round_num - 1}) ===
{_proposals_to_json(current_proposals)}

=== CRITIC FEEDBACK ===
{verdict_json}

The Critic DISAGREED with {len(disagreements)} code(s).
For each disagreed code you must either:
  a) Replace it with the Critic's suggested code if clinically correct, OR
  b) Replace it with a better-supported code you identify, OR
  c) Keep it ONLY if you can provide substantially stronger documentation evidence.

Also keep all codes the Critic agreed with.
You may add newly identified codes if clearly supported by the chart.

Return the REVISED JSON array of ICD-10 codes (same schema as before).
"""
        print(f"\n[ACTOR] Round {round_num} — revising after Critic feedback …")
        raw = _extract_text(self.client.messages.create(
            model="claude-opus-4-6", max_tokens=4096,
            thinking={"type": "adaptive"}, system=ACTOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ))
        proposals = self._parse(raw)
        print(f"[ACTOR] Revised to {len(proposals)} code(s).")
        return proposals

    def _parse(self, raw: str) -> list[ICD10Proposal]:
        data = _parse_json_array(raw, "ACTOR")
        return [
            ICD10Proposal(
                code=item.get("code", ""),
                description=item.get("description", ""),
                rationale=item.get("rationale", ""),
                confidence=float(item.get("confidence", 0.5)),
            )
            for item in data if item.get("code")
        ]


# ─────────────────────────────────────────────
# Critic Agent
# ─────────────────────────────────────────────

CRITIC_SYSTEM = (
    "You are a senior ICD-10 auditor and Clinical Documentation Improvement (CDI) "
    "specialist. You review proposed ICD-10 codes and decide whether you AGREE or "
    "DISAGREE with each one, providing detailed feedback. When you disagree, suggest "
    "the correct code. If the Actor has revised a code you previously rejected and "
    "the new code is better, you should now agree. Return ONLY valid JSON, no markdown fences."
)


class CriticAgent:

    def __init__(self, client: anthropic.Anthropic, bank: ReasoningBank):
        self.client = client
        self.bank = bank

    def evaluate(
        self,
        chart: str,
        proposals: list[ICD10Proposal],
        round_num: int,
        prior_rounds: list[DebateRound],
    ) -> tuple[list[CriticVerdict], bool]:
        """
        Returns (verdicts, consensus).
        consensus=True when Critic agrees with ALL proposed codes.
        """
        bank_ctx = self.bank.format_for_prompt(chart)
        history = ""
        if prior_rounds:
            history = "\n=== PRIOR DEBATE ROUNDS ===\n" + _debate_history_to_text(prior_rounds) + "\n"

        prompt = f"""{bank_ctx}
{history}
=== MEDICAL CHART ===
{chart}
=== END CHART ===

=== ACTOR'S CURRENT PROPOSALS (Round {round_num}) ===
{_proposals_to_json(proposals)}

For EACH proposed code, return a JSON array where each element has:
{{
  "code": "<the proposed code>",
  "agreed": <true if you agree this code is correct | false if not>,
  "score": <float 0.0–1.0 — your confidence in this code>,
  "feedback": "<specific clinical/coding rationale for your verdict>",
  "suggested_replacement": "<better ICD-10 code if you disagree, otherwise null>"
}}

Agree when: code is correct, well-supported by documentation, and appropriately specific.
Disagree when: code is wrong, too vague, not supported, sequenced incorrectly, or a
  combination code should be used instead.

Be consistent: if the Actor adopted your previous suggestion, agree with it now.
"""
        print(f"\n[CRITIC] Round {round_num} — reviewing {len(proposals)} code(s) …")
        raw = _extract_text(self.client.messages.create(
            model="claude-opus-4-6", max_tokens=4096,
            thinking={"type": "adaptive"}, system=CRITIC_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ))
        verdicts = self._parse(raw)
        agreed_count = sum(1 for v in verdicts if v.agreed)
        consensus = agreed_count == len(verdicts) and len(verdicts) > 0
        status = "CONSENSUS ✓" if consensus else f"{agreed_count}/{len(verdicts)} agreed"
        print(f"[CRITIC] Round {round_num} — {status}")
        return verdicts, consensus

    def _parse(self, raw: str) -> list[CriticVerdict]:
        data = _parse_json_array(raw, "CRITIC")
        return [
            CriticVerdict(
                code=item.get("code", ""),
                agreed=bool(item.get("agreed", False)),
                score=float(item.get("score", 0.0)),
                feedback=item.get("feedback", ""),
                suggested_replacement=item.get("suggested_replacement"),
            )
            for item in data if item.get("code")
        ]


# ─────────────────────────────────────────────
# Arbiter Agent
# ─────────────────────────────────────────────

ARBITER_SYSTEM = (
    "You are the final ICD-10 arbitration authority — a physician advisor with deep "
    "coding expertise. You receive the full Actor-Critic debate history and the medical "
    "chart. Your job is to produce the definitive, correctly sequenced ICD-10 code list. "
    "Return ONLY valid JSON, no markdown fences."
)


class ArbiterAgent:

    def __init__(self, client: anthropic.Anthropic, bank: ReasoningBank):
        self.client = client
        self.bank = bank

    def arbitrate(
        self,
        chart: str,
        debate_rounds: list[DebateRound],
    ) -> tuple[list[FinalCode], str]:
        bank_ctx = self.bank.format_for_prompt(chart)
        debate_text = _debate_history_to_text(debate_rounds)
        last_proposals = _proposals_to_json(debate_rounds[-1].proposals)
        consensus_reached = debate_rounds[-1].consensus

        prompt = f"""{bank_ctx}

=== MEDICAL CHART ===
{chart}
=== END CHART ===

=== FULL DEBATE HISTORY ({len(debate_rounds)} round(s)) ===
{debate_text}

=== ACTOR'S FINAL PROPOSALS ===
{last_proposals}

Consensus reached: {consensus_reached}

Your tasks:
1. Use the final agreed codes as the foundation.
2. For any remaining disagreements: make the authoritative call based on chart evidence.
3. Correct any sequencing errors (principal diagnosis first).
4. Add any codes clearly documented in the chart that both agents missed.
5. Remove any codes not supported by the documentation.

Return a JSON object:
{{
  "final_codes": [
    {{
      "code": "<ICD-10 code>",
      "description": "<official description>",
      "confidence": <float 0.0–1.0>,
      "source": "<debate-consensus | actor-revised | arbiter-added | arbiter-overridden>",
      "rationale": "<brief justification>"
    }}
  ],
  "arbiter_notes": "<summary of debate outcome, key decisions, overrides, additions>"
}}
"""
        print("\n[ARBITER] Producing final code list …")
        raw = _extract_text(self.client.messages.create(
            model="claude-opus-4-6", max_tokens=4096,
            thinking={"type": "adaptive"}, system=ARBITER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ))
        data = _parse_json_object(raw, "ARBITER")
        codes = [
            FinalCode(
                code=item.get("code", ""),
                description=item.get("description", ""),
                confidence=float(item.get("confidence", 0.5)),
                source=item.get("source", "unknown"),
                rationale=item.get("rationale", ""),
            )
            for item in data.get("final_codes", []) if item.get("code")
        ]
        notes = data.get("arbiter_notes", "")
        print(f"[ARBITER] Finalised {len(codes)} code(s).")
        return codes, notes


# ─────────────────────────────────────────────
# Orchestrator — Debate Loop
# ─────────────────────────────────────────────

class ICD10Detector:
    """
    Runs the Actor↔Critic debate loop (up to MAX_DEBATE_ROUNDS), then
    passes the full debate history to the Arbiter for the final ruling.
    """

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.bank = ReasoningBank()
        self.actor = ActorAgent(self.client, self.bank)
        self.critic = CriticAgent(self.client, self.bank)
        self.arbiter = ArbiterAgent(self.client, self.bank)

    def detect(self, chart: str) -> DiagnosisResult:
        print("=" * 60)
        print(f"ICD10 DEBATE PIPELINE  (max {MAX_DEBATE_ROUNDS} rounds)")
        print("=" * 60)

        debate_rounds: list[DebateRound] = []
        consensus = False

        # Round 1 — Actor's initial proposal
        proposals = self.actor.propose(chart)
        verdicts, consensus = self.critic.evaluate(chart, proposals, 1, [])
        debate_rounds.append(DebateRound(1, proposals, verdicts, consensus))

        # Rounds 2–N — debate until consensus or limit
        for rnd in range(2, MAX_DEBATE_ROUNDS + 1):
            if consensus:
                break
            proposals = self.actor.revise(chart, proposals, verdicts, rnd)
            verdicts, consensus = self.critic.evaluate(chart, proposals, rnd, debate_rounds)
            debate_rounds.append(DebateRound(rnd, proposals, verdicts, consensus))

        rounds_to_consensus = next(
            (r.round_num for r in debate_rounds if r.consensus), 0
        )
        if consensus:
            print(f"\n✓ Consensus reached in {rounds_to_consensus} round(s).")
        else:
            print(f"\n⚠ No full consensus after {MAX_DEBATE_ROUNDS} rounds — Arbiter will decide.")

        # Arbiter sees the full debate
        final_codes, arbiter_notes = self.arbiter.arbitrate(chart, debate_rounds)

        return DiagnosisResult(
            debate_rounds=debate_rounds,
            final_codes=final_codes,
            arbiter_notes=arbiter_notes,
            rounds_to_consensus=rounds_to_consensus,
        )

    # ── Reporting ──────────────────────────────

    def print_report(self, result: DiagnosisResult) -> None:
        print("\n" + "=" * 60)
        print("DEBATE REPORT")
        print("=" * 60)

        for r in result.debate_rounds:
            print(f"\n{'─'*55}")
            consensus_tag = " ✓ CONSENSUS" if r.consensus else ""
            print(f"  ROUND {r.round_num}{consensus_tag}")
            print(f"{'─'*55}")

            print("  ACTOR proposals:")
            for p in r.proposals:
                print(f"    {p.code:<12} {p.description:<45} conf={p.confidence:.2f}")
                print(f"               {p.rationale[:90]}")

            print("  CRITIC verdicts:")
            for v in r.verdicts:
                icon = "✓ AGREED   " if v.agreed else "✗ DISAGREED"
                print(f"    [{icon}] {v.code:<12} score={v.score:.2f}")
                print(f"                     {v.feedback[:90]}")
                if v.suggested_replacement:
                    print(f"                     → Suggests: {v.suggested_replacement}")

        if result.rounds_to_consensus:
            print(f"\n  Consensus after {result.rounds_to_consensus} round(s).")
        else:
            print(f"\n  No consensus after {MAX_DEBATE_ROUNDS} round(s) — Arbiter overrides.")

        src_icon = {
            "debate-consensus": "✅",
            "actor-revised":    "🔄",
            "arbiter-added":    "➕",
            "arbiter-overridden": "⚖️",
        }
        print(f"\n{'─'*55}")
        print("  FINAL ICD-10 CODES  (Arbiter)")
        print(f"{'─'*55}")
        for i, fc in enumerate(result.final_codes, 1):
            icon = src_icon.get(fc.source, "❓")
            print(f"  {i}. {icon} {fc.code:<12} {fc.description}")
            print(f"        conf={fc.confidence:.2f}  source={fc.source}")
            print(f"        {fc.rationale[:110]}")

        print(f"\n{'─'*55}")
        print("  ARBITER NOTES")
        print(f"{'─'*55}")
        print(f"  {result.arbiter_notes}")
        print("=" * 60)


# ─────────────────────────────────────────────
# Sample charts
# ─────────────────────────────────────────────

SAMPLE_CHARTS = {
    "chest_pain": """
PATIENT: 67-year-old male
CHIEF COMPLAINT: Chest pain and shortness of breath × 3 hours

HISTORY OF PRESENT ILLNESS:
Mr. Johnson presents with sudden onset substernal chest pressure radiating to
the left arm, accompanied by diaphoresis and dyspnea. He has a 30 pack-year
smoking history, type 2 diabetes mellitus with peripheral neuropathy
(last HbA1c 9.2%), and known hypertension on lisinopril 10 mg daily.

PHYSICAL EXAM:
BP 158/94, HR 102, RR 22, SpO2 94% on room air, Temp 98.6°F.
Lungs: bilateral basilar crackles. Heart: S3 gallop present.
Extremities: 2+ pitting oedema bilateral lower extremities.

LABS:
Troponin I: 2.8 ng/mL (elevated × 2). BNP: 1,840 pg/mL (markedly elevated).
HbA1c: 9.4%. BMP: K+ 3.2, Cr 1.8 (baseline 1.2), Na 138.
CBC: Hgb 10.2 g/dL (microcytic), MCV 74.

EKG: ST elevation in leads V1-V4; new LBBB pattern.

IMPRESSION:
1. ST-elevation myocardial infarction (anterior STEMI) — taken emergently to
   cardiac catheterisation lab; drug-eluting stent placed in LAD.
2. Acute systolic heart failure (EF 35% on bedside echo) precipitated by MI.
3. Type 2 diabetes mellitus with peripheral neuropathy; hyperglycaemia.
4. Hypertensive urgency.
5. Acute kidney injury (Cr 1.8, baseline 1.2) — likely cardiorenal.
6. Iron deficiency anaemia (microcytic, Hgb 10.2).
7. Hypokalaemia (K+ 3.2).

PLAN:
Dual antiplatelet therapy, heparin drip, IV furosemide, insulin protocol,
IV iron sucrose, potassium repletion, nephrology consult.
""",
    "sepsis": """
PATIENT: 54-year-old female
CHIEF COMPLAINT: Fever, confusion, right leg pain × 2 days

HISTORY:
Mrs. Patel has type 2 diabetes (poorly controlled), hypertension, and CKD stage 3.
She noticed redness and swelling over the right lower leg 4 days ago which worsened
rapidly. She became febrile (T 39.8°C), tachycardic (HR 118), hypotensive
(BP 88/52) and confused in the ED.

PHYSICAL EXAM:
Right leg: 15×20 cm area of erythema, warmth, induration, and crepitus on palpation.
Skin: dusky discolouration with bullae formation. Marked tenderness.

LABS:
WBC 24,000 with 18% bands. Lactate 4.6 mmol/L. Procalcitonin 48 ng/mL.
Creatinine 3.4 (baseline 1.9). Blood cultures: Streptococcus pyogenes (Group A Strep).
Glucose 480 mg/dL.

IMAGING:
CT right leg: gas in soft tissues consistent with necrotising fasciitis.

IMPRESSION:
1. Septic shock secondary to necrotising fasciitis of the right lower leg,
   caused by Group A Streptococcus (Streptococcus pyogenes).
2. Type 2 diabetes mellitus with hyperglycaemia (glucose 480).
3. Acute kidney injury (Cr 3.4, baseline 1.9).
4. Hypertension (documented history, on amlodipine).

PLAN:
Emergency surgical debridement, broad-spectrum IV antibiotics (piperacillin/
tazobactam + clindamycin), vasopressors for shock, insulin drip, ICU admission.
""",
}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import sys

    detector = ICD10Detector()

    chart_key = "chest_pain"
    if len(sys.argv) > 1 and sys.argv[1] in SAMPLE_CHARTS:
        chart_key = sys.argv[1]

    chart = SAMPLE_CHARTS[chart_key]
    print(f"\nProcessing chart: '{chart_key}' ({len(chart)} chars)\n")

    result = detector.detect(chart)
    detector.print_report(result)

    # Save full JSON output
    output = {
        "chart": chart_key,
        "rounds_to_consensus": result.rounds_to_consensus,
        "debate_rounds": [
            {
                "round": r.round_num,
                "consensus": r.consensus,
                "proposals": [
                    {"code": p.code, "description": p.description,
                     "rationale": p.rationale, "confidence": p.confidence}
                    for p in r.proposals
                ],
                "verdicts": [
                    {"code": v.code, "agreed": v.agreed, "score": v.score,
                     "feedback": v.feedback,
                     "suggested_replacement": v.suggested_replacement}
                    for v in r.verdicts
                ],
            }
            for r in result.debate_rounds
        ],
        "final_codes": [
            {"code": fc.code, "description": fc.description,
             "confidence": fc.confidence, "source": fc.source,
             "rationale": fc.rationale}
            for fc in result.final_codes
        ],
        "arbiter_notes": result.arbiter_notes,
    }
    out_path = f"icd10_result_{chart_key}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON result saved → {out_path}")


if __name__ == "__main__":
    main()
