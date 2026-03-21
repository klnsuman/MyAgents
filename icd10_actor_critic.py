"""
ICD10 Code Detection App — Actor-Critic-Arbiter with Reasoning Bank

Architecture:
  Actor        → Proposes ICD10 codes from the medical chart
  Critic       → Evaluates each proposed code (accuracy, completeness, specificity)
  Arbiter      → Resolves conflicts and produces the final code list
  ReasoningBank→ Stores cross-case patterns and rules that inform each agent
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional
import anthropic

# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class ICD10Proposal:
    code: str
    description: str
    rationale: str
    confidence: float          # 0–1


@dataclass
class CriticVerdict:
    code: str
    approved: bool
    score: float               # 0–1
    feedback: str
    suggested_replacement: Optional[str] = None


@dataclass
class FinalCode:
    code: str
    description: str
    confidence: float
    source: str                # "actor-accepted" | "actor-revised" | "arbiter-added"
    rationale: str


@dataclass
class DiagnosisResult:
    proposed: list[ICD10Proposal]
    verdicts: list[CriticVerdict]
    final_codes: list[FinalCode]
    arbiter_notes: str


# ─────────────────────────────────────────────
# Reasoning Bank
# ─────────────────────────────────────────────

class ReasoningBank:
    """
    A curated store of ICD10 coding rules, clinical pearls, and common patterns.
    In production this would be a vector DB / retrieval system; here it is
    an in-memory knowledge base that agents can query.
    """

    RULES = [
        {
            "id": "R01",
            "category": "specificity",
            "rule": (
                "Always prefer the most specific ICD10 code available. "
                "E.g., use J18.9 (Pneumonia, unspecified organism) only when the "
                "causative agent is unknown; otherwise use organism-specific codes "
                "such as J15.0 (Pneumonia due to Klebsiella pneumoniae)."
            ),
        },
        {
            "id": "R02",
            "category": "sequencing",
            "rule": (
                "The principal diagnosis (reason for admission after study) is listed "
                "first. Comorbidities that affect treatment or management are coded "
                "additionally."
            ),
        },
        {
            "id": "R03",
            "category": "combination_codes",
            "rule": (
                "Use combination codes when available. E.g., E11.65 (Type 2 DM with "
                "hyperglycemia) captures both conditions in one code rather than "
                "coding each separately."
            ),
        },
        {
            "id": "R04",
            "category": "excludes",
            "rule": (
                "Always check Excludes1 and Excludes2 notes. Excludes1 means the two "
                "conditions cannot be coded together; Excludes2 means they can."
            ),
        },
        {
            "id": "R05",
            "category": "signs_symptoms",
            "rule": (
                "Do not code signs and symptoms that are routinely associated with a "
                "disease process unless instructed otherwise. Code the definitive "
                "diagnosis when established."
            ),
        },
        {
            "id": "R06",
            "category": "laterality",
            "rule": (
                "Specify laterality when the code set requires it "
                "(e.g., M79.621 for right carpal tunnel syndrome vs M79.622 left)."
            ),
        },
        {
            "id": "R07",
            "category": "chronic_acute",
            "rule": (
                "When both acute and chronic forms of a condition exist and the patient "
                "has both, code the acute condition first unless guidelines specify "
                "otherwise."
            ),
        },
        {
            "id": "R08",
            "category": "completeness",
            "rule": (
                "Review the chart for secondary diagnoses: complications, co-existing "
                "conditions, anaemia, malnutrition, pressure ulcers, DVT prophylaxis "
                "indications, and substance use disorders that may affect care."
            ),
        },
    ]

    COMMON_PATTERNS = [
        {
            "pattern": "chest_pain_workup",
            "keywords": ["chest pain", "troponin", "EKG", "rule out MI"],
            "typical_codes": ["R07.9", "I25.10", "Z87.891"],
            "notes": "If MI ruled out, code the symptom. If confirmed, code the MI type.",
        },
        {
            "pattern": "sepsis",
            "keywords": ["sepsis", "SIRS", "bacteremia", "septic shock"],
            "typical_codes": ["A41.9", "R65.20", "R65.21"],
            "notes": (
                "Code the underlying infection first, then the sepsis code. "
                "Septic shock (R65.21) requires an additional code for the shock."
            ),
        },
        {
            "pattern": "diabetes_complications",
            "keywords": ["diabetes", "DM", "hyperglycemia", "neuropathy", "retinopathy"],
            "typical_codes": ["E11.65", "E11.40", "E11.311"],
            "notes": "Use combination codes (E10/E11 + 4th/5th digit) to capture complications.",
        },
        {
            "pattern": "copd_exacerbation",
            "keywords": ["COPD", "exacerbation", "acute bronchitis", "emphysema"],
            "typical_codes": ["J44.1", "J44.0"],
            "notes": "J44.1 = with acute exacerbation; J44.0 = with acute lower respiratory infection.",
        },
        {
            "pattern": "heart_failure",
            "keywords": ["heart failure", "CHF", "EF", "BNP", "edema"],
            "typical_codes": ["I50.9", "I50.30", "I50.32", "I50.22"],
            "notes": "Specify systolic/diastolic, acute/chronic/acute-on-chronic.",
        },
    ]

    def get_relevant_rules(self, chart_text: str) -> list[dict]:
        """Return all rules (could be filtered by keywords in a larger system)."""
        return self.RULES

    def get_relevant_patterns(self, chart_text: str) -> list[dict]:
        """Return patterns whose keywords appear in the chart text."""
        chart_lower = chart_text.lower()
        matched = []
        for pattern in self.COMMON_PATTERNS:
            if any(kw in chart_lower for kw in pattern["keywords"]):
                matched.append(pattern)
        return matched

    def format_for_prompt(self, chart_text: str) -> str:
        rules = self.get_relevant_rules(chart_text)
        patterns = self.get_relevant_patterns(chart_text)

        lines = ["=== REASONING BANK ===", "", "--- Coding Rules ---"]
        for r in rules:
            lines.append(f"[{r['id']}] ({r['category'].upper()}): {r['rule']}")

        if patterns:
            lines += ["", "--- Matched Clinical Patterns ---"]
            for p in patterns:
                lines.append(f"Pattern: {p['pattern']}")
                lines.append(f"  Typical codes: {', '.join(p['typical_codes'])}")
                lines.append(f"  Notes: {p['notes']}")

        lines.append("=== END REASONING BANK ===")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────

class ActorAgent:
    """
    Reads the medical chart and proposes the initial set of ICD10 codes.
    Uses adaptive thinking to reason through complex presentations.
    """

    SYSTEM = (
        "You are a board-certified medical coder (RHIA/CCS) specialising in "
        "ICD-10-CM/PCS coding. Your task is to carefully read the provided medical "
        "chart and propose ALL clinically relevant ICD10 codes. "
        "Err on the side of completeness — the Critic will prune invalid codes. "
        "Return ONLY valid JSON, no markdown fences."
    )

    def __init__(self, client: anthropic.Anthropic, reasoning_bank: ReasoningBank):
        self.client = client
        self.bank = reasoning_bank

    def propose(self, chart: str) -> list[ICD10Proposal]:
        bank_context = self.bank.format_for_prompt(chart)

        user_prompt = f"""{bank_context}

=== MEDICAL CHART ===
{chart}
=== END CHART ===

Analyse the chart and return a JSON array of proposed ICD10 codes.
Each element must follow this schema:
{{
  "code": "<ICD10 code, e.g. J18.9>",
  "description": "<official ICD10 description>",
  "rationale": "<why this code is supported by the chart>",
  "confidence": <float 0.0–1.0>
}}

Identify ALL relevant diagnoses: principal, secondary, comorbidities, and
complications that affected care. Apply the reasoning bank rules above.
"""

        print("\n[ACTOR] Proposing ICD10 codes …")
        response = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=self.SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text content (skip thinking blocks)
        raw = ""
        for block in response.content:
            if block.type == "text":
                raw = block.text
                break

        proposals = self._parse(raw)
        print(f"[ACTOR] Proposed {len(proposals)} code(s).")
        return proposals

    def _parse(self, raw: str) -> list[ICD10Proposal]:
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find a JSON array in the text
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                print(f"[ACTOR] WARNING: Could not parse JSON. Raw output:\n{raw[:500]}")
                return []

        return [
            ICD10Proposal(
                code=item.get("code", ""),
                description=item.get("description", ""),
                rationale=item.get("rationale", ""),
                confidence=float(item.get("confidence", 0.5)),
            )
            for item in data
            if item.get("code")
        ]


class CriticAgent:
    """
    Evaluates each proposed code for correctness, specificity, and clinical
    validity. May suggest replacement codes.
    """

    SYSTEM = (
        "You are a senior ICD-10 auditor and clinical documentation improvement "
        "(CDI) specialist. Your role is to critically review proposed ICD10 codes "
        "and identify: incorrect codes, codes lacking specificity, sequencing errors, "
        "missing combination codes, and codes not supported by documentation. "
        "Return ONLY valid JSON, no markdown fences."
    )

    def __init__(self, client: anthropic.Anthropic, reasoning_bank: ReasoningBank):
        self.client = client
        self.bank = reasoning_bank

    def evaluate(self, chart: str, proposals: list[ICD10Proposal]) -> list[CriticVerdict]:
        bank_context = self.bank.format_for_prompt(chart)
        proposals_json = json.dumps(
            [
                {
                    "code": p.code,
                    "description": p.description,
                    "rationale": p.rationale,
                    "confidence": p.confidence,
                }
                for p in proposals
            ],
            indent=2,
        )

        user_prompt = f"""{bank_context}

=== MEDICAL CHART ===
{chart}
=== END CHART ===

=== ACTOR'S PROPOSED CODES ===
{proposals_json}
=== END PROPOSED CODES ===

For EACH proposed code, return a JSON array where each element has:
{{
  "code": "<the proposed code>",
  "approved": <true | false>,
  "score": <float 0.0–1.0 — your confidence in this code being correct>,
  "feedback": "<concise clinical/coding rationale for your verdict>",
  "suggested_replacement": "<alternative ICD10 code if the original is wrong, else null>"
}}

Be rigorous. Flag codes that are:
- Not supported by documented findings
- Too unspecific when a more specific code exists
- Sequenced incorrectly relative to each other
- Missing required 7th characters or laterality
"""

        print("\n[CRITIC] Evaluating proposed codes …")
        response = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=self.SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = ""
        for block in response.content:
            if block.type == "text":
                raw = block.text
                break

        verdicts = self._parse(raw)
        approved = sum(1 for v in verdicts if v.approved)
        print(f"[CRITIC] {approved}/{len(verdicts)} code(s) approved.")
        return verdicts

    def _parse(self, raw: str) -> list[CriticVerdict]:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                print(f"[CRITIC] WARNING: Could not parse JSON.\n{raw[:500]}")
                return []

        return [
            CriticVerdict(
                code=item.get("code", ""),
                approved=bool(item.get("approved", False)),
                score=float(item.get("score", 0.0)),
                feedback=item.get("feedback", ""),
                suggested_replacement=item.get("suggested_replacement"),
            )
            for item in data
            if item.get("code")
        ]


class ArbiterAgent:
    """
    Synthesises the Actor's proposals and Critic's verdicts into a final,
    authoritative ICD10 code list. Resolves conflicts, accepts replacements,
    and may add overlooked codes.
    """

    SYSTEM = (
        "You are the final ICD-10 arbitration authority — a physician advisor with "
        "deep coding expertise. You receive a medical chart, the Actor's proposed "
        "codes, and the Critic's verdicts. Your job is to produce the definitive, "
        "sequenced list of ICD10 codes that will be submitted for billing. "
        "Return ONLY valid JSON, no markdown fences."
    )

    def __init__(self, client: anthropic.Anthropic, reasoning_bank: ReasoningBank):
        self.client = client
        self.bank = reasoning_bank

    def arbitrate(
        self,
        chart: str,
        proposals: list[ICD10Proposal],
        verdicts: list[CriticVerdict],
    ) -> tuple[list[FinalCode], str]:
        bank_context = self.bank.format_for_prompt(chart)

        proposals_json = json.dumps(
            [{"code": p.code, "description": p.description, "rationale": p.rationale, "confidence": p.confidence}
             for p in proposals],
            indent=2,
        )
        verdicts_json = json.dumps(
            [{"code": v.code, "approved": v.approved, "score": v.score,
              "feedback": v.feedback, "suggested_replacement": v.suggested_replacement}
             for v in verdicts],
            indent=2,
        )

        user_prompt = f"""{bank_context}

=== MEDICAL CHART ===
{chart}
=== END CHART ===

=== ACTOR PROPOSALS ===
{proposals_json}

=== CRITIC VERDICTS ===
{verdicts_json}

Your tasks:
1. Accept codes the Critic approved (approved=true, high score).
2. For codes with suggested_replacement, use the replacement if clinically sound.
3. Reject codes the Critic flagged as unsupported unless you disagree with strong evidence.
4. Add any codes the Actor missed that are clearly documented in the chart.
5. Sequence codes correctly (principal diagnosis first).

Return a JSON object with two keys:
{{
  "final_codes": [
    {{
      "code": "<ICD10 code>",
      "description": "<official description>",
      "confidence": <float 0.0–1.0>,
      "source": "<actor-accepted | actor-revised | arbiter-added>",
      "rationale": "<brief justification>"
    }}
  ],
  "arbiter_notes": "<free-text summary of key decisions, conflicts resolved, codes added/dropped>"
}}
"""

        print("\n[ARBITER] Producing final code list …")
        response = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=self.SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = ""
        for block in response.content:
            if block.type == "text":
                raw = block.text
                break

        return self._parse(raw)

    def _parse(self, raw: str) -> tuple[list[FinalCode], str]:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                print(f"[ARBITER] WARNING: Could not parse JSON.\n{raw[:500]}")
                return [], "Parse error in arbiter output."

        final_codes = [
            FinalCode(
                code=item.get("code", ""),
                description=item.get("description", ""),
                confidence=float(item.get("confidence", 0.5)),
                source=item.get("source", "unknown"),
                rationale=item.get("rationale", ""),
            )
            for item in data.get("final_codes", [])
            if item.get("code")
        ]
        notes = data.get("arbiter_notes", "")
        print(f"[ARBITER] Finalised {len(final_codes)} code(s).")
        return final_codes, notes


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

class ICD10Detector:
    """
    Orchestrates the Actor → Critic → Arbiter pipeline for a medical chart.
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
        print("ICD10 ACTOR-CRITIC-ARBITER PIPELINE")
        print("=" * 60)

        # Stage 1 — Actor proposes
        proposals = self.actor.propose(chart)

        # Stage 2 — Critic evaluates
        verdicts = self.critic.evaluate(chart, proposals)

        # Stage 3 — Arbiter finalises
        final_codes, arbiter_notes = self.arbiter.arbitrate(chart, proposals, verdicts)

        return DiagnosisResult(
            proposed=proposals,
            verdicts=verdicts,
            final_codes=final_codes,
            arbiter_notes=arbiter_notes,
        )

    def print_report(self, result: DiagnosisResult) -> None:
        print("\n" + "=" * 60)
        print("PIPELINE REPORT")
        print("=" * 60)

        print(f"\n{'─'*50}")
        print("ACTOR PROPOSALS")
        print(f"{'─'*50}")
        for p in result.proposed:
            flag = "✓" if any(v.code == p.code and v.approved for v in result.verdicts) else "✗"
            print(f"  {flag} {p.code:<12} {p.description:<50} (conf: {p.confidence:.2f})")
            print(f"       Rationale: {p.rationale[:100]}")

        print(f"\n{'─'*50}")
        print("CRITIC VERDICTS")
        print(f"{'─'*50}")
        for v in result.verdicts:
            status = "APPROVED" if v.approved else "REJECTED"
            print(f"  [{status}] {v.code:<12} score={v.score:.2f}")
            print(f"           {v.feedback[:120]}")
            if v.suggested_replacement:
                print(f"           → Suggested replacement: {v.suggested_replacement}")

        print(f"\n{'─'*50}")
        print("FINAL ICD10 CODES  (Arbiter Decision)")
        print(f"{'─'*50}")
        for i, fc in enumerate(result.final_codes, 1):
            src_emoji = {"actor-accepted": "✅", "actor-revised": "🔄", "arbiter-added": "➕"}.get(fc.source, "❓")
            print(f"  {i}. {src_emoji} {fc.code:<12} {fc.description}")
            print(f"        Confidence: {fc.confidence:.2f}  |  Source: {fc.source}")
            print(f"        {fc.rationale[:120]}")

        print(f"\n{'─'*50}")
        print("ARBITER NOTES")
        print(f"{'─'*50}")
        print(f"  {result.arbiter_notes}")
        print("=" * 60)


# ─────────────────────────────────────────────
# Sample charts & main
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


def main():
    import sys

    detector = ICD10Detector()

    chart_key = "chest_pain"
    if len(sys.argv) > 1 and sys.argv[1] in SAMPLE_CHARTS:
        chart_key = sys.argv[1]

    chart = SAMPLE_CHARTS[chart_key]
    print(f"\nProcessing chart: '{chart_key}'")
    print(f"Chart length: {len(chart)} characters\n")

    result = detector.detect(chart)
    detector.print_report(result)

    # Also save JSON output
    output = {
        "chart": chart_key,
        "proposed": [
            {"code": p.code, "description": p.description,
             "rationale": p.rationale, "confidence": p.confidence}
            for p in result.proposed
        ],
        "verdicts": [
            {"code": v.code, "approved": v.approved, "score": v.score,
             "feedback": v.feedback, "suggested_replacement": v.suggested_replacement}
            for v in result.verdicts
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
    print(f"\nJSON result saved to: {out_path}")


if __name__ == "__main__":
    main()
