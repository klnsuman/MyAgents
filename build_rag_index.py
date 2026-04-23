"""
Build (or rebuild) the static RAG index for the ICD-10 pipeline.

Run this ONCE before first use, and again whenever:
  - CMS publishes new ICD-10 codes (Oct 1 each year)
  - You add new coding guidelines
  - You want to seed the reasoning bank with MIMIC cases

Usage:
    python build_rag_index.py           # build/rebuild
    python build_rag_index.py --reset   # wipe and rebuild from scratch
"""

import sys
from rag_store import ICD10RAGStore, ReasoningBankStore

# ─────────────────────────────────────────────────────────────────────────────
# Starter ICD-10 codes
# Replace / extend with the full CMS tabular list (cms.gov) for production.
# Format: {code, description, notes (optional)}
# ─────────────────────────────────────────────────────────────────────────────

STARTER_CODES = [
    # ── Cardiovascular ───────────────────────────────────────────────────────
    {"code": "I10",    "description": "Essential (primary) hypertension"},
    {"code": "I16.0",  "description": "Hypertensive urgency"},
    {"code": "I16.1",  "description": "Hypertensive emergency"},
    {"code": "I21.01", "description": "ST elevation MI involving left anterior descending coronary artery",
     "notes": "STEMI LAD — most common anterior STEMI site"},
    {"code": "I21.09", "description": "ST elevation MI involving other coronary artery of anterior wall"},
    {"code": "I21.11", "description": "ST elevation MI involving right coronary artery"},
    {"code": "I21.19", "description": "ST elevation MI involving other coronary artery of inferior wall"},
    {"code": "I21.3",  "description": "ST elevation MI of unspecified site"},
    {"code": "I21.4",  "description": "Non-ST elevation (NSTEMI) myocardial infarction"},
    {"code": "I22.0",  "description": "Subsequent ST elevation MI of anterior wall"},
    {"code": "I25.10", "description": "Atherosclerotic heart disease of native coronary artery without angina"},
    {"code": "I25.110","description": "Atherosclerotic heart disease of native coronary artery with unstable angina"},
    {"code": "I48.0",  "description": "Paroxysmal atrial fibrillation"},
    {"code": "I48.11", "description": "Longstanding persistent atrial fibrillation"},
    {"code": "I48.20", "description": "Chronic atrial fibrillation, unspecified"},
    {"code": "I50.21", "description": "Acute systolic (congestive) heart failure",
     "notes": "Use when EF reduced and onset acute"},
    {"code": "I50.22", "description": "Chronic systolic (congestive) heart failure"},
    {"code": "I50.23", "description": "Acute on chronic systolic (congestive) heart failure"},
    {"code": "I50.31", "description": "Acute diastolic (congestive) heart failure"},
    {"code": "I50.32", "description": "Chronic diastolic (congestive) heart failure"},
    {"code": "I50.33", "description": "Acute on chronic diastolic (congestive) heart failure"},
    {"code": "I50.9",  "description": "Heart failure, unspecified"},
    {"code": "I63.9",  "description": "Cerebral infarction, unspecified"},
    {"code": "I82.401","description": "Acute DVT of unspecified deep veins of right lower extremity"},
    {"code": "I82.402","description": "Acute DVT of unspecified deep veins of left lower extremity"},
    {"code": "I26.09", "description": "Other pulmonary embolism without acute cor pulmonale"},
    {"code": "I26.99", "description": "Other pulmonary embolism with acute cor pulmonale"},

    # ── Respiratory ──────────────────────────────────────────────────────────
    {"code": "J18.9",  "description": "Pneumonia, unspecified organism",
     "notes": "Use only when causative organism unknown; prefer organism-specific codes"},
    {"code": "J15.0",  "description": "Pneumonia due to Klebsiella pneumoniae"},
    {"code": "J15.1",  "description": "Pneumonia due to Pseudomonas"},
    {"code": "J15.211","description": "Pneumonia due to methicillin susceptible Staphylococcus aureus (MSSA)"},
    {"code": "J15.212","description": "Pneumonia due to methicillin resistant Staphylococcus aureus (MRSA)"},
    {"code": "J15.4",  "description": "Pneumonia due to other streptococci"},
    {"code": "J44.0",  "description": "COPD with acute lower respiratory infection"},
    {"code": "J44.1",  "description": "COPD with (acute) exacerbation"},
    {"code": "J44.9",  "description": "COPD, unspecified"},
    {"code": "J45.50", "description": "Severe persistent asthma, uncomplicated"},
    {"code": "J96.00", "description": "Acute respiratory failure, unspecified"},
    {"code": "J96.01", "description": "Acute respiratory failure with hypoxia"},
    {"code": "J96.11", "description": "Chronic respiratory failure with hypoxia"},

    # ── Endocrine / Diabetes ─────────────────────────────────────────────────
    {"code": "E11.9",  "description": "Type 2 diabetes mellitus without complications"},
    {"code": "E11.65", "description": "Type 2 diabetes mellitus with hyperglycemia",
     "notes": "Combination code — do not separately code hyperglycemia (R73.09)"},
    {"code": "E11.40", "description": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified"},
    {"code": "E11.41", "description": "Type 2 diabetes mellitus with diabetic mononeuropathy"},
    {"code": "E11.51", "description": "Type 2 diabetes mellitus with diabetic peripheral angiopathy without gangrene"},
    {"code": "E11.311","description": "Type 2 diabetes mellitus with unspecified diabetic retinopathy with macular edema"},
    {"code": "E11.00", "description": "Type 2 diabetes mellitus with hyperosmolarity without coma"},
    {"code": "E10.65", "description": "Type 1 diabetes mellitus with hyperglycemia"},
    {"code": "E87.1",  "description": "Hyponatremia"},
    {"code": "E87.6",  "description": "Hypokalemia"},
    {"code": "E86.0",  "description": "Dehydration"},
    {"code": "E66.01", "description": "Morbid (severe) obesity due to excess calories"},
    {"code": "E78.5",  "description": "Hyperlipidemia, unspecified"},

    # ── Kidney ───────────────────────────────────────────────────────────────
    {"code": "N17.9",  "description": "Acute kidney injury, unspecified",
     "notes": "Cardiorenal AKI common with heart failure — code both"},
    {"code": "N18.3",  "description": "Chronic kidney disease, stage 3 (moderate)"},
    {"code": "N18.4",  "description": "Chronic kidney disease, stage 4 (severe)"},
    {"code": "N18.5",  "description": "Chronic kidney disease, stage 5"},
    {"code": "N18.6",  "description": "End stage renal disease"},
    {"code": "N39.0",  "description": "Urinary tract infection, site not specified"},

    # ── Infections ───────────────────────────────────────────────────────────
    {"code": "A40.0",  "description": "Sepsis due to Streptococcus, group A (Streptococcus pyogenes)"},
    {"code": "A40.1",  "description": "Sepsis due to Streptococcus, group B"},
    {"code": "A41.01", "description": "Sepsis due to methicillin susceptible Staphylococcus aureus"},
    {"code": "A41.02", "description": "Sepsis due to methicillin resistant Staphylococcus aureus (MRSA)"},
    {"code": "A41.51", "description": "Sepsis due to Escherichia coli"},
    {"code": "A41.52", "description": "Sepsis due to Pseudomonas"},
    {"code": "A41.89", "description": "Other specified sepsis"},
    {"code": "A41.9",  "description": "Sepsis, unspecified organism"},

    # ── Sepsis severity ──────────────────────────────────────────────────────
    {"code": "R65.20", "description": "Severe sepsis without septic shock",
     "notes": "Code underlying infection first, then R65.20"},
    {"code": "R65.21", "description": "Severe sepsis with septic shock",
     "notes": "Code underlying infection first, then R65.21; add shock code if needed"},

    # ── Skin / Fascia ────────────────────────────────────────────────────────
    {"code": "M72.6",  "description": "Necrotizing fasciitis",
     "notes": "Code causative organism additionally (e.g. A40.0 for Group A Strep)"},
    {"code": "L03.115","description": "Cellulitis of right lower limb"},
    {"code": "L03.116","description": "Cellulitis of left lower limb"},
    {"code": "L89.90", "description": "Pressure ulcer of unspecified site, unspecified stage"},

    # ── Blood / Anaemia ──────────────────────────────────────────────────────
    {"code": "D50.0",  "description": "Iron deficiency anaemia secondary to blood loss (chronic)"},
    {"code": "D50.9",  "description": "Iron deficiency anaemia, unspecified"},
    {"code": "D62",    "description": "Acute posthemorrhagic anaemia"},
    {"code": "D64.9",  "description": "Anaemia, unspecified"},

    # ── Signs / Symptoms ─────────────────────────────────────────────────────
    {"code": "R07.9",  "description": "Chest pain, unspecified",
     "notes": "Use only when MI/angina ruled out; if MI confirmed use I21.x"},
    {"code": "R50.9",  "description": "Fever, unspecified"},
    {"code": "R41.3",  "description": "Other amnesia — includes acute confusion"},
    {"code": "R73.09", "description": "Other abnormal glucose",
     "notes": "Do NOT code separately if E11.65 (T2DM with hyperglycemia) is used"},

    # ── Tobacco ──────────────────────────────────────────────────────────────
    {"code": "F17.210","description": "Nicotine dependence, cigarettes, uncomplicated"},
    {"code": "Z87.891","description": "Personal history of nicotine dependence"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Starter coding guidelines
# Each string is one indexed chunk.
# ─────────────────────────────────────────────────────────────────────────────

STARTER_GUIDELINES = [
    # Sequencing
    "SEQUENCING: The principal diagnosis — the condition established after study to be "
    "chiefly responsible for the admission — is listed first. Secondary diagnoses that "
    "affect patient management, treatment, or length of stay are coded additionally.",

    # Specificity
    "SPECIFICITY: Always assign the most specific ICD-10 code available. Use unspecified "
    "codes (e.g. J18.9 Pneumonia unspecified) only when the causative organism or site "
    "is genuinely unknown after workup. Prefer organism-specific codes (J15.0 Klebsiella, "
    "J15.1 Pseudomonas) when culture or clinical findings identify the pathogen.",

    # Combination codes
    "COMBINATION CODES: Use a combination code when one code fully classifies two "
    "conditions or a condition + manifestation. Example: E11.65 (Type 2 DM with "
    "hyperglycemia) captures both diagnoses — do not separately code R73.09. "
    "E11.40 captures T2DM + neuropathy. Always check the tabular for available "
    "combination codes before assigning two separate codes.",

    # Excludes notes
    "EXCLUDES NOTES: Check Excludes1 and Excludes2 notes before coding two conditions "
    "together. Excludes1 = the two conditions cannot be reported together (mutually "
    "exclusive). Excludes2 = the excluded condition is not included here but may be "
    "coded additionally when both are present.",

    # Signs and symptoms
    "SIGNS AND SYMPTOMS: Do not code signs or symptoms (e.g. chest pain R07.9, fever "
    "R50.9) that are integral to and routinely associated with an established definitive "
    "diagnosis. Code the definitive diagnosis instead. Code symptoms only when no "
    "confirmed diagnosis exists at discharge, or when the guideline explicitly instructs "
    "coding the symptom additionally.",

    # Laterality
    "LATERALITY: When the code set provides separate codes for right vs left side, "
    "always specify. Example: M79.621 = right carpal tunnel syndrome, M79.622 = left. "
    "Use bilateral codes when available. Use unspecified-side codes only when "
    "documentation does not indicate the side.",

    # Acute vs chronic
    "ACUTE VS CHRONIC: When both acute and chronic forms of a condition are present "
    "and separate codes exist for each, code the acute form first unless guidelines "
    "state otherwise. Example: I50.23 (acute on chronic systolic CHF) is preferred "
    "over coding I50.21 + I50.22 separately.",

    # Completeness
    "COMPLETENESS: Review the entire chart — H&P, progress notes, consults, labs, "
    "imaging, and discharge summary — for secondary diagnoses. Commonly missed: "
    "anaemia (D50.x, D64.9), hypokalemia (E87.6), AKI (N17.9), pressure ulcers "
    "(L89.x), DVT prophylaxis indication, malnutrition (E44.x), obesity (E66.x), "
    "tobacco dependence (F17.x or Z87.891), and substance use disorders.",

    # Sepsis sequencing
    "SEPSIS SEQUENCING: Code the underlying infection first (e.g. A40.0 Strep group A "
    "sepsis), then the severity code (R65.20 severe sepsis without shock, R65.21 with "
    "septic shock). Do not code sepsis (A41.9) when a more specific organism code is "
    "available. Necrotizing fasciitis (M72.6) is coded first, followed by the organism.",

    # STEMI coding
    "STEMI CODING: For ST-elevation MI, identify the specific coronary artery and wall. "
    "I21.01 = LAD (anterior), I21.09 = other anterior wall artery, I21.11 = RCA "
    "(inferior), I21.4 = NSTEMI. After PCI with stent placement, the MI code remains "
    "the principal diagnosis. Code acute systolic HF (I50.21) additionally if "
    "documented with reduced EF precipitated by the MI.",

    # Diabetes
    "DIABETES CODING: Type 2 DM uses E11 category with specific 4th/5th digits for "
    "complications. E11.65 = DM with hyperglycemia (do not add R73.09). "
    "E11.40 = DM with unspecified neuropathy. E11.51 = DM with peripheral angiopathy. "
    "When HbA1c is elevated but no complications documented, use E11.65 if hyperglycemia "
    "mentioned, otherwise E11.9.",

    # Heart failure specificity
    "HEART FAILURE SPECIFICITY: Specify systolic vs diastolic and acute vs chronic. "
    "Requires documentation of EF or explicit physician statement. "
    "Reduced EF (HFrEF) → systolic (I50.2x). Preserved EF (HFpEF) → diastolic (I50.3x). "
    "Acute = new onset or decompensation. Chronic = established, stable. "
    "Acute on chronic = known HF with acute decompensation.",

    # AKI
    "ACUTE KIDNEY INJURY: N17.9 = AKI unspecified. Code as secondary diagnosis when "
    "caused by another condition (e.g. cardiorenal syndrome from HF, or septic AKI). "
    "If CKD also present, code both N17.9 and N18.x; the AKI takes precedence "
    "sequentially unless CKD is the principal diagnosis.",

    # Z codes
    "Z CODES: Use Z codes for personal history (Z87.x), family history (Z82.x/Z83.x), "
    "status codes (Z95.x for cardiac implants, Z96.x for prosthetics), and tobacco "
    "history (Z87.891). These are secondary codes only — never principal diagnosis.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build(reset: bool = False) -> None:
    print("=" * 60)
    print("ICD-10 RAG Index Builder")
    print("=" * 60)

    rag = ICD10RAGStore()
    if reset:
        print("\nResetting existing collections …")
        rag.reset()

    # ── ICD-10 codes ─────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading {len(STARTER_CODES)} ICD-10 codes …")
    rag.add_codes(STARTER_CODES)
    print(f"      icd10_codes collection: {rag.codes_col.count()} documents")

    # ── Coding guidelines ────────────────────────────────────────────────────
    print(f"\n[2/3] Loading {len(STARTER_GUIDELINES)} guideline chunks …")
    rag.add_guidelines(STARTER_GUIDELINES)
    print(f"      coding_guidelines collection: {rag.guidelines_col.count()} documents")

    # ── Reasoning Bank (start empty, grows with each chart processed) ────────
    bank = ReasoningBankStore()
    print(f"\n[3/3] Reasoning bank: {bank.count()} case(s) "
          f"(grows automatically after each chart is processed)")

    # ── Verify retrieval ─────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("Retrieval verification:")

    test_chart = (
        "67-year-old male with anterior STEMI, acute systolic heart failure EF 35%, "
        "type 2 diabetes with neuropathy, hypertension, AKI, iron deficiency anaemia, hypokalemia."
    )
    ctx = rag.get_context_for_actor(test_chart, n_codes=5, n_guidelines=3)
    print(f"\nTest query: '{test_chart[:80]}…'")
    print("\nTop retrieved codes:")
    for line in ctx.split("\n"):
        if line.strip().startswith("I") or line.strip().startswith("E") or line.strip().startswith("N"):
            print(f"  {line.strip()}")

    print("\nTop retrieved guideline (first 120 chars):")
    in_gl = False
    for line in ctx.split("\n"):
        if "CODING GUIDELINES" in line:
            in_gl = True
            continue
        if in_gl and line.strip().startswith("•"):
            print(f"  {line.strip()[:120]}")
            break

    print("\n" + "─" * 50)
    print("Build complete. Run the pipeline with:")
    print("  python icd10_actor_critic.py chest_pain")
    print("  python icd10_actor_critic.py sepsis")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    build(reset=reset)
