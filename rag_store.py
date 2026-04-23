"""
RAG stores for the ICD-10 Actor-Critic-Arbiter pipeline.

ICD10RAGStore      — static store (icd10_codes + coding_guidelines)
                     queried BEFORE Actor to supply code candidates + guidelines
ReasoningBankStore — dynamic store (past cases + their final codes)
                     queried BY Critic each debate round to find precedents
                     updated BY Arbiter after every chart is finalised
"""

import uuid
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

EMBED_MODEL  = "all-MiniLM-L6-v2"   # free, local, no API key needed
CHROMA_DIR   = "./chroma_db"


def _make_ef() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# Static Store  —  Actor context (ICD-10 codes + coding guidelines)
# ─────────────────────────────────────────────────────────────────────────────

class ICD10RAGStore:
    """
    Holds two ChromaDB collections:
      icd10_codes       — ~70K ICD-10-CM codes with descriptions / includes/excludes
      coding_guidelines — CMS Official Coding Guidelines, chunked by section

    Usage:
      Built once with build_rag_index.py.
      Queried before every Actor call to supply relevant candidate codes and rules.
      Re-run the builder on Oct 1 each year when CMS publishes updated codes.
    """

    def __init__(self, persist_dir: str = CHROMA_DIR):
        ef = _make_ef()
        self._client = chromadb.PersistentClient(path=persist_dir)
        self.codes_col = self._client.get_or_create_collection(
            "icd10_codes", embedding_function=ef
        )
        self.guidelines_col = self._client.get_or_create_collection(
            "coding_guidelines", embedding_function=ef
        )

    def is_ready(self) -> bool:
        return self.codes_col.count() > 0 and self.guidelines_col.count() > 0

    # ── Ingestion ────────────────────────────────────────────────────────────

    def add_codes(self, records: list[dict], batch_size: int = 500) -> None:
        """
        records: list of {code, description, notes (optional)}
        Upserts so re-running is safe.
        """
        ids   = [r["code"] for r in records]
        docs  = [f"{r['code']}: {r['description']}. {r.get('notes', '')}".strip()
                 for r in records]
        metas = [{"code": r["code"], "description": r["description"]} for r in records]

        for i in range(0, len(records), batch_size):
            self.codes_col.upsert(
                ids=ids[i:i + batch_size],
                documents=docs[i:i + batch_size],
                metadatas=metas[i:i + batch_size],
            )

    def add_guidelines(self, chunks: list[str]) -> None:
        """chunks: list of plain-text guideline paragraphs."""
        ids = [f"gl_{i:04d}" for i in range(len(chunks))]
        self.guidelines_col.upsert(ids=ids, documents=chunks)

    def reset(self) -> None:
        """Full rebuild — deletes both collections and recreates them."""
        ef = _make_ef()
        self._client.delete_collection("icd10_codes")
        self._client.delete_collection("coding_guidelines")
        self.codes_col = self._client.get_or_create_collection(
            "icd10_codes", embedding_function=ef
        )
        self.guidelines_col = self._client.get_or_create_collection(
            "coding_guidelines", embedding_function=ef
        )

    # ── Query ────────────────────────────────────────────────────────────────

    def get_context_for_actor(
        self,
        chart_text: str,
        n_codes: int = 15,
        n_guidelines: int = 6,
    ) -> str:
        """
        Embed the chart, retrieve the most relevant ICD-10 codes and guidelines,
        and return a formatted context block to prepend to the Actor's prompt.
        """
        code_res = self.codes_col.query(
            query_texts=[chart_text], n_results=min(n_codes, self.codes_col.count())
        )
        gl_res = self.guidelines_col.query(
            query_texts=[chart_text], n_results=min(n_guidelines, self.guidelines_col.count())
        )

        lines = ["=== ICD-10 CODE CANDIDATES (retrieved for this chart) ==="]
        for doc, meta in zip(code_res["documents"][0], code_res["metadatas"][0]):
            lines.append(f"  {meta['code']}: {meta['description']}")

        lines += ["", "=== RELEVANT CODING GUIDELINES ==="]
        for doc in gl_res["documents"][0]:
            lines.append(f"  • {doc}")

        lines.append("=== END ACTOR CONTEXT ===")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Store  —  Reasoning Bank (past cases, queried by Critic)
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningBankStore:
    """
    Holds one ChromaDB collection:
      reasoning_bank — verified past cases: chart summary + final ICD codes + notes

    Usage:
      Queried by Critic each debate round to find clinically similar past cases
      whose final code assignments serve as precedent for approving/rejecting codes.

      Updated by Arbiter after every chart is finalised — the new case becomes
      a precedent for future charts.

    When it gets updated:
      • After every chart processed through the pipeline (Arbiter calls add_case())
      • Optionally seeded with MIMIC-verified cases via add_case_raw()
    """

    def __init__(self, persist_dir: str = CHROMA_DIR):
        ef = _make_ef()
        self._client = chromadb.PersistentClient(path=persist_dir)
        self.col = self._client.get_or_create_collection(
            "reasoning_bank", embedding_function=ef
        )

    def count(self) -> int:
        return self.col.count()

    # ── Ingestion ────────────────────────────────────────────────────────────

    def add_case(self, chart: str, final_codes, arbiter_notes: str) -> str:
        """
        Called by Arbiter after finalising a chart.
        final_codes: list[FinalCode]
        Returns the new case_id.
        """
        codes_str = ", ".join(
            f"{fc.code} ({fc.description})" for fc in final_codes
        )
        # The document that gets embedded blends chart context + outcome
        # so similarity search finds cases with matching clinical picture AND codes
        doc = (
            f"CLINICAL CASE:\n{chart[:600].strip()}\n\n"
            f"FINAL ICD-10 CODES: {codes_str}\n\n"
            f"ARBITER NOTES: {arbiter_notes}"
        )
        meta = {
            "codes":      ",".join(fc.code for fc in final_codes),
            "code_count": len(final_codes),
        }
        case_id = str(uuid.uuid4())
        self.col.add(ids=[case_id], documents=[doc], metadatas=[meta])
        return case_id

    def add_case_raw(
        self,
        chart_summary: str,
        codes: list[str],
        notes: str = "",
    ) -> str:
        """Seed the bank with MIMIC or manually verified cases."""
        doc = (
            f"CLINICAL CASE:\n{chart_summary}\n\n"
            f"FINAL ICD-10 CODES: {', '.join(codes)}\n\n"
            f"NOTES: {notes}"
        )
        meta = {"codes": ",".join(codes), "code_count": len(codes)}
        case_id = str(uuid.uuid4())
        self.col.add(ids=[case_id], documents=[doc], metadatas=[meta])
        return case_id

    # ── Query ────────────────────────────────────────────────────────────────

    def get_similar_cases(self, chart_text: str, n: int = 3) -> str:
        """
        Embed the current chart, retrieve the most similar past cases,
        and return a formatted block for the Critic's prompt.

        A chart may partially match multiple past cases (different disease
        combinations), so multiple rows are retrieved and presented.
        Returns empty string if the bank has no cases yet.
        """
        if self.col.count() == 0:
            return ""

        n_actual = min(n, self.col.count())
        results = self.col.query(
            query_texts=[chart_text],
            n_results=n_actual,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"][0]:
            return ""

        lines = [f"=== SIMILAR PAST CASES FROM REASONING BANK (top {n_actual}) ==="]
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            1,
        ):
            similarity = max(0.0, 1.0 - dist)
            lines.append(f"\n--- Precedent {i}  (similarity={similarity:.2f}) ---")
            lines.append(doc)
            lines.append(f"Codes in this case: {meta['codes']}")

        lines.append("\n=== END REASONING BANK CONTEXT ===")
        return "\n".join(lines)
