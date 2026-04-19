"""
RAG v4 (2026+): Graph RAG + Long Context
==========================================
Improvements over v3:
  - Knowledge Graph: entities and relationships extracted from documents
  - Graph traversal: multi-hop entity linking (A→B→C) for relationship queries
  - Long context: pack many relevant chunks into a single large-context GPT-4o call
  - Hybrid orchestration: route to vector RAG, Graph RAG, or long-context window based on query type

Architecture:
  ┌─────────────┐
  │   Query     │
  └──────┬──────┘
         │
  ┌──────▼──────────────────────────────────┐
  │  Query Router (GPT-4o-mini classifier)  │
  └──────┬──────┬──────────────────────────-┘
         │      │         │
    [vector] [graph]  [long-ctx]
         │      │         │
  ┌──────▼──────▼─────────▼──────┐
  │   Hybrid Orchestrator        │
  │   (merges all results)       │
  └──────────────┬───────────────┘
                 │
          ┌──────▼──────┐
          │  GPT-4o     │
          │  (answer)   │
          └─────────────┘

Requirements: openai, numpy, python-dotenv
"""

import os
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# 1. CORPUS (technology relationships)
# ─────────────────────────────────────────────
CORPUS = [
    "Python was created by Guido van Rossum at CWI Netherlands. It was heavily influenced by ABC language.",
    "PyTorch was developed by Meta AI Research. It is built on top of Python and uses dynamic computation graphs.",
    "TensorFlow was developed by the Google Brain team. Keras is the official high-level API for TensorFlow.",
    "BERT was created by Google AI Language team. BERT is built on the Transformer architecture and pre-trained on Wikipedia and BookCorpus.",
    "GPT-4 was created by OpenAI. GPT-4 is a successor to GPT-3 and uses the Transformer decoder architecture.",
    "LangChain was created by Harrison Chase. LangChain integrates with OpenAI, Anthropic, and Hugging Face.",
    "LlamaIndex was created by Jerry Liu. LlamaIndex specializes in data indexing for LLM applications.",
    "RAG was introduced by Meta AI. RAG combines a Dense Passage Retriever with a BART generator.",
    "Agentic RAG extends RAG with query decomposition and iterative retrieval loops.",
    "Graph RAG uses knowledge graphs to capture relationships between entities for richer context.",
    "OpenAI was founded by Sam Altman, Elon Musk, and others in 2015. OpenAI created GPT-4 and ChatGPT.",
    "Google created both BERT and TensorFlow. Google also developed the original Transformer architecture.",
    "Meta AI created PyTorch, RAG, and LLaMA. Meta AI Research is based in Menlo Park, California.",
    "The Transformer architecture was proposed by Google researchers in the paper 'Attention is All You Need' in 2017.",
    "FAISS (Facebook AI Similarity Search) was developed by Meta AI for efficient vector similarity search.",
    "Chroma is an open-source vector database. It is often used with LangChain for RAG applications.",
    "Pinecone is a managed vector database service. It supports hybrid search combining dense and sparse vectors.",
    "Hugging Face hosts thousands of pre-trained models including BERT, GPT-2, and LLaMA variants.",
    "LLaMA (Large Language Model Meta AI) is an open-source LLM family released by Meta AI.",
    "Anthropic was founded by former OpenAI researchers including Dario Amodei and Daniela Amodei.",
]


# ─────────────────────────────────────────────
# 2. KNOWLEDGE GRAPH
# ─────────────────────────────────────────────
@dataclass
class Entity:
    name: str
    type: str  # "person", "organization", "model", "framework", "concept"
    mentions: list[int] = field(default_factory=list)  # corpus sentence indices


@dataclass
class Relationship:
    source: str
    relation: str
    target: str
    sentence_idx: int


class KnowledgeGraph:
    """In-memory knowledge graph backed by OpenAI for extraction."""

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.adjacency: dict[str, list[Relationship]] = defaultdict(list)

    def extract_and_add(self, text: str, sentence_idx: int) -> None:
        """Use GPT to extract (entity, relation, entity) triples from text."""
        prompt = f"""Extract all (subject, relation, object) triples from the sentence.
Return JSON: {{"triples": [{{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "..."}}]}}
Types: person, organization, model, framework, concept, dataset, paper

Sentence: {text}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        triples = data.get("triples", [])

        for triple in triples:
            subj = triple.get("subject", "").strip()
            rel = triple.get("relation", "").strip()
            obj = triple.get("object", "").strip()
            subj_type = triple.get("subject_type", "concept")
            obj_type = triple.get("object_type", "concept")

            if not (subj and rel and obj):
                continue

            # Register entities
            if subj not in self.entities:
                self.entities[subj] = Entity(name=subj, type=subj_type)
            self.entities[subj].mentions.append(sentence_idx)

            if obj not in self.entities:
                self.entities[obj] = Entity(name=obj, type=obj_type)
            self.entities[obj].mentions.append(sentence_idx)

            # Register relationship
            r = Relationship(source=subj, relation=rel, target=obj, sentence_idx=sentence_idx)
            self.relationships.append(r)
            self.adjacency[subj].append(r)

    def build(self, corpus: list[str]) -> None:
        """Build the full knowledge graph from a corpus."""
        print(f"Building knowledge graph from {len(corpus)} sentences...")
        for i, sentence in enumerate(corpus):
            print(f"  [{i+1}/{len(corpus)}] Extracting from: {sentence[:60]}...")
            self.extract_and_add(sentence, i)
        print(f"  Graph built: {len(self.entities)} entities, {len(self.relationships)} relationships")

    def find_entity(self, name: str) -> str | None:
        """Case-insensitive entity lookup."""
        name_lower = name.lower()
        for key in self.entities:
            if key.lower() == name_lower or name_lower in key.lower():
                return key
        return None

    def get_neighbors(self, entity_name: str, max_hops: int = 2) -> list[dict]:
        """BFS graph traversal up to max_hops from an entity."""
        start = self.find_entity(entity_name)
        if not start:
            return []

        visited: set[str] = {start}
        results: list[dict] = []
        queue = [(start, 0)]

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            for rel in self.adjacency.get(current, []):
                results.append({
                    "source": rel.source,
                    "relation": rel.relation,
                    "target": rel.target,
                    "depth": depth + 1,
                    "source_text": CORPUS[rel.sentence_idx],
                })
                if rel.target not in visited:
                    visited.add(rel.target)
                    queue.append((rel.target, depth + 1))

        return results

    def get_entity_context(self, entity_name: str) -> list[str]:
        """Return all corpus sentences mentioning an entity."""
        entity = self.find_entity(entity_name)
        if not entity:
            return []
        indices = self.entities[entity].mentions
        return [CORPUS[i] for i in set(indices)]

    def summary(self) -> str:
        lines = [f"Knowledge Graph Summary:"]
        lines.append(f"  Entities ({len(self.entities)}): {', '.join(list(self.entities.keys())[:15])}...")
        lines.append(f"  Relationships ({len(self.relationships)}):")
        for r in self.relationships[:10]:
            lines.append(f"    {r.source} --[{r.relation}]--> {r.target}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 3. VECTOR STORE (for hybrid orchestration)
# ─────────────────────────────────────────────
_vector_store: list[dict] = []


def get_embedding(text: str) -> list[float]:
    r = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return r.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


def build_vector_store(corpus: list[str]) -> None:
    global _vector_store
    print("Building vector store...")
    _vector_store = [
        {"idx": i, "text": text, "embedding": get_embedding(text)}
        for i, text in enumerate(corpus)
    ]
    print(f"  {len(_vector_store)} vectors indexed.")


def vector_search(query: str, k: int = 5) -> list[dict]:
    q_emb = get_embedding(query)
    scored = sorted(_vector_store, key=lambda d: cosine_similarity(q_emb, d["embedding"]), reverse=True)
    return [{"text": d["text"], "score": cosine_similarity(q_emb, d["embedding"])} for d in scored[:k]]


# ─────────────────────────────────────────────
# 4. QUERY ROUTER
# ─────────────────────────────────────────────
QUERY_TYPES = {
    "entity_relationship": (
        "Query asks about how entities relate to each other, "
        "who created something, or connections between people/organizations/models."
    ),
    "factual_lookup": (
        "Simple factual question that can be answered from a single document passage."
    ),
    "global_summary": (
        "Query asks for a broad overview, comparison, or synthesis across many documents."
    ),
}


def route_query(query: str) -> str:
    """Classify the query type to choose the retrieval strategy."""
    options = "\n".join([f"- {k}: {v}" for k, v in QUERY_TYPES.items()])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Classify this query into exactly one of these types:\n{options}\n\n"
                    "Return JSON: {\"type\": \"<type>\", \"entities\": [\"...\"], \"reasoning\": \"...\"}"
                ),
            },
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    return data


# ─────────────────────────────────────────────
# 5. RETRIEVAL STRATEGIES
# ─────────────────────────────────────────────
def graph_rag_retrieve(
    entities: list[str], kg: KnowledgeGraph, max_hops: int = 2
) -> dict:
    """Retrieve via graph traversal for entity-relationship queries."""
    all_triples = []
    all_context = []

    for entity in entities:
        neighbors = kg.get_neighbors(entity, max_hops=max_hops)
        context = kg.get_entity_context(entity)
        all_triples.extend(neighbors)
        all_context.extend(context)

    # Format graph context
    graph_text = "\n".join([
        f"  {t['source']} --[{t['relation']}]--> {t['target']} (depth={t['depth']})"
        for t in all_triples[:20]
    ])
    return {
        "strategy": "graph_rag",
        "triples": all_triples[:20],
        "context": list(set(all_context))[:10],
        "graph_summary": graph_text,
    }


def long_context_retrieve(query: str, corpus: list[str], top_n: int = 15) -> dict:
    """Pack many relevant chunks into a long-context window."""
    # Use vector search to pick the most globally relevant chunks
    results = vector_search(query, k=top_n)
    all_text = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(results)])
    return {
        "strategy": "long_context",
        "chunks": results,
        "full_context": all_text,
    }


def vector_rag_retrieve(query: str, k: int = 5) -> dict:
    """Standard vector similarity retrieval for factual lookups."""
    results = vector_search(query, k=k)
    return {
        "strategy": "vector_rag",
        "chunks": results,
        "context": "\n".join([r["text"] for r in results]),
    }


# ─────────────────────────────────────────────
# 6. HYBRID ORCHESTRATOR
# ─────────────────────────────────────────────
def hybrid_orchestrate(query: str, route_info: dict, kg: KnowledgeGraph) -> dict:
    """Choose and combine retrieval strategies based on query routing."""
    query_type = route_info.get("type", "factual_lookup")
    entities = route_info.get("entities", [])

    print(f"\n[Router] type={query_type}, entities={entities}")
    print(f"         reasoning={route_info.get('reasoning', '')[:80]}")

    results = {}

    if query_type == "entity_relationship" and entities:
        # Primary: graph traversal | Secondary: vector for context
        print("[Strategy] Graph RAG + Vector supplemental")
        graph_result = graph_rag_retrieve(entities, kg, max_hops=2)
        vector_result = vector_rag_retrieve(query, k=3)
        results = {
            "primary": graph_result,
            "supplemental": vector_result,
            "combined_context": (
                f"Graph Relationships:\n{graph_result['graph_summary']}\n\n"
                f"Relevant Passages:\n{vector_result['context']}\n\n"
                f"Entity Context:\n{chr(10).join(graph_result['context'][:5])}"
            ),
        }

    elif query_type == "global_summary":
        # Long context: retrieve many chunks for broad synthesis
        print("[Strategy] Long-context retrieval")
        lc_result = long_context_retrieve(query, CORPUS, top_n=12)
        results = {
            "primary": lc_result,
            "combined_context": lc_result["full_context"],
        }

    else:  # factual_lookup
        # Standard vector RAG
        print("[Strategy] Vector RAG")
        vector_result = vector_rag_retrieve(query, k=5)
        results = {
            "primary": vector_result,
            "combined_context": vector_result["context"],
        }

    return results


# ─────────────────────────────────────────────
# 7. GENERATE WITH LONG CONTEXT GPT-4o
# ─────────────────────────────────────────────
def generate_answer_v4(query: str, context: str, query_type: str) -> str:
    """Generate answer with GPT-4o using full retrieved context."""
    system_prompts = {
        "entity_relationship": (
            "You are an expert at analyzing relationships between entities. "
            "Use the graph relationships and passages to explain connections clearly. "
            "Trace multi-hop relationships (A created B, B uses C, therefore A→C)."
        ),
        "global_summary": (
            "You are a research synthesizer. Use all provided passages to give a comprehensive, "
            "structured overview. Identify patterns, themes, and key relationships across sources."
        ),
        "factual_lookup": (
            "You are a precise assistant. Answer the question factually using only the provided context. "
            "Be concise and cite the relevant passage."
        ),
    }
    system = system_prompts.get(query_type, system_prompts["factual_lookup"])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 8. FULL RAG v4 PIPELINE
# ─────────────────────────────────────────────
def rag_v4_pipeline(query: str, kg: KnowledgeGraph) -> dict:
    print(f"\n{'='*60}")
    print(f"RAG v4 | Query: {query}")
    print("="*60)

    # Step 1: Route the query
    route_info = route_query(query)

    # Step 2: Orchestrate hybrid retrieval
    retrieval = hybrid_orchestrate(query, route_info, kg)

    # Step 3: Generate with GPT-4o (long context)
    answer = generate_answer_v4(query, retrieval["combined_context"], route_info["type"])

    print(f"\n[Answer]\n{answer}")

    return {
        "query": query,
        "query_type": route_info["type"],
        "entities_found": route_info.get("entities", []),
        "strategy": retrieval.get("primary", {}).get("strategy", "unknown"),
        "answer": answer,
    }


# ─────────────────────────────────────────────
# 9. DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Build all indexes (this will make multiple API calls for graph extraction)
    print("Phase 1: Building knowledge graph...")
    kg = KnowledgeGraph()
    kg.build(CORPUS)
    print("\n" + kg.summary())

    print("\nPhase 2: Building vector store...")
    build_vector_store(CORPUS)

    # --- Queries showcasing each strategy ---
    queries = [
        # Entity relationship → Graph RAG
        "What is the relationship between Meta AI and PyTorch and RAG?",

        # Entity relationship with multi-hop → Graph RAG
        "How is Google connected to BERT, and what architecture does BERT use?",

        # Global summary → Long context
        "Give me an overview of all the major AI organizations and what they have created.",

        # Factual lookup → Vector RAG
        "When was the Transformer architecture introduced and by whom?",
    ]

    results = []
    for q in queries:
        result = rag_v4_pipeline(q, kg)
        results.append(result)
        print(f"\n{'*'*60}")

    # Print summary table
    print("\n\n" + "="*60)
    print("SUMMARY: RAG v4 Routing Decisions")
    print("="*60)
    for r in results:
        print(f"\nQ: {r['query'][:60]}...")
        print(f"   Type:     {r['query_type']}")
        print(f"   Strategy: {r['strategy']}")
        print(f"   Entities: {r['entities_found']}")
