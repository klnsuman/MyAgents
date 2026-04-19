"""
RAG v1 (2023): Basic Retrieval
================================
Core concepts:
  - Chunk documents into fixed-size pieces
  - Embed chunks using OpenAI text-embedding-3-small
  - Store embeddings in an in-memory vector store
  - At query time: embed query → cosine similarity → top-K → LLM answer

Requirements: openai, numpy, python-dotenv
"""

import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────────
# 1. SAMPLE CORPUS
# ─────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_1",
        "text": (
            "Python is a high-level, interpreted programming language known for its clear syntax "
            "and readability. It was created by Guido van Rossum and first released in 1991. "
            "Python supports multiple programming paradigms including procedural, object-oriented, "
            "and functional programming."
        ),
    },
    {
        "id": "doc_2",
        "text": (
            "Machine learning is a subset of artificial intelligence that enables computers to learn "
            "from data without being explicitly programmed. Key algorithms include linear regression, "
            "decision trees, random forests, and neural networks. Supervised, unsupervised, and "
            "reinforcement learning are the three main categories."
        ),
    },
    {
        "id": "doc_3",
        "text": (
            "The Transformer architecture, introduced in 'Attention is All You Need' (2017), "
            "revolutionized natural language processing. It uses self-attention mechanisms to process "
            "sequences in parallel rather than sequentially. BERT, GPT, and T5 are all based on "
            "the Transformer architecture."
        ),
    },
    {
        "id": "doc_4",
        "text": (
            "Vector databases store high-dimensional vector embeddings and enable similarity search. "
            "Examples include Pinecone, Weaviate, Chroma, and FAISS. They use approximate nearest "
            "neighbor (ANN) algorithms like HNSW to efficiently find the most similar vectors among "
            "millions of entries."
        ),
    },
    {
        "id": "doc_5",
        "text": (
            "RAG (Retrieval-Augmented Generation) combines a retrieval system with a language model. "
            "The retriever finds relevant documents from a knowledge base, and the generator uses "
            "those documents as context to produce an accurate, grounded answer. RAG reduces "
            "hallucinations and keeps knowledge up-to-date without retraining the LLM."
        ),
    },
    {
        "id": "doc_6",
        "text": (
            "Cosine similarity measures the cosine of the angle between two vectors in a "
            "multi-dimensional space. It ranges from -1 (opposite) to 1 (identical). In NLP, it is "
            "widely used to measure semantic similarity between text embeddings. A value above 0.8 "
            "generally indicates high similarity."
        ),
    },
]


# ─────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` characters."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────
# 3. EMBEDDING
# ─────────────────────────────────────────────
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Return OpenAI embedding vector for a text string."""
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def embed_corpus(documents: list[dict]) -> list[dict]:
    """Embed every document chunk and return an augmented list."""
    vector_store = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            vector_store.append(
                {
                    "doc_id": doc["id"],
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": embedding,
                }
            )
    return vector_store


# ─────────────────────────────────────────────
# 4. COSINE SIMILARITY
# ─────────────────────────────────────────────
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Pure-numpy cosine similarity between two embedding vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ─────────────────────────────────────────────
# 5. TOP-K RETRIEVAL
# ─────────────────────────────────────────────
def retrieve_top_k(
    query: str, vector_store: list[dict], k: int = 3
) -> list[dict]:
    """Embed the query then return the top-K most similar chunks."""
    query_embedding = get_embedding(query)
    scored = []
    for entry in vector_store:
        score = cosine_similarity(query_embedding, entry["embedding"])
        scored.append({**entry, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# ─────────────────────────────────────────────
# 6. GENERATE ANSWER WITH GPT
# ─────────────────────────────────────────────
def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """Build a prompt from retrieved chunks and call GPT-4o-mini."""
    context = "\n\n".join(
        [f"[Source {i+1}]: {c['text']}" for i, c in enumerate(retrieved_chunks)]
    )
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 7. FULL RAG v1 PIPELINE
# ─────────────────────────────────────────────
def rag_v1_pipeline(query: str, vector_store: list[dict], k: int = 3) -> dict:
    """End-to-end RAG v1: retrieve top-K chunks then generate an answer."""
    print(f"\n{'='*60}")
    print(f"RAG v1 | Query: {query}")
    print("="*60)

    # Step 1: Retrieve
    top_chunks = retrieve_top_k(query, vector_store, k=k)

    print(f"\n[Retrieved {k} chunks]")
    for i, chunk in enumerate(top_chunks):
        print(f"  #{i+1}  score={chunk['score']:.4f}  doc={chunk['doc_id']}")
        print(f"       text: {chunk['text'][:80]}...")

    # Step 2: Generate
    answer = generate_answer(query, top_chunks)
    print(f"\n[Answer]\n{answer}")

    return {
        "query": query,
        "retrieved_chunks": top_chunks,
        "answer": answer,
    }


# ─────────────────────────────────────────────
# 8. DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Building vector store (embedding all chunks)...")
    vector_store = embed_corpus(DOCUMENTS)
    print(f"Vector store ready: {len(vector_store)} chunks indexed.\n")

    queries = [
        "What is RAG and why is it useful?",
        "How does cosine similarity work?",
        "What is the Transformer architecture?",
    ]

    for q in queries:
        result = rag_v1_pipeline(q, vector_store, k=3)

    # ── Persist the vector store (no DB needed for v1) ──
    store_path = "rag_v1_vector_store.json"
    serializable = [
        {k: v for k, v in entry.items() if k != "embedding"}
        for entry in vector_store
    ]
    with open(store_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nVector store metadata saved to {store_path}")
