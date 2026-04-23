"""
agent/rag_pipeline.py
RAG (Retrieval-Augmented Generation) pipeline for AutoStream knowledge base.
Uses a lightweight local approach with JSON + Markdown chunking and TF-IDF similarity.
No external vector DB required — runs fully offline.
"""

import json
import re
import math
from pathlib import Path
from typing import List, Dict


KB_JSON_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"
KB_MD_PATH   = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.md"


# ─── Chunking ────────────────────────────────────────────────────────────────

def load_chunks() -> List[Dict[str, str]]:
    """Load and chunk the knowledge base into searchable units."""
    chunks = []

    # ── 1. JSON structured data ──
    with open(KB_JSON_PATH, "r") as f:
        kb = json.load(f)

    # Company overview
    chunks.append({
        "id": "company_overview",
        "text": f"{kb['company']['name']}: {kb['company']['description']} Tagline: {kb['company']['tagline']}"
    })

    # Pricing plans
    for plan in kb["pricing_plans"]:
        features_str = "; ".join(plan["features"])
        limits_str   = "; ".join(plan["limitations"]) if plan["limitations"] else "None"
        chunks.append({
            "id": f"plan_{plan['name'].lower().replace(' ', '_')}",
            "text": (
                f"{plan['name']} costs ${plan['price_monthly']}/month. "
                f"Features: {features_str}. "
                f"Limitations: {limits_str}."
            )
        })

    # Policies
    for policy in kb["policies"]:
        chunks.append({
            "id": f"policy_{policy['topic'].lower().replace(' ', '_')}",
            "text": f"{policy['topic']}: {policy['details']}"
        })

    # FAQ
    for i, faq in enumerate(kb["faq"]):
        chunks.append({
            "id": f"faq_{i}",
            "text": f"Q: {faq['question']} A: {faq['answer']}"
        })

    # ── 2. Markdown sections (deduplicate with JSON) ──
    md_text = KB_MD_PATH.read_text()
    sections = re.split(r"\n#{2,3} ", md_text)
    for i, section in enumerate(sections[1:], 1):  # skip title
        lines = section.strip().splitlines()
        if lines:
            heading = lines[0].strip()
            body    = " ".join(lines[1:]).strip()
            chunks.append({
                "id": f"md_section_{i}",
                "text": f"{heading}: {body}"
            })

    return chunks


# ─── TF-IDF Retriever ─────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_tfidf(chunks: List[Dict]) -> Dict:
    """Build a simple TF-IDF index over the chunks."""
    corpus = [tokenize(c["text"]) for c in chunks]
    
    # Document frequency
    df: Dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    N = len(corpus)
    idf = {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in df.items()}

    # TF-IDF vectors
    vectors = []
    for doc in corpus:
        tf: Dict[str, float] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        vec = {term: (count / len(doc)) * idf.get(term, 1) for term, count in tf.items()}
        vectors.append(vec)

    return {"vectors": vectors, "idf": idf}


def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot   = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
    return dot / (norm_a * norm_b + 1e-9)


def retrieve(query: str, chunks: List[Dict], tfidf: Dict, top_k: int = 3) -> List[str]:
    """Retrieve the top_k most relevant chunks for a query."""
    query_tokens = tokenize(query)
    idf          = tfidf["idf"]
    vectors      = tfidf["vectors"]

    # Build query vector
    q_tf: Dict[str, float] = {}
    for term in query_tokens:
        q_tf[term] = q_tf.get(term, 0) + 1
    q_vec = {term: (count / len(query_tokens)) * idf.get(term, 1)
             for term, count in q_tf.items()}

    # Score all chunks
    scores = [cosine_sim(q_vec, vec) for vec in vectors]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [chunks[i]["text"] for i in top_indices if scores[i] > 0.01]


# ─── Singleton setup ──────────────────────────────────────────────────────────

_chunks: List[Dict] = []
_tfidf:  Dict       = {}


def init_rag():
    global _chunks, _tfidf
    _chunks = load_chunks()
    _tfidf  = build_tfidf(_chunks)


def query_kb(query: str, top_k: int = 3) -> str:
    """
    Public interface: given a natural language query, return
    a formatted context string from the knowledge base.
    """
    if not _chunks:
        init_rag()
    results = retrieve(query, _chunks, _tfidf, top_k=top_k)
    if not results:
        return "No relevant information found in the knowledge base."
    return "\n\n".join(f"- {r}" for r in results)


# ─── Dev test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_rag()
    print(f"Loaded {len(_chunks)} chunks.\n")
    test_queries = [
        "What is the Pro plan price?",
        "Do you offer refunds?",
        "Is 4K supported?",
        "24/7 support availability"
    ]
    for q in test_queries:
        print(f"Query: {q}")
        print(query_kb(q))
        print()
