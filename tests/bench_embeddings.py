"""
Embedding Model Benchmark
=========================
Compares retrieval quality of different embedding models
on the same set of test questions with known expected pages.

Run: python benchmark_embeddings.py
"""

import time
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.pdf_processing.parser import *

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PDF_PATH = "./data/raw_pdfs/ml_fundamentals.pdf"
TOP_K = 4

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
]

# Test questions with expected correct pages from ml_fundamentals.pdf
# Each question maps to pages that contain the answer
TEST_QUESTIONS = [
    {
        "question": "What are the three types of machine learning?",
        "expected_pages": [1],
        "topic": "ML basics",
    },
    {
        "question": "What is backpropagation?",
        "expected_pages": [2],
        "topic": "Neural networks",
    },
    {
        "question": "What activation functions are used in neural networks?",
        "expected_pages": [2],
        "topic": "Neural networks",
    },
    {
        "question": "What is the difference between batch and stochastic gradient descent?",
        "expected_pages": [3],
        "topic": "Optimization",
    },
    {
        "question": "How does dropout prevent overfitting?",
        "expected_pages": [3],
        "topic": "Regularization",
    },
    {
        "question": "What is the Adam optimizer?",
        "expected_pages": [3],
        "topic": "Optimization",
    },
    {
        "question": "What is the difference between precision and recall?",
        "expected_pages": [4],
        "topic": "Evaluation",
    },
    {
        "question": "Explain the bias-variance tradeoff",
        "expected_pages": [4],
        "topic": "Evaluation",
    },
    {
        "question": "How do random forests work?",
        "expected_pages": [5],
        "topic": "Algorithms",
    },
    {
        "question": "What is the kernel trick in SVM?",
        "expected_pages": [5],
        "topic": "Algorithms",
    },
    {
        "question": "What is K-nearest neighbors?",
        "expected_pages": [5],
        "topic": "Algorithms",
    },
    {
        "question": "What is cross-validation?",
        "expected_pages": [4],
        "topic": "Evaluation",
    },
]

# Out-of-scope questions — best similarity should be LOW
OUT_OF_SCOPE = [
    "How do I bake a cake?",
    "What is the capital of France?",
    "Explain quantum computing",
    "What is a transformer architecture?",
    "How do I train my dog?",
]


def build_vectorstore(chunks: list[dict], model_name: str) -> tuple:
    """Build FAISS index with given embedding model. Returns (vectorstore, embed_time)."""
    docs = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in chunks if c["text"].strip()
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "mps"},  # change to "cuda" or "cpu" if needed
        encode_kwargs={"normalize_embeddings": True},
    )

    start = time.time()
    vs = FAISS.from_documents(docs, embeddings)
    embed_time = time.time() - start

    return vs, embed_time


def evaluate_retrieval(vs: FAISS, questions: list[dict], top_k: int) -> dict:
    """Evaluate retrieval quality on test questions."""
    results = []

    for q in questions:
        start = time.time()
        docs_with_scores = vs.similarity_search_with_score(q["question"], k=top_k)
        query_time = time.time() - start

        retrieved_pages = [doc.metadata.get("page", -1) for doc, _ in docs_with_scores]
        scores = [1 / (1 + score) for _, score in docs_with_scores]

        # Hit@K: did any retrieved chunk come from an expected page?
        hit = any(p in q["expected_pages"] for p in retrieved_pages)

        # Hit@1: did the TOP chunk come from an expected page?
        hit_at_1 = retrieved_pages[0] in q["expected_pages"] if retrieved_pages else False

        # Reciprocal rank: position of first correct result
        rr = 0.0
        for i, p in enumerate(retrieved_pages):
            if p in q["expected_pages"]:
                rr = 1.0 / (i + 1)
                break

        results.append({
            "question": q["question"],
            "topic": q["topic"],
            "expected_pages": q["expected_pages"],
            "retrieved_pages": retrieved_pages,
            "scores": [f"{s:.4f}" for s in scores],
            "best_score": max(scores) if scores else 0,
            "hit_at_k": hit,
            "hit_at_1": hit_at_1,
            "reciprocal_rank": rr,
            "query_time_ms": f"{query_time * 1000:.1f}",
        })

    return results


def evaluate_out_of_scope(vs: FAISS, questions: list[str], top_k: int) -> list[dict]:
    """Check that out-of-scope questions get LOW similarity scores."""
    results = []
    for q in questions:
        docs_with_scores = vs.similarity_search_with_score(q, k=top_k)
        scores = [1 / (1 + score) for _, score in docs_with_scores]
        best = max(scores) if scores else 0
        results.append({
            "question": q,
            "best_score": best,
            "would_abstain_035": best < 0.35,
            "would_abstain_045": best < 0.45,
        })
    return results


def print_report(model_name: str, embed_time: float, results: list[dict], oos_results: list[dict]):
    """Print a formatted benchmark report for one model."""
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 70}")
    print(f"Embedding time: {embed_time:.2f}s")

    # Aggregate metrics
    hit_at_k = sum(1 for r in results if r["hit_at_k"]) / len(results)
    hit_at_1 = sum(1 for r in results if r["hit_at_1"]) / len(results)
    mrr = sum(r["reciprocal_rank"] for r in results) / len(results)
    avg_best = sum(r["best_score"] for r in results) / len(results)
    avg_time = sum(float(r["query_time_ms"]) for r in results) / len(results)

    print(f"\n--- In-Scope Retrieval Metrics ---")
    print(f"  Hit@{TOP_K}:              {hit_at_k:.0%} ({sum(1 for r in results if r['hit_at_k'])}/{len(results)})")
    print(f"  Hit@1:                {hit_at_1:.0%} ({sum(1 for r in results if r['hit_at_1'])}/{len(results)})")
    print(f"  MRR (Mean Recip Rank): {mrr:.4f}")
    print(f"  Avg Best Similarity:  {avg_best:.4f}")
    print(f"  Avg Query Time:       {avg_time:.1f}ms")

    print(f"\n--- Per-Question Breakdown ---")
    for r in results:
        status = "HIT" if r["hit_at_k"] else "MISS"
        print(f"  [{status}] {r['question']}")
        print(f"        Expected p.{r['expected_pages']} | Got p.{r['retrieved_pages']} | Best: {r['best_score']:.4f}")

    print(f"\n--- Out-of-Scope Rejection ---")
    for r in oos_results:
        print(f"  Score: {r['best_score']:.4f} | Abstain@0.45: {'YES' if r['would_abstain_045'] else 'NO '} | \"{r['question']}\"")

    avg_oos = sum(r["best_score"] for r in oos_results) / len(oos_results)
    print(f"\n  Avg OOS Score: {avg_oos:.4f} (lower is better)")

    return {
        "model": model_name,
        "embed_time": embed_time,
        "hit_at_k": hit_at_k,
        "hit_at_1": hit_at_1,
        "mrr": mrr,
        "avg_best_similarity": avg_best,
        "avg_query_time_ms": avg_time,
        "avg_oos_score": avg_oos,
    }


def main():
    print("Parsing PDF...")
    chunks = parse_pdf(PDF_PATH, include_images=False)
    print(f"Total chunks: {len(chunks)}\n")

    summaries = []

    for model_name in MODELS:
        print(f"\nBuilding index with {model_name}...")
        vs, embed_time = build_vectorstore(chunks, model_name)

        print(f"Running in-scope evaluation...")
        results = evaluate_retrieval(vs, TEST_QUESTIONS, TOP_K)

        print(f"Running out-of-scope evaluation...")
        oos_results = evaluate_out_of_scope(vs, OUT_OF_SCOPE, TOP_K)

        summary = print_report(model_name, embed_time, results, oos_results)
        summaries.append(summary)

    # Comparison
    print(f"\n{'=' * 70}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Metric':<25} | {MODELS[0].split('/')[-1]:<20} | {MODELS[1].split('/')[-1]:<20}")
    print("-" * 70)

    metrics = ["hit_at_k", "hit_at_1", "mrr", "avg_best_similarity", "avg_query_time_ms", "embed_time", "avg_oos_score"]
    labels = ["Hit@4", "Hit@1", "MRR", "Avg Similarity", "Avg Query (ms)", "Embed Time (s)", "Avg OOS Score (lower=better)"]

    for label, metric in zip(labels, metrics):
        v1 = summaries[0][metric]
        v2 = summaries[1][metric]
        fmt = ".4f" if isinstance(v1, float) and v1 < 10 else ".1f"
        print(f"  {label:<25} | {v1:{fmt}:<20} | {v2:{fmt}:<20}")

    print(f"\n{'=' * 70}")
    print("RECOMMENDATION")
    print(f"{'=' * 70}")

    if summaries[0]["mrr"] > summaries[1]["mrr"]:
        winner = MODELS[0].split("/")[-1]
    elif summaries[1]["mrr"] > summaries[0]["mrr"]:
        winner = MODELS[1].split("/")[-1]
    else:
        winner = "TIE"

    print(f"  Based on MRR (primary metric): {winner}")
    print(f"  Run this benchmark on your actual course PDFs for a final decision.\n")


if __name__ == "__main__":
    main()