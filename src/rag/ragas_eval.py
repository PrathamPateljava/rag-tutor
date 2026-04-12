"""
RAGAS Evaluation for RAG Tutor
==============================
Evaluates the full pipeline (retrieval + generation) using RAGAS metrics:
  - Faithfulness: Is the answer grounded in retrieved context?
  - Context Precision: Are retrieved chunks relevant to the question?
  - Context Recall: Did we retrieve everything needed to answer?
"""

import sys
import os
import asyncio
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.dataset_schema import SingleTurnSample

from src.rag.ingest import load_vectorstore, get_embeddings
from src.rag.chain import (
    TUTOR_PROMPT,
    TOP_K,
    MMR_FETCH_K,
    MMR_LAMBDA,
    OLLAMA_MODEL,
)
from langchain_ollama import OllamaLLM


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RESULTS_PATH = "./data/ragas_results.json"

# Test dataset: question, ground_truth answer, and expected source
TEST_DATASET = [
    {
        "question": "What are the three types of machine learning?",
        "ground_truth": "The three types are supervised learning (learns from labeled data), unsupervised learning (finds patterns in unlabeled data), and reinforcement learning (agent learns by interacting with environment and receiving rewards).",
    },
    {
        "question": "What is backpropagation?",
        "ground_truth": "Backpropagation is the algorithm used to train neural networks. It calculates the gradient of the loss function with respect to each weight by applying the chain rule of calculus, propagating the error backward through the network.",
    },
    {
        "question": "What is the difference between precision and recall?",
        "ground_truth": "Precision measures how many positive predictions were actually correct (TP/(TP+FP)). Recall measures how many actual positives were correctly identified (TP/(TP+FN)).",
    },
    {
        "question": "How does dropout prevent overfitting?",
        "ground_truth": "Dropout randomly deactivates neurons during training with a given probability (typically 0.2-0.5). This prevents neurons from co-adapting and forces the network to learn more robust features.",
    },
    {
        "question": "What is the bias-variance tradeoff?",
        "ground_truth": "High bias (underfitting) means the model is too simple and misses important patterns. High variance (overfitting) means the model is too complex and memorizes noise. The goal is to find the sweet spot that minimizes total error.",
    },
    {
        "question": "How do random forests improve on decision trees?",
        "ground_truth": "Random forests are ensembles of decision trees where each tree is trained on a random subset of data and features. The final prediction is the majority vote or average of all trees, which reduces overfitting compared to individual trees.",
    },
    {
        "question": "What is the kernel trick in SVM?",
        "ground_truth": "The kernel trick allows SVMs to handle non-linearly separable data by mapping it to a higher-dimensional space where a linear boundary can be found. Common kernels include linear, polynomial, and RBF.",
    },
    {
        "question": "What is the Adam optimizer?",
        "ground_truth": "Adam (Adaptive Moment Estimation) is an optimizer that combines momentum and adaptive learning rates. It is the most popular choice for training neural networks.",
    },
    {
        "question": "What is cross-validation?",
        "ground_truth": "K-fold cross-validation divides data into K equal parts. The model is trained K times, each time using K-1 folds for training and 1 fold for validation, providing a more robust performance estimate than a single train-test split.",
    },
    {
        "question": "What is the difference between L1 and L2 regularization?",
        "ground_truth": "L1 regularization (Lasso) adds the absolute value of weights to the loss and can produce sparse models. L2 regularization (Ridge) adds squared weights to the loss and prevents any single weight from being too large.",
    },
]


def get_ragas_llm():
    """Initialize Ollama as the RAGAS judge LLM via OpenAI-compatible API."""
    client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )
    return llm_factory(OLLAMA_MODEL, provider="openai", client=client)


def retrieve_and_generate(question: str, vectorstore) -> dict:
    """Run our RAG pipeline: retrieve chunks + generate answer."""
    # Retrieve with MMR
    docs = vectorstore.max_marginal_relevance_search(
        question,
        k=TOP_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA,
    )

    contexts = [doc.page_content for doc in docs]
    context_str = "\n\n---\n\n".join(contexts)

    # Generate answer
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)
    prompt = TUTOR_PROMPT.format(context=context_str, question=question)
    answer = llm.invoke(prompt)

    return {
        "answer": answer,
        "contexts": contexts,
    }


async def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    metrics: list,
    llm,
) -> dict:
    """Score a single question across all RAGAS metrics."""
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )

    scores = {}
    for metric in metrics:
        try:
            score = await metric.single_turn_ascore(sample)
            scores[metric.name] = round(float(score), 4)
            print(f"    {metric.name}: {scores[metric.name]}")
        except Exception as e:
            scores[metric.name] = None
            print(f"    {metric.name}: FAILED ({e})")

    return scores


async def run_evaluation():
    """Run full RAGAS evaluation on test dataset."""
    print("=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)

    # Load vectorstore
    print("\nLoading vectorstore...")
    vs = load_vectorstore()

    # Init RAGAS judge LLM
    print("Initializing RAGAS judge (Llama 3 via Ollama)...")
    llm = get_ragas_llm()

    # Init metrics with LLM set on each
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm),
        LLMContextPrecisionWithoutReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]

    results = []

    for i, test_case in enumerate(TEST_DATASET, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        print(f"\n[{i}/{len(TEST_DATASET)}] {question}")

        # Run our pipeline
        print("  Retrieving and generating...")
        output = retrieve_and_generate(question, vs)

        # Score with RAGAS
        print("  Scoring with RAGAS:")
        scores = await evaluate_single(
            question=question,
            answer=output["answer"],
            contexts=output["contexts"],
            ground_truth=ground_truth,
            metrics=metrics,
            llm=llm,
        )

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": output["answer"][:300],
            "num_contexts": len(output["contexts"]),
            "scores": scores,
        })

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE SCORES")
    print("=" * 60)

    metric_names = [m.name for m in metrics]
    for name in metric_names:
        valid_scores = [r["scores"][name] for r in results if r["scores"].get(name) is not None]
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            print(f"  {name:<40} {avg:.4f}  ({len(valid_scores)}/{len(results)} scored)")
        else:
            print(f"  {name:<40} NO SCORES")

    # Per-question breakdown
    print("\n" + "=" * 60)
    print("PER-QUESTION BREAKDOWN")
    print("=" * 60)

    for r in results:
        print(f"\n  Q: {r['question']}")
        for name, score in r["scores"].items():
            bar = "█" * int((score or 0) * 20) + "░" * (20 - int((score or 0) * 20))
            score_str = f"{score:.2f}" if score is not None else "N/A"
            print(f"    {name:<35} {bar} {score_str}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": OLLAMA_MODEL,
        "num_questions": len(TEST_DATASET),
        "results": results,
        "aggregate": {
            name: round(
                sum(r["scores"][name] for r in results if r["scores"].get(name) is not None)
                / max(1, sum(1 for r in results if r["scores"].get(name) is not None)),
                4,
            )
            for name in metric_names
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())