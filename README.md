# RAG Tutor

An AI-powered tutor that answers questions from your course PDFs using Retrieval-Augmented Generation.

## Tech Stack
- **LLM**: Llama 3 (via Ollama)
- **Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: bge-base-en-v1.5
- **PDF Processing**: PyMuPDF + pdfplumber

## Quick Start

### 1. Install Ollama & pull model
```bash
# Install from https://ollama.com/download
ollama pull llama3
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
# Add PDFs to data/raw_pdfs/ then ingest
python main.py ingest

# Ask a question
python main.py ask "What is gradient descent?"

# List ingested documents
python main.py list

# Wipe index and re-ingest everything
python main.py ingest --rebuild

# Quick demo with dummy data
python main.py demo
```

## Embedding Model Selection

We benchmarked two sentence embedding models on a 19-chunk ML fundamentals corpus using 12 in-scope questions with known answers and 5 out-of-scope questions. 
The full benchmark can be reproduced by running 
```bash
# Run Analysis to compare embedding techniques
python ./src/bench_embeddings.py
```
| Metric | all-MiniLM-L6-v2 | bge-base-en-v1.5 |
|---|---|---|
| Hit@1 | 92% | **100%** |
| MRR | 0.96 | **1.00** |
| Avg OOS Score (lower=better) | **0.39** | 0.49 |

`bge-base-en-v1.5` achieved perfect retrieval accuracy — every question returned the correct page as the top result. The tradeoff is higher similarity scores on out-of-scope queries, which we address through our hallucination prevention system (see below). We selected `bge-base-en-v1.5` as the stronger retriever for the final system.

## Hallucination Prevention

For an educational tutor, a fabricated answer is more harmful than no answer. The system enforces a strict no-hallucination policy through three layers:

**Layer 1 — Retrieval Similarity Threshold:** Before the LLM is invoked, the system checks the cosine similarity scores of retrieved chunks. If the best match falls below a tunable threshold (0.55), the system returns a refusal without calling the model at all. This blocks completely unrelated questions at minimal cost.

**Layer 2 — LLM Relevance Check:** Questions that pass the threshold may still be semantically misleading (e.g., "reduce bias in my hiring process" matches ML chunks about bias-variance tradeoff). A secondary LLM call evaluates whether the student's question genuinely relates to the retrieved course material before generating an answer. This catches vocabulary overlap tricks that the similarity threshold cannot.

**Layer 3 — Prompt Enforcement:** The tutor prompt strictly constrains the model to only use the provided context, never draw on outside knowledge, and redirect off-topic questions. This acts as the final safety net at generation time.

This architecture deliberately trades recall for precision — the system may occasionally refuse answerable questions, but it will not fabricate information. In an educational context where student trust is critical, we believe this is the correct tradeoff.

## Project Structure
```
rag-tutor/
├── main.py                  # CLI entry point
├── src/
│   ├── rag/                 # RAG pipeline (embedding, retrieval, generation)
│   │   ├── ingest.py
│   │   └── chain.py
│   ├── pdf_processing/      # PDF parsing & chunking
│   │   ├── parser.py
│   │   └── image_handler.py
│   └── utils/
│       └── config.py
├── data/
│   ├── raw_pdfs/            # Drop PDFs here
│   ├── processed/           # Ingested PDFs moved here
│   └── ingested_files.json  # Manifest
├── tests/
├── notebooks/               # Experiments
├── configs/
├── requirements.txt
└── README.md
```

## Team
- **Pratham Patel** — RAG pipeline (embedding, retrieval, generation, evaluation)
- **Prachita Patel** — PDF processing (extraction, cleaning, chunking, image handling)