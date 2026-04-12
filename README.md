# Nova — RAG-Based Intelligent Tutor

An AI-powered tutor that answers questions exclusively from your course PDFs using Retrieval-Augmented Generation. Nova refuses to fabricate answers — if the information isn't in your uploaded materials, it says so.

## Tech Stack
- **LLM**: Llama 3 (via Ollama)
- **Vision Model**: LLaVA (via Ollama) for image descriptions
- **Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: bge-base-en-v1.5
- **PDF Processing**: PyMuPDF + Tesseract OCR
- **UI**: Streamlit

## Quick Start

### 1. Install dependencies
```bash
# Install Ollama from https://ollama.com/download
ollama pull llama3
ollama pull llava
brew install tesseract

# Python dependencies
pip install -r requirements.txt
```

### 2. Run the UI
```bash
streamlit run app.py
```
### Note:
This app cannot be deployed as it requires ollama to be configured on the server. Kindly clone and run it locally with ollama downloaded on your device.
### 3. Or use the CLI
```bash
python main.py ingest                # Ingest new PDFs from data/raw_pdfs/
python main.py ingest --rebuild      # Wipe index and re-ingest everything
python main.py ask "What is...?"     # Ask a single question
python main.py chat                  # Interactive conversation mode
python main.py list                  # List ingested documents
python main.py demo                  # Quick demo with dummy data
```

## Features

### Multi-Document Ingestion
Upload multiple PDFs through the UI or CLI with built-in validation: max 10 files per ingest, 30MB per file, 100MB total, PDF-only, zip files explicitly rejected. Processed files move to `data/processed/` with a JSON manifest tracking filenames, chunk counts, and timestamps. Supports both append mode and full rebuild.

### Conversational Memory
Nova maintains a sliding window of the last 5 exchanges so students can ask follow-up questions naturally. "What is overfitting?" followed by "How do I prevent it?" works without restating context. Short follow-ups are rewritten for better retrieval, and conversation history is passed to both the relevance check and the generation prompt.

### Image & OCR Handling
Embedded images are extracted from PDFs and described using LLaVA, a vision-language model. Descriptions become searchable text chunks alongside regular content. Scanned PDF pages (image-only, no extractable text) are detected automatically and processed with Tesseract OCR.

### Source Citations
Every answer includes deduplicated citations showing document name, page number, relevance score, and a text snippet of what was retrieved. In the UI, sources are expandable under each response.

### Tutor Persona
Nova follows a structured teaching style: one-sentence summary, detailed explanation in plain language, real-world analogy, step-by-step breakdown of technical terms, and a key takeaway. The persona is warm, patient, and honest — it celebrates good questions and admits when it doesn't know.

## Embedding Model Selection

We benchmarked two sentence embedding models on a 19-chunk corpus using 12 in-scope questions with known answers and 5 out-of-scope questions. The full benchmark can be reproduced by running:
```bash
python ./src/bench_embeddings.py
```

| Metric | all-MiniLM-L6-v2 | bge-base-en-v1.5 |
|---|---|---|
| Hit@1 | 92% | **100%** |
| MRR | 0.96 | **1.00** |
| Avg OOS Score (lower=better) | **0.39** | 0.49 |

`bge-base-en-v1.5` achieved perfect retrieval accuracy. The tradeoff — higher out-of-scope similarity scores — is addressed by the hallucination prevention system.

## Hallucination Prevention

The system enforces a strict no-hallucination policy through three layers:

**Layer 1 — Retrieval Similarity Threshold:** Before the LLM is invoked, the system checks cosine similarity scores of retrieved chunks. If the best match falls below 0.55, the system refuses without calling the model.

**Layer 2 — LLM Relevance Check:** A secondary LLM call evaluates whether the question genuinely relates to the retrieved course material, catching vocabulary overlap tricks (e.g., "bias in hiring" matching ML chunks about bias-variance). Conversation history is included so follow-up questions aren't incorrectly rejected.

**Layer 3 — Prompt Enforcement:** The tutor prompt constrains the model to only use provided context, never draw on outside knowledge, and redirect off-topic questions.

This architecture trades recall for precision — the system may occasionally refuse answerable questions, but will not fabricate information.

## MMR Retrieval

Pure cosine similarity was replaced with Max Marginal Relevance, which balances relevance against diversity among retrieved chunks. This prevents returning near-duplicate passages from the same page and improves coverage across multiple ingested documents.

## RAGAS Evaluation

The system includes a RAGAS evaluation script that measures faithfulness, answer relevancy, context precision, and context recall using Llama 3 as the judge. Run it with:
```bash
python -m src.ragas_eval
```
Results are saved to `data/ragas_results.json` with per-question scores and aggregate metrics.

## Project Structure
```
rag-tutor/
├── app.py                       # Streamlit UI
├── main.py                      # CLI entry point
├── src/
│   ├── rag/
│   │   ├── ingest.py            # Embedding + FAISS (bge-base-en-v1.5, GPU auto-detect)
│   │   ├── chain.py             # 3-layer abstention, MMR retrieval, Nova persona
│   │   └── memory.py            # Conversational memory (sliding window)
│   ├── pdf_processing/
│   │   ├── parser.py            # Text extraction + chunking
│   │   └── image_handler.py     # Image extraction (LLaVA) + OCR (Tesseract)
│   ├── utils/
│   │   └── config.py
│   ├── bench_embeddings.py      # Embedding model benchmark
│   └── ragas_eval.py            # RAGAS evaluation framework
├── data/
│   ├── raw_pdfs/                # Drop PDFs here
│   ├── processed/               # Ingested PDFs moved here
│   └── ingested_files.json      # Manifest
├── tests/
├── notebooks/
├── configs/
├── requirements.txt
└── README.md
```

## Team
- **Pratham Patel** — RAG pipeline (embedding, retrieval, generation, evaluation, UI)
- **Prachita Patel** — PDF processing (extraction, cleaning, chunking, image handling, OCR)