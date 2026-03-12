# RAG Tutor

An AI-powered tutor that answers questions from your course PDFs using Retrieval-Augmented Generation.

## Tech Stack
- **LLM**: Llama 3 (via Ollama)
- **Framework**: LangChain
- **Vector Store**: FAISS
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
# Ingest a PDF
python main.py ingest path/to/textbook.pdf

# Ask a question
python main.py ask "What is gradient descent?"

# Quick demo with dummy data
python main.py demo
```

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
│   └── processed/           # Cleaned output
├── tests/
├── notebooks/               # Experiments
├── configs/
├── requirements.txt
└── README.md
```

## Team
- **Pratham Patel** — RAG pipeline (embedding, retrieval, generation)
- **Prachita Patel** — PDF processing (extraction, cleaning, chunking)