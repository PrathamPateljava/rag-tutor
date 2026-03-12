"""
PDF Processing Module
=====================
Owner: [Teammate's name]

Output contract:
  parse_pdf(filepath) -> list[dict]
  Each dict: {"text": str, "metadata": {"source": str, "page": int, "type": "text"|"image"}}

Dependencies: pymupdf pdfplumber ftfy pytesseract pillow
"""

import fitz  # PyMuPDF
import os
import re


def extract_text(filepath: str) -> list[dict]:
    """Extract text from PDF page by page using PyMuPDF."""
    doc = fitz.open(filepath)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    doc.close()
    return pages


def clean_text(text: str) -> str:
    """Basic text cleaning. Expand as needed."""
    # Fix hyphenated line breaks
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def parse_pdf(filepath: str) -> list[dict]:
    """
    Main entry point — returns chunks ready for the RAG pipeline.
    """
    filename = os.path.basename(filepath)
    pages = extract_text(filepath)

    all_chunks = []
    for page_data in pages:
        cleaned = clean_text(page_data["text"])
        chunks = chunk_text(cleaned)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "page": page_data["page"],
                    "type": "text",
                },
            })

    return all_chunks