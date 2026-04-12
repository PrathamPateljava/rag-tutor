"""
PDF Processing Module
=====================
Extracts text and images from PDFs, returns RAG-ready chunks.
"""

import fitz  # PyMuPDF
import os
import re
from .image_handler import process_pdf_images


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
    """Basic text cleaning."""
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
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


def parse_pdf(filepath: str, include_images: bool = True) -> list[dict]:
    """
    Main entry point — returns text and image chunks ready for the RAG pipeline.

    Args:
        filepath: path to the PDF file
        include_images: if True, extract and describe images using LLaVA
    """
    filename = os.path.basename(filepath)
    pages = extract_text(filepath)

    # Text chunks
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

    print(f"Text chunks: {len(all_chunks)}")

    # Image chunks
    if include_images:
        image_chunks = process_pdf_images(filepath, page_texts=pages)
        all_chunks.extend(image_chunks)
        print(f"Image chunks: {len(image_chunks)}")

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks