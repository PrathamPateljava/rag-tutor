"""Embedding + FAISS vector store management."""

import os
import logging
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
FAISS_INDEX_PATH = "./data/faiss_index"


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embeddings():
    device = get_device()
    print(f"Using device: {device}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _chunks_to_docs(parsed_chunks: list[dict]) -> list[Document]:
    return [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in parsed_chunks if c["text"].strip()
    ]


def ingest_chunks(parsed_chunks: list[dict]) -> FAISS:
    """Create a new FAISS index from chunks."""
    docs = _chunks_to_docs(parsed_chunks)
    print(f"Ingesting {len(docs)} chunks...")
    embeddings = get_embeddings()
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_INDEX_PATH)
    print(f"Index saved to {FAISS_INDEX_PATH}")
    return vs


def append_to_vectorstore(parsed_chunks: list[dict]) -> FAISS:
    """Append new chunks to an existing FAISS index."""
    docs = _chunks_to_docs(parsed_chunks)
    print(f"Appending {len(docs)} chunks to existing index...")
    embeddings = get_embeddings()
    vs = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    vs.add_documents(docs)
    vs.save_local(FAISS_INDEX_PATH)
    print(f"Index updated at {FAISS_INDEX_PATH}")
    return vs


def load_vectorstore() -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )