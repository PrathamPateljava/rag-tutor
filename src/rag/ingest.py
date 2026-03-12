"""Embedding + FAISS vector store management."""

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

EMBED_MODEL = "llama3"
FAISS_INDEX_PATH = "./data/faiss_index"


def ingest_chunks(parsed_chunks: list[dict]) -> FAISS:
    docs = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in parsed_chunks if c["text"].strip()
    ]
    print(f"Ingesting {len(docs)} chunks...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_INDEX_PATH)
    print(f"Index saved to {FAISS_INDEX_PATH}")
    return vs


def load_vectorstore() -> FAISS:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )