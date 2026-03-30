"""
RAG Tutor — Main entry point.
Usage:
  python main.py ingest                Ingest new PDFs from data/raw_pdfs/
  python main.py ingest --rebuild      Wipe index and re-ingest all PDFs
  python main.py ask "question"        Ask a question
  python main.py list                  List ingested documents
  python main.py demo                  Run demo with dummy data
"""

import sys
import os
import shutil
import json
from datetime import datetime

from src.pdf_processing import parse_pdf
from src.rag import ingest_chunks, load_vectorstore, build_rag_chain, ask
from src.rag.ingest import append_to_vectorstore

# ─────────────────────────────────────────────
# PATHS & LIMITS
# ─────────────────────────────────────────────
RAW_DIR = "./data/raw_pdfs"
PROCESSED_DIR = "./data/processed"
FAISS_INDEX_PATH = "./data/faiss_index"
MANIFEST_PATH = "./data/ingested_files.json"

MAX_FILES = 10
MAX_FILE_SIZE_MB = 30
MAX_TOTAL_SIZE_MB = 100
ALLOWED_EXTENSIONS = {".pdf"}
BLOCKED_EXTENSIONS = {".zip", ".tar", ".gz", ".7z", ".rar"}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_manifest() -> list[dict]:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return []


def save_manifest(manifest: list[dict]):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def get_new_files() -> list[str]:
    """Get list of files in raw_pdfs directory."""
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR, exist_ok=True)
        return []
    return [
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if os.path.isfile(os.path.join(RAW_DIR, f)) and not f.startswith(".")
    ]


def validate_files(files: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate new files against limits.
    Returns (valid_files, errors).
    """
    errors = []
    valid = []

    if len(files) == 0:
        errors.append("No files found in data/raw_pdfs/")
        return valid, errors

    if len(files) > MAX_FILES:
        errors.append(f"Too many files: {len(files)} (max {MAX_FILES} per ingest)")
        return valid, errors

    total_size = 0

    for filepath in files:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        size_mb = os.path.getsize(filepath) / (1024 * 1024)

        # Check blocked extensions
        if ext in BLOCKED_EXTENSIONS:
            errors.append(f"REJECTED {filename}: {ext} files are not allowed (security risk)")
            continue

        # Check allowed extensions
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"REJECTED {filename}: only PDF files are supported (got {ext})")
            continue

        # Check individual file size
        if size_mb > MAX_FILE_SIZE_MB:
            errors.append(f"REJECTED {filename}: {size_mb:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit")
            continue

        total_size += size_mb
        valid.append(filepath)

    # Check total size
    if total_size > MAX_TOTAL_SIZE_MB:
        errors.append(f"Total size {total_size:.1f}MB exceeds {MAX_TOTAL_SIZE_MB}MB limit")
        return [], errors

    return valid, errors


def move_to_processed(filepath: str):
    """Move a processed file from raw_pdfs to processed."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    dest = os.path.join(PROCESSED_DIR, os.path.basename(filepath))

    # Handle duplicate filenames
    if os.path.exists(dest):
        name, ext = os.path.splitext(os.path.basename(filepath))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(PROCESSED_DIR, f"{name}_{timestamp}{ext}")

    shutil.move(filepath, dest)
    return dest


# ─────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────
def cmd_ingest(rebuild: bool = False):
    """Ingest PDFs from data/raw_pdfs/."""

    if rebuild:
        print("REBUILD MODE — starting fresh\n")

        # Delete FAISS index
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
            print("  Deleted FAISS index")

        # Clear manifest
        if os.path.exists(MANIFEST_PATH):
            os.remove(MANIFEST_PATH)
            print("  Cleared manifest")

        # Move all processed files back to raw_pdfs
        if os.path.exists(PROCESSED_DIR):
            moved_back = 0
            for f in os.listdir(PROCESSED_DIR):
                src = os.path.join(PROCESSED_DIR, f)
                if os.path.isfile(src) and not f.startswith("."):
                    shutil.move(src, os.path.join(RAW_DIR, f))
                    moved_back += 1
            if moved_back:
                print(f"  Moved {moved_back} files back to raw_pdfs/")

        print()

    # Get and validate new files
    files = get_new_files()
    valid_files, errors = validate_files(files)

    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
        if not valid_files:
            return
        print()

    print(f"Found {len(valid_files)} valid PDF(s) to ingest:\n")
    for f in valid_files:
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f)} ({size:.1f}MB)")
    print()

    # Process each file
    manifest = load_manifest()
    all_chunks = []

    for filepath in valid_files:
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")

        chunks = parse_pdf(filepath)
        all_chunks.extend(chunks)

        # Move to processed
        dest = move_to_processed(filepath)

        # Add to manifest
        manifest.append({
            "filename": filename,
            "processed_path": dest,
            "chunks": len(chunks),
            "size_mb": round(os.path.getsize(dest) / (1024 * 1024), 2),
            "ingested_at": datetime.now().isoformat(),
        })

        print(f"  → {len(chunks)} chunks | moved to processed/\n")

    if not all_chunks:
        print("No chunks to embed.")
        return

    # Embed and store
    print(f"Embedding {len(all_chunks)} total chunks...")

    if rebuild or not os.path.exists(FAISS_INDEX_PATH):
        ingest_chunks(all_chunks)
    else:
        append_to_vectorstore(all_chunks)

    save_manifest(manifest)
    print(f"\nDone! {len(valid_files)} document(s) ingested. Ready for questions.")


def cmd_list():
    """List all ingested documents."""
    manifest = load_manifest()

    if not manifest:
        print("No documents ingested yet. Add PDFs to data/raw_pdfs/ and run: python main.py ingest")
        return

    print(f"\nIngested Documents ({len(manifest)} total):\n")
    print(f"  {'Filename':<35} {'Chunks':<10} {'Size':<10} {'Ingested':<20}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*20}")

    total_chunks = 0
    total_size = 0

    for doc in manifest:
        total_chunks += doc["chunks"]
        total_size += doc["size_mb"]
        date = doc["ingested_at"][:10]
        print(f"  {doc['filename']:<35} {doc['chunks']:<10} {doc['size_mb']:.1f}MB{'':<5} {date:<20}")

    print(f"\n  Total: {total_chunks} chunks | {total_size:.1f}MB")


def cmd_ask(question: str):
    """Ask a question."""
    if not os.path.exists(FAISS_INDEX_PATH):
        print("No documents ingested yet. Run: python main.py ingest")
        return

    vs = load_vectorstore()
    chain = build_rag_chain(vs)
    result = ask(chain, question)

    if result["abstained"]:
        print(f"\n{result['answer']}")
    else:
        print(f"\nAnswer: {result['answer']}\n")
        print("=" * 60)
        print("Sources:")
        print("=" * 60)
        for i, s in enumerate(result["sources"], 1):
            print(f"\n  [{i}] {s['source']} — Page {s['page']} ({s['type']}) | Relevance: {s['relevance']}")



def cmd_demo():
    """Quick demo with dummy data."""
    dummy = [
        {"text": "Machine learning is a subset of AI that learns from data.",
         "metadata": {"source": "demo.pdf", "page": 1, "type": "text"}},
        {"text": "Overfitting happens when a model memorizes training data noise.",
         "metadata": {"source": "demo.pdf", "page": 2, "type": "text"}},
    ]
    vs = ingest_chunks(dummy)
    chain = build_rag_chain(vs)
    result = ask(chain, "What is overfitting?")
    print(f"\nAnswer: {result['answer']}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        rebuild = "--rebuild" in sys.argv
        cmd_ingest(rebuild=rebuild)

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Provide a question: python main.py ask \"What is...\"")
            sys.exit(1)
        cmd_ask(sys.argv[2])

    elif command == "list":
        cmd_list()

    elif command == "demo":
        cmd_demo()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()