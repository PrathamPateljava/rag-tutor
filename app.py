"""
Nova — RAG Tutor UI
"""

import streamlit as st
import os
import json
import shutil
from datetime import datetime

from src.pdf_processing import parse_pdf
from src.rag import ingest_chunks, load_vectorstore, build_rag_chain, ask, ConversationMemory
from src.rag.ingest import append_to_vectorstore

# ─────────────────────────────────────────────
# CONFIG
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


def index_exists() -> bool:
    return os.path.exists(FAISS_INDEX_PATH)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chain_loaded" not in st.session_state:
    st.session_state.chain_loaded = False


def load_chain():
    """Load the RAG chain into session state."""
    if index_exists() and not st.session_state.chain_loaded:
        with st.spinner("Loading knowledge base..."):
            vs = load_vectorstore()
            st.session_state.chain = build_rag_chain(vs)
            st.session_state.chain_loaded = True


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nova — AI Tutor",
    page_icon="🎓",
    layout="wide",
)


# ─────────────────────────────────────────────
# SIDEBAR — Document Management
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Document Manager")

    # Upload section
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop your course PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max {MAX_FILES} files, {MAX_FILE_SIZE_MB}MB each, {MAX_TOTAL_SIZE_MB}MB total",
    )

    if uploaded_files:
        # Validate
        errors = []
        valid_files = []

        if len(uploaded_files) > MAX_FILES:
            errors.append(f"Too many files: {len(uploaded_files)} (max {MAX_FILES})")
        else:
            total_size = 0
            for f in uploaded_files:
                size_mb = f.size / (1024 * 1024)
                ext = os.path.splitext(f.name)[1].lower()

                if ext in BLOCKED_EXTENSIONS:
                    errors.append(f"❌ {f.name}: blocked file type")
                elif ext not in ALLOWED_EXTENSIONS:
                    errors.append(f"❌ {f.name}: not a PDF")
                elif size_mb > MAX_FILE_SIZE_MB:
                    errors.append(f"❌ {f.name}: {size_mb:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit")
                else:
                    total_size += size_mb
                    valid_files.append(f)

            if total_size > MAX_TOTAL_SIZE_MB:
                errors.append(f"Total size {total_size:.1f}MB exceeds {MAX_TOTAL_SIZE_MB}MB limit")
                valid_files = []

        if errors:
            for e in errors:
                st.warning(e)

        if valid_files and st.button(f"📥 Ingest {len(valid_files)} PDF(s)", type="primary"):
            os.makedirs(RAW_DIR, exist_ok=True)
            os.makedirs(PROCESSED_DIR, exist_ok=True)

            manifest = load_manifest()
            all_chunks = []
            progress = st.progress(0)

            for i, f in enumerate(valid_files):
                progress.progress((i) / len(valid_files), text=f"Processing {f.name}...")

                # Save uploaded file temporarily
                temp_path = os.path.join(RAW_DIR, f.name)
                with open(temp_path, "wb") as out:
                    out.write(f.getbuffer())

                # Parse
                chunks = parse_pdf(temp_path, include_images=False)
                all_chunks.extend(chunks)

                # Move to processed
                dest = os.path.join(PROCESSED_DIR, f.name)
                if os.path.exists(dest):
                    name, ext = os.path.splitext(f.name)
                    dest = os.path.join(PROCESSED_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
                shutil.move(temp_path, dest)

                manifest.append({
                    "filename": f.name,
                    "processed_path": dest,
                    "chunks": len(chunks),
                    "size_mb": round(f.size / (1024 * 1024), 2),
                    "ingested_at": datetime.now().isoformat(),
                })

            if all_chunks:
                progress.progress(0.9, text="Embedding chunks...")
                if index_exists():
                    append_to_vectorstore(all_chunks)
                else:
                    ingest_chunks(all_chunks)

                save_manifest(manifest)
                progress.progress(1.0, text="Done!")
                st.success(f"✅ Ingested {len(valid_files)} document(s) — {len(all_chunks)} chunks")

                # Reload chain
                st.session_state.chain_loaded = False
                load_chain()

    # Ingested documents
    st.divider()
    st.subheader("Ingested Documents")

    manifest = load_manifest()
    if manifest:
        total_chunks = sum(d["chunks"] for d in manifest)
        st.caption(f"{len(manifest)} document(s) · {total_chunks} chunks")

        for doc in manifest:
            with st.expander(f"📄 {doc['filename']}"):
                st.write(f"**Chunks:** {doc['chunks']}")
                st.write(f"**Size:** {doc['size_mb']}MB")
                st.write(f"**Ingested:** {doc['ingested_at'][:10]}")
    else:
        st.info("No documents ingested yet. Upload PDFs above.")

    # Rebuild button
    st.divider()
    if st.button("🔄 Rebuild Index", help="Wipe index and re-ingest all documents"):
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        if os.path.exists(MANIFEST_PATH):
            os.remove(MANIFEST_PATH)

        # Move processed files back
        if os.path.exists(PROCESSED_DIR):
            for f in os.listdir(PROCESSED_DIR):
                src = os.path.join(PROCESSED_DIR, f)
                if os.path.isfile(src) and f.endswith(".pdf"):
                    shutil.move(src, os.path.join(RAW_DIR, f))

        st.session_state.chain_loaded = False
        st.session_state.messages = []
        st.session_state.memory = ConversationMemory(max_turns=5)
        st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory = ConversationMemory(max_turns=5)
        st.rerun()


# ─────────────────────────────────────────────
# MAIN — Chat Interface
# ─────────────────────────────────────────────
st.title("🎓 Nova — AI Tutor")
st.caption("Ask questions about your course materials. Nova only answers from your uploaded PDFs.")

# Load chain if index exists
load_chain()

if not index_exists():
    st.info("👈 Upload PDFs in the sidebar to get started.")
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['source']} — Page {s['page']} | Relevance: {s['relevance']}")
                    if "snippet" in s:
                        st.text(s["snippet"])

# Chat input
if prompt := st.chat_input("Ask Nova a question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            result = ask(
                st.session_state.chain,
                prompt,
                memory=st.session_state.memory,
            )

        st.markdown(result["answer"])

        sources = result.get("sources", [])
        if sources and not result["abstained"]:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.caption(f"📄 {s['source']} — Page {s['page']} | Relevance: {s['relevance']}")
                    if "snippet" in s:
                        st.text(s["snippet"])

    # Store in session
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources if not result["abstained"] else [],
    })

    # Update memory
    if not result["abstained"]:
        st.session_state.memory.add(prompt, result["answer"])