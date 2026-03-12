"""
RAG Tutor — Main entry point.
Usage:
  python main.py ingest <pdf_path>
  python main.py ask "Your question here"
  python main.py demo
"""

import sys
from src.pdf_processing import parse_pdf
from src.rag import ingest_chunks, load_vectorstore, build_rag_chain, ask


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Ingest:  python main.py ingest <pdf_path>")
        print("  Ask:     python main.py ask \"Your question here\"")
        print("  Demo:    python main.py demo")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Provide a PDF path: python main.py ingest <pdf_path>")
            sys.exit(1)
        pdf_path = sys.argv[2]
        print(f"Parsing {pdf_path}...")
        chunks = parse_pdf(pdf_path)
        print(f"Got {len(chunks)} chunks. Embedding...")
        ingest_chunks(chunks)
        print("Done! Ready for questions.")

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Provide a question: python main.py ask \"What is...\"")
            sys.exit(1)
        question = sys.argv[2]
        vs = load_vectorstore()
        chain = build_rag_chain(vs)
        result = ask(chain, question)
        print(f"\nAnswer: {result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']} p.{s['page']}")

    elif command == "demo":
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

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()