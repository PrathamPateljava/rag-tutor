"""RAG chain: retrieval + Llama 3 generation with abstention logic."""

from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

OLLAMA_MODEL = "llama3"
TOP_K = 4
SIMILARITY_THRESHOLD = 0.55  # raised for bge-base-en-v1.5 (higher similarity scores)
MMR_LAMBDA = 0.7  # balance between relevance (1.0) and diversity (0.0)
MMR_FETCH_K = 10  # fetch more candidates, then pick diverse top-K


TUTOR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are Nova, a friendly and patient AI tutor. Here is who you are:

PERSONALITY:
- Warm and encouraging — you celebrate when students ask good questions
- Patient — you never make students feel dumb for not knowing something
- Curious — you show genuine enthusiasm for the subject matter
- Honest — you admit when you don't know rather than guessing
- Concise — you respect the student's time and avoid rambling

RULES:
1. ONLY use information from the provided context below. Do NOT use any outside knowledge.
2. If the context does not contain enough information to answer the question, say: "That's a great question, but I don't have enough information in your course materials to answer it. Try checking a different section or rephrasing your question!"
3. Never guess or make up information. If you're unsure, say so.
4. Do not reference the context directly (don't say "according to the context" or "the document says"). Teach as if you naturally know the material.
5. If the question is completely unrelated to the course material, politely redirect the student back to their studies.

TEACHING STYLE:
- Start with a simple one-sentence answer
- Then explain in more detail using plain language
- Use a real-world analogy or example to make abstract concepts click
- Break down formulas or technical terms step by step
- End with a quick key takeaway the student can remember

Context from course materials:
{context}

Student's question: {question}

Nova:"""
)

REFUSAL_MESSAGE = (
    "That's a great question, but it doesn't seem to be related to your course materials. "
    "I can only help with topics covered in your uploaded documents. "
    "Try asking something about the subjects in your study materials!"
)

RELEVANCE_CHECK_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""Determine if the following student question is genuinely about the same topic as the provided course material context.

The question might use words that appear in the context but mean something completely different (e.g., "bias in hiring" vs "bias in machine learning", "train my dog" vs "train a model").

Course material context:
{context}

Student question: {question}

Is the student's question genuinely asking about the topics covered in the course material? Answer ONLY "YES" or "NO"."""
)


def build_rag_chain(vectorstore: FAISS):
    """Build the RAG components (retriever + LLM). Returns a tuple."""
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)
    return vectorstore, llm


def ask(chain_tuple, question: str) -> dict:
    """
    Ask a question with layered abstention:
      Layer 1: Check retrieval similarity scores — refuse if too low
      Layer 2: Prompt enforcement — LLM constrained to context only
    """
    vectorstore, llm = chain_tuple

    # Layer 1: Retrieve with scores and check relevance
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=TOP_K)

    if not docs_with_scores:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "abstained": True}

    # FAISS returns L2 distance (lower = more similar)
    # Convert to similarity: similarity = 1 / (1 + distance)
    best_score = min(score for _, score in docs_with_scores)
    best_similarity = 1 / (1 + best_score)

    print(f"  Best similarity score: {best_similarity:.4f} (threshold: {SIMILARITY_THRESHOLD})")

    if best_similarity < SIMILARITY_THRESHOLD:
        print(f"  ABSTAINING — score below threshold")
        return {"answer": REFUSAL_MESSAGE, "sources": [], "abstained": True}

    # Layer 2: Relevance check — is the question actually about the course material?
    docs = [doc for doc, _ in docs_with_scores]
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    relevance_prompt = RELEVANCE_CHECK_PROMPT.format(context=context, question=question)
    relevance_check = llm.invoke(relevance_prompt).strip().upper()
    print(f"  Relevance check: {relevance_check}")

    if "NO" in relevance_check:
        print(f"  ABSTAINING — question not relevant to course material")
        return {"answer": REFUSAL_MESSAGE, "sources": [], "abstained": True}

    # Layer 3: Build prompt with retrieved context and let LLM answer
    prompt = TUTOR_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt)

    # Build source citations from MMR results
    # Match MMR docs back to scored results for relevance %
    score_lookup = {}
    for doc, score in docs_with_scores:
        key = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}_{doc.page_content[:50]}"
        score_lookup[key] = 1 / (1 + score)

    sources = []
    seen = set()
    for doc in docs:
        source_key = f"{doc.metadata.get('source', 'unknown')}_p{doc.metadata.get('page', '?')}"
        if source_key in seen:
            continue
        seen.add(source_key)

        # Try to find matching score
        lookup_key = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}_{doc.page_content[:50]}"
        similarity = score_lookup.get(lookup_key, None)
        relevance = f"{similarity:.0%}" if similarity else "MMR"

        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "type": doc.metadata.get("type", "text"),
            "relevance": relevance,
        })

    return {"answer": answer, "sources": sources, "abstained": False}