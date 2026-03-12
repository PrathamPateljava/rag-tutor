"""RAG chain: retrieval + Llama 3 generation."""

from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

OLLAMA_MODEL = "llama3"
TOP_K = 4

TUTOR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful tutor. Use the following context from the student's
course material to answer their question. If the context doesn't contain enough
information, say so honestly — don't make things up.

Explain concepts clearly and at an appropriate level. Use examples when helpful.

Context:
{context}

Student's question: {question}

Answer:"""
)


def build_rag_chain(vectorstore: FAISS) -> RetrievalQA:
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": TUTOR_PROMPT},
    )


def ask(chain: RetrievalQA, question: str) -> dict:
    result = chain.invoke({"query": question})
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "?"),
            "snippet": d.page_content[:120] + "...",
        }
        for d in result.get("source_documents", [])
    ]
    return {"answer": result["result"], "sources": sources}

