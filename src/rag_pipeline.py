"""
STEPS 3 and 4 of the RAG pipeline: given a question,
  3. RETRIEVE the most relevant chunks from the vector store
  4. GENERATE an answer by asking Claude to read those chunks and respond

This is "naive RAG" on purpose: no reranking, no summarization, no
fine-tuning - just retrieve, stuff into a prompt, and ask the LLM.
"""

from langchain_anthropic import ChatAnthropic

from src.config import CLAUDE_MODEL, RETRIEVAL_K

# The instructions we send to Claude along with the retrieved context.
PROMPT_TEMPLATE = """You are a helpful teaching assistant. Answer the question \
using ONLY the context below. If the context doesn't contain the answer, say \
you don't know.

Context:
{context}

Question: {question}

Answer:"""


def load_llm():
    """
    Connect to the Claude API. This is the ONLY step in the whole pipeline
    that costs money - every call to llm.invoke(...) is one API request.
    Requires ANTHROPIC_API_KEY to be set in your .env file.
    """
    return ChatAnthropic(model=CLAUDE_MODEL, temperature=0)


def retrieve_chunks(vectorstore, question):
    """
    STEP 3 - RETRIEVE.
    Turn the question into a vector, then find the chunks in the vector
    store whose vectors are closest to it (i.e. the most relevant text).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    return retriever.invoke(question)


def format_chunks_as_context(chunks):
    """
    Turn the retrieved chunks into one readable, labeled block of text
    that we can paste straight into the prompt.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        subject = chunk.metadata.get("subject", "unknown")
        doc_id = chunk.metadata.get("document_id", "?")
        lines.append(f"[{i}] ({subject} doc {doc_id}) {chunk.page_content}")
    return "\n\n".join(lines)


def generate_answer(llm, question, chunks):
    """
    STEP 4 - GENERATE.
    Build the final prompt (instructions + retrieved context + question)
    and ask Claude to answer it.
    """
    context = format_chunks_as_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return llm.invoke(prompt)


def answer_question(question, vectorstore, llm):
    """
    The full retrieve-then-generate flow for one question. This is the
    function ask.py and run_eval.py actually call.
    """
    chunks = retrieve_chunks(vectorstore, question)
    response = generate_answer(llm, question, chunks)

    sources = [
        {"subject": c.metadata.get("subject"), "document_id": c.metadata.get("document_id")}
        for c in chunks
    ]

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
        "usage": getattr(response, "usage_metadata", None),
    }
