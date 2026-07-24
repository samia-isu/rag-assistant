"""
STEPS 3 and 4 of the RAG pipeline: given a question,
  3. RETRIEVE the most relevant chunks from the vector store
  4. GENERATE an answer by asking Claude to read those chunks and respond

This is "naive RAG" on purpose: no reranking, no summarization, no
fine-tuning - just retrieve, stuff into a prompt, and ask the LLM.
"""

from langchain_anthropic import ChatAnthropic

from src.config import CLAUDE_MODEL, PROMPT_FILE, RETRIEVAL_K


def load_prompt_template():
    """
    Read the prompt instructions from prompt.txt. Keeping the prompt in
    its own plain text file (instead of hardcoded in Python) means you
    can tweak the wording/instructions without touching any code.
    """
    with open(PROMPT_FILE, encoding="utf-8") as f:
        return f.read()


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

    Example output for 2 chunks:
        [1] (Statistics doc 1) A population is the entire collection...

        [2] (ComputerScience doc 2) Algorithms are instructions...
    """
    context_text = ""
    chunk_number = 1

    for chunk in chunks:
        subject = chunk.metadata.get("subject", "unknown")
        doc_id = chunk.metadata.get("document_id", "?")

        # Label so we (and Claude) can always tell which document a piece
        # of text came from.
        label = f"[{chunk_number}] ({subject} doc {doc_id})"
        context_text = context_text + label + " " + chunk.page_content + "\n\n"

        chunk_number = chunk_number + 1

    return context_text.strip()


def generate_answer(llm, question, chunks):
    """
    STEP 4 - GENERATE.
    Build the final prompt (instructions + retrieved context + question)
    and ask Claude to answer it.
    """
    context = format_chunks_as_context(chunks)
    prompt_template = load_prompt_template()
    prompt = prompt_template.format(context=context, question=question)
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
