"""
STEPS 4 and 5 of the RAG pipeline: given a question,
  4. RETRIEVE the most relevant chunks from the vector store
  5. GENERATE an answer by asking Claude to read those chunks and respond

"""

from langchain_anthropic import ChatAnthropic

from src.config import CLAUDE_MODEL, PROMPT_FILE, RETRIEVAL_K


def load_prompt_template():

    with open(PROMPT_FILE, encoding="utf-8") as f:
        return f.read()


def load_llm():
    """
    Connect to the Claude API. 
    """
    return ChatAnthropic(model=CLAUDE_MODEL, temperature=0)


def retrieve_chunks(vectorstore, question):
    """
    STEP 4 - RETRIEVE.
    Turn the question into a vector, then find the chunks in the vector
    store whose vectors are closest to it ( the most relevant text).
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


def extract_answer_text(response):
    """
    Get the plain answer text out of a ChatAnthropic response.

    Usually response.content is already a plain string. But some models
    (e.g. claude-sonnet-5, especially when it adds citation markers like
    "[2]") return content as a LIST of content blocks instead, e.g.
    [{"type": "text", "text": "..."}]. This handles both shapes so the
    rest of the code always gets back a plain string either way.
    """
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    return str(content)


def generate_answer(llm, question, chunks):
    """
    STEP 5 - GENERATE.
    Build the final prompt (instructions + retrieved context + question)
    and ask Claude to answer it.
    """
    context = format_chunks_as_context(chunks)
    prompt_template = load_prompt_template()
    prompt = prompt_template.format(context=context, question=question)
    return llm.invoke(prompt)


def answer_question(question, vectorstore, llm):
    
    chunks = retrieve_chunks(vectorstore, question)
    response = generate_answer(llm, question, chunks)

    sources = [
        {"subject": c.metadata.get("subject"), "document_id": c.metadata.get("document_id")}
        for c in chunks
    ]

    return {
        "question": question,
        "answer": extract_answer_text(response),
        "sources": sources,
        "usage": getattr(response, "usage_metadata", None),
    }
