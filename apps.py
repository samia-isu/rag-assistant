"""
Funix UI for a simple RAG Learning Assistant - single page.

Run:
    funix apps.py
"""

import os
import time
import html

import funix
import IPython.display
import ipywidgets
from langchain_core.documents import Document

from src.rag_pipeline import generate_answer, load_llm, retrieve_chunks
from src.vectorstore import build_vectorstore_in_memory


# -------------------------------------------------------
# Styling helpers
# -------------------------------------------------------
STYLE_BLOCK = """
<style>
textarea, input {
    border: 2px solid #d6e4ff !important;
    border-radius: 14px !important;
    padding: 12px !important;
    font-size: 15px !important;
    background: #fbfdff !important;
    transition: all 0.25s ease-in-out !important;
}

textarea:hover, input:hover {
    border-color: #4f8cff !important;
    box-shadow: 0 0 0 4px rgba(79, 140, 255, 0.15) !important;
}

textarea:focus, input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.20) !important;
}

button {
    border-radius: 14px !important;
    font-weight: 700 !important;
}
</style>
"""

HEADER_MAIN = STYLE_BLOCK + """
<div style="
    background: linear-gradient(135deg, #eef5ff, #ffffff);
    border: 1px solid #dbeafe;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.07);
">
    <h2 style="margin:0; color:#1e3a8a;">RAG Learning Assistant</h2>
    <p style="margin-top:8px; color:#4b5563; font-size:15px;">
        Enter your Claude API key, paste your reference documents (one per line),
        and ask a question. Claude answers using ONLY the documents you provide -
        the index is built automatically each time you click Run.
    </p>
</div>
"""


def __error_card(message: str) -> str:
    return f"""
<div style="
    background:#fff1f2;
    border-left:7px solid #ef4444;
    border-radius:18px;
    padding:22px;
    min-height:120px;
    box-shadow:0 8px 22px rgba(0,0,0,0.08);
    font-size:16px;
    line-height:1.7;
">
    <h3 style="margin-top:0;color:#991b1b;">Error</h3>
    <div style="white-space:pre-wrap;color:#1f2937;">{html.escape(message)}</div>
</div>
"""


def __loading_card(message: str) -> str:
    return f"""
<div style="
    background:#eff6ff;
    border-left:7px solid #3b82f6;
    border-radius:18px;
    padding:24px;
    min-height:180px;
    box-shadow:0 8px 24px rgba(0,0,0,0.09);
    font-size:17px;
    line-height:1.8;
">
    <h3 style="margin-top:0;color:#1d4ed8;">AI is working...</h3>
    <div style="white-space:pre-wrap;color:#1f2937;">{html.escape(message)}</div>
</div>
"""


def __answer_card(answer_text: str, cursor: bool = False) -> str:
    cursor_symbol = "▌" if cursor else ""

    return f"""
<div style="
    background:#ffffff;
    border-left:8px solid #2563eb;
    border-radius:20px;
    padding:28px;
    min-height:300px;
    box-shadow:0 10px 30px rgba(0,0,0,0.10);
    font-size:18px;
    line-height:1.85;
">
    <h2 style="
        margin-top:0;
        margin-bottom:18px;
        color:#1e3a8a;
        font-size:24px;
    ">
        Claude Generated Answer
    </h2>

    <div style="
        background:#f8fafc;
        border:1px solid #dbeafe;
        border-radius:16px;
        padding:20px;
        min-height:210px;
        white-space:pre-wrap;
        color:#111827;
    ">{html.escape(answer_text)}<span style="color:#2563eb;font-weight:900;">{cursor_symbol}</span></div>
</div>
"""


def __context_card(chunks) -> str:
    context_text = ""

    for i, chunk in enumerate(chunks, start=1):
        context_text += f"[{i}] {chunk.page_content}\n\n"

    return f"""
<div style="
    background:#fffbeb;
    border-left:7px solid #f59e0b;
    border-radius:18px;
    padding:22px;
    margin-bottom:18px;
    box-shadow:0 8px 22px rgba(0,0,0,0.07);
    font-size:15px;
    line-height:1.7;
">
    <h3 style="margin-top:0;color:#92400e;">Retrieved Context</h3>
    <div style="white-space:pre-wrap;color:#1f2937;">{html.escape(context_text)}</div>
</div>
"""


# -------------------------------------------------------
# Single page: build the index and ask a question in one Run
# -------------------------------------------------------
@funix.funix(
    title="RAG Learning Assistant",
    description="Enter your Claude API key, your reference documents, and a question.",
    widgets={"documents": "textarea", "question": "textarea"},
    argument_labels={
        "claude_api_key": "Claude API Key",
        "documents": "Reference Documents (one per line)",
        "question": "Your Question",
        "show_context": "Show Retrieved Context",
    },
    input_layout=[
        [{"markdown": HEADER_MAIN, "width": 12}],
        [{"argument": "claude_api_key"}],
        [{"argument": "documents"}],
        [{"argument": "question"}],
        [{"argument": "show_context"}],
    ],
    # Heading shown above the results once you click Run.
    output_layout=[
        [{"markdown": "## Answer Generation", "width": 12}],
        [{"return_index": 0}],
    ],
)
def ask_rag(
    claude_api_key: ipywidgets.Password,
    documents: str,
    question: str,
    show_context: bool = True,
) -> IPython.display.HTML:
    if not documents.strip():
        yield __error_card("Please enter at least one reference document.")
        return

    if not question.strip():
        yield __error_card("Please enter a question.")
        return

    if not claude_api_key.value:
        yield __error_card("Please enter your Claude API key.")
        return

    os.environ["ANTHROPIC_API_KEY"] = claude_api_key.value

    try:
        yield __loading_card("Building the vector index from your documents...")

        lines = documents.split("\n")
        docs = [
            Document(page_content=line.strip(), metadata={"source": f"manual_doc_{i}"})
            for i, line in enumerate(lines, start=1)
            if line.strip()
        ]
        vectorstore = build_vectorstore_in_memory(docs)

        yield __loading_card(
            f"Indexed {len(docs)} documents. Retrieving the most relevant chunks..."
        )

        chunks = retrieve_chunks(vectorstore, question)

        yield __loading_card(
            "Retrieved relevant context. Now Claude is generating the answer..."
        )

        llm = load_llm()
        response = generate_answer(llm, question, chunks)

        if isinstance(response.content, str):
            answer = response.content
        else:
            answer = str(response.content)

        # Optional context display before final answer
        if show_context:
            yield __context_card(chunks) + __loading_card("Starting answer generation...")

        # Typewriter-style display
        words = answer.split()
        generated = ""

        for i, word in enumerate(words):
            generated += word + " "

            # Update every 2 words to make it smooth but not too slow
            if i % 2 == 0 or i == len(words) - 1:
                yield __answer_card(generated, cursor=True)
                time.sleep(0.035)

        yield __answer_card(generated, cursor=False)

    except Exception as e:
        yield __error_card(str(e))

    finally:
        os.environ["ANTHROPIC_API_KEY"] = ""
