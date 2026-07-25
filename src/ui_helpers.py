"""
Shared UI building blocks used by BOTH web UIs (apps.py's Funix version and
streamlit_app.py's Streamlit version), so they stay visually identical and
in sync - styling, card HTML, and file-reading logic only need to be
written once.
"""

import csv
import html
import io

import fitz  # PyMuPDF, for reading PDF files

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

/* "question" is the only textarea left in the form (documents is now a
   file upload, claude_api_key is a password input) - so this only
   affects the question field. */
textarea {
    background: #fff9e6 !important;
    font-size: 17px !important;
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
        Enter your Claude API key, upload your reference documents (.txt, .csv,
        or .pdf files - one file per document), and ask a question. Claude
        answers using ONLY the documents you upload - the index is built
        automatically each time you click Run.
    </p>
</div>
"""


def error_card(message: str) -> str:
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


def loading_card(message: str) -> str:
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


def answer_card(answer_text: str, cursor: bool = False) -> str:
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


def context_card(chunks) -> str:
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


def extract_text_from_file(file_bytes: bytes) -> str:
    """
    Turn one uploaded file's raw bytes into plain text, regardless of
    whether it's a .txt, .csv, or .pdf file. File type is detected from
    the content itself (not the filename, since some upload widgets don't
    preserve it):
      - PDF files start with the 4 bytes b"%PDF" (a real property of the
        PDF format), so we check for that and extract text with PyMuPDF.
      - Otherwise we assume it's plain text (covers both .txt and .csv).
        If it looks like our subject/document/sentence CSV format (same
        one data_analysis.py produces), we pull out just the sentences;
        any other text file is used as-is.
    """
    if file_bytes[:4] == b"%PDF":
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)

    text = file_bytes.decode("utf-8", errors="ignore")

    first_line = text.split("\n", 1)[0]
    looks_like_our_csv = "\t" in first_line and "sentence" in first_line.lower()

    if looks_like_our_csv:
        rows = csv.DictReader(io.StringIO(text), delimiter="\t")
        sentences = [row["sentence"].strip() for row in rows if row.get("sentence", "").strip()]
        return " ".join(sentences)

    return text
