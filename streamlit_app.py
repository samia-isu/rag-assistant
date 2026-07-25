"""
Streamlit UI for the RAG Learning Assistant - same look and features as
apps.py's Funix version, rebuilt for Streamlit so it can be deployed for
free on Streamlit Community Cloud (share.streamlit.io).

Run locally:
    streamlit run streamlit_app.py
"""

import os
import time

import streamlit as st
from langchain_core.documents import Document

from src.rag_pipeline import extract_answer_text, generate_answer, load_llm, retrieve_chunks
from src.ui_helpers import (
    HEADER_MAIN,
    answer_card,
    context_card,
    error_card,
    extract_text_from_file,
    loading_card,
)
from src.vectorstore import build_vectorstore_in_memory

st.set_page_config(page_title="RAG Learning Assistant", page_icon="📚")

# Same CSS + header banner used by the Funix version.
st.markdown(HEADER_MAIN, unsafe_allow_html=True)

claude_api_key = st.text_input("Claude API Key", type="password")

uploaded_files = st.file_uploader(
    "Reference Documents (upload .txt, .csv, or .pdf files)",
    type=["txt", "csv", "pdf"],
    accept_multiple_files=True,
)

question = st.text_area("Your Question", height=100)

show_context = st.checkbox("Show Retrieved Context", value=True)

run_clicked = st.button("Run")

st.markdown("## Answer Generation")
output_area = st.empty()

if run_clicked:
    if not uploaded_files:
        output_area.markdown(
            error_card("Please upload at least one reference document."),
            unsafe_allow_html=True,
        )
    elif not question.strip():
        output_area.markdown(error_card("Please enter a question."), unsafe_allow_html=True)
    elif not claude_api_key:
        output_area.markdown(
            error_card("Please enter your Claude API key."), unsafe_allow_html=True
        )
    else:
        os.environ["ANTHROPIC_API_KEY"] = claude_api_key

        try:
            output_area.markdown(
                loading_card("Building the vector index from your documents..."),
                unsafe_allow_html=True,
            )

            docs = []
            for i, uploaded_file in enumerate(uploaded_files, start=1):
                text = extract_text_from_file(uploaded_file.getvalue())
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text.strip(),
                            metadata={"source": f"uploaded_doc_{i}"},
                        )
                    )

            if not docs:
                output_area.markdown(
                    error_card("Could not read any text from the uploaded file(s)."),
                    unsafe_allow_html=True,
                )
            else:
                vectorstore = build_vectorstore_in_memory(docs)

                output_area.markdown(
                    loading_card(
                        f"Indexed {len(docs)} documents. Retrieving the most relevant chunks..."
                    ),
                    unsafe_allow_html=True,
                )

                chunks = retrieve_chunks(vectorstore, question)

                output_area.markdown(
                    loading_card("Retrieved relevant context. Now Claude is generating the answer..."),
                    unsafe_allow_html=True,
                )

                llm = load_llm()
                response = generate_answer(llm, question, chunks)
                answer = extract_answer_text(response)

                context_html = context_card(chunks) if show_context else ""

                # Typewriter-style display, same pacing as the Funix version.
                words = answer.split()
                generated = ""

                for i, word in enumerate(words):
                    generated += word + " "

                    if i % 2 == 0 or i == len(words) - 1:
                        output_area.markdown(
                            context_html + answer_card(generated, cursor=True),
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.035)

                output_area.markdown(
                    context_html + answer_card(generated, cursor=False),
                    unsafe_allow_html=True,
                )

        except Exception as e:
            output_area.markdown(error_card(str(e)), unsafe_allow_html=True)

        finally:
            os.environ["ANTHROPIC_API_KEY"] = ""
