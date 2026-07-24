"""
STEP 2 of the RAG pipeline: split documents into small chunks, turn each
chunk into a vector (a list of numbers that captures its meaning), and
save those vectors so we can search them later.

Nothing in this file calls a paid API - the embedding model runs on your
own computer (CPU is fine), and it is NOT trained/fine-tuned, only used.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME, INDEX_DIR


def get_embedding_model():
    """
    Load the model that turns text into vectors ("embeddings").
    This downloads a small model from Hugging Face the first time it runs,
    then reuses the cached copy after that.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def split_into_chunks(documents):
    """
    Break each document into smaller pieces (~400 characters, with some
    overlap between pieces so we don't cut a sentence awkwardly in half).

    We chunk because: (1) embedding models work better on short passages,
    and (2) we only want to feed the LLM the small relevant piece later,
    not the whole document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vectorstore(documents):
    """
    Full "build the index" step, used by build_index.py:
      1. split documents into chunks
      2. embed every chunk (turn it into a vector)
      3. store all the vectors in a FAISS index
      4. save that index to disk so we don't have to redo this next time
    """
    chunks = split_into_chunks(documents)

    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def load_vectorstore():
    """
    Load the FAISS index we already built and saved to disk, so we don't
    have to re-embed everything just to ask a question.
    """
    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
