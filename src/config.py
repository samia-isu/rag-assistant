"""Central place for paths and model settings used across the pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

DOCUMENTS_CSV = DATA_DIR / "documents.csv"
QA_FILE = DATA_DIR / "QA.txt"
INDEX_DIR = OUTPUT_DIR / "faiss_index"

# Embeddings run locally (free, no API key needed) since Claude has no
# embeddings endpoint. Small model, CPU is fine, no training involved.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Generation goes through the Claude API (this is the only paid/API-key step).
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
RETRIEVAL_K = 4
