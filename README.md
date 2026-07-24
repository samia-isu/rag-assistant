# RAG Learning Assistant

A simple, naive Retrieval-Augmented Generation (RAG) pipeline built for teaching
purposes (undergrad course project). No fine-tuning, no local LLM, no server to
run — just documents, a vector index, and a call to the Claude API.

## Pipeline

```
data/documents.csv                         data/QA.txt
      |                                    (sample questions for eval)
      v
[1] Ingest & group into passages   (src/ingest.py)
      v
[2] Chunk + embed (local, free)    (src/vectorstore.py)
      v
   FAISS index on disk             (output/faiss_index/)
      v
[3] Retrieve top-k chunks for a question
      v
[4] Generate answer with Claude    (src/rag_pipeline.py)
```

Only step 4 costs money (a Claude API call per question). Step 2's embeddings
run locally on CPU with a small sentence-transformers model — no GPU, no
training, no extra API key required.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set:
# ANTHROPIC_API_KEY=your_claude_api_key_here
```

## Usage

**1. Build the vector index** (run once, or whenever `data/documents.csv` changes):

```bash
python build_index.py
```

**2. Ask a question:**

```bash
python ask.py --question "What is a subset of the population called?"
```

**3. Run a batch of sample questions from `data/QA.txt`** and save results to
`output/qa_results.json`:

```bash
python run_eval.py --num 10
```

## The data

- `data/documents.csv` — 16 short reference documents (Statistics, Computer
  Science, Physics, Ecology), each made of a handful of sentences. This is the
  knowledge base that gets embedded and retrieved from.
- `data/QA.txt` — 100 sample question/answer pairs used only for a rough,
  eyeball-level sanity check (`run_eval.py`), not real grading.

## Example (retrieval step, no API key needed)

```
Q: What is a subset of the population called?
  (Statistics doc 1) Statistics involves collecting and analyzing data.
  The population is the entire collection of objects or individuals ...

Q: What do algorithms outline?
  (ComputerScience doc 2) allows computers to process large amounts of
  data and execute algorithms efficiently ...

Q: What is Newton's second law?
  (Physics doc 2) ... [no closely relevant passage — this dataset's
  Physics docs don't cover Newton's laws, a good example of the
  pipeline correctly having nothing to retrieve]
```

Once `ANTHROPIC_API_KEY` is set, `ask.py` turns each of these retrieved chunks
into a grounded answer instead of raw passages.

## Project structure

```
.
├── data/                 # documents.csv (knowledge base), QA.txt (sample eval questions)
├── src/
│   ├── config.py          # paths + model settings
│   ├── ingest.py           # load & group documents.csv / QA.txt
│   ├── vectorstore.py      # chunk, embed, build/load FAISS index
│   └── rag_pipeline.py     # retrieve + Claude generation
├── build_index.py         # one-time: build the FAISS index
├── ask.py                 # CLI: ask a single question
├── run_eval.py             # CLI: batch-run sample questions, save output/qa_results.json
├── output/                # generated: faiss_index/, qa_results.json (gitignored)
├── requirements.txt
├── .env.example

```

## Notes for cost estimation

- Embedding + vector search: **free**, runs locally, no API key.
- Generation: one Claude API call per question. Default model is
  `claude-haiku-4-5-20251001` (cheapest current Claude model) — check
  Anthropic's pricing page for current per-token rates, and multiply by
  expected questions/student to estimate class-wide cost.
- No fine-tuning or GPU is required anywhere in this pipeline.
