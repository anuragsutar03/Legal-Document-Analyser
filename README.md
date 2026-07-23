# Legal Analysis Assistant (RAG)

A Retrieval-Augmented Generation application over legal documents, built as an
NLP course project. Retrieves relevant statute sections / case excerpts for a
user's question, then generates a grounded, cited answer.

## Architecture

```
data/documents/*.txt --> ingest.py (chunk + embed) --> ChromaDB (vector store)
                                                              |
user question --> retrieve.py (embed query, similarity search) --> top-k chunks
                                                              |
                                    generate.py (LLM + grounding prompt) --> cited answer
                                                              |
                                          app.py (Streamlit UI)
```

- **Chunking** (`ingest.py`): splits documents on `SECTION N:` / `CASE:` headers
  rather than fixed character windows, so each chunk is a coherent legal unit
  (a full section or a full case summary) instead of a mid-sentence cut.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (small, fast, free,
  downloads automatically on first run). Swap for a legal-domain model like
  `nlpaueb/legal-bert-base-uncased` if you want to compare domain-specific vs.
  general-purpose embeddings in your report — that comparison makes a good
  evaluation section.
- **Vector store**: ChromaDB, persisted locally to `chroma_db/` — no server or
  account needed.
- **Generation**: pluggable. If `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set
  as an environment variable, it calls that model with a prompt that forces
  citation of sources and forbids answering beyond the retrieved excerpts. If
  no key is set, it falls back to showing the raw retrieved excerpts directly
  (extractive mode) so the app is fully runnable and demoable for free.

## Setup

```bash
cd legal_rag_app
pip install -r requirements.txt
```

(First run downloads the embedding model, ~90MB, from Hugging Face — needs
internet access once.)

## Build the index

```bash
python src/ingest.py
```

This reads everything in `data/documents/`, chunks it, embeds it, and writes
the vector index to `chroma_db/`. Re-run this any time you add or change
documents.

## Run the app

```bash
streamlit run src/app.py
```

Opens a browser UI where you can ask questions and see the generated answer
plus the exact retrieved excerpts it was grounded in.

### Optional: enable full LLM generation

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

Without a key, the app still works — it just shows the retrieved passages
directly instead of a synthesized answer.

## Using your own legal corpus

Replace or add `.txt` files in `data/documents/`. For best chunking, format
documents with headers like:

```
TITLE: <document title>

SECTION 1: <heading>
<text>

SECTION 2: <heading>
<text>
```

or for case law:

```
CASE: <case name>
FACTS: ...
HOLDING: ...
REASONING: ...
```

Then re-run `python src/ingest.py`.

Good free sources for a real corpus: Indian Kanoon, government bare-act PDFs,
or the CUAD contract-clause dataset (good for a contracts-focused project).

## Suggestions for your report / evaluation section

Since this is a course project, your professor will likely want more than a
working demo — an evaluation section is usually what separates a strong
submission:

1. **Retrieval quality**: hand-write 10–15 questions with known correct source
   sections, then measure precision@k / recall@k for your retriever.
2. **RAG vs. no-RAG baseline**: run the same questions through the LLM with and
   without retrieved context, and show qualitatively (or with a tool like
   RAGAS) how grounding reduces hallucination.
3. **Chunking ablation**: compare section-aware chunking (this implementation)
   against naive fixed-size chunking, and show retrieval quality differences.
4. **Embedding model comparison**: general-purpose (MiniLM) vs. legal-domain
   (Legal-BERT) embeddings on your evaluation question set.

## Notes

- The sample corpus in `data/documents/` is original, simplified educational
  content (including fictionalized case summaries), not real case law —
  meant purely to demonstrate the pipeline. Swap in your real corpus.
- The generated answers are for educational/demo purposes only, not legal
  advice — worth keeping this disclaimer if you present this to your class.
