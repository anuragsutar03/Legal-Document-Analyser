"""
app.py
Streamlit front-end for the Legal RAG Analysis Assistant.

Run:
    streamlit run src/app.py
"""

import os
import streamlit as st

st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️", layout="wide")

# --- Auto-build the vector index on first run (needed for fresh cloud deploys,
# where chroma_db/ doesn't exist yet because it's excluded from git) ---
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def _index_exists():
    # ChromaDB writes a sqlite file once a collection has been created
    return os.path.isdir(DB_DIR) and any(
        f.endswith(".sqlite3") for f in os.listdir(DB_DIR)
    ) if os.path.isdir(DB_DIR) else False


if not _index_exists():
    with st.spinner("First-time setup: building the document index... this can take a minute."):
        import ingest
        ingest.build_index()

from retrieve import retrieve
from generate import generate_answer

st.title("⚖️ Legal Analysis Assistant (RAG)")
st.caption(
    "Retrieval-Augmented Generation demo over a small sample legal corpus. "
    "Educational project - not legal advice."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=2, max_value=8, value=4)
    st.markdown("---")
    st.markdown(
        "**Corpus:** sample contract-law notes and fictionalized case summaries.\n\n"
        "**To use your own documents:** drop `.txt` files into `data/documents/` "
        "(use `SECTION N: ...` or `CASE: ...` headers for best chunking) and re-run "
        "`python src/ingest.py`."
    )
    st.markdown("---")
    if os.environ.get("ANTHROPIC_API_KEY"):
        st.success("Anthropic API key detected - full generation enabled.")
    elif os.environ.get("OPENAI_API_KEY"):
        st.success("OpenAI API key detected - full generation enabled.")
    else:
        st.warning("No LLM API key set - running in extractive fallback mode.")

query = st.text_input(
    "Ask a legal question about the corpus:",
    placeholder="e.g. What happens if a supplier says in advance they won't deliver goods?",
)

if st.button("Analyze", type="primary") and query.strip():
    with st.spinner("Retrieving relevant excerpts..."):
        chunks = retrieve(query, top_k=top_k)

    with st.spinner("Generating grounded answer..."):
        answer = generate_answer(query, chunks)

    st.subheader("Answer")
    st.write(answer)
 
    st.subheader("Retrieved sources")
    for c in chunks:
        with st.expander(f"{c['doc_title']} — {c['section_label']}  (relevance distance: {c['distance']:.3f})"):
            st.write(c["text"])
elif query.strip() == "":
    st.info("Enter a question above and click Analyze.")
 