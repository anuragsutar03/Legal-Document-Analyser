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
    "Retrieval-Augmented Generation demo over a sample legal corpus. "
    "Educational project - not legal advice."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Number of chunks to retrieve", min_value=2, max_value=8, value=4,
        help="How many relevant excerpts the system pulls from the documents before answering.",
    )
    st.markdown("---")
    st.markdown(
        "**Corpus:** contract law, tenant rights, employment, cheque bounce, "
        "traffic law, cybercrime, and fictionalized case summaries.\n\n"
        "**To use your own documents:** drop `.txt` files into `data/documents/` "
        "(use `SECTION N: ...` or `CASE: ...` headers for best chunking) and re-run "
        "`python src/ingest.py`."
    )
    st.markdown("---")
    if os.environ.get("GEMINI_API_KEY"):
        st.success("Gemini API key detected - full generation enabled.")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        st.success("Anthropic API key detected - full generation enabled.")
    elif os.environ.get("OPENAI_API_KEY"):
        st.success("OpenAI API key detected - full generation enabled.")
    else:
        st.warning("No LLM API key set - running in extractive fallback mode.")

# --- Example questions (clickable, fill the input box) ---
EXAMPLES = [
    "My landlord won't return my security deposit — what can I do?",
    "Can my employer fire me without any notice?",
    "I gave someone a cheque and it bounced — what happens?",
    "A shop sold me a defective product and won't refund me.",
]

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    with col:
        if st.button(example, use_container_width=True):
            st.session_state.query_input = example

query = st.text_input(
    "Ask a legal question about the corpus:",
    key="query_input",
    placeholder="e.g. What happens if a supplier says in advance they won't deliver goods?",
)


def _relevance_label(distance: float) -> str:
    """Converts a raw cosine distance into a human-readable relevance label."""
    if distance < 0.8:
        return "🟢 Highly relevant"
    elif distance < 1.2:
        return "🟡 Somewhat relevant"
    else:
        return "🔴 Low relevance"


if st.button("Analyze", type="primary") and query.strip():
    with st.spinner("Retrieving relevant excerpts..."):
        chunks = retrieve(query, top_k=top_k)

    with st.spinner("Generating grounded answer..."):
        answer = generate_answer(query, chunks)

    st.subheader("Answer")
    st.write(answer)

    best_relevance = _relevance_label(chunks[0]["distance"]) if chunks else "🔴 No matches"
    if "Low" in best_relevance or "No matches" in best_relevance:
        st.warning(
            "⚠️ Low confidence — this question may not be well covered by the current corpus."
        )

    st.subheader("Retrieved sources")
    for c in chunks:
        label = _relevance_label(c["distance"])
        with st.expander(f"{label}  ·  {c['doc_title']} — {c['section_label']}"):
            st.write(c["text"])
elif query.strip() == "":
    st.info("Enter a question above (or click an example) and press Analyze.")
