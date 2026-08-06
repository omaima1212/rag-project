"""
Days 6-7 - Streamlit chat interface.

Run with:
    streamlit run src\app.py

REMEMBER THE EXECUTION MODEL: Streamlit re-runs this entire file, top to
bottom, on every user interaction. There are no event handlers. A widget is a
function that returns its current value on this run.

Three consequences drive the code below:
  1. Ordinary variables do not survive a rerun -> st.session_state
  2. Expensive work would repeat every rerun   -> @st.cache_data
  3. Chat history must be re-DRAWN each rerun  -> replay from session_state,
     but only the newest answer is STREAMED; older ones are already text.
"""

import hashlib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import websearch
from config import (CHAT_MODEL, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL,
                    HEADER_INCLUDE_FILENAME, OLLAMA_URL, TOP_K,
                    USE_CONTEXT_HEADERS, WEB_SEARCH_BACKEND)
from embed import embed_text, get_collection
from generate import (WEB_SYSTEM_PROMPT, ask_model_stream, build_prompt,
                      build_web_prompt, looks_like_refusal)
from ingest import add_context_headers, load_and_chunk
from retrieve import retrieve

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"

st.set_page_config(page_title="Ask your PDF", page_icon="📄", layout="centered")

# Minimal CSS. Streamlit's theme (.streamlit/config.toml) handles colour;
# this only fixes spacing and styles the similarity chips, which have no
# native equivalent.
st.markdown("""
<style>
  .block-container { padding-top: 2.5rem; max-width: 820px; }
  .doc-bar {
      display: flex; align-items: center; gap: .6rem;
      padding: .6rem .9rem; border-radius: .6rem;
      background: rgba(124,158,245,.10);
      border: 1px solid rgba(124,158,245,.25);
      font-size: .88rem; margin-bottom: 1.2rem;
  }
  .doc-bar .name { font-weight: 600; }
  .doc-bar .meta { opacity: .65; }
  .chip {
      display: inline-block; padding: .12rem .5rem; border-radius: 1rem;
      font-size: .74rem; font-family: ui-monospace, monospace;
      background: rgba(124,158,245,.15); border: 1px solid rgba(124,158,245,.3);
      margin-right: .4rem;
  }
  .chip.weak { background: rgba(255,255,255,.06); border-color: rgba(255,255,255,.12); }
  .bar { height: 4px; border-radius: 2px; background: rgba(255,255,255,.08); margin: .35rem 0 .8rem; }
  .bar > div { height: 100%; border-radius: 2px; background: #7C9EF5; }
  /* Web fallback - deliberately a different colour (teal) from the indigo
     used everywhere else, so content that did NOT come from the document
     registers as visually distinct before the text is even read. */
  .web-block {
      border: 1px solid rgba(45,212,191,.3); background: rgba(45,212,191,.06);
      border-radius: .6rem; padding: .7rem .9rem; margin-top: .6rem;
  }
  .web-header { font-size: .8rem; font-weight: 600; color: #14b8a6; margin-bottom: .4rem; }
  .web-header .tag { font-weight: 400; opacity: .7; color: inherit; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached work
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def index_uploaded_pdf(file_bytes, file_name):
    """Extract, chunk, embed and store an uploaded PDF.

    Cached on the file's BYTES, and the collection is named by content hash.
    So: the same file indexes once however many reruns occur; re-uploading an
    identical file is instant; and two different files sharing a name get
    separate indexes rather than silently colliding.
    """
    UPLOAD_DIR.mkdir(exist_ok=True)
    path = UPLOAD_DIR / file_name
    path.write_bytes(file_bytes)

    digest = hashlib.sha256(file_bytes).hexdigest()[:12]
    collection_name = f"upload_{digest}"

    collection = get_collection(collection_name)
    if collection.count() > 0:
        return collection_name, collection.count(), True

    chunks = load_and_chunk(path, verbose=False)
    if not chunks:
        # A PDF with no extractable text - most commonly a scanned page with
        # no text layer. Left unguarded, Chroma's upsert() rejects an empty
        # ids list with a raw ValueError ("Expected IDs to be a non-empty
        # list, got 0 IDs") that would otherwise surface as an unhandled
        # Streamlit traceback instead of a readable message. Found by
        # reading Chroma's validation code, not by hitting it live.
        raise ValueError(
            f"No text could be extracted from '{file_name}'. This usually "
            "means it is a scanned document (an image with no text layer) - "
            "OCR is not supported. See README.md, Known limitations."
        )
    if USE_CONTEXT_HEADERS:
        chunks = add_context_headers(chunks, file_name,
                                     include_filename=HEADER_INCLUDE_FILENAME)

    collection.upsert(
        ids=[f"{digest}::{i}" for i in range(len(chunks))],
        embeddings=[embed_text(c) for c in chunks],
        documents=chunks,
        metadatas=[{"source": file_name, "chunk_index": i}
                   for i in range(len(chunks))],
    )
    return collection_name, len(chunks), False


@st.cache_data(show_spinner=False, ttl=30)
def ollama_is_running():
    import requests
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

def render_sources(hits):
    """Draw the retrieved chunks under an answer.

    The most important element on the page, and the justification is a
    measurement rather than a preference: Day 5 recorded a correct-looking
    answer whose rank-1 chunk did not contain the answer. A reader seeing only
    the answer would never have known. This turns an assertion into something
    verifiable.
    """
    with st.expander(f"Sources · {len(hits)} chunks retrieved"):
        st.caption(
            "Retrieved by meaning, not keywords. Compare similarity values "
            "against each other, not against a fixed threshold — measured "
            "correct answers on the test document scored 0.49–0.64, while an "
            "*irrelevant* chunk scored 0.43."
        )
        for rank, hit in enumerate(hits, start=1):
            sim = hit["similarity"]
            # Scale the bar across the range actually observed, not 0-1.
            # A 0-1 bar would show every real result as roughly half full and
            # convey nothing.
            width = max(4, min(100, (sim - 0.30) / 0.40 * 100))
            weak = "" if rank == 1 else " weak"
            st.markdown(
                f"<span class='chip{weak}'>#{rank}</span>"
                f"<span class='chip{weak}'>chunk {hit['chunk_index']}</span>"
                f"<span class='chip{weak}'>{sim:.4f}</span>"
                f"<div class='bar'><div style='width:{width:.0f}%'></div></div>",
                unsafe_allow_html=True,
            )
            st.text(hit["text"][:900] + ("…" if len(hit["text"]) > 900 else ""))
            if rank < len(hits):
                st.divider()


def render_web_block(web):
    """Draw the optional web-fallback result under an answer that refused.

    `web` is {"answer": str, "results": [...]} on success, or {"error": str}
    if the search step itself failed - the grounded refusal above it is
    still valid either way, so this never blocks or replaces it.
    """
    if "error" in web:
        st.markdown(
            f"<div class='web-block'><div class='web-header'>"
            f"🌐 Web fallback <span class='tag'>— unavailable</span></div>"
            f"{web['error']}</div>",
            unsafe_allow_html=True,
        )
        return

    sources_html = "".join(
        f"<div><b>[{i}]</b> <a href='{r['link']}' target='_blank'>{r['title']}</a></div>"
        for i, r in enumerate(web["results"], start=1)
    )
    st.markdown(
        f"<div class='web-block'><div class='web-header'>"
        f"🌐 Web fallback <span class='tag'>— not from your document</span></div>"
        f"{web['answer']}<div style='margin-top:.6rem;font-size:.82rem'>{sources_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Document")
    uploaded = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")

    st.subheader("Status")
    # st.success / st.error draw their own indicator. The `icon=` argument
    # accepts only real emoji or Material icon names - "✓" and "✕" are
    # typographic characters and are rejected.
    if ollama_is_running():
        st.success("Ollama running")
    else:
        st.error("Ollama unreachable")
        st.caption(f"Expected at {OLLAMA_URL}")

    st.subheader("Configuration")
    st.caption("The settings these answers were produced with. "
               "A result you cannot reproduce is not evidence.")
    st.code(
        f"embedding  {EMBED_MODEL}\n"
        f"generation {CHAT_MODEL}\n"
        f"chunk size {CHUNK_SIZE} words\n"
        f"overlap    {CHUNK_OVERLAP} words\n"
        f"top-k      {TOP_K}\n"
        f"headers    {'on' if USE_CONTEXT_HEADERS else 'off'}"
        + (f" ({'with' if HEADER_INCLUDE_FILENAME else 'no'} filename)"
           if USE_CONTEXT_HEADERS else "")
        + f"\nweb search {WEB_SEARCH_BACKEND}",
        language=None,
    )

    st.subheader("Web fallback")
    web_configured = websearch.is_configured()  # always True - see websearch.py
    st.session_state.web_fallback = st.toggle(
        "Search the web if not in doc",
        value=st.session_state.get("web_fallback", False),
        disabled=not web_configured,
    )

    if st.session_state.get("messages"):
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

if not ollama_is_running():
    st.title("Ask your PDF")
    st.warning("Ollama is not running. Start it, then reload this page.")
    st.stop()

if uploaded is None:
    st.title("Ask your PDF")
    st.caption("Answers come only from the document you upload. If the answer "
               "is not in the text, the system says so instead of inventing one.")
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

with st.spinner("Indexing the document…"):
    try:
        collection_name, chunk_count, was_cached = index_uploaded_pdf(
            uploaded.getvalue(), uploaded.name
        )
    except ValueError as e:
        st.title("Ask your PDF")
        st.error(str(e))
        st.stop()
    except RuntimeError as e:
        st.title("Ask your PDF")
        st.error(str(e))
        st.stop()

st.markdown(
    f"<div class='doc-bar'>📄 <span class='name'>{uploaded.name}</span>"
    f"<span class='meta'>· {chunk_count} chunks"
    f"{' · loaded from cache' if was_cached else ' · newly indexed'}</span></div>",
    unsafe_allow_html=True,
)

# Reset the conversation when a different document is opened - answers about
# a previous document would be misleading under a new document's header.
if st.session_state.get("collection") != collection_name:
    st.session_state.collection = collection_name
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.caption("Ask anything about this document. Try a factual detail, or "
               "something the document does not cover — it should decline.")

# Replay the existing conversation. These are already complete strings, so
# they are written directly; only a NEW answer is streamed.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("hits"):
            render_sources(message["hits"])
        if message.get("web"):
            render_web_block(message["web"])

question = st.chat_input("Ask about this document…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document…"):
                hits = retrieve(question, collection_name=collection_name)

            # st.write_stream consumes the generator, renders each fragment as
            # it arrives, and returns the complete text. The wait becomes
            # legible instead of looking like a freeze.
            answer = st.write_stream(
                ask_model_stream(build_prompt(question, hits))
            )
            render_sources(hits)

            # NOTE: the live path below deliberately does NOT call
            # render_web_block() - st.write_stream() already renders each
            # fragment as it arrives, so wrapping its OWN output in that
            # helper afterward would show the answer twice. render_web_block
            # exists for REPLAYING a past, already-complete message instead,
            # where there is no live stream to duplicate against.
            web = None
            if st.session_state.get("web_fallback") and looks_like_refusal(answer):
                try:
                    with st.spinner("Searching the web…"):
                        results = websearch.search_web(question)
                    if results:
                        st.markdown(
                            "<div class='web-header'>🌐 Web fallback "
                            "<span class='tag'>— not from your document</span></div>",
                            unsafe_allow_html=True,
                        )
                        web_answer = st.write_stream(
                            ask_model_stream(build_web_prompt(question, results),
                                            system_prompt=WEB_SYSTEM_PROMPT)
                        )
                        for i, r in enumerate(results, start=1):
                            st.caption(f"[{i}] [{r['title']}]({r['link']})")
                        web = {"answer": web_answer, "results": results}
                    else:
                        st.info("No web results found.")
                        web = {"error": "No web results found."}
                except RuntimeError as e:
                    # The grounded refusal above is still a valid, complete
                    # answer - a failed web step is additive information
                    # lost, not a failure of the turn.
                    st.warning(str(e))
                    web = {"error": str(e)}

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "hits": hits, "web": web}
            )
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")
