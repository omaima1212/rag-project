"""
Minimal Flask web UI - an alternative front door to the same pipeline used by
app.py (Streamlit) and generate.py (CLI). Added because Streamlit's rerun
model and its own process are unwelcome overhead on a machine that is
already tight on RAM (this project targets 8 GB, and Ollama alone reserves
~2.3 GB of it).

What is actually lighter here than app.py, precisely:
  - No Streamlit process/session-state machinery. Flask's dev server is a
    few MB; it does nothing until a request arrives.
  - No rerun-the-whole-script-on-every-interaction model. Each endpoint runs
    once per request and returns.
  - The page is one static HTML file with inline CSS/JS - no build step, no
    bundler, no framework runtime shipped to the browser.

What is NOT lighter, and cannot be made lighter by changing the UI layer:
  the two Ollama models. llama3.2:3b (~2 GB) and nomic-embed-text (~0.3 GB)
  cost the same RAM no matter what sits in front of them. If memory is the
  binding constraint, closing other applications still matters more than
  the choice of UI.

Run with:
    python src\\server.py
Then open http://localhost:5000
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, Response, jsonify, request, send_from_directory

import websearch
from config import CHAT_MODEL, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, \
    HEADER_INCLUDE_FILENAME, OLLAMA_URL, TOP_K, USE_CONTEXT_HEADERS, \
    WEB_SEARCH_BACKEND
from embed import embed_text, get_collection
from generate import (WEB_SYSTEM_PROMPT, ask_model_stream, build_prompt,
                      build_web_prompt, looks_like_refusal)
from ingest import add_context_headers, load_and_chunk
from retrieve import retrieve

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
WEB_DIR = Path(__file__).parent / "web"

app = Flask(__name__, static_folder=None)


# --------------------------------------------------------------------------
# Indexing - same content-hash-keyed design as app.py's index_uploaded_pdf,
# and for the same reason: the same file must index exactly once however
# many times it is uploaded, and two different files sharing a name must
# never collide. app.py used @st.cache_data for this; a plain Flask process
# has no rerun to cache against, so a dict keyed by digest does the same job
# for the life of the process. ChromaDB's own persistence is what actually
# avoids re-embedding across server RESTARTS - collection.count() > 0 is
# checked below before embedding, same guard app.py uses.
# --------------------------------------------------------------------------
_index_cache = {}


def index_pdf_bytes(file_bytes, file_name):
    digest = hashlib.sha256(file_bytes).hexdigest()[:12]
    if digest in _index_cache:
        return _index_cache[digest]

    UPLOAD_DIR.mkdir(exist_ok=True)
    path = UPLOAD_DIR / file_name
    path.write_bytes(file_bytes)

    collection_name = f"upload_{digest}"
    collection = get_collection(collection_name)

    if collection.count() > 0:
        result = (collection_name, collection.count(), True)
        _index_cache[digest] = result
        return result

    chunks = load_and_chunk(path, verbose=False)
    if not chunks:
        # A PDF with no extractable text - most commonly a scanned page
        # with no text layer. Caught here rather than left to surface as a
        # raw ValueError from Chroma's upsert(), which rejects an empty ids
        # list ("Expected IDs to be a non-empty list, got 0 IDs"). Found by
        # reading Chroma's validation code, not by hitting it live - the
        # same bug exists in app.py's index_uploaded_pdf, unguarded there
        # too. See BUILD_LOG.md.
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
    result = (collection_name, len(chunks), False)
    _index_cache[digest] = result
    return result


def ollama_is_running():
    import requests
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/")
def index():
    # max_age=0: found live (see BUILD_LOG.md, Day 10) - Flask's
    # send_from_directory sets caching headers by default, so a browser
    # reload can silently serve a STALE index.html without ever asking the
    # server again, even after the file on disk changed and the server was
    # restarted. Harmless for a page that rarely changes; actively
    # misleading here, mid-development, with this file changing every few
    # minutes. max_age=0 forces revalidation on every load - the browser
    # still avoids re-downloading unchanged bytes (a 304 covers that), it
    # just can no longer assume "unchanged" without checking first.
    return send_from_directory(WEB_DIR, "index.html", max_age=0)


@app.get("/api/health")
def health():
    return jsonify({
        "ollama": ollama_is_running(),
        "ollama_url": OLLAMA_URL,
        # Lets the frontend grey out / explain the web-fallback toggle
        # instead of offering it and then failing when it's clicked -
        # websearch.is_configured() is a cheap env-var check, no network call.
        "web_fallback_available": websearch.is_configured(),
        "config": {
            "embed_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "headers": USE_CONTEXT_HEADERS,
            "headers_include_filename": HEADER_INCLUDE_FILENAME,
            "web_search_backend": WEB_SEARCH_BACKEND,
        },
    })


@app.post("/api/upload")
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file field named 'pdf' in the request"}), 400

    file = request.files["pdf"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    file_bytes = file.read()
    try:
        collection_name, chunk_count, was_cached = index_pdf_bytes(
            file_bytes, file.filename
        )
    except ValueError as e:
        # No extractable text (scanned PDF) - a clear 4xx, not a 500.
        return jsonify({"error": str(e)}), 422
    except RuntimeError as e:
        # embed_text()'s own Ollama-down / bad-response messages.
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Unexpected error while indexing: {e}"}), 500

    return jsonify({
        "collection": collection_name,
        "chunks": chunk_count,
        "cached": was_cached,
        "filename": file.filename,
    })


@app.post("/api/ask")
def ask():
    """Streams newline-delimited JSON, mirroring the format Ollama itself
    uses for streaming (see generate.ask_model_stream) - one JSON object per
    line, so the client can parse incrementally without buffering the whole
    response:
        {"type": "hits", "hits": [...]}          - once, immediately
        {"type": "token", "text": "..."}         - repeated, as the grounded
                                                    answer arrives
        {"type": "web_start"}                    - only if web_fallback was
                                                    requested AND the grounded
                                                    answer looks like a refusal
        {"type": "web_results", "results": [...]}- once, before web tokens
        {"type": "web_token", "text": "..."}     - repeated, web-sourced answer
        {"type": "web_error", "message": "..."}  - web step failed; the
                                                    grounded answer above is
                                                    still valid, this is NOT
                                                    a fatal error for the turn
        {"type": "done"}                          - once, at the end
        {"type": "error", "message": "..."}      - in place of "done" on a
                                                    fatal failure (grounded
                                                    retrieval/generation itself)

    web_fallback is opt-in per request (see README.md, "Web fallback") and is
    checked here even though the frontend also gates it, because a request is
    a request - server-side code should never assume the client enforced
    anything.
    """
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    collection_name = body.get("collection")
    web_fallback = bool(body.get("web_fallback"))

    if not question:
        return jsonify({"error": "question must not be empty"}), 400
    if not collection_name:
        return jsonify({"error": "collection is required - upload a PDF first"}), 400

    def stream():
        try:
            hits = retrieve(question, collection_name=collection_name)
            yield json.dumps({"type": "hits", "hits": hits}) + "\n"

            prompt = build_prompt(question, hits)
            full_answer = ""
            for fragment in ask_model_stream(prompt):
                full_answer += fragment
                yield json.dumps({"type": "token", "text": fragment}) + "\n"
        except RuntimeError as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            return
        except Exception as e:
            yield json.dumps({"type": "error",
                             "message": f"Unexpected error: {e}"}) + "\n"
            return

        # The grounded answer above is complete and already shown to the
        # user regardless of what happens next - everything from here is
        # additive, and its own failure must not erase it.
        if web_fallback and looks_like_refusal(full_answer):
            yield json.dumps({"type": "web_start"}) + "\n"
            try:
                results = websearch.search_web(question)
                yield json.dumps({"type": "web_results", "results": results}) + "\n"
                if results:
                    web_prompt = build_web_prompt(question, results)
                    for fragment in ask_model_stream(web_prompt,
                                                     system_prompt=WEB_SYSTEM_PROMPT):
                        yield json.dumps({"type": "web_token", "text": fragment}) + "\n"
            except RuntimeError as e:
                yield json.dumps({"type": "web_error", "message": str(e)}) + "\n"
            except Exception as e:
                yield json.dumps({"type": "web_error",
                                 "message": f"Unexpected error: {e}"}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return Response(stream(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    print(f"Ollama reachable: {ollama_is_running()} ({OLLAMA_URL})")
    print("Open http://localhost:5000")
    # threaded=True: Flask's dev server is single-threaded by default, which
    # would block the health-check poll on the frontend while a generation
    # request is streaming. debug=False: the reloader spawns a SECOND
    # process that also imports this module - doubling nothing expensive at
    # import time, but needless on a memory-constrained machine.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
