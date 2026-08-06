"""
Single source of truth for every setting in the project.

Why this file exists: the embedding model MUST be identical when indexing
documents and when querying them. If they differ, similarity scores become
meaningless and the system fails SILENTLY - no crash, just wrong answers.
Defining the name once, here, makes that mistake impossible.
"""

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# PROJECT_ROOT resolves to C:\Users\souad\rag-project no matter where you run
# the script from. Never hardcode absolute paths in the other files.
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # put your test PDFs here
CHROMA_DIR = PROJECT_ROOT / "chroma_db"   # vector database persists here

# --- Ollama ----------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"   # used for BOTH indexing and querying
CHAT_MODEL = "llama3.2:3b"

# --- Chunking --------------------------------------------------------------
# Measured in words, not characters.
CHUNK_SIZE = 400      # words per chunk
CHUNK_OVERLAP = 50    # words repeated between consecutive chunks

# Prepend a context header (source, position, dates) to each chunk before
# embedding. Added on Day 5 to fix a measured failure: positional questions
# such as "what happened on the first day?" retrieved the wrong chunk first,
# because nothing in a chunk's text encodes where it sits in the document.
# Kept as a switch so the effect can be measured with it on and off.
USE_CONTEXT_HEADERS = True

# Whether that header includes the source filename, or only structural
# information (position + dates). Originally the header always included the
# filename. Measured cost (Day 5): the test document's filename contains
# "stage" (French for internship), so every chunk's header matched the word
# "internship" - inflating similarity on an UNANSWERABLE question about
# internship salary from 0.4291 to 0.4770. A follow-up A/B/C sweep
# (experiment.sweep_header_filename, see BUILD_LOG.md) measured the
# structure-only variant: it kept the positional fix (q4 rank-0, score
# slightly higher than with the filename), reduced the q8 inflation
# (0.4770 -> 0.4646, though not fully back to the 0.4291 floor), and matched
# or beat the filename variant on every topical question. Adopted as the
# default on that evidence.
HEADER_INCLUDE_FILENAME = False

# --- Retrieval -------------------------------------------------------------
TOP_K = 3             # how many chunks to retrieve per question

# --- Web fallback (optional, OFF by default) --------------------------------
# A deliberate exception to this project's core promise ("nothing leaves the
# machine"). When enabled BY THE USER, per-session, via a UI toggle, and only
# when the document genuinely does not contain the answer, the question is
# sent out to whichever backend is selected below and the answer is
# regenerated from web snippets instead - clearly labelled as NOT from the
# document. See README.md, "Web fallback", for exactly what this does and
# does not change.
#
# Two backends have now been tried and swapped between (see BUILD_LOG.md,
# Days 10-11), so the choice lives here as a setting rather than being
# hardcoded a third time:
#   "google"     - Google Custom Search API. Real, versioned results; needs
#                  GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID set as
#                  environment variables (see README.md for where to get
#                  both) - never hardcoded here, so a real key never ends up
#                  in a tracked file. Free for 100 searches/day.
#   "duckduckgo" - No signup, no key, works immediately - at the cost of
#                  scraping DuckDuckGo's HTML page instead of calling a
#                  sanctioned API, and (confirmed live, twice, not assumed)
#                  rate-limiting fast under normal repeated use.
# See websearch.py's own docstring for the full trade-off write-up.
#
# Set back to "duckduckgo" (Day 11, second time) after a real Google Cloud
# setup attempt hit dead end after dead end - an invalid key, a credential
# wizard that produced the wrong type, a "search the entire web" toggle
# that would not respond to clicks. None of that is a code problem this
# project can fix; DuckDuckGo trades that friction for a real but narrower
# one (rate-limiting under repeated use) that at least doesn't block getting
# started. "google" is left fully working above if a working credential
# ever gets sorted out - flip this one line back, nothing else changes.
#
# Flipped back to "google" (Day 11, third time), then back to "duckduckgo"
# (fourth time) in the same session once the actual root cause of the 403
# surfaced: Google now requires a billing account linked to a Cloud project
# before the Custom Search API will serve any request, free tier or not -
# confirmed by the "start your free trial" banner appearing throughout the
# console for this project. That requires a card on file. No card, no
# Google backend - a real, final stopping point, not a setup mistake to
# keep troubleshooting. "google" above is genuinely finished and correct
# code; it is waiting on a billing account that may never exist, not on
# anything more to fix here.
WEB_SEARCH_BACKEND = "duckduckgo"

GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")
WEB_FALLBACK_MAX_RESULTS = 3
