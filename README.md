# Local RAG — Ask Questions About Your PDFs

A retrieval-augmented generation system that answers questions about your own
documents. Everything runs locally by default: no API keys, no cloud, no data
leaving the machine — with one deliberate, opt-in exception: an optional web
search fallback for questions the document can't answer, off unless you turn
it on. See [Web fallback](#web-fallback-optional) below.

Built and tuned on a laptop with **8 GB of RAM and no dedicated GPU**.

---

## What it does

Upload a PDF, ask a question in natural language, get an answer drawn **only**
from that document — together with the passages the answer came from, so you
can check it.

If the document does not contain the answer, the system says so instead of
inventing one.

---

## Why not just paste the document into a chatbot?

Three reasons, and the third is the one people underestimate:

1. **Context limits.** A 3B model on 8 GB of RAM cannot hold a long document.
2. **Cost and privacy.** Nothing is sent anywhere; the documents stay on disk.
3. **Accuracy.** Even when a long document *fits*, models attend poorly to one
   relevant sentence buried in fifty pages. Retrieving 3 relevant passages
   produces better answers than supplying 200 pages of mostly-irrelevant text.

RAG reframes the problem from *storage* to **selection**.

---

## How it works

```
INDEXING (once per document)
  PDF ──► text ──► cleaned ──► chunks ──► vectors ──► ChromaDB
       PyMuPDF          400 words     nomic-embed-text
                        50 overlap     768 dimensions

QUERYING (every question)
  question ──► vector ──► top-3 nearest chunks ──► prompt ──► answer
            same model        cosine similarity     llama3.2:3b
```

The essential constraint: **the same embedding model must be used for indexing
and querying.** Different models produce vectors in unrelated coordinate
spaces, and mixing them fails *silently* — no crash, just wrong answers. This
is why every setting lives in one file, `src/config.py`.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download)
- ~5 GB free RAM while running

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

**Web interface — two options, same pipeline underneath:**

```powershell
streamlit run src\app.py     # Streamlit chat UI
python src\server.py         # Flask + plain HTML/JS, then open http://localhost:5000
```

The Flask option was added because Streamlit's rerun-the-whole-script model
and its own process are avoidable overhead on a machine that is already
tight on RAM. It is not faster to *generate* an answer — the two Ollama
models cost the same ~2.3 GB either way — but it has no framework process of
its own and no browser-tab-plus-session-state machinery sitting between you
and the pipeline. Pick whichever interface you'd rather look at; both call
the same `retrieve` / `build_prompt` / `generate` functions.

**Command line:**

```powershell
python src\embed.py "data\yourfile.pdf"          # index a document
python src\generate.py "your question"            # ask a question
python src\retrieve.py "your question"            # inspect retrieval only
python src\evaluate.py "run label"                # run the test suite
python src\experiment.py "data\yourfile.pdf"      # sweep configurations
```

---

## Project layout

| Path | Role |
|---|---|
| `src/config.py` | Every setting, in one place |
| `src/ingest.py` | PDF → text → cleaned → chunks (+ context headers) |
| `src/embed.py` | Chunks → vectors → ChromaDB |
| `src/retrieve.py` | Question → most similar chunks |
| `src/generate.py` | Chunks + question → grounded answer |
| `src/app.py` | Streamlit chat interface |
| `src/server.py` + `src/web/index.html` | Flask + plain HTML/JS interface — lighter alternative to Streamlit |
| `src/websearch.py` | Optional web fallback (Google or DuckDuckGo, see below) — the only network calls in the project that aren't to local Ollama |
| `src/evaluate.py` | Test suite over `questions.json`, writes reports |
| `src/experiment.py` | Configuration sweeps: chunking, k, header content — retrieval-only |
| `tools/make_test_pdf.py`, `make_messy_pdf.py`, `make_blank_pdf.py` | Generate synthetic test PDFs (clean, multi-column + table, textless) |
| `SPECIFICATION.md` | What is being built and why |
| `CONCEPTS.md` | How each mechanism works |
| `BUILD_LOG.md` | Full development record: every step, problem and fix |

---

## Configuration

All in `src/config.py`:

| Setting | Value | Chosen because |
|---|---|---|
| `CHUNK_SIZE` | 400 words | Specific enough to discriminate, large enough to stand alone |
| `CHUNK_OVERLAP` | 50 words | Measured: beat no-overlap on 5/5 test questions |
| `TOP_K` | 3 | k=1 failed a positional question; k=5 adds chunks at the noise floor |
| `USE_CONTEXT_HEADERS` | `True` | Fixes positional retrieval — see below |
| `HEADER_INCLUDE_FILENAME` | `False` | Filename in the header leaked into every chunk's vector, inflating scores on questions sharing vocabulary with the filename. A/B/C sweep across all 8 test questions showed structure-only headers keep the positional fix and reduce that inflation — see below |

**Changing any embedding-affecting setting requires re-indexing.** The config
describes intent; the database holds state, and they drift apart silently.

---

## Engineering notes

Development was measurement-driven rather than intuition-driven. `evaluate.py`
runs a fixed 8-question suite and writes timestamped reports to `results/`, so
any change can be compared against a recorded baseline.

### The main finding: a failure hiding behind a passing test

The test suite reported 8/8. Reading the report showed that one question —
*"What happened on the first day of the internship?"* — retrieved the correct
chunk at **rank 2**, not rank 1. The top-ranked chunk did not contain the
answer. The system was returning a correct answer only because `k=3` was wide
enough to sweep the right chunk in as a passenger.

Confirmed by a retrieval-only sweep:

```
k=1: chunks [1]        <- wrong chunk, answer unavailable
k=3: chunks [1, 0, 2]  <- correct chunk present at rank 2
```

**Cause:** embeddings encode meaning, not sequence. Nothing in a chunk's text
states "this is the start of the document," so the question's key term —
*first day* — had nothing to match against. Semantic search is structurally
weak on positional queries.

**Fix:** prepend a context header to each chunk before embedding, so the
missing information reaches the vector:

```
[Source: report.pdf | Part 1 of 8 | This is the BEGINNING of the document:
the first day, the start, the earliest entries | Dates covered: 8 juin 2026]
```

**Result**, against a success criterion set before measuring:

| | Before | After |
|---|---|---|
| Rank-1 chunk | 1 (wrong) | **0 (correct)** |
| Similarity | 0.4923 | **0.5613** |
| Answer content | missing the company name | includes it |

The fix held across all four chunk-size configurations tested.

**Measured cost:** topical questions lost ~0.03 similarity, because the header
adds boilerplate to every chunk and dilutes it. Accepted — a wrong rank-1 is a
failure, while a smaller margin on a question that already ranks correctly is
not.

**Unintended benefit:** one chunk had been acting as a *hub*, ranking second on
five of six questions while answering none — generic topic vocabulary with no
distinguishing content. Headers gave every chunk distinct coordinates, and the
hub dropped out of two result sets entirely.

### Other measured results

- **Overlap works, 5/5.** Comparing 400/50 against 400/0 — identical except
  for overlap — overlap won every question. The largest gain landed on
  end-of-document content, exactly where theory predicts a chunk is most
  likely to be awkwardly framed.
- **Similarity scores cannot be thresholded.** A *correct* top hit scored
  0.4923 while an *irrelevant* one scored 0.4291. Absolute cosine values are
  not calibrated across documents or models; only relative ordering within a
  single query is meaningful.
- **Large chunks stop being retrieval.** At 600-word chunks this document
  produced only 5 chunks, so `k=3` returned 60% of it — document stuffing with
  extra steps. `k` must scale with corpus size.

### Follow-up findings

A later pass closed out several items that had been sitting in "known
limitations" as assumptions rather than measurements.

**Multi-column and table extraction, actually tested.** A synthetic PDF with
a two-column page and a data table (`tools/make_messy_pdf.py`) went through
the full pipeline. Two different results, not one: the two-column prose
extracted in correct reading order — PyMuPDF's block sort handled this
layout fine, which corrects the standing assumption that multi-column always
interleaves. The table did not survive: cell boundaries collapsed into a
single run-on line (`Falcon-9X 214 18,400 47 24 Falcon-7 Lite 96 11,250 34
18 …`), and end-to-end this produced one confirmed wrong refusal — a fact
present in the text (a warranty period) that the model could no longer
locate once the column structure was gone — alongside two other
table-derived facts it answered correctly by counting position. Table
flattening is a real, reproducible failure; multi-column is not always one.

**Header filename leak, measured and reduced.** An A/B/C sweep
(`experiment.sweep_header_filename`) compared no headers, headers with the
source filename, and structure-only headers (position and dates, no
filename) across all 8 test questions. Structure-only headers kept the Day-5
positional fix — the target question's correct chunk stayed at rank 1, with
a *higher* score than the filename variant — and cut the filename-driven
score inflation on the unanswerable salary question roughly in half (+0.048
→ +0.026 over the no-header baseline). It did not reach zero: something
about the chunk's own content, independent of the filename, still gets a
modest boost from having any header at all. Adopted as the new default
(`HEADER_INCLUDE_FILENAME = False`) on the strength of "keeps the fix, costs
less" rather than "costs nothing."

**The test suite can now fail.** It had reported 8/8 in every configuration
ever run through it. Not because nothing was wrong, but because
`expect_any` term lists mixed strong signals (proper nouns, exact numbers)
with generic words a wrong answer could plausibly also contain — "team",
"environment", "job". Tightened to the distinguishing term for the two
affected questions, and added hedge-phrase detection so a keyword match
riding on hedging language ("it is likely…") downgrades to `CHECK` instead
of passing silently.

**A crash found by reading library source, not by triggering it.** A PDF
with no extractable text (a scanned page) yields zero chunks, and Chroma's
`upsert()` rejects an empty `ids` list outright. Neither the Streamlit nor
the Flask UI caught this — it would have surfaced as a raw exception instead
of an explanation. Fixed in both, verified with a purpose-built textless PDF
(`tools/make_blank_pdf.py`) through Flask's in-process test client, which
exercises the failure without starting a server or calling either model.

**A prompt-leakage bug, fixed.** Context blocks were labelled `[Source N -
chunk N]`, and the model occasionally echoed that internal label back
verbatim ("According to chunk 4 of the document…") — meaningless to a reader
who never sees chunk numbers. Relabelled `[Passage N]`, and the system
prompt now explicitly forbids citing internal labels.

---

## Web fallback (optional)

Off by default. A toggle in both UIs' sidebar — "Search the web if not in
doc" — that only does anything when **both** of these are true:

1. You turned it on for this session.
2. The document has already been asked, and the grounded answer looked like
   a refusal (the same `looks_like_refusal()` check `evaluate.py` uses to
   grade the test suite — one definition, used to decide behaviour in two
   places now, not two definitions that could drift apart).

When it fires, the question goes out to whichever backend `config.py` names
(`WEB_SEARCH_BACKEND`), the top few results are shown as sources, and a
*second*, clearly separate answer is generated from those snippets —
labelled "Web fallback — not from your document" in both UIs, teal instead
of the usual accent colour, so it can never be mistaken for a grounded
answer at a glance. If the search step itself fails, the grounded refusal
above it is untouched — a failed web step is a lost extra, not a failed
turn.

**This is the one deliberate exception to "nothing leaves the machine."**
Turning it on means your question is sent out when the document doesn't
answer it. It does not change what happens for questions the document
*does* answer — those never touch the network, on or off — and it does not
change the default: everything above this section describes the system
with the toggle off, which is how it behaves unless you act.

### Two backends, chosen in `config.py`, not one hardcoded choice

This has genuinely been swapped four times in one session (see
`BUILD_LOG.md`, Days 10–11) — Google first, DuckDuckGo after an
invalid-key wall, back to Google once a real key finally worked, back to
DuckDuckGo once Google's actual final blocker turned up: **it requires a
billing account — a card on file — linked to the Cloud project before the
Custom Search API will serve any request, free tier or not.** No card, no
Google backend, full stop; that's not a setup mistake to keep
troubleshooting, it's a real prerequisite worth knowing *before* spending
twenty minutes in the Cloud console rather than after. Both
implementations stay in `src/websearch.py`, and `config.py`'s
`WEB_SEARCH_BACKEND` picks one, so this doesn't need rewriting a fifth time
if the trade-off ever matters again:

| | `"google"` | `"duckduckgo"` (current default) |
|---|---|---|
| Setup | API key + Search Engine ID + **a billing account (card required)** | None |
| Cost | Free for 100 searches/day *if billing is linked*, then paid | Free |
| Contract | Official, versioned API | Scraped HTML — can break without notice |
| Reliability, measured | Untested past setup — never got a working credential past the billing wall | Rate-limits under repeated use, and the block can last **well over 50 minutes**, not "a minute or two" as first assumed (measured live) |

**Setup for `"google"`**, if you have a card and want to try it:

1. [console.cloud.google.com](https://console.cloud.google.com) → **check
   for a "start your free trial" banner on the project first.** If it's
   there, billing isn't linked yet, and every API call will 403 with
   `"This project does not have the access to Custom Search JSON API"`
   regardless of how correct the key and API setup are — this is what
   actually blocked setup in this session, discovered only after the key
   and API-enablement were already both individually correct. Fix this
   before anything else, via that banner (requires a card) or
   console.cloud.google.com/billing.
2. Enable the "Custom Search API" for the project.
3. Create an API key from **Credentials → "+ Create Credentials" → "API
   key"** directly — not the guided "Help me choose" wizard, which produces
   an OAuth client ID or a service account instead of a plain key. Those
   look similar in the console but are not interchangeable, and this is a
   second real way this setup goes wrong silently.
4. [programmablesearchengine.google.com](https://programmablesearchengine.google.com)
   → create a search engine (adding one placeholder site is required to
   finish creation) → then, in the created engine's **"Display options in
   search results" → "Increasing results"**, toggle **"Search the entire
   web"** on — without this it only ever searches the placeholder site.
   Copy the **Search Engine ID** from the engine's Overview page.
5. Set both as environment variables before starting either UI — not in
   `config.py`, so a real key never sits in a tracked file:

   ```powershell
   $env:GOOGLE_SEARCH_API_KEY = "..."
   $env:GOOGLE_SEARCH_ENGINE_ID = "..."
   ```

6. Restart whichever UI you're running — environment variables are only
   read at process start.

Without both set, the toggle shows disabled in the sidebar with an
explanation, rather than being offered and then failing when clicked.
`src/websearch.py` can be run directly to sanity-check whichever backend is
active, without going through either UI: `python src\websearch.py "your query"`.

**If you see `"API key not valid. Please pass a valid API key."`** — this is
Google rejecting the credential itself, not a quota or permissions problem
(those return different, more specific messages). Check, in order: (1) the
string is from the **API Keys** section of the Credentials page, not an
OAuth client ID; (2) "Custom Search API" is actually **enabled** on the same
Google Cloud project the key belongs to — having a key doesn't enable an
API, that's a separate step; (3) no stray space or quote character got
included when the environment variable was set; (4) the key and the Search
Engine ID weren't swapped into each other's slot.

**Switching to `"duckduckgo"`** needs no setup at all — set
`WEB_SEARCH_BACKEND = "duckduckgo"` in `config.py` and restart. Worth
knowing before relying on it: a second search from the same machine, about
a minute after the first, measured HTTP 202 — DuckDuckGo's anomaly/bot
detection, not a normal result page. One question worked; the next one
shortly after didn't. `search_web()` recognises that specific response and
says so plainly rather than a generic error, but it means this option
trades a credential that can be misconfigured for a scrape that throttles
under exactly the kind of repeated use a real question-answering session
involves.

**What this does *not* change:** `questions.json` and `evaluate.py` are
untouched by this feature — the fallback is opt-in and the evaluation suite
never turns it on, so q7 and q8 (the two questions specifically testing that
the system refuses rather than answering from outside knowledge) still test
exactly what they always did.

---

## Known limitations

- **Scanned PDFs are not supported.** No text layer; would require OCR. This
  now fails with a clear message instead of a crash (see above) — it is
  still not *handled*, only *reported*.
- **Tables extract poorly, confirmed.** Column structure is lost on
  extraction; a small model can sometimes recover a fact by counting
  position in the flattened row, but not reliably — see "Follow-up
  findings" above for a concrete failure. Multi-column *prose*, by contrast,
  extracted correctly in the one layout tested — that specific concern was
  weaker than assumed.
- **Positional and exact-identifier queries remain weaker than topical ones.**
  Context headers reduce this but do not remove it; hybrid keyword search
  would address it properly.
- **Header text still leaks into the semantic space, reduced but not
  eliminated.** Dropping the filename from the header (see above) roughly
  halved the score inflation on a topically-related unanswerable question;
  it did not remove it.
- **Validated on small documents.** The core test document is 11 pages, 8
  chunks; the messy-PDF test added a 2-page synthetic document with 1 chunk.
  Retrieval has not been tested at a scale where index search itself is
  under pressure — every test so far returns the entire corpus within `k=3`.
- **200/25 chunking is untested end to end.** It wins every *retrieval-only*
  comparison recorded in `BUILD_LOG.md`, but a full 8-question generation
  run at that setting was attempted and abandoned mid-run when the
  development machine ran low on memory. Retrieval score is a proxy, not a
  verdict — this one is still open.
- **Neither UI has been driven through a live browser session recently.**
  Both `app.py` and `server.py` have had their logic verified — the shared
  pipeline functions via the CLI and test suite, the two UIs' own routes and
  error paths via direct calls and (for Flask) an in-process test client —
  but not a real click-through with a model actually generating on screen,
  to avoid competing with other running applications for RAM on an 8 GB
  machine.
- **Cross-document retrieval is not built.** `embed.index_pdf` can index
  several files into one collection, and every retrieved chunk carries its
  source filename, but both UIs' upload flow creates one collection per
  uploaded file and queries only that one per session.
- **Slow.** 30–100 seconds per answer on CPU, with high variance from model
  loading and memory pressure.

---

## Possible extensions

- OCR for scanned documents
- Hybrid retrieval (semantic + BM25) for exact identifiers and dates
- Reranking with a cross-encoder
- Cross-document retrieval with per-answer source attribution
- Conversational memory across turns
- Automated evaluation metrics (RAGAS and similar)
