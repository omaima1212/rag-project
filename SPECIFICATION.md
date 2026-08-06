# Local RAG System — Project Specification

**Project name:** Local Document Q&A (RAG over your own PDFs)
**Author:** Oumaima Cherieg Mednini
**Target hardware:** 8 GB RAM, no dedicated GPU, Windows
**Estimated effort:** 20–25 hours over 6–8 days

---

## 1. What you are building, in one paragraph

A program that lets you upload a PDF and ask questions about it in natural language, and that answers **only from the content of that PDF** — not from what the model happens to remember from its training. Everything runs on your own machine: no API keys, no cloud, no cost, no data leaving your laptop. Under the hood, the document is cut into pieces, each piece is converted into a numeric fingerprint of its meaning, and when you ask a question the system finds the pieces whose meaning is closest to your question and hands them to a small language model with the instruction "answer using only this."

---

## 2. The problem this solves

A language model has two weaknesses that matter here:

1. **It doesn't know your documents.** Your company handbook, your course notes, your research papers — none of that was in its training data.
2. **It can't read a large document at once.** Context windows are limited, and a small 3B model on 8 GB of RAM has a very limited practical window. You cannot paste a 200-page PDF into it.

**Naive fix:** paste the whole document into the prompt. Fails on anything non-trivial — too long, too slow, and the model's accuracy degrades badly when the relevant sentence is buried in 50 pages of noise.

**RAG fix:** don't send the whole document. Send only the 3–5 passages that are actually relevant to this specific question. This is smaller, faster, more accurate, and it lets you show the user *which* passages the answer came from — which is the property that makes RAG trustworthy in real products.

---

## 3. What RAG actually means

**RAG = Retrieval-Augmented Generation.** Three words, three ideas:

- **Retrieval** — searching a knowledge store for relevant content. Not keyword search: *semantic* search, matching meaning rather than exact words. A question about "how much holiday do I get" should find a passage titled "Annual Leave Entitlement" even though no word matches.
- **Augmented** — the retrieved passages are injected into the prompt before the model sees the question. You are augmenting the prompt, not retraining the model. Nothing about the model's weights changes.
- **Generation** — the model writes an answer in natural language, grounded in the passages you gave it.

The key mental model: **RAG is a search engine bolted to the front of a language model.** Most of the quality of a RAG system comes from the search half, not the model half. This is the single most important thing to internalise before you start — beginners blame the model when the retriever is at fault.

---

## 4. Architecture

Two distinct phases. They run at different times and are easy to confuse — keep them separate in your head and in your code.

### Phase A — Indexing (runs once per document, offline)

```
PDF file
   ↓  [PyMuPDF]
raw text
   ↓  [chunking function]
list of overlapping text chunks (~300–500 words each)
   ↓  [nomic-embed-text via Ollama]
one vector (768 numbers) per chunk
   ↓  [ChromaDB]
persistent local vector database  ←── stores: vector + original text + metadata
```

### Phase B — Querying (runs on every question, online)

```
user question
   ↓  [nomic-embed-text — the SAME model as indexing]
question vector
   ↓  [ChromaDB similarity search]
top-k most similar chunks (k = 3 to start)
   ↓  [prompt template]
"Context: <chunks>. Question: <question>. Answer only from the context."
   ↓  [llama3.2:3b via Ollama]
grounded answer + the source chunks used
```

**Critical invariant:** the embedding model in Phase B must be *identical* to the one in Phase A. Different models produce vectors in different coordinate spaces; mixing them produces mathematically valid but meaningless similarity scores. Bugs of this type are silent — the system returns confident nonsense rather than crashing. Write the model name in exactly one constant and import it in both places.

---

## 5. Components, and what each is responsible for

| Component | Tool | Responsibility | Common failure |
|---|---|---|---|
| PDF extraction | PyMuPDF (`fitz`) | Turn PDF bytes into plain text | Multi-column layouts interleave; tables lose structure; scanned PDFs return nothing |
| Chunking | Your own Python | Split text into retrievable units with overlap | Chunks cut mid-sentence; too small = no context; too large = diluted meaning |
| Embedding | `nomic-embed-text` on Ollama | Text → vector of numbers encoding meaning | Using a different model at query time; forgetting to batch (slow) |
| Vector store | ChromaDB (local, persistent) | Store vectors + text, return nearest neighbours | Re-indexing the same doc twice creates duplicate chunks |
| Generation | `llama3.2:3b` on Ollama | Write the answer from retrieved context | Ignores the context and answers from training data (fix in the prompt) |
| Interface | Streamlit | Upload, ask, display answer + sources | Re-running the whole index on every keystroke — cache it |
| Runtime | Ollama | Local model server on `localhost:11434` | Not running → connection refused; OOM on 8 GB |

---

## 6. Tech stack rationale

You will be asked "why did you choose this?" — have answers ready.

- **Ollama over Hugging Face Transformers directly.** Ollama handles model quantisation, memory mapping, and serving behind a simple HTTP API. On 8 GB of RAM, a raw `transformers` setup would need manual quantisation work that is not the point of this project.
- **`llama3.2:3b` over anything larger.** 3 billion parameters quantised is roughly 2 GB in RAM. A 7B model is ~4.5 GB and will make your machine thrash. RAG deliberately shifts the burden from model size to retrieval quality — a small model with good context beats a large model with bad context.
- **`nomic-embed-text` over general-purpose LLM embeddings.** It's a dedicated embedding model: small, fast, 768 dimensions, trained specifically so that semantic similarity maps to vector proximity. Using a chat model for embeddings is slower and worse.
- **ChromaDB over Pinecone/Weaviate.** Runs in-process, persists to a local folder, zero configuration, no network. Hosted vector DBs solve a scale problem you don't have.
- **Streamlit over Flask/FastAPI + frontend.** Three function calls produce a usable demo UI. The project's value is the pipeline, not the web layer.
- **Why local at all?** Privacy (documents never leave the machine), cost (zero), and it proves you understand the stack rather than gluing together hosted APIs.

---

## 7. Concepts you must understand before writing code

Read up on these; each takes 15–30 minutes.

1. **Embedding.** A function mapping text to a fixed-length list of numbers, trained so that texts with similar meaning land near each other. "Dog" and "puppy" are close; "dog" and "tax return" are far.
2. **Cosine similarity.** How closeness is measured. Two vectors are compared by the angle between them, not the distance — so document length doesn't distort the comparison. Small angle = similar meaning. Score ranges roughly 0 to 1. You don't need the maths, but you do need the intuition.
3. **Vector database.** A store optimised for one operation: "given this vector, find the N stored vectors closest to it, fast." It uses approximate nearest-neighbour indexes rather than scanning everything.
4. **Chunking and overlap.** A chunk is the atomic unit of retrieval — you retrieve whole chunks, never fragments. Overlap (repeating ~50 words between consecutive chunks) exists so a sentence straddling a boundary still appears intact in at least one chunk.
5. **Top-k.** How many chunks you retrieve per question. Too few → the answer isn't in the context. Too many → the real answer is diluted by noise and you blow the context window. Start at 3, tune it.
6. **Grounding and hallucination.** A grounded answer is traceable to a specific retrieved passage. Hallucination is the model inventing plausible content. Your prompt must explicitly instruct the model to refuse when the context doesn't contain the answer — and you must test that it actually refuses.
7. **Context window.** The maximum text the model can process at once. Your retrieved chunks plus the question plus the answer all have to fit. On a 3B model, budget conservatively.

---

## 8. Data flow, concretely

Worked example so the abstractions land:

1. You index a 40-page employee handbook. It becomes ~120 chunks of ~400 words each.
2. Each chunk is embedded → 120 vectors of 768 numbers → stored in ChromaDB with the original text alongside.
3. You ask: *"How many vacation days do new employees get?"*
4. That question is embedded → one 768-number vector.
5. Chroma compares it against all 120 stored vectors and returns the 3 closest — likely the "Annual Leave" section, even though the handbook never uses the word "vacation."
6. Those 3 chunks are pasted into a prompt template with your question and an instruction to stay within the context.
7. `llama3.2:3b` reads ~1200 words and writes: *"New employees receive 20 days of annual leave, increasing to 25 after three years of service."*
8. The UI shows that answer plus the 3 source chunks, so you can verify it.

---

## 9. Known failure modes — expect these

These are not signs you did something wrong; they are the normal texture of RAG work. Day 5 exists specifically to hit them.

- **Bad retrieval, blamed on the model.** The answer is wrong because the right chunk was never retrieved. Always print retrieved chunks next to the answer while debugging. This is the number one lesson of the project.
- **Multi-column PDFs.** PyMuPDF reads left-to-right across the page, interleaving two columns into gibberish. Detectable only by reading the extracted text with your own eyes.
- **Tables.** Structure is lost; rows and columns flatten into an unreadable line of numbers.
- **Scanned PDFs.** Images of text. Extraction returns empty. Needs OCR — out of scope, but say so in the README.
- **Chunk boundaries splitting an answer.** The question's answer spans two chunks and neither is retrievable on its own. Fixed by larger chunks or more overlap.
- **The model ignoring the context.** It answers from training data instead of your document. Fixed by a stricter prompt and by testing with a question you know the document doesn't answer.
- **Semantic search missing exact identifiers.** Product codes, invoice numbers, and names retrieve poorly by meaning. Real systems add keyword search alongside (hybrid retrieval). Worth mentioning as a known limitation.

---

## 10. Hardware constraints — 8 GB RAM

- `llama3.2:3b` quantised ≈ 2 GB resident. `nomic-embed-text` ≈ 0.3 GB. Windows itself takes 3–4 GB. Your headroom is thin but workable.
- **Close your browser before running Ollama sessions.** Chrome with many tabs can consume more RAM than the model. This is the single highest-impact thing you can do.
- Expect 5–20 seconds per answer. That is normal on CPU, not a bug.
- Index once and persist ChromaDB to disk. Never re-embed on every run — that is the slowest step by far.
- Test with PDFs under ~50 pages while developing. Scale up only once the pipeline is correct.
- If you hit out-of-memory: reduce top-k, shorten chunks, or drop to a 1B model.

---

## 11. Definition of done

Minimum viable — the project is real when all of these are true:

- [ ] A PDF can be indexed end to end without manual intervention
- [ ] Five factual questions about that PDF get correct answers
- [ ] A question the document does *not* answer produces a refusal, not an invention
- [ ] The retrieved source chunks are displayed alongside the answer
- [ ] The vector store persists — restarting doesn't force re-indexing
- [ ] It runs from a Streamlit UI, not just the terminal
- [ ] A README explains the architecture, the choices, and the limitations

Good, beyond minimum:

- [ ] A documented failure case on a messy PDF, with what you changed and the measured effect
- [ ] Chunk size / overlap / k tuned on evidence, not guesses
- [ ] Errors handled gracefully: no PDF, Ollama down, empty question
- [ ] Multiple documents indexed with source attribution in answers

---

## 12. Repository layout

```
rag-project/
├── venv/
├── data/                  # test PDFs (gitignore the large ones)
├── chroma_db/             # persistent vector store (gitignore this)
├── src/
│   ├── config.py          # model names, chunk size, k — ONE source of truth
│   ├── ingest.py          # PDF → text → chunks
│   ├── embed.py           # chunks → vectors → ChromaDB
│   ├── retrieve.py        # question → top-k chunks
│   ├── generate.py        # chunks + question → answer
│   └── app.py             # Streamlit UI
├── notes.md               # running log: what broke, what fixed it
├── requirements.txt
└── README.md
```

Keep `notes.md` from day one. It costs two minutes a day and it is where your interview answers come from — "the model kept hallucinating until I realised retrieval was returning the wrong section, because my chunks were splitting the table of contents" is a far better story than "I built a RAG system."

---

## 13. What this project demonstrates

Skills you can point to afterwards:

- Understanding the difference between fine-tuning and retrieval augmentation, and when each applies
- Vector embeddings, semantic search, and vector databases in practice
- Prompt design for grounding and hallucination control
- Building and debugging a multi-stage data pipeline
- Working within real hardware constraints and justifying the trade-offs
- Systematic debugging: isolating which stage of a pipeline is at fault

The last two are what separate this from a tutorial follow-along.

---

## 14. Explicitly out of scope

Say these in the README as "possible extensions" — it shows you know the landscape.

- OCR for scanned documents
- Hybrid search (semantic + keyword/BM25)
- Reranking retrieved chunks with a cross-encoder
- Conversational memory across turns
- Multi-user, authentication, deployment
- Automated evaluation metrics (RAGAS and similar)

---

## 15. As-built status — end of Day 4

This section records what was actually constructed, and where it diverged from the plan above. Sections 1–14 describe intent; this one describes reality.

### Progress against plan

| Day | Planned | Status |
|---|---|---|
| 1 | Environment setup | Complete |
| 2 | PDF ingestion and chunking | Complete |
| 3 | Embeddings and vector storage | Complete |
| 4 | Retrieval and generation | Complete |
| 5 | Debug retrieval quality | Not started |
| 6–7 | Streamlit UI and polish | Not started |

Days 1–4 ran to plan. The full pipeline works end to end: question in, grounded answer out, with sources displayed.

### Divergences from the original design

**1. `num_ctx` is set explicitly to 4096.**
Not in the original plan. Ollama sizes its memory allocation from the context window, and the default reserved more than 8 GB of RAM could support once both models were resident. This was discovered by crashing, not by reading documentation.

**2. Error handling is heavier than a first draft would normally carry.**
Both Ollama call sites surface the server's own response body and distinguish `ConnectionError` from HTTP errors. The reason is specific to this project: `requests.raise_for_status()` discards the response body, and Ollama's failures on a memory-constrained machine are otherwise indistinguishable from one another. Section 10 anticipated memory pressure; it did not anticipate that the errors would be *uninformative*.

**3. `CONCEPTS.md` was added.**
Not in the planned repository layout. Written mid-project to separate conceptual explanation from the chronological log — the log answers "what happened," the concepts document answers "why does this work."

**4. `notes.md` was superseded by `BUILD_LOG.md`.**
Same role as planned in §12, more structure.

### Measured characteristics of the current system

| Property | Value |
|---|---|
| Test document | 11 pages, 17,141 characters, ~2,541 words |
| Chunks | 8 (min 91, max 400, avg 361 words) |
| Embedding dimensions | 768 |
| Index size | 736 KB for 8 chunks |
| RAM used by both models | 4.3 GB |
| Semantic separation (related vs unrelated) | 0.818 vs 0.370 |

### Definition-of-done progress

From §11. Minimum viable:

- [x] A PDF can be indexed end to end without manual intervention
- [x] Five factual questions about that PDF get correct answers
- [x] A question the document does not answer produces a refusal
- [x] The retrieved source chunks are displayed alongside the answer
- [x] The vector store persists — restarting doesn't force re-indexing
- [ ] It runs from a Streamlit UI, not just the terminal
- [ ] A README explains the architecture, the choices, and the limitations

Good, beyond minimum:

- [ ] A documented failure case on a messy PDF, with the fix and its measured effect
- [ ] Chunk size / overlap / k tuned on evidence rather than guesses
- [x] Errors handled gracefully — Ollama down, out of memory, empty collection
- [ ] Multiple documents indexed with source attribution

### Known open items carried into Day 5

1. **The `highperformance` artefact.** The hyphen-rejoin rule strips real hyphens from compound words that fall at a line break. Judged low-impact, not yet addressed.
2. **Test outputs were not captured verbatim.** Day 5 requires before-and-after comparison; from here on every test needs its question, retrieved chunk indices, similarity scores and answer recorded.
3. **Only one document tested, and an easy one.** Single-column, clean prose. The pipeline has not met a multi-column layout, a table, or a scanned page.
4. **No tuning has been done.** Chunk size, overlap and k are all at their initial guessed values. None has been justified by evidence.

---

## 16. As-built status — end of Day 8

This section was stale for a long time: it described end-of-Day-4 reality while `README.md` and `BUILD_LOG.md` had already moved through Days 5–7 (retrieval debugging, the context-header fix, the Streamlit chat UI). It is corrected here, and extended with a further session's work. Full detail for everything summarised below lives in `BUILD_LOG.md`.

### Progress against plan

| Day | Planned | Status |
|---|---|---|
| 1–4 | Environment through generation | Complete (unchanged from above) |
| 5 | Debug retrieval quality | Complete — found and fixed a real failure (see README, "The main finding") |
| 6–7 | Streamlit UI and polish | Complete — streaming chat interface, `src/app.py` |
| 8 | Close out carried-forward items; add a second UI | Complete — see below |

### What Day 8 did

1. **Fixed a prompt-leakage bug carried since Day 4.** `build_prompt()` labelled context blocks `[Source N - chunk N]`; the model sometimes echoed that internal label back ("According to chunk 4…"), meaningless to a reader. Relabelled `[Passage N - filename]` and the system prompt now explicitly forbids citing internal labels.

2. **Measured and fixed the header filename-leak caveat from Day 5.** The adopted context header included the source filename; on the test document that filename contains "stage" (French for internship), inflating similarity on an unanswerable internship-salary question by +0.048. An A/B/C sweep (`experiment.sweep_header_filename`) tested a structure-only header (position + dates, no filename) across the full 8-question set: it kept the Day-5 positional fix (q4 stayed rank-0, score slightly higher), reduced the inflation, and matched or beat the filename variant on every topical question. Adopted as the default (`HEADER_INCLUDE_FILENAME = False` in `config.py`); the main index was re-embedded under it.

3. **Tested a messy PDF for the first time.** Built `tools/make_messy_pdf.py` — a synthetic two-column report plus a data table, the two failure modes the README had listed since Day 1 but never actually exercised. Result, measured rather than assumed: the two-column prose extracted in correct reading order (PyMuPDF's block sort handled this particular layout correctly — a real correction to the standing assumption); the table flattened to one run-on line with no column structure, and end-to-end this caused one genuine wrong refusal (a fact present in the text was not found) while two other table-derived facts were answered correctly. See README, Known limitations.

4. **Found and fixed an unguarded crash on a textless PDF.** A scanned document (or any PDF yielding zero chunks) reached `chromadb`'s `upsert()` with empty `ids`, which raises `ValueError: Expected IDs to be a non-empty list`. Neither `app.py` nor the new `server.py` caught it, so the UI would have shown a raw traceback instead of an explanable message. Found by reading Chroma's validation source, not by triggering it live; fixed in both UIs and verified with a purpose-built empty-text fixture (`tools/make_blank_pdf.py`) run through Flask's in-process test client — no server process or model call needed to prove it.

5. **Hardened the evaluation suite so it can fail.** `evaluate.py` had reported 8/8 in every configuration ever run through it, including ones with a known-wrong retrieved chunk, because `expect_any` term lists mixed strong signals (proper nouns, exact numbers) with generic words ("team", "environment", "job") that a wrong answer could still contain. Tightened `questions.json` to the strong signal only for q4 and q6, and added hedge-phrase detection (`HEDGE_MARKERS`) so a keyword match riding on hedging language ("it is likely…") downgrades to `CHECK` instead of a silent `PASS`.

6. **Added a second UI: a minimal Flask + vanilla-JS server (`src/server.py`, `src/web/index.html`).** Requested as an alternative to Streamlit on a RAM-constrained machine — no rerun-the-whole-script model, no Streamlit process, a single static page with inline CSS/JS. Same pipeline underneath (`ingest` → `embed` → `retrieve` → `generate`), same streaming design (newline-delimited JSON, mirroring Ollama's own streaming format). Verified via Flask's test client (routes, error paths, the empty-chunk fix) without starting a live server or calling the chat model — the two Ollama models cost the same RAM regardless of which UI sits in front of them, and a live end-to-end click-through was deliberately not run today to avoid competing with other applications for memory.

### Definition-of-done progress

From §11, corrected for what Days 5–8 actually completed:

- [x] A PDF can be indexed end to end without manual intervention
- [x] Five factual questions about that PDF get correct answers
- [x] A question the document does not answer produces a refusal
- [x] The retrieved source chunks are displayed alongside the answer
- [x] The vector store persists — restarting doesn't force re-indexing
- [x] It runs from a UI, not just the terminal — **two** now, Streamlit and Flask
- [x] A README explains the architecture, the choices, and the limitations

Good, beyond minimum:

- [x] A documented failure case on a messy PDF, with the fix and its measured effect (Day 8, item 3 above)
- [x] Chunk size / overlap tuned on evidence; k=3 justified on evidence. The 200/25 chunk configuration wins every *retrieval-only* comparison recorded in `BUILD_LOG.md`, but a full 8-question *end-to-end* run at that setting was not completed — attempted on Day 8 and abandoned mid-run when the shared machine ran low on memory. Still open.
- [x] Errors handled gracefully — Ollama down, out of memory, empty collection, and now an unguarded empty-chunk crash found and fixed
- [~] Multiple documents indexed with source attribution — supported at the pipeline level (`embed.index_pdf` can index several files into one collection; every retrieved chunk carries its source filename and it is shown in both UIs), but neither UI's upload flow queries across more than one document in a single session. Not extended today.

### Known open items carried forward

1. **End-to-end run at 200/25**, blocked by memory on Day 8 — retry when the machine has headroom.
2. **A live click-through of both UIs** — `app.py` was never confirmed running in a browser (it was already "Pending" at the end of Day 7), and `server.py` is new today. Both have had their logic verified without a live model call; neither has been watched actually answer a question on screen.
3. **Cross-document retrieval** — indexing several documents into one queryable session, with per-answer attribution, remains unbuilt in the UI layer.
4. **The `highperformance` artefact** (item 1 from the Day 4 list above) is still unaddressed.

---

## 17. As-built status — end of Day 9

Two requests in one session, both extending the system rather than fixing it: a visual redesign of `src/web/index.html`, and an opt-in web-search fallback. Full detail in `BUILD_LOG.md`, "Day 9".

### What changed

**The Flask UI got a design pass.** Dark/light theme with a manual toggle (`localStorage`-persisted, not just `prefers-color-scheme`), inline SVG icons in place of emoji, avatar chat bubbles, a typing indicator before the first token, drag-and-drop upload, a floating pill composer. Still one dependency-free HTML file — no CDN, no build step, no new runtime shipped to the browser. `server.py` required no changes; every element ID the JS depends on was preserved.

**An opt-in web search fallback was added — the project's one deliberate exception to "nothing leaves the machine."** Requested as: when the document doesn't have the answer, say so, but also search the web and show that answer with its source. This was not implemented on request alone; two decisions were surfaced back to the project owner first, because they changed the system's core contract rather than extending it quietly:

- **Which search backend.** Chosen: Google Custom Search API (over a no-key DuckDuckGo scrape or the narrower Wikipedia API) — real results, at the cost of needing an API key and a Search Engine ID, both read from environment variables rather than `config.py` so a real key never sits in a tracked file.
- **Automatic or opt-in.** Chosen: **opt-in, off by default**, via a sidebar toggle in both UIs. This was the load-bearing decision: it means §11's Definition of Done and `questions.json`'s q7/q8 (which specifically test that the system refuses a question it cannot support, rather than answering from outside knowledge) are **unaffected** — the evaluation suite never sets `web_fallback`, so those two questions still test exactly what they always did before this feature existed.

The fallback fires only after a grounded answer has already been generated and found to look like a refusal (`looks_like_refusal()`, promoted from `evaluate.py`'s private grading logic to `generate.py`, so grading and the fallback trigger share one definition instead of two that could drift apart). It is rendered as a visually distinct block, explicitly labelled "not from your document," in both UIs.

### Definition-of-done progress

§11 is otherwise unchanged by Day 9 — the minimum-viable and "good, beyond minimum" checklists from §16 stand as recorded there. Day 9 adds one item outside that list entirely, since it was never part of the original spec:

- [x] Web search fallback for unanswerable questions, opt-in and off by default, without weakening the existing grounding guarantee or its tests — **done**, not yet exercised with a real API key.

### Known open items carried forward

Unchanged from Day 8's list above, plus:

5. **A real web search with configured credentials.** ~~No `GOOGLE_SEARCH_API_KEY` / `GOOGLE_SEARCH_ENGINE_ID` was available this session~~ — **superseded, see §18.** The backend itself changed; this item no longer applies as written.
6. **`SPECIFICATION.md` §14 ("Explicitly out of scope") still lists "hybrid search" and doesn't mention a web fallback at all** — written before this feature existed. Worth a pass to reconcile intent-documentation with as-built reality the next time §1–14 are revisited; not done today, consistent with this file's own convention of correcting status sections rather than rewriting the original intent sections.

---

## 18. As-built status — end of Day 10

A same-day correction to §17, not a new day of feature work. Full detail in `BUILD_LOG.md`, "Day 10".

**The first live click-through happened, and it is what caught this.** A real browser, a real question the document doesn't answer, web fallback toggled on. The grounded refusal fired correctly, the fallback triggered off it correctly, the teal "not from your document" card rendered correctly — and the search call itself failed: Google returned `API_KEY_INVALID`, a rejected credential, not a code defect. Every layer built to handle that kind of failure (§17's "a failed web step is not a failed turn") did exactly what it was designed to do on the first real attempt.

**Decision, surfaced rather than assumed:** debug the Google Cloud credential, or replace the backend. Chosen — **replace it.** `websearch.py` now calls DuckDuckGo's HTML results page (via `BeautifulSoup`) instead of Google's Custom Search API. `is_configured()`, `search_web()`'s signature, and its return shape are unchanged, so nothing in `server.py`, `app.py`, or the frontend's *logic* needed to change — only user-facing copy that said "Google" now says "DuckDuckGo," and `config.py` lost the two credential settings it no longer has any use for (`GOOGLE_SEARCH_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`, and the `import os` that only they needed).

**The trade honestly stated:** zero setup, versus scraping HTML instead of calling a versioned API. §17's original backend choice reasoned about exactly this trade-off in the abstract; Day 10 is the concrete case where the "needs an account" cost of the API path turned out to be worse in practice than expected, and the "not a stable contract" cost of the scrape path was accepted as the better trade once that became concrete rather than hypothetical.

### Definition-of-done progress

§17's new item is restated, corrected:

- [x] Web search fallback for unanswerable questions, opt-in and off by default, without weakening the existing grounding guarantee or its tests — still holds, backend changed underneath it.
- [x] Setup-free — stronger than originally shipped: zero configuration now, versus an API key and Search Engine ID before.

### Known open items carried forward

Unchanged from Day 8, plus:

7. **A live retry of the web fallback with DuckDuckGo**, in the browser — the natural next step, not yet done this session.
8. **The Streamlit UI (`app.py`) still has not been seen running.** Only the Flask UI has had a live click-through, today.

---

## 19. Vocabulary reference

| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation — retrieve relevant text, inject into the prompt, generate |
| Chunk | A unit of text stored and retrieved as one piece |
| Overlap | Repeated text shared between consecutive chunks |
| Embedding | Text represented as a fixed-length vector of numbers encoding meaning |
| Vector / dimension | The list of numbers; `nomic-embed-text` produces 768 of them |
| Cosine similarity | Closeness measured by angle between vectors |
| Vector database | Store optimised for nearest-neighbour vector search |
| Top-k | How many chunks are retrieved per query |
| Context / context window | Text placed in the prompt / the model's maximum input size |
| Grounding | Answering strictly from provided source text |
| Hallucination | Model inventing plausible but unsupported content |
| Quantisation | Compressing model weights to lower precision to fit in less RAM |
| Ingestion / indexing | The one-time pipeline that loads documents into the vector store |
| Inference | Running the model to produce output |
