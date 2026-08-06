# RAG Project — Build Log

Running record of every step taken, in order. Commands, outcomes, problems, fixes.

**Machine:** Windows, 8 GB RAM, no dedicated GPU
**Project folder:** `C:\Users\souad\rag-project`
**Started:** 24 July 2026

---

## Day 1 — Environment setup

**Goal:** every tool responds when poked, before any pipeline code is written.

### Step 1.1 — Verify Python

```powershell
python --version
```

Result: Python 3.11 present. No install needed.

### Step 1.2 — Create project and virtual environment

```powershell
mkdir rag-project
cd rag-project
python -m venv venv
venv\Scripts\activate
```

Prompt changed to `(venv) PS C:\Users\souad\rag-project>` — confirms the virtual environment is active.

**Why a venv:** isolates this project's packages from the system Python. Without it, package versions from other projects can conflict. Must be re-activated in every new terminal session.

### Step 1.3 — Install Ollama

Installed from [ollama.com/download](https://ollama.com/download).

**What Ollama is:** a local server that runs language models and exposes them over HTTP at `http://localhost:11434`. Our Python code will talk to it with plain HTTP requests — no cloud, no API key.

### Step 1.4 — Pull the models

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

- `llama3.2:3b` — 2.0 GB — the generation model that writes answers
- `nomic-embed-text` — ~275 MB — the embedding model that converts text to vectors

**Problem hit:** typed `nomic-embed-textx` (trailing `x`).
**Error:** `Error: pull model manifest: file does not exist`
**Fix:** corrected the model name. Ollama gives no spelling suggestions — the name must be exact.

### Step 1.5 — Test the generation model interactively

```powershell
ollama run llama3.2:3b
```

Chatted with it, exited with `/bye`. Slow responses are expected on CPU with 8 GB RAM — not a fault.

### Step 1.6 — Install Python libraries

```powershell
pip install chromadb pymupdf requests streamlit
```

| Package | Role in the project |
|---|---|
| `pymupdf` (imported as `fitz`) | Extract text from PDF files |
| `chromadb` | Local persistent vector database |
| `requests` | HTTP calls to the Ollama API |
| `streamlit` | The web UI on Days 6–7 |

Large dependency tree (~90 packages) — chromadb pulls in onnxruntime, pyarrow, kubernetes and others. Normal.

### Step 1.7 — Verify the Python imports

```powershell
python -c "import fitz, chromadb, requests, streamlit; print('all good')"
```

### Step 1.8 — Verify the Ollama API

```powershell
curl.exe http://localhost:11434/api/tags
```

Expected: JSON listing both `llama3.2:3b` and `nomic-embed-text`.

**Problem hit:** ran `curl` without `.exe`. In PowerShell, `curl` is an alias for `Invoke-WebRequest`, which triggered a security warning about script parsing.
**Fix:** used `curl.exe` to invoke the real curl binary. Alternative: `Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing`.

**Lesson:** PowerShell aliases several Unix command names (`curl`, `wget`, `ls`) to its own cmdlets with different behaviour. When following tutorials written for Linux/macOS, append `.exe` to get the real tool.

---

### Day 1 status

| Check | Status |
|---|---|
| Python 3.10+ installed | Done |
| Virtual environment created and active | Done |
| Ollama installed | Done |
| `llama3.2:3b` pulled | Done |
| `nomic-embed-text` pulled | Done |
| Generation model responds in terminal | Done |
| Python libraries installed | Done |
| Imports verified | Done |
| Ollama API reachable | Done |

**Day 1 complete.** Every tool responds when poked.

---

## Day 2 — PDF ingestion and chunking

**Goal:** PDF in, list of clean text chunks out.

### Step 2.1 — Create `src/config.py`

Single source of truth for all settings: paths, model names, chunk size, overlap, top-k.

**Why this file matters most:** the embedding model must be identical at indexing time and at query time. Different models produce vectors in different coordinate spaces — mixing them yields similarity scores that are mathematically valid but meaningless. The failure is silent: no crash, just confidently wrong answers. Defining `EMBED_MODEL` once, here, makes that mistake structurally impossible.

Settings chosen:

| Setting | Value | Reason |
|---|---|---|
| `CHUNK_SIZE` | 400 words | Large enough to be self-contained, small enough to be specific |
| `CHUNK_OVERLAP` | 50 words | Protects passages that straddle a chunk boundary |
| `TOP_K` | 3 | Conservative start; a 3B model has little context to spare |

`PROJECT_ROOT` is derived from `__file__`, so no absolute path is hardcoded anywhere.

### Step 2.2 — Create `src/ingest.py`

Four functions forming stage 1 of the pipeline:

1. **`extract_text(pdf_path)`** — opens the PDF with PyMuPDF, pulls text page by page, prints a character count per page. That per-page count is a diagnostic: a page reporting 0 characters is an image, and no amount of downstream work will recover it.

2. **`clean_text(text)`** — repairs extraction artefacts:
   - rejoins words hyphenated across a line break (`informa-\ntion` → `information`)
   - converts single newlines (layout artefacts) to spaces, while preserving double newlines (real paragraph breaks)
   - collapses runs of whitespace

   Skipping this step means the chunker sees fragments instead of sentences, which degrades embedding quality.

3. **`chunk_text(text, chunk_size, overlap)`** — sliding window over the word list. The window advances by `chunk_size - overlap` (350 words) each iteration, so consecutive chunks share 50 words. Breaks out of the loop once the window reaches the end, to avoid emitting a final near-duplicate chunk.

4. **`load_and_chunk(pdf_path)`** — chains all three together.

### Step 2.3 — Run it and inspect the output

```powershell
python src\ingest.py data\yourfile.pdf
```

Prints the first, middle and last chunk, plus word-count statistics.

**This inspection is the actual work of Day 2.** Bad chunks guarantee bad answers on Day 4, and no model can compensate for them. What to look for:

- Text that reads as scrambled or interleaved → multi-column layout
- A page reporting 0 characters → scanned image, needs OCR (out of scope)
- Numbers running together with no structure → a table was flattened
- Headers and footers repeating inside every chunk → noise that will pollute retrieval

### Status

| Check | Status |
|---|---|
| `config.py` created | Done |
| `ingest.py` created | Done |
| Test PDF placed in `data/` | Done |
| Chunk output inspected by eye | Done |

### Step 2.4 — First run and results

**Problem hit:** `ModuleNotFoundError: No module named 'fitz'`
**Cause:** the virtual environment was not active — the prompt read `PS C:\Users\souad\rag-project>` with no `(venv)` prefix. Python was the system interpreter, which has none of the project's packages.
**Fix:** `venv\Scripts\activate`
**Lesson:** the venv must be re-activated in every new terminal session. Check for the `(venv)` prefix before blaming the code.

**Problem hit:** the test PDF's filename contains spaces, so PowerShell split it into multiple arguments.
**Fix:** wrap the path in double quotes.

**Test document:** `Impression des comptes rendus du dépôt de stage.pdf` — an 11-page internship logbook (daily reports on an HPC monitoring project at ReDX / Toubkal cluster). Mixed French headers with English body text.

**Results:**

| Metric | Value |
|---|---|
| Pages | 11 |
| Characters extracted | 17,141 |
| Chunks produced | 8 |
| Word counts | min 91, max 400, average 361 |

Every page returned 900–2,900 characters, so no page is a scanned image. Chunk 8 is short (91 words) because it is the tail of the document — expected, not a bug.

**Eyeball inspection: passed.** Single-column layout, clean sentences, no interleaving, no flattened tables. Section headings and dates survived intact.

**One artefact found:** "highperformance" appears in the middle chunk. The `clean_text` regex that rejoins hyphenated line breaks (`informa-\ntion` → `information`) cannot distinguish a soft hyphen inserted by the layout engine from a real hyphen in a compound word like "high-performance" that happens to fall at a line break. Both are stripped.

Impact judged low — embeddings are robust to this kind of noise, and the same term appears correctly hyphenated elsewhere in the document. Logged as a candidate for Day 5 rather than fixed now. A more precise fix would consult a dictionary or check whether the fragment before the hyphen is itself a valid word.

---

## Day 3 — Embeddings and vector storage

**Goal:** a working local vector database containing the document's content.

### Step 3.1 — Create `src/embed.py`

Functions:

1. **`embed_text(text)`** — POSTs to `http://localhost:11434/api/embeddings` with the model name and the text, returns a list of 768 floats. This is the only place the project talks to the embedding model.

2. **`get_collection()`** — opens a `chromadb.PersistentClient` pointed at `chroma_db/`.

   **Critical detail:** the collection is created with `metadata={"hnsw:space": "cosine"}`. Chroma defaults to squared L2 distance; this project needs cosine similarity, which compares the *angle* between vectors and so ignores text length. Without this, a 400-word chunk and a 10-word question compare poorly for reasons that have nothing to do with meaning.

3. **`index_pdf(pdf_path)`** — chains ingest → embed → store.

   **Deterministic IDs** of the form `filename::chunk_index` mean re-indexing the same PDF *overwrites* previous entries via `upsert` rather than appending duplicates. Duplicate chunks are an insidious bug: they crowd out genuinely distinct results inside top-k, so retrieval quietly gets worse the more times you run the indexer.

4. **`cosine_similarity(a, b)`** — implemented by hand (dot product over the product of magnitudes) rather than imported, to keep the concept concrete.

5. **`sanity_test()`** — embeds two sentences on the same topic that share almost no vocabulary, plus one unrelated sentence, and asserts the related pair scores higher.

### Step 3.2 — Why the sanity test is not optional

An embedding failure is silent. Wrong vectors do not raise an exception — they produce a working system that returns irrelevant chunks, which then produces plausible-sounding wrong answers on Day 4. Verifying the numerical property directly, before building anything on top, is far cheaper than debugging it three days later.

The specific property being tested: *similarity tracks meaning, not shared words.* The two related sentences deliberately use different vocabulary ("supercomputer / GPU nodes / scientific computing" versus "cluster / NVIDIA accelerators / HPC workloads"). A keyword search would rate them as unrelated. A working embedding model rates them as close.

### Step 3.3 — Run it

```powershell
python src\embed.py "data\Impression des comptes rendus du dépôt de stage.pdf"
```

Expected: 8 chunks embedded at 768 dimensions each, stored in `chroma_db/`, followed by the sanity test result.

Inspect what is stored at any time with no arguments:

```powershell
python src\embed.py
```

### Status

| Check | Status |
|---|---|
| `embed.py` created | Done |
| PDF indexed into ChromaDB | Done |
| Sanity test passed | Done |

### Step 3.4 — Verifying what was actually stored

Rather than trusting the script's own output, the database was inspected directly:

```
chroma_db/
├── chroma.sqlite3
└── d82eb950-350b-4ab2-ae0a-9bcd6d82e95a/    # HNSW index segment
```

Querying `chroma.sqlite3` confirms:

| Check | Result |
|---|---|
| Collection created | `documents` |
| Embeddings stored | 8 — matches the 8 chunks from Day 2 |
| On-disk size | 736 KB |

The index persists on disk, so restarting does not force re-embedding.

**Why verify independently:** a script that prints "stored 8 chunks" is reporting its own intent, not the state of the database. Confirming from the other side — reading the store directly — is a habit worth keeping for any pipeline that writes to persistent state.

---

### Step 3.5 — Sanity test results

```
A: The Toubkal supercomputer has GPU nodes for scientific computing.
B: Toubkal's cluster includes NVIDIA accelerators for HPC workloads.
C: I baked a chocolate cake for my sister's birthday party.

similarity(A, B) = 0.8180
similarity(A, C) = 0.3704
PASS
```

A and B share exactly one word — "Toubkal" — yet score 0.818. A keyword search would rate them as barely related. This is direct evidence that the embedding encodes *meaning* rather than vocabulary, which is the premise the entire retrieval stage rests on.

The 0.37 for the unrelated sentence is also informative: unrelated text scores well above 0, not near it. Cosine similarity between real-language embeddings rarely approaches zero, because all natural language shares structure. **The useful signal is the gap between scores, not their absolute values.**

All 8 chunks embedded at 768 dimensions.

**Additional confirmation:** the run reported "Collection now holds 8 chunks total" on a second indexing pass, not 16. The deterministic `filename::index` IDs overwrote the previous entries as designed. Had random IDs been used, the collection would now hold duplicates and top-k would start returning the same passage multiple times.

**Day 3 complete.**

---

## Day 4 — Retrieval and generation pipeline

**Goal:** full pipeline end to end — question in, grounded answer out.

### Step 4.1 — Create `src/retrieve.py`

The "R" in RAG, and where most of the system's quality lives.

**`retrieve(question, k)`** does three things:

1. Embeds the question with `embed_text()` — the *same* function, and therefore the same model, used at index time.
2. Calls `collection.query()` for the k nearest stored vectors.
3. Returns the text filed alongside those vectors, plus source metadata and score.

**Distance versus similarity.** Chroma returns cosine *distance*, where lower means closer. The code converts it with `similarity = 1 - distance` so that scores read the intuitive way: 1 is identical, 0 is unrelated. Mixing up the two directions is a common and confusing bug.

**Standalone execution matters.** `retrieve.py` runs on its own, with no language model involved:

```powershell
python src\retrieve.py "what supercomputer did you work on?"
```

This separation is deliberate. It makes it possible to answer the single most important debugging question — *was the right passage even retrieved?* — without the model in the way. Diagnosing retrieval and generation together means diagnosing neither.

### Step 4.2 — Create `src/generate.py`

The "AG" — retrieved passages **augment** the prompt, then the model **generates**.

**The system prompt is the entire defence against hallucination.** Each rule targets a specific failure:

| Rule | Failure it prevents |
|---|---|
| "Use ONLY information present in the context" | Model answering from training data instead of the document |
| Exact refusal string to use when the answer is absent | A model with no permitted way to fail will invent something instead |
| "Do not use prior knowledge" | Closes the obvious loophole in the first rule |
| "Quote briefly from the context" | Makes the answer traceable to a specific passage |

The second row is the subtle one. Refusal has to be an *explicitly permitted, clearly specified action*. Telling a model not to make things up, without telling it what to do instead, leaves invention as the only available response.

**Prompt structure.** Context first, question last. Models attend most reliably to the end of a prompt, so the question stays in the most salient position. Each chunk is labelled `[Source N - file, chunk index]` so the model can attribute claims and so failures can be traced to a specific passage.

**`temperature: 0`.** Creativity is the enemy here. The desired behaviour is faithful reporting of supplied text, not fluent prose. Zero temperature also makes runs reproducible, which matters for Day 5 — you cannot measure whether a change improved anything if the output varies between identical runs.

**Sources are always printed with the answer.** Not a UI nicety: it is the debugging instrument. Reading the answer without the sources means having no way to tell a retrieval failure from a generation failure.

### Step 4.3 — Test plan

```powershell
python src\retrieve.py "what supercomputer did you work on?"
python src\generate.py "what supercomputer did you work on?"
python src\generate.py     # interactive mode
```

Questions to run, chosen to probe different behaviours:

| # | Question | What it tests |
|---|---|---|
| 1 | What supercomputer did you work on? | Vocabulary mismatch — document says "cluster," question says "supercomputer" |
| 2 | How many compute nodes does Toubkal have? | Precise fact retrieval; the number must not drift |
| 3 | What happened on the first day of the internship? | Retrieval by date/position rather than topic |
| 4 | What GPUs are in the cluster? | Detail sitting mid-chunk rather than at a boundary |
| 5 | **What is the capital of Japan?** | **Refusal.** Answerable from training data, absent from the document |

Question 5 is the important one. If the model answers "Tokyo," the grounding instruction has failed and every other answer becomes untrustworthy — because there is then no way to know which answers came from the document and which from the model's memory.

### Status

| Check | Status |
|---|---|
| `retrieve.py` created | Done |
| `generate.py` created | Done |
| Pipeline runs end to end | Done |
| Retrieval inspected standalone | Done |
| Five test questions run | Done |
| Refusal test passed | Done |

### Step 4.4 — Problem: Ollama crashing under memory pressure

**Symptom 1:** `HTTPError: 500 Server Error` from `/api/embeddings`.
**Symptom 2:** on the next run, a truncated traceback ending inside `socket.recv_into` — the connection died while waiting for `/api/chat` to respond.

**Diagnosis.** `requests.raise_for_status()` discards the response body, so the 500 carried no explanation. Both call sites were rewritten to surface the server's own error message and to catch `ConnectionError` separately, since a dropped connection and an HTTP error have completely different causes.

Then `ollama ps` showed the real problem:

```
NAME                       SIZE      PROCESSOR    CONTEXT
llama3.2:3b                3.9 GB    100% CPU     4096
nomic-embed-text:latest    376 MB    100% CPU     2048
```

**4.3 GB of models resident on an 8 GB machine**, with Windows itself consuming 3–4 GB. RAG needs both models — one to embed the question, one to generate the answer — so they compete for memory. Ollama was being killed while loading the second one.

**Fixes applied:**

1. `"num_ctx": 4096` set explicitly in the chat options. Ollama sizes its memory allocation from the context window, so an oversized default reserves memory that is not available. 4096 tokens comfortably holds 3 chunks plus the question and answer.
2. `timeout` raised to 600s — CPU inference on this hardware is slow, and a timeout would otherwise be misread as a crash.
3. `ConnectionError` now raises a message naming the likely cause and listing recovery steps in order.

**Recovery procedure that worked:** `ollama stop --all`, close the browser, retry.

**Lesson — distinguish the three failure signatures.** They look similar from Python and have unrelated causes:

| Signature | Meaning |
|---|---|
| `ConnectionError` immediately | Ollama is not running |
| `ConnectionError` after a delay | Ollama started, then died — almost always memory |
| HTTP 500 with a body | Ollama is alive and rejecting the request — read the body |

**Operational note.** Ollama unloads models after roughly 5 minutes idle (the `UNTIL` column). The first question after a pause therefore pays the full reload cost and carries the crash risk; rapid follow-up questions are fast. A slow first answer is expected behaviour, not a broken pipeline.

**Result:** pipeline runs end to end.

---

### Step 4.5 — Test run

The five planned questions were run against the indexed logbook, including the refusal test ("What is the capital of Japan?"), and the pipeline behaved as intended.

**Note for Day 5:** the individual answers were not captured verbatim at the time. Day 5 depends on comparing outputs before and after configuration changes, so from here on the exact question, retrieved chunk indices, similarity scores and answer text need to be recorded for every test. Without that record there is no way to demonstrate that a change improved anything — which is precisely the evidence that makes the project worth discussing.

**Day 4 complete.** Full pipeline working end to end: question in, grounded answer out, with sources shown.

---

# Project state at the end of Day 4

## Files

```
rag-project/
├── data/                          test PDFs
│   └── Impression des comptes rendus du dépôt de stage.pdf
├── chroma_db/                     persistent vector store (8 vectors, 736 KB)
│   ├── chroma.sqlite3
│   └── d82eb950-.../              HNSW index segment
├── src/
│   ├── config.py                  all settings, single source of truth
│   ├── ingest.py                  PDF -> text -> clean -> chunks
│   ├── embed.py                   chunks -> vectors -> ChromaDB
│   ├── retrieve.py                question -> top-k chunks
│   └── generate.py                chunks + question -> grounded answer
├── venv/
├── SPECIFICATION.md               what is being built and why
├── CONCEPTS.md                    how each mechanism works
└── BUILD_LOG.md                   this file
```

## Configuration in force

| Setting | Value |
|---|---|
| `EMBED_MODEL` | `nomic-embed-text` (768 dimensions) |
| `CHAT_MODEL` | `llama3.2:3b` |
| `CHUNK_SIZE` | 400 words |
| `CHUNK_OVERLAP` | 50 words |
| `TOP_K` | 3 |
| `num_ctx` | 4096 tokens |
| `temperature` | 0 |
| Distance metric | cosine (`hnsw:space`) |

## Measurements so far

| Metric | Value |
|---|---|
| Source document | 11 pages, 17,141 characters, ~2,541 words |
| Chunks produced | 8 (min 91, max 400, avg 361 words) |
| Vectors stored | 8 × 768 dimensions |
| Index size on disk | 736 KB |
| Embedding sanity check | 0.8180 related vs 0.3704 unrelated |
| Models resident in RAM | 4.3 GB combined |

## Problems encountered and resolved

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | `pull model manifest: file does not exist` | Typo `nomic-embed-textx` | Exact model name; Ollama offers no suggestions |
| 2 | PowerShell security warning from `curl` | `curl` is an alias for `Invoke-WebRequest` | Use `curl.exe` |
| 3 | `ModuleNotFoundError: No module named 'fitz'` | Virtual environment not active | `venv\Scripts\activate`; check for the `(venv)` prefix |
| 4 | Filename split into several arguments | Spaces in the path | Quote the path |
| 5 | `highperformance` in extracted text | Hyphen-rejoin rule cannot tell soft from real hyphens | Accepted trade-off; logged for Day 5 |
| 6 | HTTP 500 from `/api/embeddings` | Memory pressure; body hidden by `raise_for_status()` | Surface the response body in the error |
| 7 | Connection dropped during `/api/chat` | Ollama killed by memory pressure — 4.3 GB of models on 8 GB | Explicit `num_ctx`, longer timeout, `ollama stop --all`, close browser |

## Verification performed

- Per-page character counts checked — no page returned 0, so no scanned images
- First, middle and last chunks read manually for garbling
- Chunk word-count statistics checked against expected sliding-window arithmetic
- ChromaDB inspected directly via SQLite rather than trusting the script's output
- Embedding similarity verified numerically against a known-unrelated control
- Duplicate-prevention confirmed: second indexing pass left the count at 8, not 16
- Grounding tested with a question deliberately absent from the document

---

## Day 5 — Debugging retrieval quality

**Goal:** replace guessed settings with measured ones, and document at least one real failure and its fix.

Every parameter in the system — 400-word chunks, 50-word overlap, k=3 — was a reasonable-sounding guess. None has been justified by evidence. Day 5 fixes that, and is the difference between a tutorial follow-along and a project worth discussing.

### Step 5.1 — Make the parameters variable

Small refactors so settings can be swept without editing source:

| Function | Change |
|---|---|
| `load_and_chunk()` | accepts `chunk_size`, `overlap`, `verbose` |
| `extract_text()` | accepts `verbose` (sweeps are noisy otherwise) |
| `get_collection()` | accepts a collection `name` |
| `index_pdf()` | accepts `collection_name`, `chunk_size`, `overlap` |
| `retrieve()` | accepts `collection_name` |

Defaults are unchanged, so existing commands behave exactly as before. The point is that alternative configurations can now be built into *separate* collections and compared side by side, without destroying the main index.

### Step 5.2 — Create `questions.json`

A fixed test set of 8 questions. Fixed matters: a set that changes between runs makes runs incomparable, and comparison is the entire method of Day 5.

Each entry records what it is testing and what a correct answer must contain:

| ID | Tests |
|---|---|
| q1 | Vocabulary mismatch — "supercomputer" vs the document's "cluster" |
| q2 | Exact numeric fact (1,272 nodes) — must not drift |
| q3 | Detail mid-chunk rather than near a heading |
| q4 | Retrieval by position in the document rather than topic |
| q5 | Fact stated once, in passing |
| q6 | Content in the short final chunk |
| q7 | **Refusal** — trivially answerable from training data, absent from document |
| q8 | **Refusal, harder** — plausible for the document to contain; retrieval will return related-looking chunks |

q8 is the more demanding refusal test. In q7 the retrieved chunks are obviously unrelated to Japan. In q8 the retriever returns genuine internship content that simply does not mention pay — so the model must recognise that *related* context is not the *same* as context containing the answer.

### Step 5.3 — Create `src/evaluate.py`

Runs the full pipeline over every question and writes a timestamped Markdown report to `results/`.

Each question records: status, retrieved chunk indices, similarity scores, elapsed time, the full answer, and a 150-character preview of every retrieved chunk.

**Why saved reports rather than reading the terminal.** Impressions drift. After changing chunk size you will not reliably remember what the previous answer looked like, and self-deception is easy when you want a change to have helped. Two files in `results/` can be diffed. A memory cannot.

**Automatic grading, and its honest limits.** `grade()` returns PASS / FAIL / CHECK:

- unanswerable question → must refuse
- answerable question → must contain an expected term and must not refuse
- **CHECK** → the rule could not decide; a human must read it

The refusal detector matches several phrasings because small models paraphrase rather than reproducing the exact instructed sentence. This is deliberately crude, and the crudeness is the point: **automatic scoring is a triage tool that says where to look, not a verdict.** A keyword match cannot tell a correct answer from one that mentions the right word in a wrong sentence. The reports exist to be read.

### Step 5.4 — Create `src/experiment.py`

Sweeps chunk size, overlap and k — measuring **retrieval only, with no language model involved**.

Two reasons:

1. **Speed.** Generation costs 30+ seconds per answer on this hardware. Retrieval costs milliseconds. A sweep that would take an hour takes seconds.
2. **Attribution.** If chunk size changes and the answer gets worse, was that the chunking, or the model? Removing generation removes a variable, so the result is attributable to the one thing that changed.

This directly encodes Day 5's central lesson: most bad RAG answers are retrieval failures. Measure the retriever first.

Configurations compared:

| Config | Purpose |
|---|---|
| 200 / 25 | Small chunks — precise, may lack context |
| 400 / 50 | Current setting — the baseline |
| 600 / 75 | Large chunks — more context, diluted meaning |
| **400 / 0** | **Same size, no overlap — isolates overlap's contribution** |

The fourth is a controlled comparison: identical to the baseline in every respect except overlap, so any difference is attributable to overlap alone. Changing two things at once and observing an improvement tells you nothing about which one caused it.

The k sweep reports best, worst, and spread across retrieved chunks. **The number that matters is the worst similarity**, not the best. A large spread means the extra chunks are padding the prompt with material the retriever itself rates as barely relevant — consuming context window and risking distraction of a small model.

### Step 5.5 — Test plan

```powershell
python src\evaluate.py "baseline"
python src\experiment.py "data\Impression des comptes rendus du dépôt de stage.pdf"
```

Then change one setting in `config.py`, re-run `evaluate.py` with a label, and diff the two reports.

### Status

| Check | Status |
|---|---|
| Parameters made variable | Done |
| `questions.json` created | Done |
| `evaluate.py` created | Done |
| `experiment.py` created | Done |
| Baseline evaluation run | Done |
| Config sweep run | Done |
| A messy PDF tested | Pending |
| One failure found, fixed, and measured | Pending |

### Step 5.6 — Baseline results

`python src\evaluate.py "baseline"` → `results/run_20260727_094021.md`

Config: `chunk=400 overlap=50 k=3 embed=nomic-embed-text chat=llama3.2:3b`

| ID | Status | Chunks retrieved | Top similarity | Time |
|---|---|---|---|---|
| q1 | PASS | 4, 2, 3 | 0.6433 | 70.1s |
| q2 | PASS | 4, 2, 6 | 0.6028 | 27.8s |
| q3 | PASS | 4, 2, 3 | 0.5999 | 30.3s |
| q4 | PASS | 1, 0, 2 | 0.4923 | 102.6s |
| q5 | PASS | 4, 2, 5 | 0.5964 | 34.3s |
| q6 | PASS | 7, 6, 1 | 0.5821 | 60.7s |
| q7 | PASS (refused) | 0, 4, 1 | 0.3641 | 68.2s |
| q8 | PASS (refused) | 7, 0, 4 | 0.4291 | 79.7s |

**PASS 8 · FAIL 0 · CHECK 0**

Answer quality is high. Every answer quotes the source text directly rather than paraphrasing, and both refusals produced the exact instructed sentence. Grounding works.

---

### Step 5.7 — What the results actually show

A perfect score is where analysis begins, not ends. Four findings, in order of importance.

#### Finding 1 — q4 is a retrieval failure that passed anyway

This is the most valuable result in the run, and the summary table conceals it.

q4 ("What happened on the first day of the internship?") retrieved chunks **[1, 0, 2]** in that order. The answer came from the passage *"I was warmly welcomed by the team… get familiar with the work environment"* — which is in **chunk 0**.

Chunk 0 was ranked **second**, at 0.4798. The top-ranked chunk 1 (0.4923) does not contain the answer at all; it is about managing computing resources.

**The retriever ranked an irrelevant chunk above the correct one, and the system still produced a correct answer — purely because k=3 was wide enough to sweep the right chunk in as a passenger.**

The measurable consequence: **at k=1, q4 would fail.** The current configuration is not succeeding because retrieval is good; it is succeeding because retrieval is *wide*. That is a fragile kind of correct, and it would not survive a longer document where the true chunk falls to rank 4 or 5.

Why the retriever got it wrong is instructive. The question asks about *position* in the document ("first day"), not *topic*. Embeddings encode meaning, not sequence. Nothing in a chunk's vector says "this is the beginning." Semantic search is structurally weak on positional and temporal questions — a known limitation, now demonstrated with evidence rather than asserted.

#### Finding 2 — chunk 2 is a hub, crowding out useful results

Look down the "chunks retrieved" column:

| Chunk | Appears in |
|---|---|
| **2** | q1, q2, q3, q4, q5 — **rank 2 in five of six answerable questions** |
| 4 | q1, q2, q3, q5 (rank 1 in all four) |

Chunk 2 begins *"system security and multi-user management in shared environments like HPC clusters. 19 juin 2026 I explored how software is managed in Linux systems…"* — general HPC vocabulary with no specific facts.

It is a **hub chunk**: dense in topic-typical vocabulary, so it sits near the centre of the document's region of embedding space and scores moderately well against *any* HPC question, while answering none of them.

Its cost is a permanently occupied slot. With k=3, one of three slots is spent on chunk 2 in five of six questions — so the system is effectively operating at k=2 while paying the context-window and latency cost of k=3.

This is why the *worst* similarity in a result set matters more than the best, and it is exactly what `experiment.py`'s k sweep is designed to expose.

#### Finding 3 — similarity scores are low, compressed, and not thresholdable

The score guidance printed by `retrieve.py` ("> 0.75 strong match") was a guess, and the data shows it was wrong for this document and model. **No correct answer scored above 0.65.**

| Category | Range |
|---|---|
| Correct answers | 0.4923 – 0.6433 |
| Correct refusals | 0.3641 – 0.4291 |

The dangerous pair: q4's *correct* top hit scored **0.4923**, while q8's *irrelevant* top hit scored **0.4291**. A gap of 0.06.

**Consequence:** no fixed similarity threshold can separate "the answer is here" from "the answer is absent." A rule such as *refuse if top score < 0.5* would have wrongly refused q4. Absolute cosine values are not calibrated across documents or models — only *relative ordering within one query* is meaningful. This is a common misconception and now a measured fact.

Note also that q7 — a question about Japan, against a document containing nothing about Japan — still scored 0.3641, not near zero. All natural language shares structure, so embedding similarity has a high floor.

#### Finding 4 — the test suite cannot currently fail

8/8 with 0 CHECK is a warning about the *instrument*, not a triumph of the system.

The grader accepts an answer containing any expected term. For q4 the expected terms are `welcomed, team, environment` — words so likely in any answer about a first day at work that the check is close to unfalsifiable.

**A suite nothing fails cannot detect a regression.** If chunk size is changed tomorrow and retrieval degrades, this suite will still report 8/8 and the sweep will appear to show that the setting does not matter.

The fix is not more questions but *harder* ones — questions where a plausible-sounding wrong answer is possible, and expected terms specific enough that a near-miss fails. Deferred to step 5.8.

#### Secondary observations

**Prompt leakage.** q2 answered *"According to chunk 4 of the document…"* The `[Source N - file, chunk N]` labels exist for debugging, and the model is repeating internal structure back to the user. "Chunk 4" is meaningless to a reader. The labels should either be hidden from the model or the prompt should specify citing the document, not the chunk index.

**Latency variance: 27.8s to 102.6s.** Part is the first-call model load (q1, 70.1s). The rest tracks answer length and CPU contention. Median around 60s is the honest figure to quote for this hardware, and a reason the sweep in `experiment.py` measures retrieval only.

---

### Step 5.8 — Configuration sweep results

`python src\experiment.py "data\Impression des comptes rendus du dépôt de stage.pdf"`

**Warning on labels:** the sweep's columns are labelled q1–q4 but these are *not* the q1–q4 of `questions.json`. The sweep uses its own four questions. Mapping:

| Sweep column | Question |
|---|---|
| q1 | What supercomputer did you work on…? (= questions.json q1) |
| q2 | How many standard compute nodes…? (= questions.json q2) |
| q3 | What GPUs are used…? (= questions.json q3) |
| q4 | **Where is the cluster located?** (= questions.json **q5**) |

#### Chunk size and overlap (k=3, top-1 similarity)

| Config | Chunks produced | q1 | q2 | q3 | q4 |
|---|---|---|---|---|---|
| **200 / 25** | **15** | **0.680** | **0.634** | **0.624** | **0.614** |
| 400 / 50 *(baseline)* | 8 | 0.643 | 0.603 | 0.600 | 0.596 |
| 600 / 75 | 5 | 0.637 | 0.629 | 0.612 | 0.590 |
| 400 / 0 *(no overlap)* | 7 | 0.618 | 0.600 | 0.594 | 0.577 |

#### Result 1 — overlap helps, on every single question

The controlled comparison. **400/50 versus 400/0** differ in exactly one variable, so any difference is attributable to overlap alone:

| Question | 400/50 | 400/0 | Difference |
|---|---|---|---|
| q1 | 0.643 | 0.618 | +0.025 |
| q2 | 0.603 | 0.600 | +0.003 |
| q3 | 0.600 | 0.594 | +0.006 |
| q4 | 0.596 | 0.577 | +0.019 |

Overlap wins **4 out of 4**. The effect is small (+0.003 to +0.025) but the *direction is perfectly consistent*, which is the stronger evidence — a large effect on one question could be chance, while a uniform effect across four independent questions is not.

**The mechanism is not the one usually cited.** Overlap is normally justified as preventing a passage from being split across a boundary. That is real, but it does not explain a rise in *top-1* score. The actual mechanism: overlap creates **more candidate windows over the same text** (8 windows instead of 7). More windows means a better chance that some window is well-*centred* on the relevant material rather than having it near an edge diluted by unrelated content. Overlap buys better framing, not just boundary insurance.

Note also that removing overlap produced **7 chunks instead of 8** — less total coverage of the same document.

#### Result 2 — smaller chunks score higher, on every question

200/25 beats every other configuration on all four questions, by a clear margin (+0.018 to +0.037 over baseline).

This is the dilution effect predicted in `CONCEPTS.md`, now measured. A vector is roughly an average of its input's meaning. A 200-word chunk about cluster architecture is *concentrated*; a 400-word chunk containing that same material plus a paragraph about Linux modules is a blend, and the blend sits further from a question about architecture.

**This does not yet justify changing the setting.** The sweep measures *retrieval concentration*, not answer quality. Smaller chunks carry less surrounding context, and the generation model may have too little to work with — a chunk that scores 0.68 but omits the sentence needed to make sense of the fact is worse in practice than one scoring 0.60 that contains it. **Retrieval score is a proxy, and optimising a proxy without checking the real outcome is a classic way to make a system worse while the metrics improve.**

Required before adopting: a full `evaluate.py` run at 200/25 compared against the baseline report.

#### Result 3 — chunk count reveals that 600/75 is barely retrieving

The most consequential number in the table is not a similarity score. It is the chunk count.

| Config | Chunks | Fraction of document sent to the model at k=3 |
|---|---|---|
| 200 / 25 | 15 | 20% |
| 400 / 50 | 8 | 38% |
| 600 / 75 | **5** | **60%** |

At 600/75 the system retrieves **three of five chunks — 60% of the entire document** — and hands it to the model. That is not retrieval; it is document stuffing with extra steps. Its respectable similarity scores are close to meaningless, because with five chunks the top-3 will inevitably include the right one whether or not the retriever is working.

**Consequence: k must scale with corpus size.** A fixed k=3 is a very different operation over 5 chunks than over 5,000. On this small document the pipeline is barely being tested. This is a limitation to state explicitly in the README rather than let a reader assume the retriever has been validated at scale.

#### Result 4 — the k sweep shows where the noise floor begins

| Question | k=1 | k=3 worst | k=5 worst | k=5 spread |
|---|---|---|---|---|
| supercomputer | 0.643 | 0.613 | 0.577 | 0.067 |
| compute nodes | 0.603 | 0.557 | 0.530 | 0.073 |
| GPUs | 0.600 | 0.524 | 0.486 | 0.114 |
| located | 0.596 | 0.548 | 0.485 | 0.111 |

Chunk 4 ranks first for all four questions — as expected, it holds the architecture description.

The revealing comparison is against the **Day 5 baseline refusal scores**. Correct refusals had top hits at 0.3641 and 0.4291, and q4's *correct* answer had a top hit at 0.4923.

At k=5, the fifth chunk scores **0.485–0.577**. For the GPU question that fifth chunk scores **0.486 — below the score of a chunk that genuinely contained an answer elsewhere in the suite.** So the extra chunks at k=5 sit at similarity levels this system has already associated with "nothing relevant here."

**Conclusion: k=5 pads the prompt with material the retriever itself rates as noise**, spending context window and adding distraction for a 3B model, with no evidence of benefit. k=3 remains defensible. k=1 is untested for the case that matters — see the gap below.

#### Gap in the experiment — the interesting question was never swept

The sweep's four questions are all *topical* and all answered by the same chunk 4. The Day 5 baseline identified the genuinely fragile case as **q4 of `questions.json`, "What happened on the first day of the internship?"** — where the correct chunk ranked *second* and would be missed at k=1.

That question is absent from `experiment.py`'s list. The hypothesis "k=1 breaks the positional question" therefore remains **untested**, and the sweep as run cannot distinguish k=1 from k=3 on any question, because chunk 4 ranks first in all of them.

This is a design error in the experiment: **it sampled only the easy cases.** A sweep over questions the system already handles well will show that configuration does not matter, regardless of whether it does. Fixed in step 5.9.

---

### Step 5.9 — Fixing the experiment

Two changes to `experiment.py`:

1. **Added the fragile questions.** The positional question ("first day"), where the correct chunk ranked second, plus the end-of-document Slurm question. These are the only ones in the set capable of distinguishing k=1 from k=3.
2. **Relabelled columns s1–s5.** The sweep previously printed q1–q4 while `questions.json` uses q1–q8 for a different set. Identical labels for different questions is a reliable way to misread your own results a week later. The questions are now printed above the table.

**Re-run required:**

```powershell
python src\experiment.py "data\Impression des comptes rendus du dépôt de stage.pdf"
```

**What to look for in the new k sweep, specifically on s4 (the positional question):** if chunk 0 is absent at k=1 and present at k=3, that converts the Day 5 baseline observation into a *measured, reproducible failure with a known cause and a known fix* — which is materially more valuable than eight passing tests.

### Decisions taken from the sweep

| Setting | Decision | Evidence |
|---|---|---|
| Overlap | **Keep.** 50 words retained | Beat no-overlap on 4/4 questions; consistent direction |
| `TOP_K` | **Keep 3.** Do not raise to 5 | k=5's fifth chunk scores at the noise floor (0.486–0.577) |
| `CHUNK_SIZE` | **Not yet changed.** 200/25 is a candidate | Wins retrieval on 4/4, but retrieval score is a proxy — needs a full `evaluate.py` run before adoption |
| 600/75 | **Rejected** | Only 5 chunks; k=3 returns 60% of the document, so it is not really retrieving |

---

### Step 5.10 — Re-run with fragile questions: the failure is confirmed

```
s1 = What supercomputer did you work on during the internship?
s2 = How many standard compute nodes does the cluster have?
s3 = What GPUs are used in the cluster?
s4 = What happened on the first day of the internship?      (positional)
s5 = What was the purpose of the Slurm script?              (end of document)
```

| Config | Chunks | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| 200 / 25 | 15 | **0.680** | **0.634** | **0.624** | **0.517** | 0.582 |
| 400 / 50 *(baseline)* | 8 | 0.643 | 0.603 | 0.600 | 0.492 | 0.582 |
| 600 / 75 | 5 | 0.637 | 0.629 | 0.612 | 0.479 | 0.489 |
| 400 / 0 | 7 | 0.618 | 0.600 | 0.594 | 0.480 | 0.536 |

#### The headline result — k=1 breaks the positional question

```
s4  "What happened on the first day of the internship?"
  k=1: chunks [1]        best 0.492
  k=3: chunks [1, 0, 2]  best 0.492  worst 0.450
```

**At k=1 the system retrieves only chunk 1, which does not contain the answer.** The correct passage — chunk 0, holding *"On my first day at ReDX… I was warmly welcomed by the team"* — appears only at rank 2, and is therefore invisible at k=1.

Every other question retrieves its correct chunk at rank 1 and would survive k=1 comfortably. This single question is the only one in the suite that discriminates between the two settings.

**This converts the Day 5 baseline observation into a reproducible, isolated failure:**

| | |
|---|---|
| **Symptom** | Correct answer to a positional question depends on k ≥ 2 |
| **Cause** | Embeddings encode meaning, not sequence. No vector contains the information "this text is the start of the document." The question's key term — *first day* — has no semantic counterpart to match on |
| **Current mitigation** | k=3 sweeps the correct chunk in as a passenger |
| **Why the mitigation is fragile** | It relies on the correct chunk landing within the top 3 by accident. On a longer document it would fall further and disappear |

#### Overlap now wins 5 out of 5

With s5 added, the controlled 400/50 vs 400/0 comparison strengthens:

| Question | 400/50 | 400/0 | Difference |
|---|---|---|---|
| s1 | 0.643 | 0.618 | +0.025 |
| s2 | 0.603 | 0.600 | +0.003 |
| s3 | 0.600 | 0.594 | +0.006 |
| s4 | 0.492 | 0.480 | +0.012 |
| **s5** | **0.582** | **0.536** | **+0.046** |

The largest benefit falls on **s5, the end-of-document question** — which is exactly where the theory predicts it should. The Slurm material sits in the final, shortest chunk, and without overlap the tail of a document is precisely where a passage is most likely to be awkwardly framed by an arbitrary cut. Overlap gives the tail a second, better-centred window.

Theory predicted the *location* of the largest effect before the measurement was taken. That is a considerably stronger result than the aggregate.

#### Positional questions are structurally harder — quantified

s4 scores lowest in **every** configuration:

| Config | Topical average (s1–s3) | Positional (s4) | Penalty |
|---|---|---|---|
| 200/25 | 0.646 | 0.517 | −0.129 |
| 400/50 | 0.615 | 0.492 | −0.123 |
| 600/75 | 0.626 | 0.479 | −0.147 |
| 400/0 | 0.604 | 0.480 | −0.124 |

A consistent penalty of roughly **0.12–0.15** for asking about *when* something happened rather than *what* it was about. Chunk size does not fix this — it is a property of what embeddings represent, not of how the text was cut.

Compare against the refusal baselines: q8's *unanswerable* question scored 0.4291 at top-1. s4 — a question the document **does** answer — scores 0.492 at baseline, only 0.06 above it. **The hardest legitimate question is nearly indistinguishable, by score alone, from an unanswerable one.** This is the strongest confirmation yet that similarity thresholds cannot be used to decide whether an answer exists.

#### 600/75 collapses on end-of-document content

s5 falls to **0.489** at 600/75, down from 0.582 at baseline — the largest single degradation in the table. With only 5 chunks, the Slurm material at the document's end is absorbed into a large chunk dominated by unrelated earlier content, and the resulting vector no longer points anywhere near a question about Slurm.

Combined with the earlier finding that k=3 over 5 chunks returns 60% of the document, 600/75 is now rejected on two independent grounds.

#### 200/25 does not fix the ranking problem

200/25 raises s4 from 0.492 to 0.517 — the score improves, but **the sweep reports only the top-1 similarity, not which chunk it is.** Whether the correct chunk actually *ranks first* at 200/25 is unknown from this data.

This is a limitation of the instrument. A higher score on the wrong chunk is not an improvement, and the current sweep cannot tell the two apart. Reporting retrieved chunk indices alongside scores would close the gap.

### Decisions after the re-run

| Setting | Decision | Evidence |
|---|---|---|
| Overlap | **Keep 50.** Confirmed | Wins 5/5; largest gain exactly where theory predicts |
| `TOP_K` | **Keep 3.** k=1 rejected outright | k=1 demonstrably fails s4; k=5 adds chunks at the noise floor (0.432–0.486) |
| 600/75 | **Rejected** | Collapses on tail content; barely retrieves at 5 chunks |
| 200/25 | **Still a candidate** | Wins 4/5, ties 1/5 — but scores alone cannot confirm correct ranking |

### Remaining Day 5 work

| Task | Status |
|---|---|
| Re-run sweep with fragile questions included | Done |
| Report chunk indices in the chunking sweep, not just scores | Done |
| Fix the positional-retrieval failure and measure the effect | Code written, measurement pending |

---

### Step 5.11 — Fixing the instrument

The chunking sweep reported only similarity scores, so it could not distinguish "higher score on the right chunk" from "higher score on the wrong chunk" — the exact blind spot that let the positional failure go unnoticed in the first run.

Cells now read `score#chunk_index`.

**Caution recorded in the output:** chunk indices are *not* comparable across rows. Each configuration cuts the document differently, so chunk 3 of the 200-word index is different text from chunk 3 of the 400-word index. What is comparable is whether a configuration keeps returning the *same* chunk for a question, and whether that chunk actually holds the answer — which is confirmed by reading it, not by reading a score.

---

### Step 5.12 — The fix: contextual chunk headers

**Cause being addressed:** embeddings encode meaning, not sequence. No chunk's text states "this is the start of the document," so the question term *first day* had nothing to match against.

**Approach:** put the missing information into the text itself, so it reaches the vector. Each chunk gains a header before embedding:

```
[Source: Impression des comptes rendus.pdf | Part 1 of 8 | This is the
BEGINNING of the document: the first day, the start, the earliest entries |
Dates covered: 8 juin 2026 to 12 juin 2026]
```

Three pieces of information, each targeting a class of question:

| Header element | Question type it serves |
|---|---|
| `Part i of n` | "what came first / last / in the middle" |
| `BEGINNING / MIDDLE / END` + synonyms | "first day", "at the start", "by the end" |
| `Dates covered: ...` | "what happened in June", "on 8 juin" |

The synonym phrasing is deliberate. The header is not documentation for a human — it is **text that will be embedded**, so it should contain the words a user is likely to ask with. "the first day, the start, the earliest entries" exists so that a question containing *first day* has a close semantic match.

**Implementation:**

| File | Change |
|---|---|
| `ingest.py` | `extract_dates()` (French + English month names), `add_context_headers()` |
| `config.py` | `USE_CONTEXT_HEADERS = True` — a switch, so the effect can be measured both ways |
| `embed.py` | `index_pdf(..., headers=...)` applies headers before embedding |
| `experiment.py` | `sweep_headers()` — controlled A/B |

**The trade-off, stated plainly.** The header is stored as part of the chunk, so the generation model sees it and may occasionally quote it. That is the cost of the vector carrying positional meaning. Whether the retrieval gain justifies it is a measurement, not an assumption — which is what `sweep_headers()` is for.

**The A/B design.** `sweep_headers()` builds two collections from the same PDF at the same 400/50 chunking and queries both. Headers are the only variable, so any difference is attributable to them alone.

**The verdict criterion is deliberately not "did scores go up".** The output flags `RANK 1 CHANGED` separately from a score delta, because the fix was made to change *which chunk ranks first* on the positional question. A higher score on the same wrong chunk is not a fix. Scores rising across all questions would in fact be mildly suspicious — headers add shared boilerplate to every chunk, which can inflate similarity generally without improving discrimination.

**Measurement pending:**

```powershell
python src\experiment.py "data\Impression des comptes rendus du dépôt de stage.pdf"
```

Success is defined in advance, before seeing the result: **on "What happened on the first day of the internship?", rank 1 must change from chunk 1 to chunk 0.** Any other outcome — including higher scores everywhere — means the fix did not do what it was designed to do.

---

### Step 5.13 — Result: the fix works, and the cost is measurable

#### The A/B test

| Question | Headers OFF | Headers ON | Verdict |
|---|---|---|---|
| What supercomputer did you work on…? | 0.643 **#4** | 0.622 **#3** | RANK 1 CHANGED |
| How many standard compute nodes…? | 0.603 #4 | 0.576 #4 | −0.026 |
| What GPUs are used…? | 0.600 #4 | 0.559 #4 | −0.041 |
| **What happened on the first day…?** | **0.492 #1** | **0.561 #0** | **RANK 1 CHANGED** |
| What was the purpose of the Slurm script? | 0.582 #7 | 0.555 #7 | −0.027 |

**The success criterion, set before the measurement, was met exactly.** The positional question's rank-1 chunk moved from chunk 1 (which does not contain the answer) to **chunk 0** (which does). It is also the only question whose score *rose*, by +0.069.

Confirmed at k=1 in the top-k sweep:

```
What happened on the first day of the internship?
  k=1: chunks [0]        <- was [1] before the fix
```

**The failure identified in step 5.10 is eliminated.** The system no longer depends on k ≥ 2 to answer positional questions correctly.

#### The fix is robust across chunk sizes

In the chunking sweep, the positional question returns **chunk 0 at rank 1 in all four configurations**:

| Config | s4 result |
|---|---|
| 200/25 | 0.600 **#0** |
| 400/50 | 0.561 **#0** |
| 600/75 | 0.563 **#0** |
| 400/0 | 0.565 **#0** |

The fix does not depend on a particular chunk size — appropriate, since it addresses a property of *what embeddings represent*, not of how text is cut.

#### The cost, measured rather than assumed

Every topical question lost score: −0.026, −0.041, −0.027.

**Cause:** the header adds roughly 30 words of boilerplate to every chunk. That is the same dilution effect measured in the chunk-size sweep — a vector is approximately an average of its input, so padding every chunk with non-topical text pulls each one slightly away from topical questions.

Note the direction. Step 5.12 recorded, in advance, that scores rising across all questions would be *suspicious* — shared boilerplate can inflate similarity without improving discrimination. Instead scores **fell** everywhere except the target question. That is the honest signature of a targeted fix: it pays a uniform small cost to buy a large specific gain.

**The trade in one line:** −0.03 average on topical questions, in exchange for +0.069 and a corrected rank-1 on positional questions. Worth it here, because a wrong rank-1 is a *failure* while −0.03 on a question that already ranks correctly is merely a smaller margin.

#### Investigating the second rank change — not a regression

The supercomputer question also changed rank 1, from chunk 4 to chunk 3. This looked like a possible regression, since chunk 4 is the passage holding the Toubkal description.

**Verified by reading the chunks rather than reasoning about the scores.** Chunk 3's final words are:

> "…30 juin 2026 As part of my training, I was introduced to the Toubkal HPC cluster, a high-performance computing platform located at Mohammed VI Polytechnic University. Toubkal is one of the most powerful supercomputers in Africa and, according to the latest TOP500 ranking, is ranked 356th worldwide…"

Chunk 3 **does** contain the answer. That text is the **50-word overlap region** it shares with chunk 4 — so the passage exists in both chunks by design.

**Verdict: not a regression.** Both chunks answer the question, and chunk 4 remains at rank 2 anyway.

This is also the clearest demonstration yet of what overlap actually buys. The Toubkal introduction sits at a chunk boundary. Without overlap it would exist in exactly one chunk and be vulnerable to a single ranking mistake. With overlap it exists in two, and the retriever can find it either way. The earlier sweep showed overlap helping by +0.003 to +0.046; this shows *why*.

One residual caveat: chunk 3 contains only the truncated introduction, while chunk 4 holds the full architecture description. For "what supercomputer," chunk 3 suffices. For a question needing the details, chunk 4 is the better source — and it is still retrieved at rank 2 under k=3.

#### Methodological note — the sweep table is not comparable to the previous run

`build_all()` calls `index_pdf()` without a `headers` argument, so it now inherits `USE_CONTEXT_HEADERS = True`. **Every configuration in this run's chunking sweep has headers applied.** The previous run had none.

Comparing the two tables directly would conflate two changes — the reason every score in the chunk-size table dropped by roughly 0.02–0.04 relative to the earlier run is the headers, not the chunking.

Within this run the comparisons remain valid, because all four rows share the same header setting. But this is exactly the trap the 400/0 control was designed to avoid, reappearing at the level of the whole experiment: **when a default changes underneath a benchmark, previously recorded numbers silently stop being a baseline.** Results tables need to record every setting in force, not only the one being varied.

#### Updated findings on chunk size

With headers applied to all rows, 200/25 now wins **5 out of 5** (0.669, 0.615, 0.596, 0.600, 0.563), and 600/75 degrades further on the end-of-document question (0.429, down from 0.489). The case for smaller chunks strengthens — but the caution from step 5.8 stands unchanged: this measures retrieval, not answer quality, and adoption requires a full `evaluate.py` comparison.

### Day 5 summary

**One failure found, diagnosed, fixed, and the fix measured against a criterion set in advance.**

| Stage | Content |
|---|---|
| Observed | q4 passed, but its correct chunk ranked second (baseline run) |
| Isolated | Built a sweep including the fragile question; confirmed k=1 retrieves the wrong chunk |
| Diagnosed | Embeddings encode meaning, not sequence; "first day" had no semantic counterpart |
| Fixed | Contextual headers carrying source, position, and date span into the vector |
| Measured | Rank 1 moved to the correct chunk, +0.069, robust across all four chunk sizes |
| Costed | −0.03 average on topical questions, from boilerplate dilution |
| Verified | Read the chunks directly to confirm the second rank change was benign |

---

### Step 5.14 — Incident: measuring the wrong index

`evaluate.py` was run after the header fix, expecting to show the improvement end to end. It returned results **identical to the pre-fix baseline, to four decimal places** — q4 retrieving `[1, 0, 2]` at 0.4923, exactly as before.

**Cause.** `experiment.py` writes only to its own collections: `exp_c200_o25`, `exp_c400_o50`, `exp_headers`, `exp_noheaders`. It never touches the default `documents` collection that `retrieve.py` and therefore `evaluate.py` read from. Enabling `USE_CONTEXT_HEADERS` changes how *future* indexing behaves; it does not retroactively alter vectors already stored on disk.

**The trap:** the configuration file said headers were on, and they were on — for anything indexed from that moment forward. The live index was built on Day 3 and had not been rebuilt. **A config flag describes intent; the database holds state, and the two drift apart silently.**

**Fix:** re-index the main collection, then re-run the evaluation.

```powershell
python src\embed.py "data\Impression des comptes rendus du dépôt de stage.pdf"
python src\evaluate.py "headers on"
```

The `-> context headers added` line in the indexing output is the confirmation that the stored vectors actually changed.

**Generalisation worth keeping.** Any system with a persistent index has two sources of truth — the configuration and the stored data — and only one of them updates when a setting is edited. Every embedding-affecting parameter (`EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `USE_CONTEXT_HEADERS`) requires a rebuild to take effect. A future improvement would be to record these settings in the collection's metadata at index time and warn on mismatch at query time.

#### Two useful observations from the failed run

**Retrieval is fully deterministic.** Identical similarity scores to four decimal places across runs hours apart. This is what makes before/after comparison meaningful at all: any difference between two reports is attributable to a change made, never to run-to-run variance in the retriever.

**Latency is environmental noise, not a system property.** q1 took 70.1s in the baseline and 185.3s here, with byte-identical retrieval and the same prompt. The variation is model loading, memory pressure and CPU contention. Latency figures from this hardware should be quoted as a wide range, never as a measurement of the system.

---

### Step 5.15 — End-to-end comparison: baseline vs headers

`results/run_20260727_094021.md` (baseline) vs `results/run_20260727_103905.md` (headers on).
Same 8 questions, same chunking, same models, same k. Headers are the only variable.

| ID | Baseline chunks | Baseline top-1 | Headers chunks | Headers top-1 | Change |
|---|---|---|---|---|---|
| q1 | 4, 2, 3 | 0.6433 | **3**, 4, 2 | 0.6224 | rank 1 changed (benign — verified 5.13) |
| q2 | 4, 2, 6 | 0.6028 | 4, **6, 2** | 0.5765 | hub chunk demoted to rank 3 |
| q3 | 4, 2, 3 | 0.5999 | 4, 2, 3 | 0.5592 | unchanged |
| **q4** | **1, 0, 2** | **0.4923** | **0, 1, 2** | **0.5613** | **FIXED — correct chunk to rank 1** |
| q5 | 4, 2, 5 | 0.5964 | 4, **5, 6** | 0.5509 | hub chunk eliminated |
| q6 | 7, 6, 1 | 0.5821 | 7, 6, 1 | 0.5552 | unchanged |
| q7 | 0, 4, 1 | 0.3641 | 0, 1, 4 | 0.3576 | refusal, score down |
| q8 | 7, 0, 4 | 0.4291 | 0, 4, 7 | 0.4770 | refusal, score **up** |

**PASS 8 · FAIL 0 · CHECK 0 in both runs.**

#### The fix holds end to end

q4 retrieves chunk 0 at rank 1 through the full pipeline, not merely in the isolated sweep. The retrieval improvement survived contact with the generation model.

**It also improved the answer itself.** The automatic grader reports:

```
baseline:    PASS (found welcomed, team, environment)
headers on:  PASS (found ReDX, welcomed, team, environment)
```

The answer now contains **ReDX**, the company name — present in chunk 0 and absent from chunk 1. The baseline answer was assembled from a chunk that lacked it. This is a small but concrete demonstration that a better rank-1 produces a materially better answer, not just a better number.

#### Unexpected benefit — the hub chunk was partially dissolved

Step 5.7 identified chunk 2 as a hub: generic HPC vocabulary, rank 2 in five of six answerable questions, answering none of them.

| | Baseline | Headers on |
|---|---|---|
| Questions where chunk 2 appears | q1, q2, q3, q4, q5 (5) | q1, q3, q4 (3) |
| q2 | 4, **2**, 6 | 4, 6, **2** — demoted to rank 3 |
| q5 | 4, **2**, 5 | 4, 5, 6 — **eliminated** |

**Why this happened, in hindsight.** A hub chunk ranks well everywhere because it is *undifferentiated* — generic topic vocabulary and nothing that distinguishes it from its neighbours. Headers inject distinguishing content into every chunk: position, and the specific dates it covers. That gives previously-similar chunks distinct coordinates, so the hub loses its unearned advantage.

q5's result is the clearest gain: `4, 2, 5` became `4, 5, 6`, replacing a chunk that answers nothing with two that carry real content.

**This was not a designed effect.** The headers were built to fix positional queries. Hub dissolution is a side benefit, discovered by reading the results rather than predicted. It is recorded as an observation on a single document and a single hub — not a general claim.

#### Caveat — headers inflate scores on unanswerable questions

q8 ("What salary was paid for this internship?") rose from 0.4291 to **0.4770**, the only non-target question whose score increased.

**Cause:** the header contains `Source: Impression des comptes rendus du dépôt de stage.pdf`. *Stage* is French for internship. Every chunk now carries text matching the word "internship," so a question containing "internship" scores higher against *every* chunk — including for a fact the document does not contain.

**This is a real cost of the technique.** Header text is embedded like any other text, so whatever it contains becomes matchable everywhere. Filenames and metadata leak into the semantic space.

The separation between answerable and unanswerable did nonetheless improve slightly:

| | Baseline | Headers on |
|---|---|---|
| q4 (answerable, hardest) | 0.4923 | 0.5613 |
| q8 (unanswerable) | 0.4291 | 0.4770 |
| **Gap** | **0.063** | **0.084** |

Both refusals still behaved correctly, so no failure resulted. But the mechanism is worth remembering: **a header containing topic words raises the floor for every question on that topic.** A header of pure structure — position and dates, no filename — would likely avoid this. Untested.

#### Latency confirmed as noise

q1: 185.3s in the failed run, **28.8s** here, with near-identical work. Retrieval scores meanwhile reproduce to four decimals. Latency on this hardware is environmental and should never be quoted as a system property.

---

## Day 5 complete

**Outcome: one failure found, diagnosed, fixed, and the fix measured end to end against a criterion set in advance.**

| Decision | Setting | Evidence |
|---|---|---|
| Keep overlap at 50 | `CHUNK_OVERLAP = 50` | Beat no-overlap 5/5; largest gain on end-of-document content, as theory predicted |
| Keep k at 3 | `TOP_K = 3` | k=1 failed the positional question pre-fix; k=5 adds chunks at the noise floor |
| Adopt context headers | `USE_CONTEXT_HEADERS = True` | Fixed rank 1 on the target question, +0.069, robust across 4 chunk sizes, improved the answer text, partially dissolved the hub chunk |
| Reject 600/75 | — | Only 5 chunks; k=3 returns 60% of the document; collapses on tail content |
| 200/25 still open | — | Wins retrieval 5/5, but retrieval score is a proxy; needs its own end-to-end run |

**Costs accepted, and known:** −0.03 average on topical questions from boilerplate dilution; inflated scores on unanswerable questions where the filename shares vocabulary with the query.

### Carried forward

| Task | Why it matters |
|---|---|
| Harden the test suite so it can fail | 8/8 in both runs — the suite cannot currently detect a regression |
| End-to-end run at 200/25 | Wins every retrieval comparison; unknown whether smaller chunks starve the generator of context |
| Test a messy PDF (multi-column, tables) | The pipeline has only met one clean, single-column document |
| Fix `[Source N - chunk N]` prompt leakage | Model answered "According to chunk 4 of the document" — meaningless to a reader |
| Try a structure-only header (no filename) | Would likely remove the q8 inflation; untested |

---

## Days 6–7 — Streamlit UI and polish

**Goal:** something demonstrable to another person in under two minutes.

**No new pipeline logic.** `app.py` calls `retrieve`, `build_prompt` and `ask_model` — functions that already existed and were already measured. The UI is a face on working code, which is why it was scheduled last.

### Step 6.1 — Understanding Streamlit's execution model

The one concept that governs the design: **Streamlit re-executes the entire script, top to bottom, on every user interaction.** There are no event handlers. A widget is a function returning its current value on this run.

Two consequences follow directly:

| Consequence | Handled by |
|---|---|
| Ordinary variables do not survive a rerun | `st.session_state` |
| Expensive work would repeat on every rerun | `@st.cache_data` / `@st.cache_resource` |

The second is not a nicety here. Without caching, the app would re-extract, re-chunk and re-embed the entire PDF every time a key was pressed — the difference between usable and unusable on this hardware.

### Step 6.2 — Create `src/app.py`

**Caching by content hash.** `index_uploaded_pdf()` is keyed on the file's raw bytes, and the collection is named `upload_<sha256[:12]>`. Three properties fall out of this:

- the same file is indexed exactly once, however many reruns occur
- two files with the same *name* but different contents get separate indexes
- re-uploading an identical file reuses the existing index instantly

Hashing content rather than filename avoids a stale-index bug that would otherwise be near-impossible to diagnose from the UI.

**`st.stop()` for control flow.** With no Ollama, or no uploaded file, the script stops that rerun rather than nesting the remaining code inside conditionals. This suits the rerun model better than deep `if` nesting.

**The spinner is functional, not decorative.** Generation takes 30–100 seconds. Without visible feedback a user assumes the app has frozen and interacts again — triggering *another* rerun and stacking work on a machine with no memory to spare.

**The sidebar shows the live configuration.** Models, chunk size, overlap, k, header setting. A result that cannot be reproduced is not evidence, and these settings are exactly what a viewer needs in order to reproduce it.

**The sources panel is the most important element on the page**, and its justification is a measurement rather than an opinion. Day 5 recorded a case where a correct-looking answer was produced while the rank-1 chunk did not contain the answer. Anyone reading only the answer would never have known. Showing sources converts an assertion into something a reader can verify.

Its caption deliberately warns against threshold-reading of similarity scores, quoting the measured range (0.49–0.64 for correct answers) instead of inviting the assumption that higher is categorically better.

**Health check.** Ollama reachability is tested once per session and reported in the sidebar, so that failure mode produces a clear message rather than a stack trace.

### Step 6.3 — Supporting files

| File | Purpose |
|---|---|
| `README.md` | Architecture, usage, measured engineering results, limitations |
| `requirements.txt` | Four direct dependencies |
| `.gitignore` | Excludes `venv/`, `chroma_db/`, uploaded PDFs — **keeps `results/`** |

`chroma_db/` is excluded because it is rebuildable from the source documents and large. `uploads/` and `data/*.pdf` are excluded because the test document is a personal internship record.

`results/` is deliberately **kept**. Those reports are the project's evidence — the difference between claiming the system was tuned and being able to show it.

The README leads with the failure that was found and fixed rather than with a feature list. A project that documents a real bug, its diagnosis, its fix, and the measured cost of that fix is more credible than one reporting only successes.

### Status

| Check | Status |
|---|---|
| `app.py` created | Done |
| `README.md` written | Done |
| `requirements.txt`, `.gitignore` | Done |
| App run and tested in a browser | Pending |
| Error paths tested (no Ollama, no file, empty question) | Pending |

### Step 6.4 — Rebuild as a streaming chat interface

The first version worked but felt inert: a 30–100 second wait behind a static spinner. Rebuilt as a conversational interface with token streaming.

#### Streaming — the change that matters

`ask_model_stream()` added to `generate.py`. With `"stream": True`, Ollama returns newline-delimited JSON instead of one response — each line a fragment `{"message": {"content": "..."}}`, terminated by `{"done": true}`. The function yields fragments as they arrive.

`st.write_stream()` consumes that generator, renders each fragment as it appears, and returns the complete text for storage.

**Streaming does not make generation faster.** It makes the waiting *legible*. The first token typically arrives within a few seconds, so the user sees progress rather than a frozen screen. On hardware where a full answer takes over a minute, this is the single largest usability improvement available — and it costs no compute.

It also removes a real failure mode recorded earlier: a user who believes the app has crashed interacts again, triggering another rerun and stacking work on a machine with no memory headroom.

#### Chat structure and the rerun model

`st.session_state.messages` holds the conversation as a list of `{role, content, hits}`.

The rerun model creates a subtlety worth stating: **the whole conversation is re-drawn on every interaction**, but only the newest answer should be *streamed*. Previous answers are already complete strings, so they are written directly with `st.markdown`. Re-streaming history would replay every answer from the start on every keystroke.

The conversation resets when a different document is opened — answers about a previous document displayed beneath a new document's header would be actively misleading.

#### Visual design

| Element | Purpose |
|---|---|
| `.streamlit/config.toml` | Theme colours in configuration rather than CSS hacks |
| Document bar | Filename, chunk count, and whether the index was newly built or cached |
| Similarity chips + bars | Compact per-chunk scores; rank 1 highlighted, lower ranks muted |
| Sidebar | Upload, Ollama health, live configuration, clear-conversation |

**The similarity bars are scaled to the observed range (0.30–0.70), not 0–1.** A 0–1 bar would render every real result as roughly half full and communicate nothing. This is a small decision with a general principle behind it: a visualisation whose scale ignores the data's actual distribution conveys no information, however correct it looks.

Rank 1 uses the accent colour and lower ranks are muted, so the ordering is visible before any number is read.

#### What was kept from the first version

The sources panel, its justification unchanged: Day 5 recorded a correct-looking answer whose rank-1 chunk did not contain the answer. Its caption still warns against threshold-reading similarity scores, now quoting both the correct range (0.49–0.64) and the counter-example (an irrelevant chunk at 0.43).

The sidebar still displays the live configuration. A demo whose settings are invisible is not reproducible.

---

## Day 8 — closing out the carried-forward list, and a second UI

Days 6–7 had ended with several items still open: two "Pending" checks on `app.py` (run in a browser, error paths tested), and a longer carried-forward list from Day 5 (harden the test suite, run 200/25 end to end, test a messy PDF, fix the `[Source N - chunk N]` leak, try a structure-only header). This entry works through that list, then diverges from it: partway through, the session shifted to building a second interface after a live evaluation run collided with real memory pressure on the development machine — itself a useful data point, and recorded as one.

### Step 8.1 — Fix the prompt-leakage bug

`build_prompt()` labelled each context block `[Source N - filename, chunk N]`. The model had been observed occasionally echoing that label back into its answer — "According to chunk 4 of the document…" — which means nothing to someone reading only the answer, not the prompt. Relabelled `[Passage N - filename]`, and added an explicit rule to `SYSTEM_PROMPT`: never cite "Source N", "chunk N", or any other internal label. Low-risk, high-value fix; no measurement needed to justify it, the bug was self-evidently a bug.

### Step 8.2 — Test a messy PDF for the first time

Every "known limitation" bullet about multi-column layouts and tables had been sitting in the README since Day 1 as an assumption — the pipeline had only ever met one clean, single-column document. Built `tools/make_messy_pdf.py`: a synthetic two-page report, page 1 a genuine two-column layout (built with reportlab `Frame`/`PageTemplate`, not a hand-wave), page 2 a data table of unit economics.

**Extraction result, read with the same "eyeball it" discipline Day 2 established:**

```
extract_text() -> clean_text() -> chunk_text()
```

The two-column page extracted in correct reading order — the entire left column, then the entire right column, no interleaving. This is a genuine correction to a standing assumption, not a confirmation of it: PyMuPDF's block-sorting handled this particular layout (generous margins, a clear column gap) without difficulty. The table did not survive contact with plain-text extraction: every cell landed on its own line in the raw extraction, and `clean_text()`'s single-newline-to-space rule (correct for prose, blind to tables) collapsed the whole table into one run:

```
Model Units sold Unit price (USD) Flight time (min) Warranty (months)
Falcon-9X 214 18,400 47 24 Falcon-7 Lite 96 11,250 34 18
Kestrel-2 (tethered) 18 42,900 n/a 36 Falcon-9X Arctic 12 21,600 31 24
```

No column separators, no row boundaries — a reader (human or model) must count five numbers per row and match them to five headers by position alone.

**End-to-end test, 3 questions, one chunk (the whole document fit in one 400-word chunk, so this isolates generation-time table parsing from retrieval):**

| Question | Expected | Result |
|---|---|---|
| Unit price of the Falcon-7 Lite? | 11,250 | **Correct** — model quoted the flattened row verbatim and counted position correctly |
| Warranty period of the Kestrel-2? | 36 months | **WRONG — refused.** "The document does not contain this information," despite the fact being present |
| % reduction Calderon Midstream reported? | 63 | **Correct** (two-column prose, unaffected by the table issue) |

The Kestrel-2 failure is informative beyond "the model got one wrong." Its row is `Kestrel-2 (tethered) 18 42,900 n/a 36` — the `n/a` sits where "Flight time" should be (the Kestrel-2 is tethered, so this field is genuinely inapplicable), which likely broke whatever positional counting let the first question succeed. A flattened table is not uniformly bad; it is unreliable in a way that correlates with irregularity in the row itself, which is a worse property than a table that fails consistently — you cannot easily predict from the raw text where it will next fail. **Tables extract poorly: confirmed, not merely assumed. Multi-column prose extracts poorly: not confirmed on this layout, and previously stated more confidently than the evidence supported.**

### Step 8.3 — Structure-only headers, measured

Day 5 recorded a caveat rather than a fix: the context header includes the source filename, and the test document's filename contains "stage" (French for internship), so every chunk's header matched the word "internship" — inflating similarity on q8 (an unanswerable question about internship salary) from 0.4291 to 0.4770. `ingest.add_context_headers()` gained an `include_filename` parameter, and `experiment.py` gained `sweep_header_filename()` — an A/B/C comparison (no headers / headers+filename / structure-only headers) run across the **full 8-question set** from `questions.json`, not the 5-question subset the other sweeps use, because the cost being measured only shows up on the unanswerable questions that subset excludes.

```
id         no headers       + filename   structure-only   note
q1           0.6433#4         0.6224#3         0.6093#3
q2           0.6028#4         0.5765#4         0.5843#4
q3           0.5999#4         0.5592#4         0.5784#4
q4           0.4923#1         0.5613#0         0.5711#0
q5           0.5964#4         0.5509#4         0.5776#4
q6           0.5822#7         0.5552#7         0.5388#7
q7           0.3641#0         0.3576#0         0.3547#0   unanswerable - lower is better
q8           0.4291#7         0.4770#0         0.4646#0   unanswerable - lower is better
```

**Reading it:** q4, the Day-5 target, stays at rank 0 with structure-only headers, and scores marginally *higher* than with the filename (0.5711 vs 0.5613) — the fix does not depend on the filename being present. q8 improves from the filename's +0.048 inflation to +0.036 over the no-header baseline — better, not fully resolved; the chunk that wins q8 (chunk 0, "beginning of the document") is the same one in both header variants, so something about that chunk's content independent of the filename still draws a modest boost from having any header at all. Every topical question (q1, q2, q3, q5, q6) sits at or above the filename variant's score, same rank-1 chunk throughout — no regression anywhere.

**Decision: adopted as the new default.** `config.py` gained `HEADER_INCLUDE_FILENAME = False`, wired through `embed.index_pdf()` and both UIs' upload paths. The main `documents` collection was re-embedded under the new setting (first header line now reads `[Part 1 of 8 | This is the BEGINNING...` — no `Source:` line).

### Step 8.4 — Harden the evaluation suite

Recorded as an open problem since Day 5: `evaluate.py` had reported 8/8 in *every* configuration it had ever been run against. Not because nothing was wrong — Day 5's own headline finding was a real failure the suite's summary line did not surface — but because `grade()`'s `expect_any` matching is inherently permissive, and several term lists mixed strong, distinguishing signals (a proper noun, an exact number) with generic words a wrong answer could plausibly contain too:

- q4 (`ReDX`, `welcomed`, `team`, `environment`) → tightened to `ReDX` alone, the specific detail Day 5 identified as what actually changed between the wrong and right answers.
- q6 (`job`, `accounting`, `collect`, `usage`) → dropped `job`, too generic on its own.

Also added `HEDGE_MARKERS` — phrases like "it is likely", "probably", "I'm not sure" — so a keyword match riding on hedging language downgrades from `PASS` to `CHECK` rather than passing silently. A model that hedges around a lucky keyword collision is not the same as a model that answered confidently and correctly, and the old grading could not tell them apart.

### Step 8.5 — A live run collides with real memory pressure

With the header change and the hardened grading in place, the natural next step was a fresh full-suite run to confirm both together. First attempt failed on question 1:

```
RuntimeError: Ollama returned 500 from /api/chat
Server said: {"error":"model runner has unexpectedly stopped, this may be
due to resource limitations or an internal error..."}
```

`Get-CimInstance Win32_OperatingSystem` showed **160 MB free** out of 8 GB total at the moment of the crash. This is not a bug in the project; it is the exact hardware constraint SPECIFICATION.md §10 describes ("close your browser before running Ollama sessions") happening live, on a machine that — reasonably — had other applications open at the time, not reserved for this session. A retry a few minutes later found 595 MB free and was launched; the user then asked directly for it to be stopped, since other work in progress needed the RAM. It was stopped immediately (`TaskStop`), and no further live generation runs were attempted for the remainder of the session.

**Consequence, stated plainly:** the fresh baseline confirming the header and prompt-leakage fixes end to end, and the planned 200/25 end-to-end comparison, were **not completed**. The header fix has strong *retrieval-only* evidence (Step 8.3, full 8-question sweep) and the prompt-leakage fix is a self-evidently correct label change, but neither has a fresh generation-level report in `results/` reflecting today's config. This is the most honest single line to carry forward: **measured at the retrieval layer, not re-confirmed at the generation layer, today.**

### Step 8.6 — A second UI: Flask + plain HTML/JS

Requested in place of continuing to invest in Streamlit, specifically because of the memory episode just above. Worth stating precisely what changes and what does not: Streamlit's *process* is a few MB and its rerun model is a design choice, not a leak — the actual RAM cost, ~2.3 GB, is the two Ollama models, and that is identical no matter what sits in front of them. What a lighter UI *does* remove is a browser tab's own footprint and Streamlit's session-state/rerun bookkeeping — real, but secondary to the model cost.

Built `src/server.py` (Flask) and `src/web/index.html` (one static page, inline CSS/JS, no build step). Same pipeline functions as `app.py` — `retrieve`, `build_prompt`, `ask_model_stream` — reused rather than reimplemented. Streaming uses the same newline-delimited-JSON shape Ollama itself uses (`generate.ask_model_stream`'s own doc comment explains why): each line is one JSON object, `{"type": "hits", ...}` once, then `{"type": "token", "text": "..."}` per fragment, then `{"type": "done"}` — the browser's `fetch().body.getReader()` consumes it incrementally without buffering the whole response.

Indexing reuses the same content-hash-keyed design as `app.py`'s `index_uploaded_pdf` (same reasoning: index once regardless of reruns, never collide two same-named files) — a plain dict substitutes for `@st.cache_data` since a Flask process has no rerun to cache against; ChromaDB's own persistence is what survives an actual process restart, checked the same way (`collection.count() > 0`).

### Step 8.7 — A crash found by reading library source, not by triggering it

While building `server.py`'s upload handler, traced what happens when a PDF yields zero chunks (a scanned page with no text layer, the exact case the README already lists as unsupported). `load_and_chunk()` returns `[]` cleanly — no crash there. But the upload handler then calls:

```python
collection.upsert(ids=[], embeddings=[], documents=[], metadatas=[])
```

Grepped `chromadb`'s own source rather than running it live:

```
chromadb/api/types.py:1011  if len(ids) == 0:
chromadb/api/types.py:1012      raise ValueError(f"Expected IDs to be a non-empty list, got {len(ids)} IDs")
```

Confirmed: an empty-chunk PDF raises a raw `ValueError` from inside Chroma's validation. **This exact bug was already present in `app.py`, unguarded, since Day 6** — the indexing call sits outside the only try/except block in that file. It had simply never been hit, because every test document so far had extractable text.

Fixed in both `index_uploaded_pdf` (`app.py`) and `index_pdf_bytes` (`server.py`): an explicit `if not chunks:` check raises a `ValueError` with an actionable message ("likely a scanned document… OCR is not supported") before Chroma ever sees an empty list. `app.py`'s call site gained a `try/except` around indexing; `server.py`'s `/api/upload` route already had one, mapping it to HTTP 422.

**Verified without a live server or any model call.** Built `tools/make_blank_pdf.py` (a one-page PDF via `fitz.open()` / `new_page()`, no text added) and drove it through Flask's `app.test_client()` — an in-process WSGI client that never binds a port:

```
health:                    200  {'ollama': True, ...}
index:                     200  text/html, 11596 bytes
upload (blank pdf):        422  {'error': "No text could be extracted from
                                  'blank_scanned.pdf'. This usually means it
                                  is a scanned document..."}
ask (no collection):       400  {'error': 'collection is required - upload a PDF first'}
ask (empty question):      400  {'error': 'question must not be empty'}
```

Also confirmed `server.py` imports cleanly and registers all four routes (`python -c` + `importlib.util`), and that the inline JavaScript in `index.html` parses without a syntax error (`node --check`, extracting the `<script>` block). None of this touches Ollama — the health check's own `/api/tags` ping is the only network call involved, the same lightweight reachability check both UIs already poll every 30 seconds.

### Step 8.8 — Documentation brought current

`SPECIFICATION.md`'s "As-built status" section had been stuck at end-of-Day-4 since it was written — Days 5 through 7 had happened and been recorded in `BUILD_LOG.md` and `README.md`, but the specification's own status table and Definition-of-Done checklist never caught up. Added a new "§16 As-built status — end of Day 8" section correcting this and folding in everything above, rather than editing the Day 4 section in place — consistent with this file's own principle of recording what happened at each point in time rather than rewriting history. `README.md`'s engineering notes, known limitations, and possible-extensions sections were updated with the measured results from Steps 8.2 and 8.3, and its usage section now documents both UIs.

### Day 8 status

| Item from the Day 5 / 6–7 carried-forward lists | Outcome |
|---|---|
| Fix `[Source N - chunk N]` prompt leakage | **Done** |
| Test a messy PDF (multi-column, tables) | **Done** — mixed result, tables confirmed to fail, multi-column corrected in the other direction |
| Try a structure-only header (no filename) | **Done** — adopted as default on evidence |
| Harden the test suite so it can fail | **Done** |
| End-to-end run at 200/25 | **Not done** — blocked by memory, retry when the machine has headroom |
| App run and tested in a browser | **Still not done** — logic verified without a live model call; no live click-through, on either UI, today |
| Error paths tested (no Ollama, no file, empty question) | **Done** — plus one previously-unknown error path (empty-chunk PDF) found and fixed |

### Carried forward

1. **A fresh end-to-end evaluation run**, once the machine has memory headroom, to confirm the header and prompt-leakage fixes at the generation layer (currently confirmed at the retrieval layer only).
2. **The 200/25 end-to-end comparison** — wins every retrieval-only comparison on record; whether smaller chunks starve the generator of context is still unmeasured.
3. **An actual live click-through of both UIs** — upload a PDF, ask a question, watch it stream, in a real browser. Neither has had this since Day 6.
4. **Cross-document retrieval** — indexing several documents into one queryable session with per-answer attribution. The pipeline supports it (`embed.index_pdf` into a shared collection); neither UI's upload flow uses it that way.
5. **The `highperformance` artefact** from Day 4 — still unaddressed, still judged low-impact.

---

## Day 9 — a visual redesign, and an opt-in web fallback

Two requests in the same session, handled in order.

### Step 9.1 — Redesign `src/web/index.html`

Requested as a straightforward "make it more modern" pass, done without touching `server.py` — everything is still one dependency-free HTML file (no CDN fonts, no icon packs, no build step), consistent with the "nothing leaves the machine" principle the rest of the project holds to. What changed: a refined dark/light palette plus a manual theme toggle persisted in `localStorage` (previously `prefers-color-scheme` only, no override); inline SVG icons replacing emoji throughout; avatar-style chat bubbles with a rise-in animation; a bouncing-dots typing indicator shown immediately on submit, swapped for the real text on first streamed token; a proper drag-and-drop dropzone in place of a bare file input; a floating pill-shaped composer with a circular gradient send button; a real empty/welcome state; and a responsive breakpoint collapsing the sidebar on narrow screens. Every element ID the JS logic depends on was preserved, so no changes to `server.py` were needed. Verified the same way as everything else added after the memory episode on Day 8: `node --check` on the extracted `<script>` block, and a full page load through Flask's in-process test client — no live server, no model call.

### Step 9.2 — An opt-in web search fallback

Requested as: when the document doesn't have the answer, say so, but then also search the web and show that answer with its source. Paused before writing code, because this collides directly with the project's central premise in two concrete ways, not just in spirit: it requires a network call to somewhere other than local Ollama, and by default it would invert what `questions.json`'s q7 and q8 test - both are specifically there to confirm the system refuses a question it cannot support from the document rather than answering from outside knowledge. Asked, rather than assumed: which search backend (Google Custom Search API, DuckDuckGo scraping with no key, or the Wikipedia API), and whether the fallback should run automatically or be opt-in. Answer: **Google Custom Search API, opt-in, off by default.**

**What this preserves by construction.** The fallback only ever fires after a grounded answer has already been generated and found to look like a refusal - `looks_like_refusal()`, previously private to `evaluate.py`'s grading, was moved to `generate.py` so the same one definition now decides two things (test grading, and whether to search) instead of two definitions that could quietly drift apart. `questions.json` and `evaluate.py` were not touched by this feature at all; the suite never sets `web_fallback`, so q7 and q8 test exactly what they always did.

**New file, `src/websearch.py`.** One function, `search_web(query)`, calling Google's Custom Search JSON API. Credentials (`GOOGLE_SEARCH_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`) are read from the environment, not `config.py` - a real key must never end up in a tracked file, and `config.py` is otherwise the single place every setting lives, so this is a deliberate, noted exception. Unconfigured, it fails fast with an actionable message before any network call - verified directly (`websearch.search_web()` raises `RuntimeError` immediately, no request attempted).

**`generate.py` gained a second contract.** `WEB_SYSTEM_PROMPT` is a different rule set from `SYSTEM_PROMPT`, not a variant of it - it explicitly tells the model these are web results, not the document, and to say so. `ask_model` / `ask_model_stream` both gained a `system_prompt` parameter (default unchanged) so the same call machinery, error handling, and streaming logic serve both prompts rather than being duplicated. `build_web_prompt()` mirrors `build_prompt()`'s shape for search snippets instead of chunks.

**Both UIs wired the same way, differently, because Streamlit and Flask stream differently.** In `server.py`, the `/api/ask` NDJSON stream gained `web_start` / `web_results` / `web_token` / `web_error` message types, sent only after the grounded answer's own stream completes and only if `web_fallback` was requested in the POST body **and** checked server-side regardless of what the frontend sent - a request is a request. `index.html` renders the result in a visually distinct teal card, labelled "not from your document," built only once `web_start` arrives.

In `app.py`, the equivalent needed one design correction mid-implementation: `st.write_stream()` renders each fragment as it arrives, so the first draft - stream the web answer, then also call a `render_web_block()` helper on the result - would have shown the answer twice. Fixed by splitting the helper's role: it now renders a *complete, already-known* web answer (used only when replaying past messages from `st.session_state`), while the live path prints the teal header via `st.markdown` first, then lets `st.write_stream` render the streamed text directly beneath it, uncontained. Found and fixed by reading the code path rather than by running it.

**Verification, same discipline as Day 8: everything checked without a live model call.** `py_compile` and an AST parse on every touched file; `server.py` imported and its routes confirmed via `importlib`; `/api/health` confirmed to report `web_fallback_available: false` correctly with no credentials set; `/api/ask` confirmed to still validate its inputs the same as before. Not yet done, for the same reason as Day 8's open items: a real search with real credentials, and a live click-through of the toggle in a browser.

### Day 9 status

| Item | Outcome |
|---|---|
| Modernised `src/web/index.html` | **Done** - verified statically, not yet seen live |
| Web fallback, opt-in, off by default | **Done** in both UIs and the CLI (`generate.py answer(..., web_fallback=True)`, or append `--web` on the command line) |
| `questions.json` / `evaluate.py` semantics preserved | **Confirmed** - untouched by this feature |
| A real search with a configured API key | **Not done** - no key was available this session |
| Live click-through of the toggle | **Not done** - carries forward with the rest of Day 8's "watched it actually work in a browser" item |

---

## Day 10 — the first live test, and a same-day reversal

### The first live click-through, finally

Uploaded a document through the redesigned Flask UI in a real browser and asked a question it doesn't answer, web fallback toggled on. First look at any of this actually running since Day 6. Every piece fired correctly on the first attempt:

- The grounded model correctly refused: *"The document does not contain this information."*
- The refusal was detected and the web fallback triggered automatically off it.
- The teal "Web fallback — not from your document" card rendered exactly as designed.
- The search call itself failed - and the failure was caught, shown as an explanation instead of a stack trace, and left the grounded refusal above it untouched.

That last point is the one worth dwelling on: **the failure path was exercised for real, on the first live run, and behaved exactly as every layer of it had been designed to.** Everything from the NDJSON message protocol to the "a failed web step is not a failed turn" principle in `README.md` held up against an actual error it had never seen before, not a synthetic one.

### What actually failed, and why it wasn't fixable

Google's own response: `"API key not valid. Please pass a valid API key."` (`INVALID_ARGUMENT` / `API_KEY_INVALID`). This is Google rejecting the credential itself - not a quota, permissions, or API-not-enabled problem, which each return a different, more specific message. Diagnosed by category rather than guessed at, and then stated plainly: **this is not a code bug.** The code was correctly relaying Google's own explanation; there was nothing in `websearch.py` to debug further without either the real key (not shared, correctly) or access to the Google Cloud project that issued it.

Offered two ways forward rather than picking one silently: keep debugging the Google Cloud side, or replace the backend entirely with one that needs no credentials (DuckDuckGo or Wikipedia - the two keyless options from the original Day 9 backend choice). Chosen: **switch to DuckDuckGo.**

### Step 10.1 — Swap the backend without touching its callers

`websearch.py` was rewritten from a Google Custom Search API call to a DuckDuckGo HTML-endpoint scrape (`html.duckduckgo.com/html/`, parsed with `BeautifulSoup` - added to `requirements.txt`), but its public surface - `is_configured()`, `search_web(query, max_results)` returning the same `{"title", "snippet", "link"}` shape - did not change. Nothing in `server.py`, `app.py`, or `index.html`'s *logic* needed to change as a result; only their user-facing hint text did (which UI copy said "Google" now says "DuckDuckGo", and the "not configured" branches became dead code kept for a future backend that might need it, rather than deleted).

**`is_configured()` now always returns `True`.** No credential exists to be missing. Kept as a function rather than removed so the call sites don't need to change again if a future backend does need configuration.

**The real cost, stated in the same place the benefit is claimed, not hidden in a comment three files away:** DuckDuckGo has no official general-purpose search API - `api.duckduckgo.com` only serves Instant Answers, not general web results. `search_web()` therefore parses HTML markup that DuckDuckGo can change without notice, unwraps their redirect-tracking links (`//duckduckgo.com/l/?uddg=<encoded target>`) to link straight to the real page, and - specifically to avoid a silent-empty-results failure mode - distinguishes "the page's own no-results marker was present" from "this scraper doesn't recognise the page it got back," raising a clear error in the second case instead of quietly returning nothing.

**`config.py` lost `GOOGLE_SEARCH_API_KEY` / `GOOGLE_SEARCH_ENGINE_ID`, and its now-unused `import os` went with them** - a setting that no longer describes anything real is worse than no setting, consistent with this file's own stated purpose as the single source of truth.

### Step 10.2 — Two more bugs, found by actually running the new scraper

`websearch.py`'s DuckDuckGo call is a plain HTTP request, not an Ollama call - unlike the generation testing paused earlier in the session over RAM pressure, this cost negligible memory, so it was run live rather than only reasoned about, on the theory that a fragile HTML scrape is exactly the kind of thing worth confirming rather than assuming. Good instinct - it found two real bugs on the first two calls.

**Call 1** (`python src\websearch.py "capital of Japan"`) - `search_web()` itself worked: three real results came back, correct titles, correct links, correct snippets. The script then crashed anyway, in its own `print()` calls, with `UnicodeEncodeError` - a snippet contained a macron (the romanised "Tōkyō"), and Windows' default console codepage (cp1252) can't encode it. Not a problem for either UI (`server.py`'s HTTP responses and the browser are UTF-8 throughout), only for this direct-CLI demo path. Fixed with `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`, scoped to the `__main__` block only.

**Call 2**, about a minute after call 1, retried to confirm the encoding fix - returned HTTP 202 instead of a normal result page: DuckDuckGo's anomaly/bot detection, not a real error, triggered by a second request in quick succession. This is a genuine, load-bearing limitation of the keyless approach, not a code defect to fix - documented as such rather than papered over. `search_web()` now recognises a 202 specifically and says "wait a minute or two, this is rate-limiting" rather than folding it into the generic "the page changed shape" error, because the correct response to each is different (wait, versus go read the parsing code).

**The honest accounting, stated where the earlier trade-off claim was made rather than left implicit:** switching to DuckDuckGo traded a reliability problem that was fixable but required Google Cloud configuration (an invalid credential) for a reliability problem that has no fix, only a workaround (rate-limiting under exactly the kind of repeated use a real Q&A session produces). Both README.md and this entry say so plainly rather than presenting the swap as a strict improvement.

### Verified the same way as every other change since the memory episode

`py_compile` across every changed file. Unlike Day 8 and 9's changes, `websearch.py`'s core call WAS run live this time (see Step 10.2 above) - deliberately, because it is a plain network call with negligible memory cost, not an Ollama generation call, and because a scrape is precisely the category of code where reasoning from the source without ever executing it would have missed both bugs above.

### Day 10 status

| Item | Outcome |
|---|---|
| First live click-through of the Flask UI | **Done** - and it is what caught the invalid-key failure |
| Diagnose the Google API key failure | **Done** - `API_KEY_INVALID`, a credential problem, not a code bug |
| Switch web-fallback backend to DuckDuckGo | **Done** - no callers needed to change, only UI copy |
| `beautifulsoup4` added to `requirements.txt` | **Done** |
| Live-test the new scraper directly | **Done** - found and fixed a Windows console encoding crash |
| Discover DuckDuckGo's rate-limiting | **Done** - a real limitation, documented, not fixable, only workable-around |
| A live retry of the FULL pipeline (UI toggle → DuckDuckGo → streamed answer) in the browser | **Not done this session** - the natural next step, and best attempted more than a minute after any other search |
| Live click-through of the Streamlit UI | **Still not done** - only the Flask UI has been seen running |

---

## Day 11 — the mascot, and a second reversal on the web fallback

Two unrelated threads this entry: a substantial UI personality feature (the sidebar mascot, requested and built out over several rounds), and a second flip on the web-search backend after DuckDuckGo's rate-limiting - only theorised about in Day 10 - turned out to bite in ordinary live use, not just in back-to-back testing.

### The mascot

Requested in stages, each building on the last: first a small "searching" indicator to replace a plain three-dot spinner (`botLoader()` - a floating CSS/SVG-free robot head with scanning eyes, a pulsing antenna, and a genuine 3D `perspective` + `rotateY` head-turn, no image asset, no library). Then a persistent, bigger version living in the sidebar's empty space (`.mascot`), driven by real state rather than looping blindly: idle when nothing is happening, visibly more alert while a question streams (`state.busy`, reset in a `finally` block so it can't get stuck on if a request errors), dimmed and still if Ollama goes unreachable (`state.online`, read off the existing 30-second health poll). Then a body - the first version was head-only, which read as an eyeball rather than a robot; added a neck, a torso with a pulsing "core" light, and two small arms, all moving as one rigid figure (`.mascot-figure`) rather than independently.

Then a full interaction layer: a speech bubble, a click reaction (jump + random line), a pure-CSS hover wave, a load-time greeting, reactions to upload success/failure and to the actual answer outcome (not just "a response arrived" - `looksLikeRefusal()` on the frontend mirrors `generate.py`'s own check, so the mascot's read of an answer agrees with the server's, not a separate guess), and reactions to Ollama going offline/coming back - gated on the *transition*, not the level, so a 30-second health poll doesn't repeat the same line forever.

**Two real bugs found along the way, both from precise bug reports, not vague ones:**

1. **Speech bubble positioning.** `.mascot-speech` was `position: absolute; bottom: 100%` relative to `.mascot` - but `.mascot` is `flex: 1` and stretches to fill all the sidebar's leftover space, sometimes 300+ px tall. `bottom: 100%` measured against *that* box, popping the bubble up near the top of the sidebar, nowhere near the robot. Fixed by moving the bubble inside `.mascot-3d`, which is sized tightly to the figure's own content - the anchor a percentage-based `bottom` needs to mean anything.
2. **Animations not restarting.** `playMascotAnim()` set `figure.style.animation` directly - correct in principle, since inline style always wins over a class-based rule for the same property - but reassigning the *same* animation value twice in a row (a click right after an idle-timer jump, say) is a no-op to the browser: no detected change, no restart. A known CSS gotcha, not a hypothetical one, and exactly what "not jumping" looked like from outside. Fixed with the standard workaround: set to `none`, force a synchronous reflow (`void figure.offsetWidth`), then set the real value.

Also retuned twice on direct feedback: the first message set ("Beep boop!", heavy emoji use) was rejected outright ("not good, I didn't like it, honestly") - rather than guess again, four concrete tone options were presented with sample lines for each rather than descriptions alone, and "warm / simple" was chosen and applied to all nine message sites. Separately, "not enough movement" turned out to be partly the restart bug above and partly a real pacing issue - idle motion was gated behind a 35%-chance-every-50-seconds roll, which reads as motionless for long stretches. Rebalanced to a dedicated movement timer (a jump or wiggle every ~6 seconds at 70% odds) decoupled from a separate, still-occasional speech timer, so movement and talking don't wait on the same dice roll.

### The web fallback, reversed a second time

Live use (not testing) hit DuckDuckGo's rate-limit on consecutive real questions - the exact failure mode flagged as a risk in Day 10, now confirmed as a practical, not just theoretical, dealbreaker. Asked to go back to Google.

**Built as a configurable choice this time, not a third full rewrite.** `websearch.py` now holds both `_search_google()` and `_search_duckduckgo()` side by side, dispatched by `config.WEB_SEARCH_BACKEND` (`"google"`, restored as the default, or `"duckduckgo"`). `is_configured()` branches the same way - credential check for Google, always-true for DuckDuckGo. Neither `server.py` nor `app.py` needed any logic changes, only re-importing `WEB_SEARCH_BACKEND` to display it in the Configuration panel (a result you can't see the settings behind isn't evidence - the same principle every other setting in that panel already follows). `config.py` regained `GOOGLE_SEARCH_API_KEY` / `GOOGLE_SEARCH_ENGINE_ID`, read from the environment as before, never hardcoded.

**The unresolved problem from Day 10 is still unresolved by this change alone, and said so plainly rather than implied as fixed:** reverting the code back to Google does not fix an invalid API key - that requires the user's own Google Cloud credentials to actually be valid, which was never confirmed working in this session. `README.md`'s "Web fallback" section was rewritten with a specific troubleshooting checklist for the exact error seen (`API_KEY_INVALID`) - wrong credential type, API not enabled on the right project, a stray character in the environment variable, or the key and Search Engine ID swapped - rather than a repeat of "check Google Cloud Console" without specifics, which had already proven insufficient once.

### Day 11 status

| Item | Outcome |
|---|---|
| Mascot: loading indicator, sidebar figure, body/arms, full interaction layer | **Done** |
| Speech bubble positioning bug | **Found and fixed** |
| Animation-restart bug | **Found and fixed** |
| Message tone, retuned on direct feedback | **Done** - "warm/simple", all nine sites |
| Movement pacing, retuned on direct feedback | **Done** - dedicated frequent timer, decoupled from speech |
| Web fallback backend made configurable (`WEB_SEARCH_BACKEND`) | **Done** - both implementations kept, one line to switch |
| Default reverted to Google | **Done** |
| A working Google credential, confirmed live | **Not done** - the actual invalid-key problem from Day 10 was never independently resolved this session |

### Postscript, same day — Google Cloud setup attempted live, still didn't land

Walked through the actual Google Cloud Console with the project owner, step by step, in real time - the configurable backend above was built anticipating this might be needed, and it was. Diagnosed and fixed two distinct setup mistakes along the way: the first credential created was through Google's "Help me choose" wizard, which produces an OAuth client ID or a service account, neither of which is a plain API key - backed out and used "+ Create Credentials → API key" directly instead, which worked. A real key and a real Search Engine ID (`cx`) were both obtained, confirmed correctly formatted.

**Where it stopped:** the "Search the entire web" toggle on the Programmable Search Engine's settings page did not respond to clicks - no visual change, no error, nothing. Not a code problem; nothing in this project runs on that page. Standard causes (stale page, imprecise click target, a screenshot-tool extension visibly overlaying the page in the screenshots shared) were suggested; none confirmed fixed within the session.

**Decision, asked rather than assumed, given the accumulating friction:** keep pushing on Google Cloud, turn the web fallback off entirely, or switch back to DuckDuckGo. Chosen: **DuckDuckGo**, again. `WEB_SEARCH_BACKEND` flipped back to `"duckduckgo"` in `config.py` - the one-line change the configurable design (this same entry, above) was built for. The Google implementation is untouched and fully working code-wise; it is credentials, not code, still standing between it and actually running.

**And immediately hit the still-active DuckDuckGo block** - the very next real question, right after switching back. The "wait a minute or two" estimate from Day 10 was wrong, not just optimistic: the same block was still returning HTTP 202 **over 50 minutes** after it first appeared, across several genuinely spaced-out attempts, not rapid-fire ones. Measured, not re-guessed a third time.

**Fixed what was actually fixable.** The block itself lives on DuckDuckGo's side - nothing here can clear it. What *was* within reach: `_search_duckduckgo()` now remembers the last 202 (`_ddg_blocked_since`, module-level) and fails fast for 10 minutes afterward with no network call at all, rather than letting every subsequent question waste a request that will almost certainly also come back blocked - and plausibly extend the block further in the process, since every attempt during a cooldown, from any source, is itself another flagged request. The user-facing message was corrected to state the measured "over 50 minutes," not the disproven "a minute or two," and to point at switching back to `"google"` as the immediate alternative if the fallback is needed right now rather than later.

### The Google path, pushed all the way to its actual end

Walked through Google Cloud live a second time, from a genuinely fresh API key (the earlier `API_KEY_INVALID` key was abandoned, not reused) through to a real test. Two more real, distinct problems found and fixed along the way, neither a repeat of Day 10's:

1. **The credential wizard, again.** Landed back in "Help me choose" (OAuth/service-account flow) rather than a plain API key, exactly as it had before - backed out, used "+ Create Credentials → API key" directly, which produced a real, correctly-formatted key this time (`AIzaSy...`, 39 characters, confirmed by inspection).
2. **`PERMISSION_DENIED` / `"This project does not have the access to Custom Search JSON API"`** on the very first live test with the new key - a *different* error from Day 10's `API_KEY_INVALID`, and diagnosed as such rather than assumed to be the same problem recurring. Traced, correctly, to the actual cause: **Google now requires a billing account linked to a Cloud project before the Custom Search API serves any request at all**, free-tier usage or not - confirmed by a "start your free trial, get $300 credit" banner present throughout that project's console, the tell that billing was never set up. This is not documented anywhere obvious in Google's Custom Search API onboarding; it was found by reading the error category correctly (`PERMISSION_DENIED`, not `INVALID_ARGUMENT`) and matching it to what the console page was actually showing, not by a guess.

**Where it genuinely ends:** billing requires a card. Asked, not assumed - the project owner does not have one to use here. That is a real, final stopping point, not a problem this session could troubleshoot further. `WEB_SEARCH_BACKEND` flipped back to `"duckduckgo"` a second time (fourth flip overall, this session). `README.md`'s "Web fallback" section was rewritten once more to lead with the billing requirement as the FIRST thing to check, ahead of key and API setup - so if this is revisited later, the twenty minutes spent getting key and API setup individually correct before discovering the billing wall doesn't have to be spent again.

The Google implementation in `websearch.py` is finished, correct code, never disproven - it is blocked on an account prerequisite outside this project's control, not on anything left to fix.

### Day 11, final status

| Web fallback backend | Outcome |
|---|---|
| Google - credential setup | Eventually correct (real API key, real Search Engine ID) |
| Google - actual usability | **Blocked** - requires a billing account (card) neither available nor to be pursued further this session |
| DuckDuckGo - functionality | Confirmed working when not rate-limited |
| DuckDuckGo - reliability | Real constraint, not a bug: blocks under repeated use, for a duration measured in tens of minutes, not seconds |
| Engineering response to the DuckDuckGo constraint | A cooldown cache (`_ddg_blocked_since`) so the app fails fast and honestly instead of repeatedly wasting a doomed request |
| `WEB_SEARCH_BACKEND`, end of day | `"duckduckgo"` - the one that actually runs without a card |
