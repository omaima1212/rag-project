# Concepts — How This System Actually Works

A conceptual companion to the code. `BUILD_LOG.md` records *what* was done and *when*;
this document explains *why* each mechanism exists and what problem it solves.

Examples throughout use the real test document: an 11-page internship logbook,
17,141 characters, which became 8 chunks and 8 vectors.

---

## Contents

1. [The core problem](#1-the-core-problem)
2. [Stage 1 — Extraction: a PDF is not text](#2-stage-1--extraction-a-pdf-is-not-text)
3. [Stage 2 — Cleaning: removing layout scars](#3-stage-2--cleaning-removing-layout-scars)
4. [Stage 3 — Chunking: why documents must be cut up](#4-stage-3--chunking-why-documents-must-be-cut-up)
5. [Stage 4 — Embeddings: meaning as geometry](#5-stage-4--embeddings-meaning-as-geometry)
6. [Stage 5 — Cosine similarity: why angle, not distance](#6-stage-5--cosine-similarity-why-angle-not-distance)
7. [Stage 6 — The vector database: what is actually stored](#7-stage-6--the-vector-database-what-is-actually-stored)
8. [How the stages depend on each other](#8-how-the-stages-depend-on-each-other)
9. [Questions to test your own understanding](#9-questions-to-test-your-own-understanding)

---

## 1. The core problem

A language model has two limitations that this project works around.

**It does not know your documents.** Your internship logbook was never in its training data. Asked about it directly, the model has nothing to draw on and will either refuse or invent something plausible.

**It cannot read a large document at once.** Every model has a context window — a maximum amount of text it can process in one go. `llama3.2:3b` running on 8 GB of RAM has a small practical window. An 11-page document might fit; a 200-page manual will not.

The obvious fix — paste the whole document into the prompt — fails for a third, less obvious reason: **accuracy degrades when the relevant sentence is buried in noise.** Even when a long document technically fits, models attend poorly to a single relevant line surrounded by fifty pages of irrelevant ones.

So the real problem is not storage. It is **selection**: given a question, find the small number of passages that actually bear on it, and show the model only those.

That reframing is what RAG is. Everything below is machinery in service of selection.

---

## 2. Stage 1 — Extraction: a PDF is not text

The most common misconception about this stage is that a PDF *contains* text and extraction merely copies it out. It does not.

A PDF is a set of **drawing instructions**. Roughly: *place glyph 'T' at coordinate (72, 340) in Calibri 11pt; place glyph 'h' at (79, 340);* and so on for every character on the page. The file records where marks go on a canvas.

What a PDF does **not** record:

- where one word ends and the next begins (that is inferred from horizontal gaps)
- what order text should be read in
- that a page has two columns
- which text is a heading, a footnote, a page number, or a table cell
- paragraph boundaries

PyMuPDF reconstructs text by walking those instructions in the order the file happens to store them and inferring structure from geometry. **That reconstruction is a guess.** It is usually a good guess for simple single-column documents and a poor one otherwise.

This explains the classic failures:

| Failure | Cause |
|---|---|
| Two columns interleaved line by line | Nothing in the file says "this is column two"; the reader crosses the page horizontally |
| Tables become an unreadable run of numbers | Cell structure is drawn, not encoded; extraction sees only positioned characters |
| A scanned document yields nothing | There are no glyph instructions at all — the page is one image. Needs OCR |
| Headers and footers repeat mid-sentence | They are drawn like any other text and appear in reading order |

**The practical consequence:** the per-page character count printed by `extract_text()` is a diagnostic, not decoration. A page reporting 0 characters is an image. A page reporting far fewer characters than it visibly contains has lost something. In the test document every page returned 900–2,900 characters, which is why extraction was judged sound.

**The habit to keep:** always read a sample of extracted text with your own eyes before building on it. No downstream stage can detect that its input is garbled — it will faithfully process nonsense.

---

## 3. Stage 2 — Cleaning: removing layout scars

Extraction leaves artefacts caused purely by how text was laid out on a page. They carry no meaning and actively mislead the next stage.

**Hyphenation across line breaks.** A word that did not fit at the end of a line was split by the layout engine: `informa-` / `tion`. Extraction preserves the break, so the text contains `informa-\ntion` — which is two meaningless tokens instead of one real word.

**Newlines inside sentences.** A wrapped sentence arrives with newlines in the middle of it. But a *paragraph* break is also a newline. The two must be distinguished, and the only available signal is how many consecutive newlines there are: one is usually a wrap, two or more is usually a real break.

**Runs of whitespace.** Justified text and tab-based alignment produce long spans of spaces that mean nothing semantically.

The cleaning rules address exactly these three, in that order.

### The unavoidable trade-off

The hyphen rule cannot distinguish a **soft hyphen** (inserted by the layout engine because a word did not fit) from a **real hyphen** in a compound word that merely happened to fall at a line break. Both look identical after extraction.

In the test document this cost one word: `high-performance` became `highperformance`.

That is the correct trade to make. Soft hyphens are far more common than compound words breaking at line ends, and the damage is asymmetric — a broken word (`informa` + `tion`) is meaningless to an embedding model, whereas a merged compound (`highperformance`) still sits close in meaning to the correct form. Embeddings are robust to that kind of noise in a way they are not robust to shattered words.

**The general lesson:** text cleaning rules are heuristics, not truths. Each one trades a common error for a rarer one. Know which trade you made and write it down, because on a different document the balance may flip.

---

## 4. Stage 3 — Chunking: why documents must be cut up

This is the conceptual heart of the ingestion pipeline, and the stage most often done thoughtlessly.

### Why not embed the whole document as one vector?

Because an embedding is, loosely, an *average* of the meaning of its input. A single vector representing 11 pages would blend HPC cluster architecture, first-day impressions of the company, Slurm accounting scripts, and everything else into one point in space. That point is a poor match for any specific question, because it is a mediocre representation of all of them at once.

**Meaning must stay local to remain searchable.** The longer the text behind a vector, the more diluted and less discriminating that vector becomes.

### Why not embed each sentence?

The opposite failure. A lone sentence often lacks the context needed to be understood or to be useful as an answer. "It consists of 1,272 standard compute nodes" — what does *it* refer to? Retrieved alone, that sentence cannot answer anything, and the model receiving it has no way to recover the missing subject.

### The chunk as the atomic unit of retrieval

A chunk is the smallest thing the system can return. You retrieve whole chunks, never fragments. So a chunk must be:

- **small enough to be specific** — about roughly one topic, so its vector is discriminating
- **large enough to be self-contained** — carrying enough context to stand alone as an answer

400 words sits in a reasonable middle for prose. There is no universal correct value; it depends on the document. Dense technical reference material often wants smaller chunks; narrative or discursive text often wants larger.

### Overlap, and why it is not optional

A chunk boundary is an **arbitrary cut that knows nothing about content**. It falls at word 400 regardless of whether word 400 is mid-sentence, mid-definition, or mid-table.

Without overlap, a passage straddling a boundary is split in half and retrievable from **neither** chunk — each half is incomplete, so neither embeds close to a question about it. The information is present in the database and yet unfindable. This is a silent failure: nothing errors, the answer is simply missing.

Overlap fixes it by repeating the last N words of each chunk at the start of the next. **Any passage shorter than the overlap appears intact in at least one chunk**, guaranteed.

### The sliding window, concretely

With `CHUNK_SIZE = 400` and `CHUNK_OVERLAP = 50`, the window advances by `400 - 50 = 350` words each step.

The test document held roughly 2,541 words:

```
chunk 0:  words    0 – 400     (400 words)
chunk 1:  words  350 – 750     (400 words)  ← shares 50 words with chunk 0
chunk 2:  words  700 – 1100    (400 words)
chunk 3:  words 1050 – 1450    (400 words)
chunk 4:  words 1400 – 1800    (400 words)
chunk 5:  words 1750 – 2150    (400 words)
chunk 6:  words 2100 – 2500    (400 words)
chunk 7:  words 2450 – 2541    ( 91 words)  ← the tail
```

8 chunks, totalling 2,891 words for a 2,541-word document. The 350-word difference is exactly the duplication introduced by 7 overlaps of 50 words each.

**That duplication is the price of overlap** — more storage, more embedding calls, and a chance the same passage is retrieved twice. It buys the guarantee that no short passage is unfindable. The trade is worth it, but it is a trade.

The final chunk being short (91 words) is expected — it is the tail of the document, not a bug.

---

## 5. Stage 4 — Embeddings: meaning as geometry

### The problem embeddings solve

Given the question *"What supercomputer did you work on?"*, find the relevant chunk.

Keyword search fails immediately. The document says **Toubkal cluster**, **compute nodes**, **HPC platform**. The question says **supercomputer**. Zero word overlap. A keyword index returns nothing, and it is not close to working — the vocabularies simply do not intersect.

This is not an edge case. It is the normal state of affairs: **people ask questions in different words than documents use to answer them.**

### What an embedding is

An embedding model maps any text to a fixed-length list of numbers. `nomic-embed-text` produces **768 numbers** for any input, whether that input is three words or four hundred.

Treat those 768 numbers as **coordinates in a 768-dimensional space**. Every piece of text becomes a point in that space. The model was trained so that **texts with similar meaning land near each other**, regardless of the vocabulary they use.

Your 8 chunks became 8 points in that space — 6,144 numbers in total.

### The dimensions have no names

There is no "dimension 47 = technicality" or "dimension 112 = past tense." The axes are learned during training and are not individually interpretable. Nobody designed them; they emerged from the training objective.

What carries meaning is only **relative position**. The absolute coordinates of a single vector tell you nothing. The distance between two vectors tells you everything.

This is worth internalising because it explains a rule that otherwise looks arbitrary — see §8 on why the same model must be used at index time and query time.

### Why 768 dimensions?

Enough capacity to separate fine distinctions in meaning; small enough to store and compare quickly. It is an engineering trade-off chosen by whoever trained the model, not a property of language. Other models use 384, 1024, or 1536.

### An embedding is not a summary

You cannot read a vector, and you cannot reconstruct the original text from it. It is a lossy, one-way projection designed for exactly one operation: **comparing meaning**. That is why the original text must be stored alongside the vector — see §7.

---

## 6. Stage 5 — Cosine similarity: why angle, not distance

Once texts are points in space, "relevant" becomes "close." But *close* has to be defined, and the definition matters.

### Two ways to measure closeness

**Euclidean distance (L2)** — straight-line distance between two points, the everyday notion of distance.

**Cosine similarity** — the angle between the two vectors, measured from the origin. Ranges from -1 (opposite) through 0 (unrelated) to 1 (identical direction).

### Why angle wins here

Embedding vectors have both a **direction** and a **magnitude**. Direction encodes *what the text is about*. Magnitude correlates with incidental properties — notably **length of the input**.

That matters because of a structural asymmetry in this system:

- your **question** is around 10 words
- your **chunks** are around 400 words

A 400-word chunk tends to produce a longer vector than a 10-word question. Under Euclidean distance, **every chunk looks far from every question** — not because the meanings differ, but because the magnitudes do. Length noise drowns out the meaning signal.

Cosine similarity divides out magnitude entirely. Only direction survives the comparison. A 10-word question and a 400-word chunk about the same subject point the same way, and score high.

### The practical consequence in the code

ChromaDB defaults to **squared L2 distance**. The collection is therefore created explicitly with:

```python
metadata={"hnsw:space": "cosine"}
```

Leaving the default would not raise an error. It would silently degrade every query in the system, for a reason very hard to trace back from the symptom ("answers are a bit off"). This is a recurring theme: **the dangerous failures in RAG are the ones that do not crash.**

### What the sanity test demonstrates

```
A: "The Toubkal supercomputer has GPU nodes for scientific computing."
B: "Toubkal's cluster includes NVIDIA accelerators for HPC workloads."
C: "I baked a chocolate cake for my sister's birthday party."
```

A and B share almost no vocabulary — `supercomputer/cluster`, `GPU nodes/NVIDIA accelerators`, `scientific computing/HPC workloads`. A keyword system rates them unrelated. A working embedding model rates them close, and rates C far from both.

If that ordering does not hold, every stage built on top is worthless — and would still appear to work, returning irrelevant chunks that the model then answers from confidently. **Verifying the numerical property directly, before building on it, is far cheaper than debugging it three days later.**

---

## 7. Stage 6 — The vector database: what is actually stored

A vector database is optimised for exactly one operation: *given this vector, return the N stored vectors closest to it, quickly*. It uses approximate nearest-neighbour indexes (HNSW, in Chroma's case) rather than comparing against every stored vector one by one.

Four things are stored per chunk:

| Field | Example | Purpose |
|---|---|---|
| `id` | `Impression...::3` | Unique key. Deterministic — see below |
| `embedding` | 768 floats | What actually gets searched |
| `document` | the original chunk text | What gets returned and shown to the model |
| `metadata` | `{source: "...pdf", chunk_index: 3}` | Lets an answer cite where it came from |

### The row that is easiest to overlook

**`document` — the original text — is stored alongside the vector, because the vector cannot be reversed.**

Chroma does not search your text. It searches vectors, finds the closest ones, and hands back the text it filed next to them. The vector is an **index**; the text is the **payload**. Retrieval is a two-step move: compare numbers, return words.

### Why deterministic IDs matter

IDs are built as `filename::chunk_index` rather than randomly generated. Combined with `upsert`, re-indexing the same PDF **overwrites** its previous entries instead of appending a second copy.

With random IDs, indexing the same document three times gives three copies of every chunk. Then a top-3 query returns *the same passage three times* — the system appears to retrieve three sources while actually seeing one. Retrieval quality quietly degrades every time you run the indexer, which is a genuinely difficult bug to notice.

### Why persistence matters

`PersistentClient` writes to `chroma_db/` on disk. Embedding is by far the slowest step in the project; re-running it because a terminal was closed is pure waste. Index once, query many times.

### Verifying independently

After indexing, the database was inspected directly rather than trusting the script's own output — confirming a `documents` collection holding 8 embeddings, 736 KB on disk.

A script printing "stored 8 chunks" reports its *intent*, not the *state of the database*. Confirming from the other side is a habit worth keeping for any pipeline that writes to persistent state.

---

## 8. How the stages depend on each other

```
   PDF  ──►  text  ──►  clean text  ──►  chunks  ──►  vectors  ──►  database
        (1)        (2)             (3)          (4)           (5)
```

### Errors flow downhill and cannot be repaired upstream

Every stage trusts its input completely. None can detect that what it received is damaged.

- If extraction interleaves two columns, cleaning tidies the whitespace of gibberish.
- If chunking splits an answer across a boundary, embedding faithfully encodes two half-answers.
- If a chunk is meaningless, its vector is a precise representation of meaninglessness.
- The generation model then answers confidently from whatever it is handed.

**No model, however capable, can recover information that the pipeline destroyed before it arrived.** This is why the eyeball inspection at Day 2 was a real quality gate and not busywork — it is the last point at which a human can still read the data before it turns into numbers.

### The one rule that must never be broken

**The same embedding model must be used when indexing and when querying.**

Recall from §5 that embedding dimensions have no intrinsic meaning — only relative position matters, and that positioning is specific to one trained model. Two different models produce vectors in **entirely unrelated coordinate spaces**. Comparing across them is like comparing a latitude to a temperature: the arithmetic runs, the result is a number, and the number means nothing.

The failure is silent. No crash, no warning — just a system that retrieves essentially random chunks and answers from them. This is why `EMBED_MODEL` is defined once in `config.py` and imported everywhere, rather than typed out at each call site.

---

## 9. Questions to test your own understanding

If you can answer these without looking, the concepts have landed. These are also, almost verbatim, what an interviewer would ask.

1. Why can't the whole document be embedded as a single vector?
2. Why does overlap exist, and what does it cost?
3. Your document is 2,541 words. With 400-word chunks and 50-word overlap, how many chunks result, and why?
4. Why does keyword search fail on the question "what supercomputer did you work on?"
5. Why cosine similarity instead of Euclidean distance?
6. Why must the original text be stored next to the vector?
7. What happens if a different embedding model is used at query time, and how would you notice?
8. Why would random chunk IDs degrade retrieval quality over time?
9. A question gets a wrong answer. What do you check first — the prompt, the model, or the retrieved chunks? Why?
10. What kinds of document would break this pipeline, and at which stage?

*(Answer to 9, since it is the one that matters most: check the retrieved chunks. Most bad RAG answers are retrieval failures wearing a generation-failure costume. If the right passage was never retrieved, no prompt change and no larger model will fix it.)*
