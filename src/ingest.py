"""
Day 2 - Stage 1 of the pipeline: PDF file  ->  list of clean text chunks.

Run it directly to inspect what comes out:
    python src\ingest.py data\yourfile.pdf
"""

import re
import sys
from pathlib import Path

import fitz  # this is PyMuPDF - the package is 'pymupdf', the import is 'fitz'

from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text(pdf_path, verbose=True):
    """Open a PDF and return all of its text as one string.

    Each page is separated by a form-feed marker so you can see page
    boundaries when you print the output. PyMuPDF reads the page's internal
    text objects in document order - which is why multi-column layouts often
    come out interleaved and wrong. That is expected; you will hit it on Day 5.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No such PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append(text)
        if verbose:
            print(f"  page {page_number}: {len(text)} characters")
    doc.close()

    full_text = "\n\n".join(pages)
    if verbose:
        print(f"  -> extracted {len(full_text)} characters "
              f"from {len(pages)} pages")
    return full_text


def clean_text(text):
    """Tidy up the raw extraction.

    PDFs are full of artefacts: hyphens splitting words across line breaks,
    single newlines inside a sentence, runs of whitespace. Cleaning here means
    the chunker sees sentences rather than fragments.
    """
    # Re-join words hyphenated across a line break: "informa-\ntion" -> "information"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # A single newline inside a paragraph is a layout artefact, not a break.
    # Two or more newlines is a real paragraph break - keep those.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse runs of spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse three or more blank lines down to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of roughly `chunk_size` words.

    Why chunks: a chunk is the atomic unit of retrieval. You retrieve whole
    chunks, never fragments, so a chunk must be small enough to be specific
    and large enough to be self-contained.

    Why overlap: a sentence that straddles a chunk boundary would otherwise be
    split across two chunks and be retrievable from neither. Repeating the last
    `overlap` words at the start of the next chunk guarantees any short passage
    appears intact in at least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap   # how far the window advances each time

    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        # Stop once the window has reached the end, otherwise the overlap
        # would produce a final near-duplicate chunk.
        if start + chunk_size >= len(words):
            break

    return chunks


# French month names, because the test document dates its entries in French.
# Extend this list for documents in other languages.
MONTHS = ("janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
          "septembre|octobre|novembre|décembre|decembre|"
          "January|February|March|April|May|June|July|August|September|"
          "October|November|December")
DATE_RE = re.compile(rf"\b\d{{1,2}}\s+({MONTHS})\s+\d{{4}}\b", re.IGNORECASE)


def extract_dates(text):
    """Return every date-like string in the text, in order of appearance."""
    return [m.group(0) for m in DATE_RE.finditer(text)]


def add_context_headers(chunks, source_name, include_filename=True):
    """Prepend a short context header to each chunk BEFORE embedding.

    THE PROBLEM THIS SOLVES (measured on Day 5):
    The question "What happened on the first day of the internship?" failed at
    k=1. The correct chunk ranked SECOND, because embeddings encode meaning
    and nothing in a chunk's text says "this is the start of the document".
    The question's key term - "first day" - had no counterpart to match on.

    THE FIX:
    Put the missing information into the text itself, so it ends up in the
    vector. Each chunk gains a header naming its source, its position in the
    document, whether it is the beginning/middle/end, and the dates it covers.
    "beginning of the document ... the first day" now gives "first day" in a
    question something to match against.

    THE TRADE-OFF:
    The header is stored as part of the chunk, so the model sees it too and
    may occasionally quote it. That is the price of the vector carrying
    positional meaning. Measure whether the retrieval gain is worth it -
    do not assume.

    include_filename=False - STRUCTURE-ONLY variant, added after a Day 5
    caveat was measured rather than merely suspected: the header's filename
    text leaks into the semantic space. The test document's filename
    contained "stage" (French for internship), so every chunk's header
    matched the word "internship" - inflating similarity scores on q8, an
    UNANSWERABLE question about internship salary, from 0.4291 to 0.4770.
    This flag drops the filename from the header, keeping only position and
    dates, to test whether the positional fix survives without that cost.
    See BUILD_LOG.md, "structure-only header" experiment, for the measured
    A/B result and the decision on whether this became the default.
    """
    total = len(chunks)
    out = []
    for i, chunk in enumerate(chunks):
        parts = [f"Part {i + 1} of {total}"]
        if include_filename:
            parts.insert(0, f"Source: {source_name}")

        if i == 0:
            parts.append("This is the BEGINNING of the document: "
                         "the first day, the start, the earliest entries")
        elif i == total - 1:
            parts.append("This is the END of the document: "
                         "the last day, the conclusion, the final entries")
        else:
            parts.append("This is the MIDDLE of the document")

        dates = extract_dates(chunk)
        if dates:
            span = (dates[0] if len(dates) == 1
                    else f"{dates[0]} to {dates[-1]}")
            parts.append(f"Dates covered: {span}")

        out.append("[" + " | ".join(parts) + "]\n" + chunk)

    return out


def load_and_chunk(pdf_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                   verbose=True):
    """The full Day 2 pipeline: PDF path in, list of chunks out.

    chunk_size and overlap are parameters rather than fixed constants so that
    Day 5 can sweep them and MEASURE their effect instead of guessing.
    """
    if verbose:
        print(f"Reading {pdf_path}")
    raw = extract_text(pdf_path, verbose=verbose)
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
    if verbose:
        print(f"  -> {len(chunks)} chunks "
              f"({chunk_size} words each, {overlap} word overlap)")
    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src\\ingest.py path\\to\\file.pdf")
        sys.exit(1)

    chunks = load_and_chunk(sys.argv[1])

    # Sanity check: READ THESE WITH YOUR OWN EYES.
    # You are looking for garbled text, interleaved columns, missing sections,
    # or chunks that cut off mid-thought in a way that destroys the meaning.
    # Bad chunks here guarantee bad answers on Day 4 - no model can fix them.
    print("\n" + "=" * 70)
    print("FIRST CHUNK")
    print("=" * 70)
    print(chunks[0][:1200])

    print("\n" + "=" * 70)
    print("MIDDLE CHUNK")
    print("=" * 70)
    print(chunks[len(chunks) // 2][:1200])

    print("\n" + "=" * 70)
    print("LAST CHUNK")
    print("=" * 70)
    print(chunks[-1][:1200])

    lengths = [len(c.split()) for c in chunks]
    print("\n" + "=" * 70)
    print(f"Total chunks : {len(chunks)}")
    print(f"Word counts  : min {min(lengths)}, max {max(lengths)}, "
          f"average {sum(lengths) // len(lengths)}")
