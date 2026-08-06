"""
Day 3 - Stage 2 of the pipeline: chunks -> vectors -> ChromaDB.

Run it to index a PDF:
    python src\embed.py "data\yourfile.pdf"

Run it with no arguments to see what is already stored:
    python src\embed.py
"""

import sys
from pathlib import Path

import chromadb
import requests

from config import CHROMA_DIR, EMBED_MODEL, OLLAMA_URL
from ingest import load_and_chunk

COLLECTION_NAME = "documents"


def embed_text(text):
    """Turn one piece of text into a vector using Ollama.

    An embedding is a list of numbers that encodes MEANING. Texts with similar
    meaning produce vectors that point in similar directions, which is what
    makes semantic search possible: we can find relevant text without any
    keyword overlap at all.

    nomic-embed-text returns 768 numbers per text.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is it running?\n"
            "Start it with:  ollama serve"
        ) from None

    if response.status_code != 200:
        # raise_for_status() hides the server's explanation. Ollama returns a
        # JSON body like {"error": "..."} that says what actually went wrong -
        # usually a bad model name or not enough memory. Always surface it.
        raise RuntimeError(
            f"Ollama returned {response.status_code} from /api/embeddings\n"
            f"Model requested: {EMBED_MODEL!r}\n"
            f"Server said: {response.text.strip()}"
        )

    return response.json()["embedding"]


def get_collection(name=COLLECTION_NAME):
    """Open (or create) a persistent ChromaDB collection.

    The name is a parameter so Day 5 can build several collections with
    different chunking settings and compare them side by side, without
    destroying the main index.

    PersistentClient writes to disk, so the index survives restarting the
    script. Embedding is by far the slowest step in the project - you never
    want to redo it just because you closed a terminal.

    hnsw:space="cosine" is important. Chroma defaults to squared L2 distance;
    we want COSINE similarity, which compares the ANGLE between vectors and
    therefore ignores text length. Two passages about the same topic should
    match whether one is 50 words and the other 400.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def index_pdf(pdf_path, collection_name=COLLECTION_NAME,
              chunk_size=None, overlap=None, headers=None, verbose=True):
    """Full indexing pipeline: PDF -> chunks -> vectors -> stored in Chroma."""
    from config import (CHUNK_OVERLAP, CHUNK_SIZE, HEADER_INCLUDE_FILENAME,
                       USE_CONTEXT_HEADERS)
    from ingest import add_context_headers

    chunk_size = chunk_size or CHUNK_SIZE
    overlap = CHUNK_OVERLAP if overlap is None else overlap
    headers = USE_CONTEXT_HEADERS if headers is None else headers

    pdf_path = Path(pdf_path)
    chunks = load_and_chunk(pdf_path, chunk_size=chunk_size,
                            overlap=overlap, verbose=verbose)

    if headers:
        chunks = add_context_headers(chunks, pdf_path.name,
                                     include_filename=HEADER_INCLUDE_FILENAME)
        if verbose:
            print("  -> context headers added; first chunk now begins:")
            print(f"     {chunks[0].splitlines()[0]}")

    collection = get_collection(collection_name)

    if verbose:
        print(f"\nEmbedding {len(chunks)} chunks with {EMBED_MODEL}...")
    ids, vectors, metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        # Deterministic IDs: re-indexing the same PDF OVERWRITES the old
        # entries instead of creating duplicate copies. Duplicate chunks are a
        # nasty bug - they crowd out genuinely different results in top-k.
        ids.append(f"{pdf_path.stem}::{i}")
        vectors.append(embed_text(chunk))
        metadatas.append({"source": pdf_path.name, "chunk_index": i})
        if verbose:
            print(f"  chunk {i + 1}/{len(chunks)} embedded "
                  f"({len(vectors[-1])} dimensions)")

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
    )

    if verbose:
        print(f"\nStored {len(chunks)} chunks in {CHROMA_DIR}")
        print(f"Collection '{collection_name}' now holds "
              f"{collection.count()} chunks total")
    return collection


def cosine_similarity(a, b):
    """Similarity between two vectors, from -1 (opposite) to 1 (identical).

    Written out by hand rather than imported, so the concept is concrete:
    it is the dot product divided by the product of the two lengths, which
    is exactly the cosine of the angle between the vectors.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def sanity_test():
    """Prove the embeddings actually encode meaning.

    Do not skip this. If these numbers come out wrong, everything built on top
    of them is worthless, and the failure is silent - you would only notice it
    on Day 4 as mysteriously bad answers.
    """
    print("\n" + "=" * 70)
    print("SANITY TEST - do similar texts really produce closer vectors?")
    print("=" * 70)

    a = "The Toubkal supercomputer has GPU nodes for scientific computing."
    b = "Toubkal's cluster includes NVIDIA accelerators for HPC workloads."
    c = "I baked a chocolate cake for my sister's birthday party."

    va, vb, vc = embed_text(a), embed_text(b), embed_text(c)

    sim_ab = cosine_similarity(va, vb)
    sim_ac = cosine_similarity(va, vc)

    print(f"\nA: {a}")
    print(f"B: {b}")
    print(f"C: {c}")
    print(f"\nsimilarity(A, B) = {sim_ab:.4f}   <- same topic, different words")
    print(f"similarity(A, C) = {sim_ac:.4f}   <- unrelated topic")

    if sim_ab > sim_ac:
        print("\nPASS - the embedding model encodes meaning, not vocabulary.")
        print("Note that A and B share almost no words, yet score high.")
    else:
        print("\nFAIL - something is wrong with the embedding setup.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        index_pdf(sys.argv[1])
        sanity_test()
    else:
        collection = get_collection()
        print(f"Collection '{COLLECTION_NAME}' holds "
              f"{collection.count()} chunks.")
        if collection.count():
            peek = collection.peek(limit=3)
            for doc_id, meta in zip(peek["ids"], peek["metadatas"]):
                print(f"  {doc_id}  <- {meta['source']}")
