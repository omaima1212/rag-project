"""
Day 4 - Stage 3 of the pipeline: question -> the most relevant chunks.

This is the "R" in RAG, and it is where most of the quality of the whole
system lives. If the right chunk is not retrieved here, nothing downstream
can recover it.

Run it on its own to inspect retrieval WITHOUT any model involved:
    python src\retrieve.py "what supercomputer did you work on?"
"""

import sys

from config import TOP_K
from embed import embed_text, get_collection


def retrieve(question, k=TOP_K, collection_name=None):
    """Find the k chunks most similar in meaning to the question.

    Three steps:
      1. Embed the question with the SAME model used to index the documents.
         (Different models = unrelated coordinate spaces = meaningless scores.)
      2. Ask Chroma for the k nearest stored vectors.
      3. Return the TEXT filed alongside those vectors.

    Note step 3: the vector is only an index. The text is the payload.
    """
    collection = (get_collection(collection_name) if collection_name
                  else get_collection())

    if collection.count() == 0:
        raise RuntimeError(
            "The collection is empty. Index a PDF first:\n"
            '    python src\\embed.py "data\\yourfile.pdf"'
        )

    question_vector = embed_text(question)

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns a list-per-query; we only sent one query, so take index 0.
    hits = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # With hnsw:space="cosine", Chroma reports cosine DISTANCE.
        # Convert to the similarity score we reason about: 1 = identical.
        hits.append({
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "similarity": 1 - distance,
        })

    return hits


def print_hits(hits, preview_chars=400):
    """Display retrieved chunks for inspection.

    Get in the habit of looking at this BEFORE looking at the model's answer.
    Most bad RAG answers are retrieval failures wearing a generation-failure
    costume - the model was simply never shown the right passage.
    """
    print(f"\nRetrieved {len(hits)} chunks:\n")
    for rank, hit in enumerate(hits, start=1):
        print("=" * 70)
        print(f"#{rank}  similarity {hit['similarity']:.4f}  "
              f"| chunk {hit['chunk_index']} of {hit['source']}")
        print("=" * 70)
        print(hit["text"][:preview_chars].strip())
        if len(hit["text"]) > preview_chars:
            print("   [...]")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src\\retrieve.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}")
    print_hits(retrieve(question))

    # MEASURED on the internship logbook with nomic-embed-text (Day 5).
    # An earlier guess here said "> 0.75 = strong match". That was wrong: no
    # correct answer in the whole evaluation scored above 0.65.
    print("Measured score ranges for this document + model:")
    print("  correct answers   0.49 - 0.64")
    print("  correct refusals  0.36 - 0.43")
    print()
    print("Do NOT threshold on these numbers. A correct hit scored 0.4923")
    print("while an irrelevant one scored 0.4291 - a gap of 0.06. Absolute")
    print("cosine values are not calibrated across documents or models;")
    print("only the RELATIVE ordering within a single query is meaningful.")
