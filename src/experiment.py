"""
Day 5 - Config sweep: measure the effect of chunk size, overlap and k
on RETRIEVAL ONLY. No language model involved.

Why retrieval only: generation is slow (30s+ per answer on this hardware) and
it adds a second source of variation. If you change chunk size and the answer
gets worse, was that the chunking or the model having a bad day? Isolating
retrieval makes each run take seconds instead of minutes, and makes the result
attributable to the one thing you changed.

This reflects the central lesson of Day 5: most bad RAG answers are retrieval
failures. Measure the retriever first.

    python src\experiment.py "data\yourfile.pdf"
"""

import sys
from pathlib import Path

from embed import embed_text, get_collection, index_pdf
from retrieve import retrieve

# (chunk_size, overlap) pairs to compare.
CONFIGS = [
    (200, 25),    # small chunks: precise but may lack context
    (400, 50),    # current setting
    (600, 75),    # large chunks: more context but diluted meaning
    (400, 0),     # same size as current, NO overlap - isolates overlap's effect
]

K_VALUES = [1, 3, 5]


def collection_name_for(chunk_size, overlap):
    return f"exp_c{chunk_size}_o{overlap}"


def build_all(pdf_path):
    """Index the same PDF once per configuration, into separate collections."""
    print("Building one index per configuration...\n")
    built = []
    for chunk_size, overlap in CONFIGS:
        name = collection_name_for(chunk_size, overlap)
        collection = index_pdf(pdf_path, collection_name=name,
                               chunk_size=chunk_size, overlap=overlap,
                               verbose=False)
        print(f"  {name:20s} -> {collection.count():3d} chunks")
        built.append((chunk_size, overlap, name, collection.count()))
    print()
    return built


def sweep_chunking(questions):
    """How does chunk size and overlap affect retrieval scores?"""
    print("=" * 78)
    print("CHUNK SIZE / OVERLAP SWEEP  (k=3, higher similarity is better)")
    print("=" * 78)

    # Label columns s1..sN, not q1..qN - questions.json already uses q1..q8
    # for a DIFFERENT set, and identical labels for different questions is a
    # reliable way to misread your own results later.
    for i, question in enumerate(questions, start=1):
        print(f"  s{i} = {question}")
    print()

    header = f"{'config':>14s} {'chunks':>7s}  " + \
             "  ".join(f"{'s' + str(i + 1):>6s}" for i in range(len(questions)))
    print(header)
    print("-" * len(header))

    for chunk_size, overlap in CONFIGS:
        name = collection_name_for(chunk_size, overlap)
        count = get_collection(name).count()
        cells = []
        for question in questions:
            hits = retrieve(question, k=3, collection_name=name)
            # Report WHICH chunk ranked first, not only how well it scored.
            # A higher score on the wrong chunk is not an improvement, and a
            # score-only table cannot tell the two apart - which is exactly
            # the blind spot that hid the positional failure in the first run.
            cells.append(f"{hits[0]['similarity']:.3f}#{hits[0]['chunk_index']}")
        row = f"{chunk_size:>6d}/{overlap:<7d} {count:>7d}  " + \
              "  ".join(f"{c:>8s}" for c in cells)
        print(row)

    print("\nCells read as  score#chunk_index.")
    print("Chunk indices are NOT comparable across rows - each configuration")
    print("cuts the document differently, so chunk 3 of the 200-word index is")
    print("different text from chunk 3 of the 400-word index. What IS")
    print("comparable: whether a configuration keeps returning the SAME chunk")
    print("for the same question, and whether that chunk actually holds the")
    print("answer - which you confirm by reading it, not by reading a score.\n")


def sweep_k(questions, collection_name=None):
    """How much does the top-k choice change what the model sees?"""
    print("=" * 78)
    print("TOP-K SWEEP  (which chunks come back, and how good is the worst one)")
    print("=" * 78)

    for question in questions:
        print(f"\n{question}")
        for k in K_VALUES:
            hits = retrieve(question, k=k, collection_name=collection_name)
            indices = ", ".join(str(h["chunk_index"]) for h in hits)
            best = hits[0]["similarity"]
            worst = hits[-1]["similarity"]
            print(f"  k={k}: chunks [{indices}]  "
                  f"best {best:.3f}  worst {worst:.3f}  "
                  f"spread {best - worst:.3f}")

    print("\nThe number that matters is the WORST similarity at each k.")
    print("A large spread means the extra chunks are padding the prompt with")
    print("material the retriever itself rates as barely relevant - which")
    print("costs context window and can distract a small model.\n")


def sweep_headers(pdf_path, questions):
    """A/B test: context headers ON versus OFF, everything else identical.

    This is the controlled experiment for the Day 5 fix. Chunk size, overlap,
    k, and the models are all held constant. The ONLY difference is whether
    each chunk carries a position header. Any change is therefore attributable
    to the headers and to nothing else.
    """
    print("=" * 78)
    print("CONTEXT HEADERS A/B  (400/50, k=3 - headers are the only variable)")
    print("=" * 78)

    index_pdf(pdf_path, collection_name="exp_noheaders",
              chunk_size=400, overlap=50, headers=False, verbose=False)
    index_pdf(pdf_path, collection_name="exp_headers",
              chunk_size=400, overlap=50, headers=True, verbose=False)

    print(f"\n{'question':<46s} {'OFF':>14s} {'ON':>14s}   verdict")
    print("-" * 96)

    for question in questions:
        off = retrieve(question, k=3, collection_name="exp_noheaders")
        on = retrieve(question, k=3, collection_name="exp_headers")

        off_cell = f"{off[0]['similarity']:.3f}#{off[0]['chunk_index']}"
        on_cell = f"{on[0]['similarity']:.3f}#{on[0]['chunk_index']}"

        if off[0]["chunk_index"] != on[0]["chunk_index"]:
            verdict = "RANK 1 CHANGED"
        elif on[0]["similarity"] > off[0]["similarity"]:
            verdict = f"+{on[0]['similarity'] - off[0]['similarity']:.3f}"
        else:
            verdict = f"{on[0]['similarity'] - off[0]['similarity']:.3f}"

        print(f"{question[:44]:<46s} {off_cell:>14s} {on_cell:>14s}   {verdict}")

    print("\nThe cell that matters is the positional question. Headers were")
    print("added to fix ITS rank-1 chunk, not to raise scores generally.")
    print("A higher score on the same wrong chunk is not a fix.\n")


def sweep_header_filename(pdf_path, questions):
    """A/B/C test: no headers vs filename headers vs structure-only headers.

    Added to chase a caveat recorded on Day 5 rather than left as a guess:
    the adopted header format embeds the source filename, and on the test
    document that filename contains "stage" (French for internship) - so
    every chunk's header matches the word "internship", inflating similarity
    on q8, an UNANSWERABLE question about internship salary. This sweep
    checks whether dropping the filename (position + dates only) keeps the
    Day 5 fix (q4, the positional question) while removing that inflation.

    `questions` should be the full questions.json set, id included, so the
    unanswerable questions (which is where the cost showed up) are covered -
    not just the five topical/positional ones used elsewhere in this file.
    """
    print("=" * 78)
    print("HEADER FILENAME A/B/C  (400/50, k=3 - header CONTENT is the only "
          "variable)")
    print("=" * 78)

    index_pdf(pdf_path, collection_name="exp_hdr_none",
              chunk_size=400, overlap=50, headers=False, verbose=False)

    # headers=True always adds context headers via index_pdf -> we call
    # add_context_headers ourselves for the two header variants so we can
    # control include_filename, which index_pdf does not expose.
    from embed import embed_text, get_collection
    from ingest import add_context_headers, load_and_chunk

    for coll_name, include_filename in [("exp_hdr_full", True),
                                        ("exp_hdr_struct", False)]:
        chunks = load_and_chunk(pdf_path, chunk_size=400, overlap=50,
                                verbose=False)
        chunks = add_context_headers(chunks, Path(pdf_path).name,
                                     include_filename=include_filename)
        collection = get_collection(coll_name)
        collection.upsert(
            ids=[f"{Path(pdf_path).stem}::{i}" for i in range(len(chunks))],
            embeddings=[embed_text(c) for c in chunks],
            documents=chunks,
            metadatas=[{"source": Path(pdf_path).name, "chunk_index": i}
                       for i in range(len(chunks))],
        )

    print(f"\n{'id':<4s} {'no headers':>16s} {'+ filename':>16s} "
          f"{'structure-only':>16s}   note")
    print("-" * 90)

    for q in questions:
        none_hit = retrieve(q["question"], k=3, collection_name="exp_hdr_none")[0]
        full_hit = retrieve(q["question"], k=3, collection_name="exp_hdr_full")[0]
        struct_hit = retrieve(q["question"], k=3, collection_name="exp_hdr_struct")[0]

        note = ""
        if not q["answerable"]:
            note = "unanswerable - lower is better"

        cells = [f"{h['similarity']:.4f}#{h['chunk_index']}"
                for h in (none_hit, full_hit, struct_hit)]
        print(f"{q['id']:<4s} {cells[0]:>16s} {cells[1]:>16s} "
              f"{cells[2]:>16s}   {note}")

    print("\nColumns: no headers at all / current default (filename + "
          "position + dates) / structure-only (position + dates, no "
          "filename). For q4 (positional) a higher score AND rank-0 chunk "
          "matters. For q7/q8 (unanswerable) a LOWER score is the win - it "
          "means the header stopped inflating the match.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src\\experiment.py "data\\yourfile.pdf"')
        sys.exit(1)

    pdf_path = Path(" ".join(sys.argv[1:]))

    # The first run of this sweep used four purely TOPICAL questions, all of
    # which are answered by the same chunk. Every configuration therefore
    # ranked that chunk first, and the sweep could not tell k=1 from k=3.
    #
    # A sweep over questions the system already handles well will report that
    # configuration does not matter - whether or not it does. Include the
    # cases known to be fragile, or the experiment measures nothing.
    questions = [
        # topical - easy, answered by one obvious chunk
        "What supercomputer did you work on during the internship?",
        "How many standard compute nodes does the cluster have?",
        "What GPUs are used in the cluster?",
        # POSITIONAL - the known-fragile case. In the Day 5 baseline the
        # correct chunk ranked SECOND here, so this question is the only one
        # in the set capable of distinguishing k=1 from k=3.
        "What happened on the first day of the internship?",
        # end-of-document content, in the shortest chunk
        "What was the purpose of the Slurm script?",
    ]

    build_all(pdf_path)
    sweep_chunking(questions)
    sweep_k(questions, collection_name=collection_name_for(400, 50))
    sweep_headers(pdf_path, questions)
