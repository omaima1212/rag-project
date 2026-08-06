"""
Day 5 - Evaluation harness: run the whole question set and SAVE the results.

Why this exists: you cannot improve what you do not measure. Asking questions
by hand and forming an impression is not evidence - impressions drift, and you
will not remember what the answer looked like before you changed a setting.
This runs a fixed set of questions, records everything, and writes a timestamped
report you can diff against later runs.

    python src\evaluate.py                  # run with current config
    python src\evaluate.py "k=2 test"       # label the run
"""

import json
import sys
import time
from datetime import datetime

from config import (CHAT_MODEL, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL,
                    PROJECT_ROOT, TOP_K)
from generate import ask_model, build_prompt, looks_like_refusal
from retrieve import retrieve

QUESTIONS_FILE = PROJECT_ROOT / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# REFUSAL_MARKERS and looks_like_refusal used to live here. Moved to
# generate.py when the web fallback was added, because that heuristic became
# load-bearing in a second place - deciding whether to offer a web search -
# and a definition used to decide behaviour has to have exactly one home.
# Imported back in above so nothing here changes.

# Added while hardening the suite (it previously reported 8/8 in every
# configuration tested, including ones with a known-wrong retrieved chunk -
# see BUILD_LOG.md). A keyword match alone cannot tell a grounded answer from
# one that hedges around a lucky keyword collision, so hedging downgrades a
# would-be PASS to CHECK rather than letting it through silently.
HEDGE_MARKERS = [
    "it is likely", "it's likely", "probably", "might be", "may be",
    "possibly", "it seems", "presumably", "i am not sure", "i'm not sure",
    "i believe", "appears to",
]


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def grade(question, answer):
    """Return PASS / FAIL / CHECK for one answer.

    - unanswerable question -> must refuse
    - answerable question   -> must contain at least one expected term
                               AND must not refuse
    CHECK means the automatic rule could not decide; a human must read it.
    """
    refused = looks_like_refusal(answer)

    if not question["answerable"]:
        return ("PASS", "correctly refused") if refused else \
               ("FAIL", "answered a question the document cannot support")

    if refused:
        return "FAIL", "refused a question the document does answer"

    low = answer.lower()
    hits = [term for term in question["expect_any"] if term.lower() in low]
    if not hits:
        return "CHECK", "no expected term found - read the answer yourself"

    hedges = [m for m in HEDGE_MARKERS if m in low]
    if hedges:
        return ("CHECK", f"found {', '.join(hits)}, but hedged with "
                f"'{hedges[0]}' - read the answer yourself")

    return "PASS", f"found {', '.join(hits)}"


def run_suite(label=""):
    questions = load_questions()
    RESULTS_DIR.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_line = (f"chunk={CHUNK_SIZE} overlap={CHUNK_OVERLAP} k={TOP_K} "
                   f"embed={EMBED_MODEL} chat={CHAT_MODEL}")

    print(f"Running {len(questions)} questions")
    print(f"Config: {config_line}\n")

    rows = []
    for q in questions:
        print(f"[{q['id']}] {q['question']}")
        start = time.time()

        hits = retrieve(q["question"])
        answer = ask_model(build_prompt(q["question"], hits)).strip()

        elapsed = time.time() - start
        status, reason = grade(q, answer)

        print(f"    {status}  ({reason})  {elapsed:.1f}s")
        print(f"    chunks {[h['chunk_index'] for h in hits]}  "
              f"top similarity {hits[0]['similarity']:.4f}\n")

        rows.append({
            "q": q, "hits": hits, "answer": answer,
            "status": status, "reason": reason, "elapsed": elapsed,
        })

    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = sum(1 for r in rows if r["status"] == "FAIL")
    check = sum(1 for r in rows if r["status"] == "CHECK")

    print("=" * 70)
    print(f"PASS {passed}   FAIL {failed}   CHECK {check}   of {len(rows)}")
    print("=" * 70)

    path = RESULTS_DIR / f"run_{stamp}.md"
    write_report(path, label, config_line, rows, passed, failed, check)
    print(f"\nReport written to {path}")
    return rows


def write_report(path, label, config_line, rows, passed, failed, check):
    """Save a readable report. Markdown so successive runs can be diffed."""
    lines = [
        f"# Evaluation run — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"**Label:** {label or '(none)'}",
        f"**Config:** `{config_line}`",
        f"**Result:** PASS {passed} · FAIL {failed} · CHECK {check} "
        f"of {len(rows)}",
        "",
        "## Summary",
        "",
        "| ID | Status | Chunks retrieved | Top similarity | Time |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        chunks = ", ".join(str(h["chunk_index"]) for h in r["hits"])
        lines.append(
            f"| {r['q']['id']} | {r['status']} | {chunks} | "
            f"{r['hits'][0]['similarity']:.4f} | {r['elapsed']:.1f}s |"
        )

    lines += ["", "## Detail", ""]
    for r in rows:
        q = r["q"]
        lines += [
            f"### {q['id']} — {q['question']}",
            "",
            f"*Tests:* {q['tests']}",
            "",
            f"**{r['status']}** — {r['reason']}",
            "",
            "**Answer:**",
            "",
            "> " + r["answer"].replace("\n", "\n> "),
            "",
            "**Retrieved:**",
            "",
            "| Rank | Chunk | Similarity | First 150 characters |",
            "|---|---|---|---|",
        ]
        for rank, h in enumerate(r["hits"], start=1):
            preview = " ".join(h["text"].split())[:150].replace("|", "\\|")
            lines.append(
                f"| {rank} | {h['chunk_index']} | {h['similarity']:.4f} | "
                f"{preview}... |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_suite(" ".join(sys.argv[1:]))
