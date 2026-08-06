"""
Day 4 - Stage 4, the final stage: retrieved chunks + question -> grounded answer.

This is the "AG" in RAG: the retrieved passages AUGMENT the prompt, and the
model GENERATES an answer from them. The model's own training knowledge is
deliberately suppressed in favour of the supplied text.

Ask a single question:
    python src\generate.py "what supercomputer did you work on?"

Or run without arguments for an interactive loop.
"""

import sys

import requests

from config import CHAT_MODEL, OLLAMA_URL, TOP_K
from retrieve import retrieve

# The instruction block is the entire defence against hallucination.
# Each line exists for a specific failure it prevents:
#   - "ONLY the context"      -> stops the model answering from training data
#   - explicit refusal string -> gives it a safe action when the answer is absent,
#                                because a model with no permitted way to fail
#                                will invent something instead
#   - "do not use prior knowledge" -> closes the obvious loophole
#   - "quote briefly"         -> makes the answer traceable to the source
SYSTEM_PROMPT = """You answer questions using ONLY the context provided by the user.

Rules:
- Use only information present in the context. Do not use prior knowledge.
- If the context does not contain the answer, reply exactly:
  "The document does not contain this information."
- Do not guess, infer beyond the text, or fill gaps with plausible detail.
- Quote briefly from the context to support your answer.
- Never refer to "Source N", "chunk N", or any other internal label from the
  context in your answer - the reader never sees those labels, only your
  answer. Refer to the document by name if you need to attribute something.
- Be concise."""

# Substring markers used to detect a refusal. Deliberately crude, and moved
# here (from evaluate.py, where this lived until the web fallback was added)
# because it is now load-bearing in TWO places that must agree: the
# evaluation harness's grading, and the web-fallback trigger in the UIs. One
# definition, two callers - the config.py principle applied to behaviour
# instead of settings. The system prompt asks for one exact sentence; small
# models paraphrase, so several markers catch paraphrases at the cost of
# occasionally matching a genuine answer that happens to contain one.
REFUSAL_MARKERS = [
    "does not contain",
    "not contain this information",
    "no information",
    "not mentioned",
    "cannot answer",
]


def looks_like_refusal(answer_text):
    """Did the model decline to answer from the document?

    A TRIAGE signal, not a verdict - see evaluate.py's grade() for where
    this matters for scoring, and server.py / app.py for where it now also
    decides whether to offer the optional web fallback.
    """
    low = answer_text.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


# A different contract from SYSTEM_PROMPT, not a variant of it: this prompt
# is used ONLY after the document has already been asked and found not to
# contain the answer, and ONLY when the user has explicitly opted in to the
# web fallback (see websearch.py, README.md "Web fallback"). It must never
# be confused for document grounding - the rules below say so explicitly, and
# the UI layer labels its output as "from the web" wherever it is shown.
WEB_SYSTEM_PROMPT = """You answer questions using ONLY the web search results
provided by the user.

Rules:
- These results are from a live web search, NOT from the user's document.
  Say so if it is relevant to the answer.
- Base your answer only on the provided snippets, not on unstated prior
  knowledge.
- If the snippets do not answer the question either, say so plainly instead
  of guessing.
- Mention which result number(s) support your answer, e.g. "(result 1)".
- Be concise."""


def build_prompt(question, hits):
    """Assemble the retrieved chunks and the question into one user message.

    Structure matters. Context comes FIRST, question LAST - models attend most
    reliably to the end of a prompt, so the question stays fresh. Each chunk is
    labelled so YOU can tell which passage an answer came from when debugging
    (see SOURCES USED in the CLI, or the sources panel in the UI) - but the
    label is deliberately NOT "chunk N": that internal indexing detail was
    leaking into answers verbatim ("According to chunk 4 of the document..."),
    which means nothing to a reader who never sees chunk numbers. "Passage N"
    reads naturally if the model quotes it back, and the system prompt now
    also tells it explicitly not to cite internal labels at all.
    """
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"[Passage {i} - {hit['source']}]\n"
            f"{hit['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""CONTEXT:
{context}

---

QUESTION: {question}

Answer using only the context above."""


def build_web_prompt(question, results):
    """Same shape as build_prompt, but for web search results instead of
    document chunks - kept as a separate function rather than a branch
    inside build_prompt because the two must never accidentally merge: one
    produces document-grounded prompts, the other explicitly does not.
    """
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[Result {i} - {r['title']} ({r['link']})]\n{r['snippet']}")
    context = "\n\n---\n\n".join(blocks)

    return f"""WEB SEARCH RESULTS:
{context}

---

QUESTION: {question}

Answer using only the results above."""


def ask_model(prompt, system_prompt=SYSTEM_PROMPT):
    """Send the assembled prompt to llama3.2:3b via Ollama.

    system_prompt defaults to the document-grounding contract (SYSTEM_PROMPT)
    but the web fallback passes WEB_SYSTEM_PROMPT instead - same call
    machinery and error handling, a different set of rules for the model.
    """
    print(f"Asking {CHAT_MODEL} "
          f"(first call loads the model into RAM - can take a minute)...")

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "num_predict": 200,
                    # temperature 0 = as deterministic as possible. Creativity
                    # is the enemy here; we want faithful reporting of the
                    # context, not prose. Also makes runs reproducible.
                    "temperature": 0,
                    # Cap the context window. Ollama sizes its memory
                    # allocation from this, so on a low-RAM machine an
                    # oversized default is what pushes it into a crash.
                    # 4096 tokens comfortably fits 3 chunks plus the question.
                    "num_ctx": 4096,
                },
            },
            timeout=600,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Lost the connection to Ollama while generating.\n"
            "On a low-RAM machine this almost always means Ollama ran out of\n"
            "memory and the server process died.\n\n"
            "Try, in order:\n"
            "  1. ollama ps            (see what is loaded)\n"
            "  2. ollama stop --all    (free every loaded model)\n"
            "  3. close your browser, then retry\n"
            "  4. lower TOP_K to 2 in src/config.py"
        ) from None

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returned {response.status_code} from /api/chat\n"
            f"Model requested: {CHAT_MODEL!r}\n"
            f"Server said: {response.text.strip()}"
        )

    return response.json()["message"]["content"]


def ask_model_stream(prompt, system_prompt=SYSTEM_PROMPT):
    """Same call as ask_model, but yields the answer token by token.

    Ollama with "stream": True returns newline-delimited JSON rather than one
    response: each line is a fragment {"message": {"content": "..."}}, ending
    with {"done": true}.

    Why this matters more here than in most apps: generation takes 30-100
    seconds on this hardware. A user staring at a frozen spinner assumes the
    application has crashed. Streaming does not make it faster - it makes the
    waiting legible, and the first token typically arrives in a few seconds.

    system_prompt: see ask_model - same default, same web-fallback override.
    """
    import json

    with requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            # Keep the model resident for 30 minutes instead of Ollama's
            # 5-minute default. Without this, a pause between questions lets
            # the model unload, and the next question pays a full 3.9 GB
            # reload from disk before generating a single token. This is the
            # main reason answers in the UI felt slower than from the CLI.
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                # Cap the answer length. On CPU every token is expensive, and
                # a grounded answer quoting a source rarely needs more than
                # this. Prevents the model rambling for two extra minutes.
                "num_predict": 200,
            },
        },
        stream=True,
        timeout=600,
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned {response.status_code} from /api/chat\n"
                f"Server said: {response.text.strip()}"
            )

        for line in response.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("done"):
                break
            fragment = payload.get("message", {}).get("content", "")
            if fragment:
                yield fragment


def answer(question, k=TOP_K, show_sources=True, web_fallback=False):
    """The complete RAG pipeline, end to end.

    web_fallback: opt-in only (see README.md, "Web fallback"). Only fires
    when the grounded answer looks like a refusal - a question the document
    DOES answer never touches the network, whether or not this is True.
    """
    hits = retrieve(question, k=k)
    prompt = build_prompt(question, hits)
    reply = ask_model(prompt)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(reply.strip())

    if show_sources:
        print("\n" + "=" * 70)
        print("SOURCES USED (always check these against the answer)")
        print("=" * 70)
        for i, hit in enumerate(hits, start=1):
            preview = " ".join(hit["text"].split())[:220]
            print(f"\n[{i}] similarity {hit['similarity']:.4f} | "
                  f"chunk {hit['chunk_index']} of {hit['source']}")
            print(f"    {preview}...")

    if web_fallback and looks_like_refusal(reply):
        print("\n" + "=" * 70)
        print("WEB FALLBACK (NOT from your document - see README, 'Web fallback')")
        print("=" * 70)
        try:
            import websearch
            results = websearch.search_web(question)
            if not results:
                print("No web results found.")
            else:
                web_reply = ask_model(build_web_prompt(question, results),
                                      system_prompt=WEB_SYSTEM_PROMPT)
                print(web_reply.strip())
                print("\nSources:")
                for i, r in enumerate(results, start=1):
                    print(f"  [{i}] {r['title']} - {r['link']}")
        except RuntimeError as e:
            print(str(e))

    return reply, hits


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        web = "--web" in args
        args = [a for a in args if a != "--web"]
        answer(" ".join(args), web_fallback=web)
    else:
        print("RAG over your indexed documents. Blank line or Ctrl+C to quit.")
        print('Append "--web" to a question to allow a web search fallback '
              "if the document doesn't have the answer.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            web = question.endswith("--web")
            if web:
                question = question[:-len("--web")].strip()
            answer(question, web_fallback=web)
            print()
