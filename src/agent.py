"""ADAS assistant agent using local Ollama tools and the shared RAG index."""
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import (ADAS_COLLECTION_NAME, ADAS_SCOPE_KEYWORDS, CHAT_MODEL,
                    MAX_AGENT_ITERATIONS, OLLAMA_URL, TOP_K,
                    AGENT_KEEP_ALIVE, AGENT_NUM_CTX, AGENT_NUM_PREDICT,
                    AGENT_REQUEST_TIMEOUT_SECONDS, QUIZ_MAX_QUESTIONS,
                    ADAS_VEHICLE_MAKES, WEB_COOLDOWN_MINUTES)
from retrieve import retrieve, reindex_missing_page_metadata
from embed import get_collection
from citations import source_citation
import websearch

logger = logging.getLogger(__name__)
_web_cooldown_until = 0.0
_web_consecutive_failures = 0
WEB_FAILURES_BEFORE_COOLDOWN = 2
REACTION_TIME_SECONDS = 1.5

ASSISTANT_POLICY = """You are Sentinel, an assistant that helps people understand ADAS.
You are an ADAS and closely related vehicle-safety assistant.
Scope: ADAS concepts, operation, limitations, safety, regulations, terminology,
comparisons, vehicle safety systems, and closely related automotive technology.
For every in-scope user request, search_docs has already been called first.
When the passages list periods or eras, include all of them in order with their exact dates; do not merge or skip any.
Never draw ASCII art or text diagrams. A chart is attached by the app when relevant.

Answer in this order:
1. If retrieved passages support the answer, answer from those passages and cite
   them as the supplied source of evidence. Never invent a citation.
2. If the passages do not cover the ADAS question, use only the web snippets
   supplied in the prompt. If none are supplied, answer from general knowledge
   cautiously. Never present external facts as document facts.
3. For unrelated questions, politely decline in one sentence and suggest ADAS
   topics. Do not use tools for unrelated questions.

Do not write labels, URLs, or source lists. The application adds answer labels
and citations after you respond.

Car-specific features, availability, thresholds and exact speed ranges vary by
manufacturer, model and year. Do not state these exact facts from memory; ask
which vehicle when unknown, and otherwise direct the user to its owner's manual.
When a user asks what a particular car supports but gives no make/model/year,
call ask_user and pause with a question asking for those details.
For numeric stopping-distance requests use stopping_distance. It is simplified,
not a prediction of the vehicle. Remind the user that ADAS assists the driver
and does not replace them. Be concise and plain-spoken."""

TUTOR_POLICY = ASSISTANT_POLICY + """

Tutor mode: explain a requested ADAS system step by step in simple language.
Offer a short quiz when useful. Use generate_quiz when asked for a quiz. When
the user submits quiz answers, grade them and explain mistakes. State whether
the quiz is based on documents or general knowledge."""

TOOLS = [
    {"type": "function", "function": {"name": "stopping_distance",
     "description": "Estimate simplified stopping distance; not vehicle-specific.",
     "parameters": {"type": "object", "properties": {
         "speed_kmh": {"type": "number"},
         "road_condition": {"type": "string", "enum": ["dry", "wet", "snow", "ice"]}},
         "required": ["speed_kmh", "road_condition"]}}},
    {"type": "function", "function": {"name": "ask_user",
     "description": "Pause and ask a needed clarification, especially the vehicle model/year.",
     "parameters": {"type": "object", "properties": {
         "question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "generate_quiz",
     "description": "Retrieve source passages for a multiple-choice quiz about an ADAS system.",
     "parameters": {"type": "object", "properties": {
         "system_name": {"type": "string"},
         "n": {"type": "integer", "minimum": 1, "maximum": QUIZ_MAX_QUESTIONS}}}}},
]


def stopping_distance(speed_kmh, road_condition):
    """Illustrative estimate: reaction distance + constant-deceleration braking."""
    factors = {"dry": 0.8, "wet": 0.5, "snow": 0.2, "ice": 0.1}
    condition = str(road_condition).lower()
    speed_kmh = float(speed_kmh)
    if condition not in factors or speed_kmh < 0:
        raise ValueError("Use a nonnegative speed and dry/wet/snow/ice condition.")
    speed = speed_kmh / 3.6
    reaction = speed * REACTION_TIME_SECONDS
    braking = speed * speed / (2 * 9.81 * factors[condition])
    return {"reaction_m": round(reaction, 1), "braking_m": round(braking, 1),
            "total_m": round(reaction + braking, 1),
            "reaction_time_s": REACTION_TIME_SECONDS,
            "friction_coefficient": factors[condition],
            "road_condition": condition,
            "note": "Simplified estimate; actual distance varies. ADAS assists the driver and does not replace them."}


def _web_is_on_cooldown():
    return time.monotonic() < _web_cooldown_until


def _record_web_failure(error):
    global _web_cooldown_until, _web_consecutive_failures
    _web_consecutive_failures += 1
    text = str(error).lower()
    rate_limited = (re.search(r"\b(?:http\s*)?(?:202|429)\b", text) is not None
                    or "rate limit" in text or "rate-limit" in text
                    or "too many requests" in text or "throttl" in text)
    if rate_limited or _web_consecutive_failures >= WEB_FAILURES_BEFORE_COOLDOWN:
        _web_cooldown_until = time.monotonic() + max(0, WEB_COOLDOWN_MINUTES) * 60


def _record_web_success():
    global _web_consecutive_failures
    _web_consecutive_failures = 0


def _parse_arguments(raw):
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return raw


def run_tool(name, arguments, source_log=None, document_hits=None):
    """Execute a tool and record citation-ready sources; errors are values."""
    try:
        arguments = _parse_arguments(arguments)
        if name == "search_docs":
            if arguments.get("strategy") == "document_overview":
                hits = _document_overview_hits()
            else:
                hits = retrieve(arguments["query"], k=TOP_K,
                                collection_name=ADAS_COLLECTION_NAME,
                                system=arguments.get("system"),
                                car_model=arguments.get("car_model"))
            sources = [{"type": "document", "source": h["source"],
                        "page": h.get("page"),
                        "pdf_page": h.get("pdf_page", h.get("page")),
                        "printed_page": h.get("printed_page"),
                        "neighbor": h.get("neighbor", False),
                        "system": h.get("system"),
                        "car_model": h.get("car_model"),
                        "chunk_index": h["chunk_index"]} for h in hits]
            if source_log is not None:
                source_log.extend(sources)
            return {"hits": [{"text": h["text"], **sources[i],
                               "similarity": h["similarity"]}
                              for i, h in enumerate(hits)],
                    "found": bool(hits)}
        if name == "web_search":
            try:
                raw_results = websearch.search_web(arguments["query"])
                _record_web_success()
                results = [r for r in raw_results
                           if r.get("snippet", "").strip() and r.get("link")]
                sources = [{"type": "web", "title": r.get("title", "Web result"),
                            "url": r["link"]} for r in results]
                if source_log is not None:
                    source_log.extend(sources)
                if not results:
                    return {"error": "Web search returned no useful results."}
                return {"results": results, "found": bool(sources)}
            except Exception as exc:
                _record_web_failure(exc)
                return {"error": f"Web search failed: {exc}"}
        if name == "stopping_distance":
            return stopping_distance(arguments["speed_kmh"],
                                     arguments["road_condition"])
        if name == "ask_user":
            question = str(arguments["question"]).strip()
            if not question:
                raise ValueError("question must not be empty")
            return {"ask_user": question}
        if name == "generate_quiz":
            system_name = str(arguments.get("system_name") or "").strip()
            n = int(arguments.get("n", 3))
            if not 1 <= n <= QUIZ_MAX_QUESTIONS:
                raise ValueError(f"Provide n from 1 to {QUIZ_MAX_QUESTIONS}")
            if system_name:
                hits = retrieve(system_name, k=max(TOP_K, n),
                                collection_name=ADAS_COLLECTION_NAME,
                                system=system_name)
            else:
                hits = list(document_hits or [])
            sources = [{"type": "document", "source": h["source"],
                        "page": h.get("page"),
                        "pdf_page": h.get("pdf_page", h.get("page")),
                        "printed_page": h.get("printed_page"),
                        "neighbor": h.get("neighbor", False),
                        "system": h.get("system"),
                        "car_model": h.get("car_model"),
                        "chunk_index": h["chunk_index"]} for h in hits]
            if source_log is not None:
                source_log.extend(sources)
            return {"system": system_name, "number_of_questions": n,
                    "basis": "documents" if hits else "general_knowledge",
                    "instruction": ("Write multiple-choice questions with four options and the correct answer. Use the page metadata for citations."
                        if hits else "No matching document passages. Clearly label the quiz as not from the documents."),
                    "hits": [{"text": h["text"], **sources[i]}
                             for i, h in enumerate(hits)]}
        raise ValueError(f"Unknown tool: {name}")
    except Exception as exc:
        return {"error": f"{name} tool error: {exc}"}


def _ollama_chat(messages, tools=None, model=None, keep_alive=AGENT_KEEP_ALIVE):
    payload = {"model": model or CHAT_MODEL, "messages": messages,
               "stream": False, "keep_alive": keep_alive,
               "options": {"temperature": 0, "num_ctx": AGENT_NUM_CTX,
                            "num_predict": AGENT_NUM_PREDICT}}
    if tools is not None:
        payload["tools"] = tools
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload,
                             timeout=AGENT_REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")
    return response.json().get("message", {})


DOCUMENT_SCOPE_WORDS = re.compile(
    r"\b(?:documents?|docs?|pdfs?|files?|guides?|manuals?|texts?|pages?|"
    r"summar(?:y|ies|ize|ise|izing|ising|ized|ised)|quizzes?|"
    r"test\s+me|questions?|explains?|explained|explaining|explain|"
    r"it|that|this|them)\b", re.IGNORECASE)

CLEAR_OFF_TOPIC_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:recipe|cooking|cook|couscous|bake|baking)\b",
    r"\bwho\s+won\b|\b(?:world cup|sports? results?|score of the game|league standings)\b",
    r"\b(?:politics?|election|president|prime minister|politician)\b",
    r"\b(?:celebrity|celebrities|actor|actress|singer|gossip)\b",
    r"\b(?:cover letter|resume|résumé|curriculum vitae)\b",
    r"\b(?:homework|math problem|algebra problem)\b",
))


def _has_scope_word(question):
    lower = str(question or "").lower()
    if DOCUMENT_SCOPE_WORDS.search(lower):
        return True
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lower)
               for term in ADAS_SCOPE_KEYWORDS)


def _looks_in_scope(question):
    """Return true unless a clear off-topic pattern has no scope context."""
    lower = str(question or "").lower()
    clearly_off_topic = any(pattern.search(lower) for pattern in CLEAR_OFF_TOPIC_PATTERNS)
    return not (clearly_off_topic and not _has_scope_word(lower))


def _needs_vehicle_details(question):
    lower = question.lower()
    asks_about_own_car = re.search(
        r"\b(my|our|this)\s+(car|vehicle|auto)\b", lower) is not None
    if not asks_about_own_car:
        return False
    has_year = re.search(r"\b(?:19|20)\d{2}\b", lower) is not None
    has_make = any(re.search(rf"\b{re.escape(make)}\b", lower)
                   for make in ADAS_VEHICLE_MAKES)
    return not (has_year and has_make)


def _previous_turn_in_scope(history):
    history = history or []
    user_turns = [item for item in history if item.get("role") == "user"
                  and item.get("content")]
    if not user_turns:
        return False
    previous = str(user_turns[-1]["content"])
    # An explicitly declined prior turn should not establish ADAS context.
    assistant_turns = [item for item in history if item.get("role") == "assistant"]
    if assistant_turns and "declined" in str(assistant_turns[-1].get("answer_source", "")):
        return False
    return _has_scope_word(previous) or not any(
        pattern.search(previous) for pattern in CLEAR_OFF_TOPIC_PATTERNS)


def _scope_check(question, model=None, history=None):
    """Decline only clear off-topic requests without ADAS/document context."""
    follow_up = (len(str(question).split()) < 12 or
                 re.search(r"\b(?:it|that|this|them)\b", str(question), re.I))
    if follow_up and _previous_turn_in_scope(history):
        return True
    return _looks_in_scope(question)


def _is_document_meta_request(question):
    lower = question.lower()
    return (bool(re.search(r"\b(?:summary|summarize|quiz|test\s+me|document|doc|pdf|file|guide|manual|text|page|questions?)\b", lower))
            or bool(re.search(r"\bwhat\s+(?:is|['’]s)\s+(?:it|that|this)\s+about\b", lower)))


def _document_overview_hits():
    """Return a small, page-ordered sample spread across the shared corpus."""
    collection = get_collection(ADAS_COLLECTION_NAME)
    if not collection.count():
        return []
    data = collection.get(include=["documents", "metadatas"])
    if reindex_missing_page_metadata(
            collection, ADAS_COLLECTION_NAME, data.get("metadatas") or []):
        data = collection.get(include=["documents", "metadatas"])
    rows = []
    for text, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
        meta = meta or {}
        rows.append({"text": text or "", "source": meta.get("source", "document"),
                     "page": meta.get("pdf_page", meta.get("page")),
                     "pdf_page": meta.get("pdf_page", meta.get("page")),
                     "printed_page": meta.get("printed_page") or None,
                     "neighbor": False, "chunk_index": meta.get("chunk_index", 0),
                     "system": meta.get("system", "general"),
                     "source_type": meta.get("source_type", "reference"),
                     "car_model": meta.get("car_model", ""), "similarity": None})
    rows.sort(key=lambda hit: (hit["source"], hit.get("page") or 0,
                               hit.get("chunk_index") or 0))
    if len(rows) <= 8:
        return rows
    # Keep the opening context, then sample evenly through the remaining text.
    indices = {0, 1, 2}
    indices.update(round(i * (len(rows) - 1) / 5) for i in range(1, 6))
    indices = sorted(indices)[:8]
    return [rows[index] for index in indices]


def _answer_from_passages(question, hits, model=None, quiz=False, overview=False):
    if not hits:
        return "NO_ANSWER"
    passages = "\n\n".join(
        f"[{source_citation(hit)}] {hit['text']}" for hit in hits)
    task_rules = "Answer the user's question directly from these passages."
    if quiz:
        task_rules = (
            "Create exactly 3 multiple-choice questions using only these passages. "
            "Each must have exactly four options (A-D), identify the correct answer, "
            "and the application will add the source page to each question. Do not "
            "write page citations yourself. Do not say the document is unavailable "
            "because passages are supplied.")
    elif overview:
        task_rules = (
            "Summarize what the supplied document passages say. Use only these "
            "passages and do not say the document is unavailable when passages exist.")
    task_rules += (
        " When the passages list periods or eras, include all of them in order "
        "with their exact dates; do not merge or skip any.")
    try:
        msg = _ollama_chat([
            {"role": "system", "content": (
                "You answer only from the supplied passages. If the passages do not "
                "contain the answer, reply exactly NO_ANSWER. Do not write labels, "
                "URLs or source lists. " + task_rules)},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages}"},
        ], tools=None, model=model)
        return _clean_model_answer(msg.get("content", "")) or "NO_ANSWER"
    except Exception:
        logger.exception("Document answer gate failed")
        return "NO_ANSWER"


def _passages_contain_answer(question, hits, model=None):
    """Compatibility helper retained for callers that need a boolean gate."""
    return _answer_from_passages(question, hits, model=model) != "NO_ANSWER"


def _clean_model_answer(answer):
    answer = str(answer or "").strip()
    answer = _strip_ascii_diagram_blocks(answer)
    labels = (r"not found in your documents(?:\s*[,.:;-]\s*|\s+)?",
              r"from the web(?:\s*:\s*|\s+)?",
              r"from general knowledge(?:\s*\(unverified\))?(?:\s*:\s*|\s+)?",
              r"from your documents(?:\s*:\s*|\s+)?",
              r"calculated with sentinel's simplified formula\s*\(not from the documents\)(?:\s*:\s*|\s+)?")
    label_pattern = r"(?:" + "|".join(labels) + r")"
    # Models sometimes repeat the routing label; strip each leading copy.
    previous = None
    while answer != previous:
        previous = answer
        answer = re.sub(label_pattern, "", answer,
                        flags=re.IGNORECASE).strip()
    answer = re.sub(r"^[\s:;,.-]+", "", answer)
    answer = re.sub(r"(?im)^\s*(?:assistant|user|system)\s*:\s*", "", answer)
    answer = re.sub(r"(?im)^\s*(?:sources?|citations?)\s*:\s*.*$", "", answer)
    answer = re.sub(r"\([^()\n]{1,120},\s*p\.\s*\d+\)", "", answer)
    answer = re.sub(r"\[[^\]]*\]\([^)]+\)", "", answer)
    answer = re.sub(
        r"(?i)\b(?:https?://|www\.)[^\s<>()]+|"
        r"\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.(?:com|org|net|io|gov|edu|co|ai|info|dev|app)(?:/[^\s]*)?",
        "", answer)
    answer = re.sub(r"(?im)(?:^|\s+)(?:assistant|user|system)[.!]?\s*$", "", answer)
    return re.sub(r"[ \t]+\n", "\n", answer).strip()


def _strip_ascii_diagram_blocks(answer):
    """Remove fenced ASCII box/flow diagrams while preserving prose and code."""
    fence = re.compile(r"(?ms)^```[^\n]*\n(.*?)^```[ \t]*(?:\n|$)")

    def remove_diagram(match):
        block = match.group(1)
        lines = block.splitlines()
        has_box_border = any(re.search(r"^\s*\+[-=]{2,}\+\s*$", line)
                             for line in lines)
        has_box_sides = any("|" in line for line in lines)
        has_flow_arrow = any(re.search(r"(?:^|\s)v(?:\s|$)", line)
                             for line in lines)
        if (has_box_border and has_box_sides) or (
                sum("|" in line for line in lines) >= 2 and has_flow_arrow):
            return ""
        return match.group(0)

    return fence.sub(remove_diagram, answer).strip()


def _is_adas_timeline_request(question):
    has_visual_intent = re.search(
        r"\b(?:graphs?|charts?|diagrams?|timelines?|evolution|history\s+over\s+time)\b",
        str(question or ""), re.IGNORECASE)
    has_adas_term = re.search(
        r"\b(?:adas|advanced\s+driver\s+assistance(?:\s+systems?)?|"
        r"driver\s+assistance\s+systems?)\b", str(question or ""), re.IGNORECASE)
    return bool(has_visual_intent and has_adas_term)


def _adas_timeline_image(question):
    """Build an optional chart image; chart failures never interrupt an answer."""
    if not _is_adas_timeline_request(question):
        return []
    guide_loaded = False
    try:
        collection = get_collection(ADAS_COLLECTION_NAME)
        data = collection.get(where={"source": "2021-ADAS-Guide-Preview.pdf"},
                              limit=1, include=["metadatas"])
        guide_loaded = bool(data.get("ids"))
    except Exception:
        logger.exception("Could not check whether the ADAS guide is indexed")
    try:
        from charts import adas_timeline_chart
        chart_path = adas_timeline_chart(guide_loaded=guide_loaded)
        return [{"url": f"/charts/{Path(chart_path).name}",
                 "caption": "ADAS evolution timeline",
                 "source": "2021 ADAS Guide", "printed_page": "4",
                 "source_status": "not currently loaded" if not guide_loaded else "",
                 "kind": "chart"}]
    except Exception:
        logger.exception("Could not generate ADAS evolution chart")
        return []


def _add_document_citations(answer, sources, is_quiz=False):
    citations = []
    for source in sources:
        if source.get("type") != "document" or not source.get("source"):
            continue
        citation = f"({source_citation(source)})"
        if citation not in citations:
            citations.append(citation)
    if not citations:
        return answer
    if is_quiz:
        pages = citations
        index = [0]
        def cite_question(match):
            citation = pages[index[0] % len(pages)]
            index[0] += 1
            return f"{match.group(1)} {citation}"
        return re.sub(r"(?im)^((?:From your documents:\s*)?Question\s+\d+\s*:\s*.*)$",
                      cite_question, answer)
    return answer + "\n\nSources: " + "; ".join(citations)


def _dedupe_sources(sources):
    unique = []
    seen = set()
    for source in sources:
        key = (source.get("type"), source.get("source"),
               source.get("chunk_index"), source.get("printed_page"),
               source.get("pdf_page", source.get("page")), source.get("url"))
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def ask_agent(question, mode="assistant", history=None, model=None,
              keep_alive=AGENT_KEEP_ALIVE):
    """Run deterministic document-first routing and the limited agent tool loop."""
    mode = "tutor" if mode == "tutor" else "assistant"
    trace, source_log, images = [], [], []
    if not _scope_check(question, model=model, history=history):
        return _build_result(
            "I can help with ADAS, driver assistance, vehicle safety systems, and related automotive technology. What would you like to know about one of those?",
            [], [], mode, "declined", images=[])

    images = _adas_timeline_image(question)

    # Document-wide requests need representative chunks, not similarity-ranked
    # passages that may over-represent one section.
    overview = _is_document_meta_request(question)
    search_args = {"query": question}
    if overview:
        search_args["strategy"] = "document_overview"
    initial = run_tool("search_docs", search_args, source_log)
    hits = initial.get("hits", [])
    trace.append({"tool": "search_docs", "arguments": search_args,
                  "result": "results" if hits else "no hits",
                  "pages": [{"source": hit.get("source"),
                            "pdf_page": hit.get("pdf_page", hit.get("page")),
                            "printed_page": hit.get("printed_page"),
                            "neighbor": hit.get("neighbor", False)}
                            for hit in hits]})
    source_log.extend({"type": "document", "source": hit.get("source"),
                       "page": hit.get("page"),
                       "pdf_page": hit.get("pdf_page", hit.get("page")),
                       "printed_page": hit.get("printed_page"),
                       "neighbor": hit.get("neighbor", False),
                       "system": hit.get("system"),
                       "car_model": hit.get("car_model"),
                       "chunk_index": hit.get("chunk_index")}
                      for hit in hits)

    if _needs_vehicle_details(question):
        clarification = run_tool("ask_user", {"question":
            "Which vehicle make, model, and year do you mean? ADAS features vary by vehicle."})
        trace.append({"tool": "ask_user", "arguments": {}, "result": "clarification requested"})
        response = _build_result(clarification["ask_user"], trace, source_log,
                                 mode, "general_knowledge", images=[])
        response["clarification_question"] = clarification["ask_user"]
        return response

    quiz_request = bool(re.search(r"\b(?:quiz|test\s+me)\b", question, re.I))
    pages = [{"source": hit.get("source"),
              "pdf_page": hit.get("pdf_page", hit.get("page")),
              "printed_page": hit.get("printed_page"),
              "neighbor": hit.get("neighbor", False)}
             for hit in hits]
    trace.append({"tool": "document_gate", "pages": pages,
                  "decision": "checking" if hits else "no_answer"})
    gate_answer = (_answer_from_passages(question, hits, model=model,
                                         quiz=quiz_request, overview=overview)
                   if hits else "NO_ANSWER")
    has_document_answer = gate_answer.strip() != "NO_ANSWER"
    trace[-1]["decision"] = "answer" if has_document_answer else "no_answer"
    if has_document_answer:
        result = _build_result(gate_answer, trace, source_log, mode,
                               "documents", images=images)
        result["answer"] = _add_document_citations(
            result["answer"], source_log, is_quiz=quiz_request)
        return result

    web_results = []
    if _web_is_on_cooldown():
        cooldown_left = max(1, int((_web_cooldown_until - time.monotonic()) / 60))
        trace.append({"tool": "web_search",
                      "result": f"skipped: web is on cooldown ({cooldown_left} min remaining)"})
        answer_source = "general_knowledge"
        evidence = ("No relevant document passages were found and web search is on "
                    "cooldown. Answer from general knowledge, clearly qualify uncertainty, "
                    "and avoid URLs or source claims.")
    else:
        web_result = run_tool("web_search", {"query": question}, source_log)
        trace.append({"tool": "web_search", "arguments": {"query": question},
                      "result": "results" if web_result.get("found") else
                      web_result.get("error", "no useful results")})
        web_results = web_result.get("results", [])
        if web_results:
            answer_source = "web"
            evidence = "WEB SNIPPETS (the only factual source for this answer):\n" + "\n\n".join(
                f"[{item.get('title', 'Web result')}] {item.get('snippet', '')} URL: {item['link']}"
                for item in web_results)
        else:
            answer_source = "general_knowledge"
            evidence = "No relevant document passages or usable web results were found. Answer from general knowledge, clearly qualify uncertainty, and avoid URLs or source claims."

    policy = TUTOR_POLICY if mode == "tutor" else ASSISTANT_POLICY
    rules = ("Use only the supplied evidence when its tier is documents or web. "
             "Do not write labels, URLs or source lists. Do not invent citations. "
             "Answer concisely and do not emit role names or control tokens.\n\n" + evidence)
    messages = [{"role": "system", "content": policy + "\n\n" + rules}]
    history = (history or [])[-6:]
    for item in history:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            messages.append({"role": item["role"], "content": str(item["content"])[:2000]})
    messages.append({"role": "user", "content": question})
    calculation_details = None
    for _ in range(MAX_AGENT_ITERATIONS):
        try:
            message = _ollama_chat(messages, tools=TOOLS, model=model,
                                   keep_alive=keep_alive)
        except Exception as exc:
            trace.append({"tool": "ollama", "result": f"error: {exc}"})
            final = "I couldn't complete the request because the local model call failed."
            result = _build_result(
                final, trace, source_log, mode,
                "calculation" if calculation_details else answer_source,
                images=images)
            return _add_calculation_constants(result, calculation_details)
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            answer = _clean_model_answer(message.get("content", ""))
            result = _build_result(
                answer, trace, source_log, mode,
                "calculation" if calculation_details else answer_source,
                images=images)
            _add_calculation_constants(result, calculation_details)
            if answer_source == "documents":
                result["answer"] = _add_document_citations(
                    result["answer"], source_log,
                    is_quiz=overview and "quiz" in question.lower())
            return result
        for call in calls:
            try:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                args = _parse_arguments(fn.get("arguments", {}))
                allowed_tools = {tool["function"]["name"] for tool in TOOLS}
                if name not in allowed_tools:
                    trace.append({"tool": "unsupported_tool_call", "result": name})
                    messages.append({"role": "tool", "content": json.dumps(
                        {"error": "Only ask_user, stopping_distance, and generate_quiz are available to the model."})})
                    continue
                if name == "generate_quiz":
                    result = run_tool(name, args, source_log,
                                      document_hits=hits if overview else None)
                else:
                    result = run_tool(name, args, source_log)
                trace_result = (result if name == "stopping_distance" else
                                result.get("error", "tool error")[:240]
                                if "error" in result else "ok")
                trace.append({"tool": name, "arguments": args,
                              "result": trace_result})
                if name == "stopping_distance" and "error" not in result:
                    calculation_details = result
                    try:
                        from charts import stopping_distance_chart
                        speed = float(args["speed_kmh"])
                        condition = str(args["road_condition"]).lower()
                        chart_path = stopping_distance_chart(speed, condition)
                        display_speed = int(speed) if speed.is_integer() else speed
                        images.append({
                            "url": f"/charts/{Path(chart_path).name}",
                            "caption": (f"Stopping distance at {display_speed} km/h on "
                                        f"{condition} road (simplified estimate)"),
                            "kind": "chart",
                        })
                    except Exception:
                        logger.exception("Could not generate stopping-distance chart")
                    answer_source = "calculation"
                if name == "ask_user" and result.get("ask_user"):
                    response = _build_result(
                        result["ask_user"], trace, source_log, mode,
                        "calculation" if calculation_details else "general_knowledge",
                        images=images)
                    _add_calculation_constants(response, calculation_details)
                    response["clarification_question"] = result["ask_user"]
                    return response
                messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
            except Exception as exc:
                # Malformed call structures and arguments are fed back as a
                # tool error so the model gets a chance to repair its call.
                trace.append({"tool": "malformed_tool_call", "result": str(exc)})
                messages.append({"role": "tool", "content": json.dumps({"error": str(exc)})})

    # Hard cap reached: ask for a plain answer with tools omitted.
    messages.append({"role": "system", "content": "Tool limit reached. Give your best concise final answer now, obeying the source labels and safety rules. Do not request or call tools."})
    try:
        final = _clean_model_answer(_ollama_chat(messages, tools=None, model=model,
                                                 keep_alive=keep_alive).get("content", ""))
    except Exception as exc:
        final = f"I couldn't complete the request after the tool limit ({exc})."
    trace.append({"tool": "iteration_limit", "result": "plain final answer requested"})
    result = _build_result(final, trace, source_log, mode,
                           "calculation" if calculation_details else answer_source,
                           images=images)
    _add_calculation_constants(result, calculation_details)
    if answer_source == "documents":
        result["answer"] = _add_document_citations(
            result["answer"], source_log,
            is_quiz=overview and "quiz" in question.lower())
    return result


def _add_calculation_constants(result, calculation):
    if calculation:
        reaction_time = calculation.get("reaction_time_s", REACTION_TIME_SECONDS)
        friction = calculation.get("friction_coefficient", "unspecified")
        condition = calculation.get("road_condition", "road")
        result["answer"] += (
            f"\n\nConstants used: reaction time {reaction_time} s; "
            f"friction coefficient {friction} for {condition} road.")
    return result


def _build_result(answer, trace, sources, mode, source_override=None, images=None):
    tools = {item.get("tool") for item in trace}
    if source_override:
        source = source_override
    elif any(item.get("type") == "web" for item in sources):
        source = "web"
    elif "web_search" in tools:
        # A failed web attempt leads to general knowledge rather than falsely
        # crediting irrelevant retrieved passages.
        source = "general_knowledge"
    elif any(item.get("type") == "document" for item in sources):
        source = "documents"
    else:
        source = "general_knowledge"
    answer = _clean_model_answer(answer) or "I couldn't produce a final answer."
    labels = {
        "documents": "From your documents: ",
        "web": "Not found in your documents. From the web: ",
        "general_knowledge": "Not found in your documents. From general knowledge (unverified): ",
        "calculation": "Calculated with Sentinel's simplified formula (not from the documents): ",
        "declined": "Out of scope: ",
    }
    if source in labels:
        answer = labels[source] + answer
    return {"answer": answer, "answer_source": source,
            "sources": _dedupe_sources(sources), "tool_trace": trace,
            "mode": mode, "images": images or []}


if __name__ == "__main__":
    try:
        while True:
            question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("You: ").strip()
            if not question:
                break
            result = ask_agent(question)
            print(result["answer"])
            if len(sys.argv) > 1:
                break
    except (EOFError, KeyboardInterrupt):
        pass
