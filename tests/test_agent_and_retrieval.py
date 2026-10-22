import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import agent
import retrieve as retrieval
from ingest import (extract_tables_text, load_pdf_page_chunks,
                    parse_printed_page)


def test_stopping_distance_known_values():
    dry = agent.stopping_distance(50, "dry")
    wet = agent.stopping_distance(50, "wet")
    assert dry["reaction_m"] == 20.8
    assert dry["braking_m"] == 12.3
    assert dry["total_m"] == 33.1
    assert dry["reaction_time_s"] == 1.5
    assert dry["friction_coefficient"] == .8
    assert wet["reaction_m"] == 20.8
    assert wet["braking_m"] == 19.7
    assert wet["total_m"] == 40.5


@pytest.mark.parametrize("speed,condition", [(-1, "dry"), (20, "mud"), ("bad", "wet")])
def test_stopping_distance_rejects_bad_inputs(speed, condition):
    with pytest.raises((ValueError, TypeError)):
        agent.stopping_distance(speed, condition)


def test_tool_errors_return_as_values():
    result = agent.run_tool("stopping_distance", {"speed_kmh": -1,
                                                   "road_condition": "dry"})
    assert "error" in result


def test_malformed_tool_call_is_returned_to_mocked_ollama_for_retry(monkeypatch):
    monkeypatch.setattr(agent, "run_tool", lambda name, arguments, source_log=None:
                        ({"hits": [], "found": False} if name == "search_docs" else
                         {"error": "offline"} if name == "web_search" else
                         agent.stopping_distance(arguments["speed_kmh"],
                                                 arguments["road_condition"])))
    replies = iter([
        {"tool_calls": [{"function": {"name": "stopping_distance",
          "arguments": {"speed_kmh": 50, "road_condition": "mud"}}}]},
        {"tool_calls": [{"function": {"name": "stopping_distance",
          "arguments": {"speed_kmh": 50, "road_condition": "wet"}}}]},
        {"content": "Not found in your documents. From general knowledge (unverified): estimate is simplified."},
    ])
    seen_messages = []

    def fake_chat(messages, **kwargs):
        seen_messages.append(messages)
        return next(replies)

    monkeypatch.setattr(agent, "_ollama_chat", fake_chat)
    monkeypatch.setattr(agent, "_passages_contain_answer", lambda *args, **kwargs: False)
    result = agent.ask_agent("ADAS stopping distance estimate", mode="assistant")
    assert len(seen_messages) == 3
    assert any("error" in message.get("content", "")
               for message in seen_messages[1] if message.get("role") == "tool")
    assert "estimate is simplified." in result["answer"]
    assert "reaction time 1.5 s" in result["answer"]


def test_web_search_failure_is_a_tool_error(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("HTTP 202 throttling")
    monkeypatch.setattr(agent.websearch, "search_web", fail)
    result = agent.run_tool("web_search", {"query": "ADAS AEB"})
    assert "HTTP 202" in result["error"]


def test_iteration_cap_requests_plain_final_answer(monkeypatch):
    monkeypatch.setattr(agent, "run_tool", lambda name, arguments, source_log=None:
                        ({"hits": [], "found": False} if name == "search_docs" else
                         {"error": "offline"} if name == "web_search" else
                             {"reaction_m": 1, "braking_m": 1, "total_m": 2,
                              "reaction_time_s": 1.5, "friction_coefficient": .8,
                              "road_condition": "dry"}))
    calls = []

    def fake_chat(messages, tools=None, **kwargs):
        calls.append(tools)
        if tools is not None:
            return {"tool_calls": [{"function": {"name": "stopping_distance",
                    "arguments": {"speed_kmh": 20, "road_condition": "dry"}}}]}
        return {"content": "Final answer with the required safety note."}

    monkeypatch.setattr(agent, "_ollama_chat", fake_chat)
    monkeypatch.setattr(agent, "_passages_contain_answer", lambda *args, **kwargs: False)
    result = agent.ask_agent("ADAS stopping distance", mode="assistant")
    expected_tools = agent.TOOLS
    assert calls[:-1] == [expected_tools] * agent.MAX_AGENT_ITERATIONS
    assert calls[-1] is None
    assert "Final answer with the required safety note." in result["answer"]
    assert "reaction time 1.5 s" in result["answer"]
    assert result["answer"].startswith(
        "Calculated with Sentinel's simplified formula (not from the documents):")


def test_quiz_retrieves_source_chunks_and_marks_general_when_empty(monkeypatch):
    hit = {"text": "AEB may apply braking.", "source": "aeb.pdf", "page": 3,
           "system": "AEB", "car_model": "", "chunk_index": 0,
           "similarity": .7}
    monkeypatch.setattr(agent, "retrieve", lambda *args, **kwargs: [hit])
    source_log = []
    result = agent.run_tool("generate_quiz", {"system_name": "AEB", "n": 3}, source_log)
    assert result["basis"] == "documents"
    assert source_log[0]["page"] == 3
    monkeypatch.setattr(agent, "retrieve", lambda *args, **kwargs: [])
    result = agent.run_tool("generate_quiz", {"system_name": "AEB"})
    assert result["basis"] == "general_knowledge"
    assert "not from the documents" in result["instruction"]


def test_ask_user_stops_agent_loop(monkeypatch):
    calls = []
    original = agent.run_tool

    def fake_tool(name, arguments, source_log=None, document_hits=None):
        if name == "search_docs":
            return {"hits": [], "found": False}
        if name == "web_search":
            return {"error": "offline"}
        return original(name, arguments, source_log)

    def fake_chat(messages, tools=None, model=None, keep_alive=None):
        calls.append(messages)
        return {"tool_calls": [{"function": {"name": "ask_user",
                    "arguments": {"question": "What is the make, model, and year?"}}}]}

    monkeypatch.setattr(agent, "run_tool", fake_tool)
    monkeypatch.setattr(agent, "_ollama_chat", fake_chat)
    monkeypatch.setattr(agent, "_passages_contain_answer", lambda *args, **kwargs: False)
    result = agent.ask_agent("Can a vehicle's AEB work in rain?", mode="assistant")
    assert result["clarification_question"] == "What is the make, model, and year?"
    assert [call["tool"] for call in result["tool_trace"]] == [
        "search_docs", "document_gate", "web_search", "ask_user"]
    assert len(calls) == 1


def test_web_failure_returns_general_fallback_without_crashing(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda *args, **kwargs: [])
    def fail(*args, **kwargs):
        raise RuntimeError("HTTP 202 throttling")
    monkeypatch.setattr(agent.websearch, "search_web", fail)
    monkeypatch.setattr(agent, "_ollama_chat", lambda *args, **kwargs:
                        {"content": "From the web: AEB can brake. https://fake.test assistant"})
    result = agent.ask_agent("Explain AEB", mode="assistant")
    assert result["answer_source"] == "general_knowledge"
    assert "web_search" in [x["tool"] for x in result["tool_trace"]]
    assert "AEB" in result["answer"]
    assert "From the web" not in result["answer"]
    assert "https://" not in result["answer"]
    assert not result["answer"].endswith("assistant")


@pytest.mark.parametrize("answer,trace,sources,expected", [
    ("Cited answer (guide.pdf, p. 3)", [],
     [{"type": "document", "source": "guide.pdf", "page": 3}], "documents"),
    ("Not found in your documents. From the web: Answer.", [],
     [{"type": "web", "url": "https://example.test"}], "web"),
    ("Not found in your documents. From general knowledge (unverified): Answer.",
     [{"tool": "web_search"}], [], "general_knowledge"),
])
def test_answer_source_is_derived_from_actual_tools(answer, trace, sources, expected):
    assert agent._build_result(answer, trace, sources, "assistant")["answer_source"] == expected


def test_answer_labels_are_code_added_once_and_model_artifacts_are_removed():
    answer = ("Not found in your documents. From the web: AEB may apply braking. "
              "https://fake.invalid www.fakesite.com assistant")
    result = agent._build_result(answer, [{"tool": "web_search"}], [], "assistant",
                                 "web")
    assert result["answer"].count("Not found in your documents.") == 1
    assert result["answer"].count("From the web:") == 1
    assert "assistant" not in result["answer"].lower()
    assert "https://" not in result["answer"]
    assert "fakesite.com" not in result["answer"]


def test_off_topic_scope_decline_skips_tools(monkeypatch):
    monkeypatch.setattr(agent, "_scope_check", lambda *args, **kwargs: False)
    result = agent.ask_agent("Tell me a cake recipe")
    assert result["answer_source"] == "declined"
    assert result["tool_trace"] == []


@pytest.mark.parametrize("question,expected_fact,source,page", [
    ("Who invented cruise control?", "Ralph Teetor invented cruise control.",
     "history.pdf", 3),
    ("What are the SAE levels of automation?", "The SAE levels range from 0 to 5.",
     "automation.pdf", 8),
])
def test_document_answers_always_retrieve_and_cite(monkeypatch, question,
                                                    expected_fact, source, page):
    hit = {"text": expected_fact, "source": source, "page": page,
           "system": "general", "source_type": "reference", "car_model": "",
           "chunk_index": 0, "similarity": .62}
    retrieved = []
    monkeypatch.setattr(agent, "retrieve", lambda q, **kwargs: retrieved.append(q) or [hit])
    monkeypatch.setattr(agent.websearch, "search_web",
                        lambda *args, **kwargs: pytest.fail("web should not run for a supported passage"))

    def fake_chat(messages, tools=None, **kwargs):
        if messages[0]["content"].startswith("Answer only YES or NO"):
            return {"content": "YES"}
        return {"content": f"From the web: {expected_fact} https://fake.invalid assistant"}

    monkeypatch.setattr(agent, "_ollama_chat", fake_chat)
    result = agent.ask_agent(question)
    assert retrieved == [question]
    assert result["answer_source"] == "documents"
    assert result["tool_trace"][0]["tool"] == "search_docs"
    assert expected_fact in result["answer"]
    assert f"({source}, PDF p. {page})" in result["answer"]
    assert "From the web" not in result["answer"]
    assert "https://" not in result["answer"]
    assert not result["answer"].endswith("assistant")


def test_page_one_automaker_naming_question_uses_document_gate(monkeypatch):
    hit = {"text": "Automakers use different names for similar ADAS features because of branding and market positioning.",
           "source": "adas-guide.pdf", "page": 1, "system": "general",
           "source_type": "reference", "car_model": "", "chunk_index": 0,
           "similarity": .6}
    monkeypatch.setattr(agent, "retrieve", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(agent.websearch, "search_web",
                        lambda *args, **kwargs: pytest.fail("documents answer this"))
    seen = []

    def fake_chat(messages, tools=None, **kwargs):
        seen.append(messages)
        assert tools is None
        assert "reply exactly NO_ANSWER" in messages[0]["content"]
        return {"content": "Automakers often choose different brand names for similar features."}

    monkeypatch.setattr(agent, "_ollama_chat", fake_chat)
    result = agent.ask_agent("Why are ADAS names so different between automakers?")
    assert result["answer_source"] == "documents"
    assert "adas-guide.pdf, PDF p. 1" in result["answer"]
    assert result["tool_trace"][-1]["decision"] == "answer"
    assert result["tool_trace"][-1]["pages"] == [
        {"source": "adas-guide.pdf", "pdf_page": 1,
         "printed_page": None, "neighbor": False}]


def test_web_cooldown_skips_search_after_http_202(monkeypatch):
    monkeypatch.setattr(agent, "_web_cooldown_until", 0.0)
    monkeypatch.setattr(agent, "_web_consecutive_failures", 0)
    monkeypatch.setattr(agent, "WEB_COOLDOWN_MINUTES", 30)
    calls = []

    def fail(query):
        calls.append(query)
        raise RuntimeError("HTTP 202 throttling")

    monkeypatch.setattr(agent.websearch, "search_web", fail)
    assert "HTTP 202" in agent.run_tool("web_search", {"query": "ADAS AEB"})["error"]
    monkeypatch.setattr(agent, "retrieve", lambda *args, **kwargs: [])
    monkeypatch.setattr(agent, "_ollama_chat", lambda *args, **kwargs:
                        {"content": "AEB is an emergency braking system."})
    result = agent.ask_agent("Explain AEB")
    assert len(calls) == 1
    web_events = [item for item in result["tool_trace"] if item["tool"] == "web_search"]
    assert len(web_events) == 1
    assert "cooldown" in web_events[0]["result"]
    assert result["answer_source"] == "general_knowledge"


def test_repeated_web_failures_also_start_cooldown(monkeypatch):
    monkeypatch.setattr(agent, "_web_cooldown_until", 0.0)
    monkeypatch.setattr(agent, "_web_consecutive_failures", 0)
    monkeypatch.setattr(agent, "WEB_COOLDOWN_MINUTES", 30)

    def fail(query):
        raise RuntimeError("network down")

    monkeypatch.setattr(agent.websearch, "search_web", fail)
    agent.run_tool("web_search", {"query": "ADAS one"})
    assert not agent._web_is_on_cooldown()
    agent.run_tool("web_search", {"query": "ADAS two"})
    assert agent._web_is_on_cooldown()


@pytest.mark.parametrize("question", [
    "make me a quiz about this document", "what is this document about",
    "make me a quiz about that document",
    "so can you make me a quiz about that document?", "quiz me on the pdf",
    "what is it about?", "summarize this guide",
])
def test_document_meta_questions_are_in_scope_and_use_spread_chunks(monkeypatch, question):
    hits = [
        {"text": "AEB uses sensors to detect possible collisions.", "source": "guide.pdf",
         "page": 2, "pdf_page": 5, "printed_page": "2", "system": "AEB", "source_type": "reference", "car_model": "",
         "chunk_index": 0, "similarity": None},
        {"text": "ACC maintains a following distance from the vehicle ahead.", "source": "guide.pdf",
         "page": 8, "pdf_page": 9, "printed_page": "8", "system": "ACC", "source_type": "reference", "car_model": "",
         "chunk_index": 6, "similarity": None},
    ]
    monkeypatch.setattr(agent, "_document_overview_hits", lambda: hits)
    calls = []

    def fake_chat(messages, tools=None, **kwargs):
        calls.append(messages)
        if "quiz" in question and tools is not None and not any(
                message.get("role") == "tool" for message in messages):
            return {"tool_calls": [{"function": {"name": "generate_quiz",
                    "arguments": {"n": 3}}}]}
        if "quiz" in question:
            return {"content": "Question 1: Which system detects a possible collision?\nA) AEB\nB) ACC\nC) ABS\nD) ESC\nAnswer: A\n\nQuestion 2: What does ACC maintain?\nA) Tire pressure\nB) Following distance\nC) Engine temperature\nD) Headlight angle\nAnswer: B\n\nQuestion 3: Which system is discussed first?\nA) ACC\nB) AEB\nC) ABS\nD) ESC\nAnswer: B"}
        return {"content": "The guide introduces AEB collision detection and later explains ACC following distance."}

    monkeypatch.setattr(agent, "_ollama_chat", fake_chat)
    result = agent.ask_agent(question, mode="tutor")
    assert result["answer_source"] == "documents"
    assert result["tool_trace"][0]["tool"] == "search_docs"
    assert result["tool_trace"][0]["arguments"]["strategy"] == "document_overview"
    assert result["sources"][0]["page"] == 2
    assert result["sources"][1]["page"] == 8
    if "quiz" in question:
        assert result["answer"].count("Question ") == 3
        assert result["answer"].count("A)") == 3
        assert result["answer"].count("B)") == 3
        assert result["answer"].count("C)") == 3
        assert result["answer"].count("D)") == 3
        assert "(guide.pdf, p. 2)" in result["answer"]
        assert "(guide.pdf, p. 8)" in result["answer"]
        assert "Answer: A" in result["answer"] and "Answer: B" in result["answer"]
        gate_prompt = calls[-1][0]["content"]
        assert "exactly 3 multiple-choice questions" in gate_prompt
        assert "only these passages" in gate_prompt
    else:
        assert "AEB" in result["answer"] and "ACC" in result["answer"]


@pytest.mark.parametrize("question", [
    "give me a couscous recipe", "who won the World Cup",
])
def test_clear_off_topic_questions_are_declined_before_retrieval(monkeypatch, question):
    monkeypatch.setattr(agent, "retrieve", lambda *args, **kwargs:
                        pytest.fail("off-topic request must be declined before retrieval"))
    result = agent.ask_agent(question)
    assert result["answer_source"] == "declined"
    assert result["tool_trace"] == []


def test_short_follow_up_inherits_previous_adas_context():
    history = [
        {"role": "user", "content": "Explain adaptive cruise control"},
        {"role": "assistant", "content": "It adjusts speed to maintain a gap."},
    ]
    assert agent._scope_check("tell me more about it", history=history)
    assert agent._scope_check("who won?", history=history)


class FakeCollection:
    def __init__(self):
        self.where = None

    def count(self):
        return 2

    def get(self, **kwargs):
        return {"ids": [], "documents": [], "metadatas": []}

    def query(self, **kwargs):
        self.where = kwargs.get("where")
        return {"documents": [["AEB limits on wet roads", "ACC keeps distance"]],
                "metadatas": [[
                    {"source": "a.pdf", "page": 4, "pdf_page": 4,
                     "printed_page": "4", "chunk_index": 0,
                     "system": "AEB", "source_type": "reference", "car_model": "Model A"},
                    {"source": "b.pdf", "page": 8, "pdf_page": 8,
                     "printed_page": "8", "chunk_index": 1,
                     "system": "ACC", "source_type": "manual", "car_model": "Model B"}]],
                "distances": [[0.1, 0.2]]}


def test_retrieval_filters_by_system_and_car_model(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(retrieval, "get_collection", lambda name=None: collection)
    monkeypatch.setattr(retrieval, "embed_text", lambda query: [0.0])
    hits = retrieval.retrieve("ADAS", collection_name="adas_documents",
                              system="AEB", car_model="Model A")
    assert collection.where == {"$and": [{"system": "AEB"}, {"car_model": "Model A"}]}
    assert hits[0]["source"] == "a.pdf" and hits[0]["page"] == 4


def test_evolution_query_expands_adjacent_era_chunks_and_respects_cap(monkeypatch):
    records = {
        0: "Earlier era content.",
        1: "2010 to 2016: advanced driver assistance.",
        2: "2016 to 2025: partially automated safety.",
        3: "Another period.",
        4: "Related ADAS notes.",
        5: "Neighbor of related notes.",
        8: "Unrelated retrieved passage.",
    }

    def meta(index):
        return {"source": "2021-ADAS-Guide-Preview.pdf", "chunk_index": index,
                "page": 4, "pdf_page": 4, "printed_page": "4",
                "system": "general", "source_type": "reference", "car_model": ""}

    class EraCollection:
        def count(self):
            return 9

        def query(self, **kwargs):
            indexes = [1, 4, 8]
            return {"documents": [[records[i] for i in indexes]],
                    "metadatas": [[meta(i) for i in indexes]],
                    "distances": [[.1, .2, .3]]}

        def get(self, where=None, include=None):
            if where is None:
                return {"ids": [], "documents": [], "metadatas": []}
            index = where["$and"][1]["chunk_index"]
            if index not in records:
                return {"ids": [], "documents": [], "metadatas": []}
            return {"ids": [f"guide::{index}"], "documents": [records[index]],
                    "metadatas": [meta(index)]}

    monkeypatch.setattr(retrieval, "get_collection", lambda name=None: EraCollection())
    monkeypatch.setattr(retrieval, "embed_text", lambda query: [0.0])
    monkeypatch.setattr(retrieval, "EXPAND_NEIGHBORS", True)
    hits = retrieval.retrieve("evolution of ADAS over time", k=3)
    assert any("2010 to 2016" in hit["text"] for hit in hits)
    assert any("2016 to 2025" in hit["text"] for hit in hits)
    assert len(hits) <= 3 + 2
    assert sum(bool(hit.get("neighbor")) for hit in hits) <= 2
    assert [hit["chunk_index"] for hit in hits] == sorted(
        hit["chunk_index"] for hit in hits)


def test_printed_page_citation_precedes_pdf_page_fallback():
    answer = agent._add_document_citations(
        "The five eras are listed.",
        [{"type": "document", "source": "guide.pdf", "pdf_page": 6,
          "printed_page": "4"}],
    )
    assert answer.endswith("Sources: (guide.pdf, p. 4)")
    fallback = agent._add_document_citations(
        "A fact.", [{"type": "document", "source": "guide.pdf",
                     "pdf_page": 6, "printed_page": None}],
    )
    assert fallback.endswith("Sources: (guide.pdf, PDF p. 6)")


def test_retrieval_reindexes_legacy_page_metadata_automatically(tmp_path, monkeypatch,
                                                               capsys):
    pdf_path = tmp_path / "legacy.pdf"
    pdf_path.write_bytes(b"mock pdf")

    class LegacyCollection:
        current = False

        def count(self):
            return 1

        def query(self, **kwargs):
            meta = {"source": "legacy.pdf", "chunk_index": 0,
                    "page": 1, "system": "general",
                    "source_type": "reference", "car_model": ""}
            if self.current:
                meta.update({"pdf_page": 1, "printed_page": ""})
            return {"documents": [["AEB passage"]], "metadatas": [[meta]],
                    "distances": [[.1]]}

        def get(self, where=None, include=None):
            if where and "chunk_index" in where.get("$and", [{}, {}])[1]:
                return {"ids": [], "documents": [], "metadatas": []}
            meta = {"source": "legacy.pdf", "source_path": str(pdf_path),
                    "page": 1, "chunk_index": 0}
            if self.current:
                meta.update({"pdf_page": 1, "printed_page": ""})
            return {"ids": ["legacy::0"], "metadatas": [meta]}

    collection = LegacyCollection()
    calls = []
    monkeypatch.setattr(retrieval, "get_collection", lambda name=None: collection)
    monkeypatch.setattr(retrieval, "embed_text", lambda query: [0.0])

    def fake_index_pdf(path, **kwargs):
        calls.append((path, kwargs))
        collection.current = True

    monkeypatch.setattr(retrieval, "index_pdf", fake_index_pdf)
    hits = retrieval.retrieve("AEB", collection_name="uploads", k=1)
    assert len(calls) == 1
    assert hits[0]["printed_page"] is None
    assert "Re-indexing legacy.pdf" in capsys.readouterr().out


def test_page_number_parser_uses_text_then_label_then_none():
    assert parse_printed_page("4\nThe Five Eras") == "4"
    assert parse_printed_page("The Five Eras", "iv") == "iv"
    assert parse_printed_page("The Five Eras", "") is None


def test_five_eras_page_cites_printed_page_four():
    pdf = ROOT / "data" / "2021-ADAS-Guide-Preview.pdf"
    page_chunk = next(item for item in load_pdf_page_chunks(pdf, verbose=False)
                      if "The Five Eras of ADAS" in item["text"])
    assert page_chunk["pdf_page"] == 5
    assert page_chunk["printed_page"] == "4"
    cited = agent._add_document_citations(
        "The guide lists the eras.",
        [{"type": "document", "source": pdf.name,
          "pdf_page": page_chunk["pdf_page"],
          "printed_page": page_chunk["printed_page"]}],
    )
    assert cited.endswith("Sources: (2021-ADAS-Guide-Preview.pdf, p. 4)")


def test_hybrid_retrieval_fuses_exact_acronym_match(monkeypatch):
    class HybridCollection(FakeCollection):
        def get(self, **kwargs):
            return {"documents": ["Adaptive cruise control keeps distance.",
                                  "SAE J3016 defines driving automation levels."],
                    "metadatas": [
                        {"source": "acc.pdf", "page": 1, "pdf_page": 1,
                         "printed_page": "1", "chunk_index": 0,
                         "system": "ACC", "source_type": "reference", "car_model": ""},
                        {"source": "sae.pdf", "page": 2, "pdf_page": 2,
                         "printed_page": "2", "chunk_index": 0,
                         "system": "general", "source_type": "standard", "car_model": ""}]}

        def query(self, **kwargs):
            return {"documents": [["Adaptive cruise control keeps distance."]],
                    "metadatas": [[{"source": "acc.pdf", "page": 1, "pdf_page": 1,
                                    "printed_page": "1", "chunk_index": 0,
                                    "system": "ACC", "source_type": "reference", "car_model": ""}]],
                    "distances": [[0.2]]}

    collection = HybridCollection()
    monkeypatch.setattr(retrieval, "get_collection", lambda name=None: collection)
    monkeypatch.setattr(retrieval, "embed_text", lambda query: [0.0])
    hits = retrieval.retrieve("SAE J3016", collection_name="adas_documents",
                              use_hybrid=True)
    assert any(hit["source"] == "sae.pdf" and hit["page"] == 2 for hit in hits)


def test_table_to_text_preserves_header_value_relationships():
    class Table:
        def extract(self):
            return [["Model", "Warranty (months)"], ["Falcon-9X", "24"]]

    class Finder:
        tables = [Table()]

    class Page:
        def find_tables(self):
            return Finder()

    text = extract_tables_text(Page())
    assert "Model: Falcon-9X" in text
    assert "Warranty (months): 24" in text
