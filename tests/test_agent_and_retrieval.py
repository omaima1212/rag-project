import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import agent
import retrieve as retrieval
from ingest import extract_tables_text


def test_stopping_distance_known_values():
    dry = agent.stopping_distance(50, "dry")
    wet = agent.stopping_distance(50, "wet")
    assert dry == {"reaction_m": 20.8, "braking_m": 12.3, "total_m": 33.1,
                   "note": dry["note"]}
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
    assert result["answer"].endswith("simplified.")


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
                         {"reaction_m": 1, "braking_m": 1, "total_m": 2}))
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
    assert result["answer"].endswith("Final answer with the required safety note.")
    assert result["answer"].startswith("Not found in your documents.")


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
    assert [call["tool"] for call in result["tool_trace"]] == ["search_docs", "web_search", "ask_user"]
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
    assert f"({source}, p. {page})" in result["answer"]
    assert "From the web" not in result["answer"]
    assert "https://" not in result["answer"]
    assert not result["answer"].endswith("assistant")


@pytest.mark.parametrize("question", [
    "make me a quiz about this document", "what is this document about",
    "make me a quiz about that document", "quiz me on the pdf",
    "what is it about?", "summarize this guide",
])
def test_document_meta_questions_are_in_scope_and_use_spread_chunks(monkeypatch, question):
    hits = [
        {"text": "AEB uses sensors to detect possible collisions.", "source": "guide.pdf",
         "page": 2, "system": "AEB", "source_type": "reference", "car_model": "",
         "chunk_index": 0, "similarity": None},
        {"text": "ACC maintains a following distance from the vehicle ahead.", "source": "guide.pdf",
         "page": 8, "system": "ACC", "source_type": "reference", "car_model": "",
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
        assert "generate_quiz" in [item["tool"] for item in result["tool_trace"]]
        assert result["answer"].count("Question ") == 3
        assert result["answer"].count("A)") == 3
        assert result["answer"].count("B)") == 3
        assert result["answer"].count("C)") == 3
        assert result["answer"].count("D)") == 3
        assert "(guide.pdf, p. 2)" in result["answer"]
        assert "(guide.pdf, p. 8)" in result["answer"]
        assert "Answer: A" in result["answer"] and "Answer: B" in result["answer"]
        quiz_result = next(message for message in calls[-1] if message.get("role") == "tool")
        assert '"number_of_questions": 3' in quiz_result["content"]
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


class FakeCollection:
    def __init__(self):
        self.where = None

    def count(self):
        return 2

    def query(self, **kwargs):
        self.where = kwargs.get("where")
        return {"documents": [["AEB limits on wet roads", "ACC keeps distance"]],
                "metadatas": [[
                    {"source": "a.pdf", "page": 4, "chunk_index": 0,
                     "system": "AEB", "source_type": "reference", "car_model": "Model A"},
                    {"source": "b.pdf", "page": 8, "chunk_index": 1,
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


def test_hybrid_retrieval_fuses_exact_acronym_match(monkeypatch):
    class HybridCollection(FakeCollection):
        def get(self, **kwargs):
            return {"documents": ["Adaptive cruise control keeps distance.",
                                  "SAE J3016 defines driving automation levels."],
                    "metadatas": [
                        {"source": "acc.pdf", "page": 1, "chunk_index": 0,
                         "system": "ACC", "source_type": "reference", "car_model": ""},
                        {"source": "sae.pdf", "page": 2, "chunk_index": 0,
                         "system": "general", "source_type": "standard", "car_model": ""}]}

        def query(self, **kwargs):
            return {"documents": [["Adaptive cruise control keeps distance."]],
                    "metadatas": [[{"source": "acc.pdf", "page": 1, "chunk_index": 0,
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
