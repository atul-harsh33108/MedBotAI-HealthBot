"""
Standalone logic test for the HealthBot LangGraph workflow.

This mirrors the graph structure defined in healthbot.ipynb but replaces the
LLM and Tavily tool with deterministic fakes, and replaces input() with a
scripted queue of patient responses. It verifies:

  1. The full linear path executes in the correct order.
  2. State fields are populated correctly by each node.
  3. The accumulated message history is actually replayed into the model, and
     that the quiz/grading nodes see the summary but NOT the raw search results.
  4. The hybrid search fallback triggers when the LLM returns no tool call.
  5. Choosing "yes" at the end loops back through reset_state and clears
     topic-specific fields before ask_topic runs again.
  6. Choosing "no" ends the session.
  7. A multi-line justification survives grade parsing intact.
  8. invoke_with_retry retries transient failures and raises a readable
     HealthBotServiceError once attempts are exhausted.
  9. render_markdown falls back to cleaned plain text outside a notebook.
 10. message_text extracts plain text from provider content blocks, excluding
     reasoning blocks and signatures.
 11. Grade parsing tolerates the decorated label formats models actually emit.
 12. Provider failover: a daily cap or provider error switches to the next
     configured key, transient limits retry in place, and the switch sticks.

Run with:
    .venv\\Scripts\\python.exe test_graph_logic.py
"""

import builtins
import contextlib
import io
import re
import sys
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, RemoveMessage


# ---------------------------------------------------------------------------
# State schema (identical to notebook)
# ---------------------------------------------------------------------------
class HealthBotState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    topic: Optional[str]
    search_queries: Optional[list[str]]
    search_used_tool_call: Optional[bool]
    search_results: Optional[str]
    summary: Optional[str]
    quiz_question: Optional[str]
    patient_answer: Optional[str]
    grade: Optional[str]
    feedback: Optional[str]
    continue_session: Optional[bool]


# ---------------------------------------------------------------------------
# Shared helpers (identical to notebook)
# ---------------------------------------------------------------------------
def _is_plain_text(message: AnyMessage) -> bool:
    return (
        isinstance(message.content, str)
        and not isinstance(message, ToolMessage)
        and not getattr(message, "tool_calls", None)
    )


def _merge_consecutive(messages: list[AnyMessage]) -> list[AnyMessage]:
    merged: list[AnyMessage] = []
    for message in messages:
        if (
            merged
            and type(merged[-1]) is type(message)
            and _is_plain_text(merged[-1])
            and _is_plain_text(message)
        ):
            merged[-1] = type(message)(content=f"{merged[-1].content}\n\n{message.content}")
        else:
            merged.append(message)
    return merged


def build_model_payload(
    system_prompt: str,
    state: HealthBotState,
    user_prompt: str,
    include_search_results: bool = True,
) -> list[AnyMessage]:
    history = list(state.get("messages") or [])
    if not include_search_results:
        history = [
            m for m in history
            if not isinstance(m, ToolMessage) and not getattr(m, "tool_calls", None)
        ]

    return _merge_consecutive(
        [SystemMessage(content=system_prompt), *history, HumanMessage(content=user_prompt)]
    )


class HealthBotServiceError(RuntimeError):
    """No configured provider could complete the request."""


class _NonRecoverable(Exception):
    """Internal marker: switching providers would not fix this error."""


TRANSIENT_MARKERS = (
    "rate limit", "429", "overloaded", "timeout", "timed out",
    "temporarily unavailable", "502", "503", "504",
)

DAILY_QUOTA_MARKERS = ("per day", "daily limit", "insufficient_quota", "exceeded your current quota")

PROVIDER_MARKERS = (
    "api key", "unauthorized", "401", "403", "permission denied", "authentication",
    "not found", "404", "does not exist", "does not support", "unsupported",
    "model_not_found", "invalid model", "tool choice", "resource_exhausted", "quota",
)

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 0  # no real sleeping in tests


def _normalize(text: str) -> str:
    return re.sub(r"[\s_]+", "", text.lower())


def _matches(error: Exception, markers: tuple[str, ...]) -> bool:
    haystack = _normalize(str(error))
    return any(_normalize(marker) in haystack for marker in markers)


def invoke_tool(tool, query, what: str = "Tavily"):
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return tool.invoke(query)
        except Exception as error:
            last_error = error
            if attempt == MAX_ATTEMPTS or not _matches(error, TRANSIENT_MARKERS):
                break
            time.sleep(BACKOFF_SECONDS * attempt)

    raise HealthBotServiceError(
        f"Sorry -- {what} could not be completed right now. "
        f"Underlying error: {type(last_error).__name__}: {last_error}"
    ) from last_error


# The graph tests use a single fake provider; TEST 10 swaps in a multi-provider
# table to exercise failover.
PREFERRED_MODEL_KEY = "1"
AVAILABLE_MODELS = {"1": {"label": "Fake provider", "provider": "fake", "model": "fake"}}
FAILOVER_ORDER = ["1"]
_model_cache: dict = {}
_active_provider = 0


def _get_model(key: str, with_tools: bool):
    """Overridden by the graph tests and by TEST 10."""
    raise NotImplementedError


def active_model_label() -> str:
    return AVAILABLE_MODELS[FAILOVER_ORDER[_active_provider]]["label"]


def _attempt(model, payload, what: str, label: str):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return model.invoke(payload)
        except Exception as error:
            if _matches(error, DAILY_QUOTA_MARKERS) or _matches(error, PROVIDER_MARKERS):
                raise
            if not _matches(error, TRANSIENT_MARKERS):
                raise _NonRecoverable(str(error)) from error
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(BACKOFF_SECONDS * attempt)


def invoke_model(payload, what: str = "the request", with_tools: bool = False):
    global _active_provider

    failures = []
    for index in range(_active_provider, len(FAILOVER_ORDER)):
        key = FAILOVER_ORDER[index]
        label = AVAILABLE_MODELS[key]["label"]

        try:
            result = _attempt(_get_model(key, with_tools), payload, what, label)
        except _NonRecoverable as error:
            raise HealthBotServiceError(
                f"Sorry -- {what} failed for a reason that changing provider will not fix: {error}"
            ) from error.__cause__
        except Exception as error:
            failures.append(f"{label} -> {type(error).__name__}: {error}")
            continue

        _active_provider = index
        return result

    raise HealthBotServiceError(
        f"Sorry -- {what} could not be completed. Every configured provider failed:\n  "
        + "\n  ".join(failures)
    )


# ---------------------------------------------------------------------------
# Model responses (identical to notebook)
# ---------------------------------------------------------------------------
def message_text(message: AnyMessage) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return str(text)

    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


# ---------------------------------------------------------------------------
# Grade parsing (identical to notebook)
# ---------------------------------------------------------------------------
GRADE_LABEL = re.compile(r"^[ \t>*_#\-]*grade\s*[:\-]\s*\**\s*", re.IGNORECASE | re.MULTILINE)
JUSTIFICATION_LABEL = re.compile(r"^[ \t>*_#\-]*justification\s*[:\-]\s*\**\s*", re.IGNORECASE | re.MULTILINE)
LETTER_GRADE = re.compile(r"^([A-F][+-]?)(?![A-Za-z0-9])", re.IGNORECASE)


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```[^\n]*\n?", "", text)
    return re.sub(r"\n?```\s*$", "", text).strip()


def parse_grade_response(result_text: str) -> tuple[str, str]:
    text = _strip_code_fence(result_text.strip())

    grade = ""
    after_grade_line = ""
    label = GRADE_LABEL.search(text)
    if label:
        newline = text.find("\n", label.end())
        end = len(text) if newline == -1 else newline
        first_line = text[label.end():end].strip()
        letter = LETTER_GRADE.match(first_line)
        grade = letter.group(1).upper() if letter else first_line
        after_grade_line = text[end:]

    justification = JUSTIFICATION_LABEL.search(text)
    if justification:
        feedback = text[justification.end():].strip()
    else:
        feedback = after_grade_line.strip() or text

    return grade or "N/A", feedback


# ---------------------------------------------------------------------------
# Patient-facing output (identical to notebook)
# ---------------------------------------------------------------------------
def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


IN_NOTEBOOK = _in_notebook()


def _to_plain_text(markdown_text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", markdown_text, flags=re.MULTILINE)
    text = re.sub(r"^---\s*$", "-" * 70, text, flags=re.MULTILINE)
    return text.replace("**", "")


def render_markdown(markdown_text: str) -> None:
    sys.stdout.flush()
    if IN_NOTEBOOK:
        from IPython.display import Markdown, display
        display(Markdown(markdown_text))
    else:
        print(_to_plain_text(markdown_text))


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeToolCallLLM:
    """Simulates an LLM bound with forced tool choice that DOES call the tool."""

    last_payload = None

    def invoke(self, messages):
        FakeToolCallLLM.last_payload = messages
        return AIMessage(
            content="",
            tool_calls=[{"name": "tavily_search_results_json", "args": {"query": "diabetes"}, "id": "call_1"}],
        )


class FakeNoToolCallLLM:
    """Simulates an LLM that fails to call the tool, to exercise the fallback path."""

    last_payload = None

    def invoke(self, messages):
        FakeNoToolCallLLM.last_payload = messages
        return AIMessage(content="I looked into it.", tool_calls=[])


class FakeTavilyTool:
    call_log = []

    def invoke(self, query):
        FakeTavilyTool.call_log.append(query)
        return [{"title": "Diabetes Overview", "url": "https://example.com", "content": "Diabetes is a condition...", "score": 0.9}]


FAKE_SUMMARY = "FAKE SUMMARY: Diabetes is a chronic condition affecting blood sugar. " * 3

# Bolded labels and a justification spanning several lines: the shape models
# actually produce, which the original parser could not read.
FAKE_GRADE_RESPONSE = (
    "**Grade:** B\n"
    '**Justification:** The answer aligns with "Diabetes is a chronic condition '
    'affecting blood sugar."\n'
    "It could have gone further by mentioning how the body handles insulin.\n"
    "Nice work overall -- you clearly read the summary."
)


class FakeGenericLLM:
    """Used for summarize / quiz / grade nodes -- returns canned, distinguishable text.

    The summary comes back as a list of content blocks, including a reasoning
    block and a signature, which is what providers such as Anthropic and Gemini
    return. This is the shape that leaked raw dicts into the patient-facing
    output before message_text() existed.
    """

    call_count = 0
    payloads: dict[str, list[AnyMessage]] = {}

    def invoke(self, messages):
        FakeGenericLLM.call_count += 1
        system_prompt = messages[0].content

        if "patient-friendly summary" in system_prompt:
            FakeGenericLLM.payloads["summarize"] = messages
            return AIMessage(content=[
                {"type": "text", "text": FAKE_SUMMARY, "extras": {"signature": "El4KXAERTTIPGEyN5yBW"}},
                {"type": "thinking", "thinking": "internal reasoning the patient must never see"},
            ])
        if "single comprehension-check question" in system_prompt:
            FakeGenericLLM.payloads["quiz"] = messages
            return AIMessage(content="What does diabetes affect?")
        if "grading a patient's answer" in system_prompt:
            FakeGenericLLM.payloads["grade"] = messages
            return AIMessage(content=FAKE_GRADE_RESPONSE)
        return AIMessage(content="FAKE GENERIC RESPONSE")


class FlakyLLM:
    """Fails with a transient error `failures` times, then succeeds."""

    def __init__(self, failures, message="429 RESOURCE_EXHAUSTED: quota exceeded"):
        self.remaining = failures
        self.message = message
        self.attempts = 0

    def invoke(self, payload):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError(self.message)
        return AIMessage(content="recovered")


llm_with_tavily = FakeToolCallLLM()
tavily_tool = FakeTavilyTool()
llm = FakeGenericLLM()


# ---------------------------------------------------------------------------
# Nodes (logic copied from healthbot.ipynb)
# ---------------------------------------------------------------------------
SEARCH_SYSTEM_PROMPT = "search for medical info"
SUMMARY_SYSTEM_PROMPT = "Write a clear, empathetic, patient-friendly summary"
QUIZ_SYSTEM_PROMPT = "Write ONE clear, specific single comprehension-check question"
GRADE_SYSTEM_PROMPT = "You are HealthBot, grading a patient's answer"

visited_nodes = []


def ask_topic(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("ask_topic")
    topic = builtins.input("What health topic...? ").strip()
    render_markdown(f"Got it -- let's learn about **{topic}**.")
    return {"topic": topic, "messages": [HumanMessage(content=f"I would like to learn about: {topic}")]}


def describe_search(ai_message, queries: list[str], used_tool_call: bool) -> str:
    shown = ", ".join(f"`{query}`" for query in queries)
    plural = "query" if len(queries) == 1 else "queries"

    if used_tool_call:
        tools = ", ".join(sorted({call["name"] for call in ai_message.tool_calls}))
        return (
            f"**{active_model_label()}** issued a tool call to `{tools}` "
            f"with {len(queries)} {plural}: {shown}"
        )
    return (
        f"**{active_model_label()}** returned no tool call, so HealthBot searched "
        f"Tavily directly for {shown}."
    )


def search_topic(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("search_topic")
    topic = state["topic"]
    ai_message = invoke_model(
        build_model_payload(SEARCH_SYSTEM_PROMPT, state, f"Topic: {topic}"),
        what="the medical search",
        with_tools=True,
    )

    used_tool_call = bool(ai_message.tool_calls)
    queries: list[str] = []
    tool_messages = []

    if used_tool_call:
        for call in ai_message.tool_calls:
            query = call["args"].get("query", topic)
            queries.append(query)
            results = invoke_tool(tavily_tool, query)
            tool_messages.append(ToolMessage(content=str(results), tool_call_id=call["id"]))
    else:
        queries.append(topic)
        results = invoke_tool(tavily_tool, topic)
        tool_messages.append(ToolMessage(content=str(results), tool_call_id="fallback"))

    render_markdown(describe_search(ai_message, queries, used_tool_call))

    search_results_text = "\n\n".join(tm.content for tm in tool_messages)
    return {
        "search_results": search_results_text,
        "search_queries": queries,
        "search_used_tool_call": used_tool_call,
        "messages": [ai_message, *tool_messages],
    }


def summarize_results(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("summarize_results")
    prompt = f"Search results:\n{state['search_results']}"
    response = invoke_model(
        build_model_payload(SUMMARY_SYSTEM_PROMPT, state, prompt),
        what="the summary",
    )
    summary = message_text(response)
    return {"summary": summary, "messages": [AIMessage(content=summary)]}


def present_summary(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("present_summary")
    render_markdown(
        f"### Here's what we found about: {state['topic']}\n\n"
        f"{state['summary']}\n\n"
        "---"
    )
    builtins.input("Press Enter when ready...")
    return {}


def generate_quiz(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("generate_quiz")
    prompt = f"Summary:\n{state['summary']}"
    response = invoke_model(
        build_model_payload(QUIZ_SYSTEM_PROMPT, state, prompt, include_search_results=False),
        what="the quiz question",
    )
    quiz_question = message_text(response)
    return {"quiz_question": quiz_question, "messages": [AIMessage(content=quiz_question)]}


def ask_quiz_question(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("ask_quiz_question")
    render_markdown(f"### Comprehension check\n\n{state['quiz_question']}")
    answer = builtins.input("Your answer: ").strip()
    return {"patient_answer": answer, "messages": [HumanMessage(content=f"My answer: {answer}")]}


def grade_answer(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("grade_answer")
    prompt = f"Patient's answer: {state['patient_answer']}"
    response = invoke_model(
        build_model_payload(GRADE_SYSTEM_PROMPT, state, prompt, include_search_results=False),
        what="the grading",
    )
    result_text = message_text(response)
    grade, feedback = parse_grade_response(result_text)

    return {"grade": grade, "feedback": feedback, "messages": [AIMessage(content=result_text)]}


def present_grade(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("present_grade")
    render_markdown(
        "### Your results\n\n"
        f"**Grade:** {state['grade']}\n\n"
        f"**Feedback:** {state['feedback']}"
    )
    return {}


def ask_continue(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("ask_continue")
    choice = builtins.input("Another topic? (yes/no): ").strip().lower()
    return {"continue_session": choice.startswith("y")}


def route_continue(state: HealthBotState) -> str:
    return "reset_state" if state.get("continue_session") else END


def reset_state(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("reset_state")
    render_markdown("---\n\nStarting a fresh session for your new topic.")
    return {
        "messages": [RemoveMessage(id=m.id) for m in state["messages"]],
        "topic": None,
        "search_queries": None,
        "search_used_tool_call": None,
        "search_results": None,
        "summary": None,
        "quiz_question": None,
        "patient_answer": None,
        "grade": None,
        "feedback": None,
        "continue_session": None,
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph():
    gb = StateGraph(HealthBotState)
    gb.add_node("ask_topic", ask_topic)
    gb.add_node("search_topic", search_topic)
    gb.add_node("summarize_results", summarize_results)
    gb.add_node("present_summary", present_summary)
    gb.add_node("generate_quiz", generate_quiz)
    gb.add_node("ask_quiz_question", ask_quiz_question)
    gb.add_node("grade_answer", grade_answer)
    gb.add_node("present_grade", present_grade)
    gb.add_node("ask_continue", ask_continue)
    gb.add_node("reset_state", reset_state)

    gb.add_edge(START, "ask_topic")
    gb.add_edge("ask_topic", "search_topic")
    gb.add_edge("search_topic", "summarize_results")
    gb.add_edge("summarize_results", "present_summary")
    gb.add_edge("present_summary", "generate_quiz")
    gb.add_edge("generate_quiz", "ask_quiz_question")
    gb.add_edge("ask_quiz_question", "grade_answer")
    gb.add_edge("grade_answer", "present_grade")
    gb.add_edge("present_grade", "ask_continue")
    gb.add_conditional_edges("ask_continue", route_continue, {"reset_state": "reset_state", END: END})
    gb.add_edge("reset_state", "ask_topic")

    return gb.compile()


captured_output = io.StringIO()


def run_test(scripted_inputs, use_fallback=False):
    global llm_with_tavily, captured_output, _get_model, _active_provider, FAILOVER_ORDER
    llm_with_tavily = FakeNoToolCallLLM() if use_fallback else FakeToolCallLLM()

    # Single fake provider for the graph tests; failover itself is TEST 10.
    FAILOVER_ORDER = ["1"]
    _active_provider = 0
    _get_model = lambda key, with_tools: llm_with_tavily if with_tools else llm

    visited_nodes.clear()
    FakeTavilyTool.call_log.clear()
    FakeGenericLLM.payloads.clear()
    captured_output = io.StringIO()

    input_queue = list(scripted_inputs)

    def fake_input(prompt=""):
        return input_queue.pop(0)

    original_input = builtins.input
    builtins.input = fake_input
    try:
        graph = build_graph()
        initial_state: HealthBotState = {
            "messages": [], "topic": None, "search_queries": None,
            "search_used_tool_call": None, "search_results": None, "summary": None,
            "quiz_question": None, "patient_answer": None, "grade": None,
            "feedback": None, "continue_session": None,
        }
        # Node output is captured rather than printed, so the test log stays
        # readable -- and so test 7 can assert on what the patient would see.
        with contextlib.redirect_stdout(captured_output):
            final_state = graph.invoke(initial_state, config={"recursion_limit": 100})
    finally:
        builtins.input = original_input

    return final_state


def main():
    print("=" * 70)
    print("TEST 1: Single topic, then exit (tool-call path)")
    print("=" * 70)
    final_state = run_test(
        scripted_inputs=["diabetes", "", "it affects blood sugar", "no"],
        use_fallback=False,
    )
    assert visited_nodes == [
        "ask_topic", "search_topic", "summarize_results", "present_summary",
        "generate_quiz", "ask_quiz_question", "grade_answer", "present_grade", "ask_continue",
    ], f"Unexpected node path: {visited_nodes}"
    assert final_state["grade"] == "B", f"Expected grade B, got {final_state['grade']}"
    assert "Diabetes is a chronic condition" in final_state["feedback"]
    assert FakeTavilyTool.call_log == ["diabetes"], f"Expected forced tool-call search, got {FakeTavilyTool.call_log}"

    # Provenance: the model issued the tool call, and that fact is in state and shown.
    assert final_state["search_used_tool_call"] is True, "tool-call path should be recorded"
    assert final_state["search_queries"] == ["diabetes"], final_state["search_queries"]
    shown = captured_output.getvalue()
    assert "issued a tool call" in shown, "tool-call provenance was not displayed"
    assert "tavily_search_results_json" in shown, "tool name missing from provenance line"
    print("PASS: single-topic path, tool-call search, grading, and exit all correct;")
    print("      search provenance recorded in state and shown to the reader.\n")

    print("=" * 70)
    print("TEST 2: Message history is replayed into the model")
    print("=" * 70)
    summarize_payload = FakeGenericLLM.payloads["summarize"]
    quiz_payload = FakeGenericLLM.payloads["quiz"]
    grade_payload = FakeGenericLLM.payloads["grade"]

    # Every payload must carry more than just [system, user] -- that is the
    # whole point of build_model_payload.
    for label, payload in (("summarize", summarize_payload), ("quiz", quiz_payload), ("grade", grade_payload)):
        assert len(payload) > 2, f"{label} payload has no replayed history: {len(payload)} messages"
        assert isinstance(payload[0], SystemMessage), f"{label} payload must start with the system prompt"

    # summarize needs the raw search results, so the tool result stays in.
    assert any(isinstance(m, ToolMessage) for m in summarize_payload), \
        "summarize payload should include the Tavily ToolMessage"

    # quiz + grade must NOT see raw search results -- the summary is their only
    # source of truth. Both halves of the tool-call pair must be gone.
    for label, payload in (("quiz", quiz_payload), ("grade", grade_payload)):
        assert not any(isinstance(m, ToolMessage) for m in payload), \
            f"{label} payload must not contain raw search results"
        assert not any(getattr(m, "tool_calls", None) for m in payload), \
            f"{label} payload must not contain an unmatched tool call"

    # ...but they must see the summary and the earlier conversation.
    assert any(FAKE_SUMMARY.strip()[:40] in m.content for m in quiz_payload if isinstance(m.content, str)), \
        "quiz payload should contain the summary"
    assert any("What does diabetes affect?" in m.content for m in grade_payload if isinstance(m.content, str)), \
        "grade payload should contain the quiz question"

    # No two consecutive messages may share a role, or strict providers reject it.
    for label, payload in (("summarize", summarize_payload), ("quiz", quiz_payload), ("grade", grade_payload)):
        for a, b in zip(payload, payload[1:]):
            assert not (type(a) is type(b) and _is_plain_text(a) and _is_plain_text(b)), \
                f"{label} payload has consecutive {type(a).__name__} messages"
    print("PASS: history is replayed; search results reach summarize but not quiz/grade;")
    print("      no consecutive same-role messages in any payload.\n")

    print("=" * 70)
    print("TEST 3: Multi-line justification survives grade parsing")
    print("=" * 70)
    feedback = final_state["feedback"]
    assert feedback.startswith("The answer aligns with"), f"Justification start lost: {feedback!r}"
    assert "mentioning how the body handles insulin" in feedback, "Second line of justification was dropped"
    assert "Nice work overall" in feedback, "Third line of justification was dropped"
    assert "Grade:" not in feedback, "Grade line leaked into the feedback"
    print("PASS: all three justification lines retained, grade line excluded.\n")

    print("=" * 70)
    print("TEST 4: Two topics via restart loop, then exit (verifies state reset)")
    print("=" * 70)
    final_state = run_test(
        scripted_inputs=[
            "asthma", "", "breathing problem", "yes",   # topic 1 -> continue
            "flu", "", "viral infection", "no",          # topic 2 -> exit
        ],
        use_fallback=False,
    )
    expected_path = (
        ["ask_topic", "search_topic", "summarize_results", "present_summary",
         "generate_quiz", "ask_quiz_question", "grade_answer", "present_grade", "ask_continue"]
        + ["reset_state"]
        + ["ask_topic", "search_topic", "summarize_results", "present_summary",
           "generate_quiz", "ask_quiz_question", "grade_answer", "present_grade", "ask_continue"]
    )
    assert visited_nodes == expected_path, f"Unexpected node path: {visited_nodes}"
    assert final_state["topic"] == "flu", f"Expected topic to be updated to 'flu', got {final_state['topic']}"
    assert final_state["continue_session"] is False

    # After the reset, topic 2's payloads must not mention topic 1.
    for label in ("summarize", "quiz", "grade"):
        joined = " ".join(m.content for m in FakeGenericLLM.payloads[label] if isinstance(m.content, str))
        assert "asthma" not in joined.lower(), f"topic 1 leaked into topic 2's {label} payload"
    print("PASS: reset_state re-enters ask_topic, topic 2 completes, and no topic-1")
    print("      content leaks into the replayed history.\n")

    print("=" * 70)
    print("TEST 5: Hybrid search fallback (LLM returns no tool call)")
    print("=" * 70)
    final_state = run_test(
        scripted_inputs=["migraine", "", "head pain", "no"],
        use_fallback=True,
    )
    assert FakeTavilyTool.call_log == ["migraine"], f"Expected fallback direct search on topic, got {FakeTavilyTool.call_log}"

    # The fallback must say so, rather than claiming the model made the call.
    assert final_state["search_used_tool_call"] is False, "fallback path should be recorded"
    fallback_shown = captured_output.getvalue()
    assert "returned no tool call" in fallback_shown, "fallback provenance was not displayed"
    assert "issued a tool call" not in fallback_shown, "fallback must not claim a model tool call"
    print("PASS: fallback path invokes Tavily directly and reports itself honestly")
    print("      rather than claiming the model made the call.\n")

    print("=" * 70)
    print("TEST 6: invoke_tool retries transient search failures")
    print("=" * 70)
    # Tavily is the only search backend, so this path retries but never fails
    # over. Model-side retry and failover are TEST 10.
    flaky = FlakyLLM(failures=2, message="503 temporarily unavailable")
    result = invoke_tool(flaky, "diabetes", what="Tavily")
    assert result.content == "recovered", "retry should eventually succeed"
    assert flaky.attempts == 3, f"expected 3 attempts, got {flaky.attempts}"

    doomed = FlakyLLM(failures=99, message="503 temporarily unavailable")
    try:
        invoke_tool(doomed, "diabetes", what="Tavily")
    except HealthBotServiceError as error:
        assert doomed.attempts == MAX_ATTEMPTS, f"expected {MAX_ATTEMPTS} attempts, got {doomed.attempts}"
        assert "Tavily" in str(error), "error message should name what failed"
    else:
        raise AssertionError("expected HealthBotServiceError after exhausting retries")

    # A non-transient error should fail fast rather than burning all retries.
    permanent = FlakyLLM(failures=99, message="401 invalid api key")
    try:
        invoke_tool(permanent, "diabetes", what="Tavily")
    except HealthBotServiceError:
        assert permanent.attempts == 1, f"non-transient error should not retry, got {permanent.attempts} attempts"
    else:
        raise AssertionError("expected HealthBotServiceError for a permanent failure")
    print("PASS: transient errors retried, exhaustion raises HealthBotServiceError,")
    print("      non-transient errors fail fast.\n")

    print("=" * 70)
    print("TEST 7: render_markdown falls back to clean plain text off-notebook")
    print("=" * 70)
    assert IN_NOTEBOOK is False, "this test process is not a notebook kernel"

    # Markup we emit must not survive into the plain-text fallback.
    sample = "### Your results\n\n**Grade:** A\n\n**Feedback:** Nice work.\n\n---"
    plain = _to_plain_text(sample)
    assert "###" not in plain, "heading markers leaked into plain text"
    assert "**" not in plain, "bold markers leaked into plain text"
    assert "Your results" in plain and "Grade: A" in plain
    assert "-" * 70 in plain, "horizontal rule should become a ruler line"

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        render_markdown("### Heading\n\n**bold** text")
    rendered = buffer.getvalue()
    assert "Heading" in rendered and "bold text" in rendered
    assert "#" not in rendered and "*" not in rendered, f"markup survived: {rendered!r}"

    # And the real run must have shown the patient the grade and feedback.
    final_state = run_test(scripted_inputs=["diabetes", "", "blood sugar", "no"])
    shown = captured_output.getvalue()
    assert "Your results" in shown, "grade block was never displayed"
    assert f"Grade: {final_state['grade']}" in shown, "grade value not shown to the patient"
    assert "Comprehension check" in shown, "quiz heading was never displayed"
    assert "**" not in shown, f"raw markup reached the patient: {shown!r}"
    print("PASS: fallback strips markup, and the run displays the quiz and grade blocks.\n")

    print("=" * 70)
    print("TEST 8: message_text extracts text from provider content blocks")
    print("=" * 70)
    blocks = AIMessage(content=[
        {
            "type": "text",
            "text": "Paragraph one.\n\nParagraph two.",
            "extras": {"signature": "El4KXAERTTIPGEyN5yBWlBcGHX4GS5HfV6"},
        },
        {"type": "thinking", "thinking": "internal reasoning that must not be shown"},
    ])
    extracted = message_text(blocks)
    assert type(extracted) is str, f"expected a plain str, got {type(extracted).__name__}"
    assert "{'type'" not in extracted, "raw content-block dict leaked into the text"
    assert "signature" not in extracted, "provider signature blob leaked into the text"
    assert "internal reasoning" not in extracted, "reasoning block leaked into the text"
    assert "\\n" not in extracted, "escaped newlines present -- still a repr, not real text"
    assert extracted == "Paragraph one.\n\nParagraph two.", "paragraph break not preserved"
    assert message_text(AIMessage(content="plain string")) == "plain string"

    # The summary the graph produced must be clean text, not a stringified list.
    assert isinstance(final_state["summary"], str)
    assert "{'type'" not in final_state["summary"], "raw blocks reached state['summary']"
    assert "internal reasoning" not in captured_output.getvalue(), \
        "reasoning block was displayed to the patient"
    print("PASS: text extracted from blocks; dicts, signatures and reasoning excluded.\n")

    print("=" * 70)
    print("TEST 9: grade parsing tolerates the formats models actually produce")
    print("=" * 70)
    fence = "```"
    cases = {
        "plain (as prompted)": ("Grade: B\nJustification: Good.", "B"),
        "bolded labels": ("**Grade:** B\n**Justification:** Good.", "B"),
        "markdown heading": ("### Grade: B\nJustification: Good.", "B"),
        "bullet prefixed": ("- Grade: B\n- Justification: Good.", "B"),
        "blockquoted with modifier": ("> Grade: A-\n> Justification: Good.", "A-"),
        "code fenced": (f"{fence}\nGrade: C\nJustification: Good.\n{fence}", "C"),
        "lowercase label": ("grade: d\njustification: Good.", "D"),
        "no justification label": ("Grade: F\nThat missed the point.", "F"),
        "trailing parenthetical": ("Grade: B (solid effort)\nJustification: Good.", "B"),
    }
    for name, (text, expected) in cases.items():
        grade, feedback = parse_grade_response(text)
        assert grade == expected, f"{name}: expected {expected!r}, got {grade!r}"
        assert "```" not in feedback, f"{name}: code fence leaked into feedback"
        assert not feedback.startswith("*"), f"{name}: stray emphasis at start of feedback"
        assert "Grade:" not in feedback, f"{name}: grade line leaked into feedback"

    # A response with no recognisable grade must not silently claim a letter.
    grade, feedback = parse_grade_response("I am not able to grade this.")
    assert grade == "N/A", f"expected N/A for an ungradeable response, got {grade!r}"
    assert feedback == "I am not able to grade this.", "feedback should fall back to the full text"
    print(f"PASS: all {len(cases)} formats parsed correctly, plus the no-grade fallback.\n")

    print("=" * 70)
    print("TEST 10: provider failover across every configured key")
    print("=" * 70)
    global AVAILABLE_MODELS, FAILOVER_ORDER, _get_model, _active_provider

    # The real error text from a Gemini free-tier daily cap.
    gemini_daily = (
        "Error calling model 'gemini-3.5-flash-lite' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
        "You exceeded your current quota. Quota exceeded for metric: generate_content_free_tier_requests, "
        "limit: 20. 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
    )

    AVAILABLE_MODELS = {
        "1": {"label": "OpenAI", "provider": "openai", "model": "m"},
        "2": {"label": "Gemini", "provider": "google_genai", "model": "m"},
        "3": {"label": "Anthropic", "provider": "anthropic", "model": "m"},
        "4": {"label": "OpenRouter", "provider": "openrouter", "model": "m"},
    }

    def wire(providers, preferred="2"):
        """Point model lookup at fakes, preferring Gemini as the reported run did."""
        global FAILOVER_ORDER, _get_model, _active_provider
        FAILOVER_ORDER = [preferred] + [k for k in AVAILABLE_MODELS if k != preferred]
        _active_provider = 0
        _get_model = lambda key, with_tools: providers[key]

    # (a) Daily quota on the preferred provider: switch without burning retries.
    providers = {
        "1": FlakyLLM(failures=0),
        "2": FlakyLLM(failures=99, message=gemini_daily),
        "3": FlakyLLM(failures=0),
        "4": FlakyLLM(failures=0),
    }
    wire(providers)
    assert invoke_model([], what="the medical search", with_tools=True).content == "recovered"
    assert providers["2"].attempts == 1, \
        f"a daily cap must not be retried, got {providers['2'].attempts} attempts"
    assert providers["1"].attempts == 1, "should have fallen back to the next provider"

    # (b) The switch sticks, so later nodes skip the exhausted provider.
    for what in ("the summary", "the quiz question", "the grading"):
        assert invoke_model([], what=what).content == "recovered"
    assert providers["2"].attempts == 1, "exhausted provider was probed again"

    # (c) A per-minute rate limit is retried in place, not failed over.
    providers = {
        "1": FlakyLLM(failures=0),
        "2": FlakyLLM(failures=2, message="429 rate limit exceeded, slow down"),
        "3": FlakyLLM(failures=0),
        "4": FlakyLLM(failures=0),
    }
    wire(providers)
    assert invoke_model([], what="the summary").content == "recovered"
    assert providers["2"].attempts == 3, f"expected 3 attempts, got {providers['2'].attempts}"
    assert providers["1"].attempts == 0, "should not fail over for a transient limit"

    # (d) Bad key and missing model both trigger a switch; the last one serves it.
    providers = {
        "1": FlakyLLM(failures=99, message="401 Unauthorized: invalid api key"),
        "2": FlakyLLM(failures=99, message=gemini_daily),
        "3": FlakyLLM(failures=99, message="404 model not found"),
        "4": FlakyLLM(failures=0),
    }
    wire(providers)
    assert invoke_model([], what="the grading").content == "recovered"

    # (e) All four down: one error naming each.
    providers = {k: FlakyLLM(failures=99, message=gemini_daily) for k in "1234"}
    wire(providers)
    try:
        invoke_model([], what="the summary")
    except HealthBotServiceError as error:
        for label in ("OpenAI", "Gemini", "Anthropic", "OpenRouter"):
            assert label in str(error), f"{label} missing from the combined error"
    else:
        raise AssertionError("expected HealthBotServiceError when every provider fails")

    # (f) An error that looks like our bug is raised at once, not tried four times.
    providers = {k: FlakyLLM(failures=99, message="TypeError: unhashable type: 'dict'") for k in "1234"}
    wire(providers)
    try:
        invoke_model([], what="the summary")
    except HealthBotServiceError as error:
        assert "will not fix" in str(error), str(error)
        assert providers["2"].attempts == 1, f"expected 1 attempt, got {providers['2'].attempts}"
        assert sum(providers[k].attempts for k in "134") == 0, "should not have tried other providers"
    else:
        raise AssertionError("expected HealthBotServiceError for a non-recoverable error")

    print("PASS: daily caps and provider errors switch without waiting, transient limits")
    print("      retry in place, the switch sticks, and our own bugs surface immediately.\n")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
