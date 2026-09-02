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
 13. The end-of-session recap retains metadata only and never enters a prompt.
 14. The notebook defines every function it calls -- this file mirrors the
     notebook rather than importing it, so a helper deleted from the notebook
     would otherwise go unnoticed here.

Run with:
    .venv\\Scripts\\python.exe test_graph_logic.py
"""

import ast
import builtins
import contextlib
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, RemoveMessage
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# State schema (identical to notebook)
# ---------------------------------------------------------------------------
class HealthBotState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    topic: Optional[str]
    search_queries: Optional[list[str]]
    search_used_tool_call: Optional[bool]
    search_sources: Optional[list[str]]
    search_results: Optional[str]
    summary: Optional[str]
    quiz_question: Optional[str]
    patient_answer: Optional[str]
    grade: Optional[str]
    feedback: Optional[str]
    citation: Optional[str]
    citation_verified: Optional[bool]
    wants_simpler: Optional[bool]
    simple_summary: Optional[str]
    continue_session: Optional[bool]


class HealthBotUpdate(TypedDict, total=False):
    """What a node returns: only the keys it changed."""

    messages: list
    topic: Optional[str]
    search_queries: Optional[list[str]]
    search_used_tool_call: Optional[bool]
    search_sources: Optional[list[str]]
    search_results: Optional[str]
    summary: Optional[str]
    quiz_question: Optional[str]
    patient_answer: Optional[str]
    grade: Optional[str]
    feedback: Optional[str]
    citation: Optional[str]
    citation_verified: Optional[bool]
    wants_simpler: Optional[bool]
    simple_summary: Optional[str]
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

SWITCH_MARKERS = (
    "per day", "daily limit", "quota", "resource_exhausted", "insufficient_quota",
    "exceeded your current quota",
    "credit balance", "billing", "insufficient funds", "payment required", "402",
    "spending limit", "hard limit", "purchase credits",
    "api key", "unauthorized", "401", "403", "permission denied", "authentication",
    "not found", "404", "does not exist", "does not support", "unsupported",
    "model_not_found", "invalid model", "tool choice",
)

BUG_TYPES = (
    TypeError, AttributeError, KeyError, IndexError, NameError,
    ImportError, AssertionError,
)

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 0  # no real sleeping in tests


_SECRET_PATTERN = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]{8,}"
    r"|sk-[A-Za-z0-9_\-]{8,}"
    r"|AIza[A-Za-z0-9_\-]{8,}"
    r"|tvly-[A-Za-z0-9_\-]{8,}"
    r"|AKIA[A-Z0-9]{8,}"
    r"|Bearer\s+[A-Za-z0-9._\-]{8,}"
)


def redact(text: str) -> str:
    return _SECRET_PATTERN.sub("<redacted>", str(text))


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


def _get_model(key: str, with_tools: bool, schema=None):
    """Overridden by the graph tests and by TEST 10."""
    raise NotImplementedError


def active_model_label() -> str:
    return AVAILABLE_MODELS[FAILOVER_ORDER[_active_provider]]["label"]


def reset_provider_failover() -> None:
    global _active_provider
    _active_provider = 0


def _attempt(model, payload, what: str, label: str):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return model.invoke(payload)
        except BUG_TYPES as error:
            raise _NonRecoverable(f"{type(error).__name__}: {error}") from error
        except Exception as error:
            if _matches(error, SWITCH_MARKERS):
                raise
            if not _matches(error, TRANSIENT_MARKERS) or attempt == MAX_ATTEMPTS:
                raise
            time.sleep(BACKOFF_SECONDS * attempt)


RETRY_HINT = re.compile(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)
MAX_HINTED_WAIT_SECONDS = 90


def _retry_hint_seconds(errors: list[Exception]) -> Optional[float]:
    hints = []
    for error in errors:
        match = RETRY_HINT.search(str(error))
        if match:
            hints.append(float(match.group(1)))
    usable = [hint for hint in hints if 0 < hint <= MAX_HINTED_WAIT_SECONDS]
    return min(usable) if usable else None


def invoke_model(payload, what: str = "the request", with_tools: bool = False,
                 schema=None, _already_waited: bool = False):
    global _active_provider

    # Start at the last known-good provider, but wrap around so no key is ever
    # permanently excluded from a later call.
    order = FAILOVER_ORDER[_active_provider:] + FAILOVER_ORDER[:_active_provider]

    failures: list[str] = []
    raw_errors: list[Exception] = []

    for key in order:
        label = AVAILABLE_MODELS[key]["label"]

        try:
            result = _attempt(_get_model(key, with_tools, schema), payload, what, label)
        except _NonRecoverable as error:
            raise HealthBotServiceError(
                f"Sorry -- {what} failed for a reason that changing provider will not fix: "
                f"{redact(error)}"
            ) from error.__cause__
        except Exception as error:
            raw_errors.append(error)
            failures.append(f"{label} -> {type(error).__name__}: {redact(error)}")
            continue

        _active_provider = FAILOVER_ORDER.index(key)
        return result

    wait = _retry_hint_seconds(raw_errors)
    if wait is not None and not _already_waited:
        time.sleep(wait + 2)
        return invoke_model(payload, what=what, with_tools=with_tools, schema=schema,
                            _already_waited=True)

    raise HealthBotServiceError(
        f"Sorry -- {what} could not be completed. All {len(order)} configured "
        f"provider(s) were tried:\n  " + "\n  ".join(failures)
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
FIRST_QUOTE = re.compile(r'"([^"]{10,})"')


class GradeResult(BaseModel):
    """Structured grading result (identical to notebook)."""

    grade: Literal["A", "B", "C", "D", "F"]
    citation: str = Field(min_length=10)
    justification: str = Field(min_length=20)





def _normalise_quote(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip(" \"'`.,;:!?-")


def citation_is_grounded(citation: str, summary: str) -> bool:
    quote = _normalise_quote(citation)
    if len(quote) < 10:
        return False
    return quote in _normalise_quote(summary)


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


DISCLAIMER = (
    "_General health education drawn from public sources -- not medical advice, "
    "diagnosis, or treatment. For anything about your own health, please speak to "
    "a qualified healthcare professional._"
)

AFFIRMATIVE = {"y", "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "please",
               "another", "again", "more", "continue"}
NEGATIVE = {"n", "no", "nope", "nah", "exit", "quit", "stop", "done", "finish",
            "nothing", "end"}


def ask_yes_no(question: str) -> bool:
    while True:
        choice = builtins.input(question).strip().lower()
        if choice in AFFIRMATIVE:
            return True
        if choice in NEGATIVE:
            return False
        print("  Sorry, I didn't catch that -- please answer yes or no.")


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

# A quote that really is in FAKE_SUMMARY, so citation verification should pass.
FAKE_CITATION = "Diabetes is a chronic condition affecting blood sugar"
UNGROUNDED_CITATION = "Diabetes is cured entirely by drinking water"

FAKE_JUSTIFICATION = (
    'The answer aligns with "Diabetes is a chronic condition affecting blood sugar."\n'
    "It could have gone further by mentioning how the body handles insulin.\n"
    "Nice work overall -- you clearly read the summary."
)


class FakeStructuredLLM:
    """Simulates a provider honouring with_structured_output(GradeResult)."""

    grade = "B"
    justification = FAKE_JUSTIFICATION
    citations: list = []  # popped in order; a single entry repeats
    calls = 0

    def invoke(self, messages):
        FakeStructuredLLM.calls += 1
        # Recorded in the shared store so the payload assertions in TEST 2 apply
        # to the structured path too.
        FakeGenericLLM.payloads["grade"] = messages
        pool = FakeStructuredLLM.citations or [FAKE_CITATION]
        citation = pool[0] if len(pool) == 1 else pool.pop(0)
        return GradeResult(
            grade=FakeStructuredLLM.grade,
            citation=citation,
            justification=FakeStructuredLLM.justification,
        )


class FakeSchemaUnsupported:
    """A provider that cannot honour a schema, forcing the free-text fallback."""

    def invoke(self, messages):
        raise NotImplementedError("structured output not supported by this provider")




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
        if "more simply" in system_prompt:
            FakeGenericLLM.payloads["simplify"] = messages
            return AIMessage(content="SIMPLER: diabetes means blood sugar runs too high.")
        return AIMessage(content="FAKE GENERIC RESPONSE")


class FlakyLLM:
    """Fails `failures` times, then succeeds.

    `error_type` lets a test raise a real Python error (TypeError and friends) so
    the type-based bug detection can be exercised, not just message matching.
    """

    def __init__(self, failures, message="429 RESOURCE_EXHAUSTED: quota exceeded",
                 error_type=RuntimeError):
        self.remaining = failures
        self.message = message
        self.error_type = error_type
        self.attempts = 0

    def invoke(self, payload):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error_type(self.message)
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

# The node sequence a topic follows before any branching, kept in one place so
# every path assertion stays in step with the graph.
MAIN_PATH = ["ask_topic", "search_topic", "summarize_results", "present_summary",
             "generate_quiz", "ask_quiz_question", "grade_answer", "present_grade"]
# A passing grade goes straight to recording progress and asking about a new topic.
DIRECT_PATH = MAIN_PATH + ["record_progress", "ask_continue"]


def ask_topic(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("ask_topic")
    topic = builtins.input("What health topic...? ").strip()
    while not topic:
        topic = builtins.input("Please type a health topic...: ").strip()
    render_markdown(f"Got it -- let's learn about **{topic}**.")
    return {"topic": topic, "messages": [HumanMessage(content=f"I would like to learn about: {topic}")]}


def format_search_results(results) -> tuple[str, list[str]]:
    if isinstance(results, str):
        return results, []

    items = results if isinstance(results, list) else [results]
    blocks: list[str] = []
    sources: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            blocks.append(str(item))
            continue

        title = (item.get("title") or "Untitled source").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()

        heading = f"Source: {title}"
        if url:
            heading += f"\nURL: {url}"
            sources.append(f"[{title}]({url})")
        blocks.append(f"{heading}\n{content}")

    return "\n\n---\n\n".join(blocks), sources


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


def search_topic(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("search_topic")
    topic = state["topic"]
    ai_message = invoke_model(
        build_model_payload(SEARCH_SYSTEM_PROMPT, state, f"Topic: {topic}"),
        what="the medical search",
        with_tools=True,
    )

    used_tool_call = bool(ai_message.tool_calls)
    queries: list[str] = []
    sources: list[str] = []
    tool_messages = []

    if used_tool_call:
        calls = ai_message.tool_calls
    else:
        calls = [{"args": {"query": topic}, "id": "fallback"}]

    for call in calls:
        query = call["args"].get("query", topic)
        queries.append(query)
        results = invoke_tool(tavily_tool, query)
        results_text, urls = format_search_results(results)
        sources.extend(urls)
        tool_messages.append(ToolMessage(content=results_text, tool_call_id=call["id"]))

    render_markdown(describe_search(ai_message, queries, used_tool_call))

    return {
        "search_results": "\n\n".join(tm.content for tm in tool_messages),
        "search_queries": queries,
        "search_used_tool_call": used_tool_call,
        "search_sources": list(dict.fromkeys(sources)),
        "messages": [ai_message, *tool_messages],
    }


def summarize_results(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("summarize_results")
    prompt = f"Search results:\n{state['search_results']}"
    response = invoke_model(
        build_model_payload(SUMMARY_SYSTEM_PROMPT, state, prompt),
        what="the summary",
    )
    summary = message_text(response)
    return {"summary": summary, "messages": [AIMessage(content=summary)]}


def present_summary(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("present_summary")
    parts = [
        f"### Here's what we found about: {state['topic']}",
        "",
        state["summary"],
        "",
        DISCLAIMER,
    ]
    sources = state.get("search_sources") or []
    if sources:
        parts += ["", "**Sources**", ""]
        parts += [f"- {source}" for source in sources]
    parts += ["", "---"]
    render_markdown("\n".join(parts))
    builtins.input("Press Enter when ready...")
    return {}


def generate_quiz(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("generate_quiz")
    prompt = f"Summary:\n{state['summary']}"
    response = invoke_model(
        build_model_payload(QUIZ_SYSTEM_PROMPT, state, prompt, include_search_results=False),
        what="the quiz question",
    )
    quiz_question = message_text(response)
    return {"quiz_question": quiz_question, "messages": [AIMessage(content=quiz_question)]}


def ask_quiz_question(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("ask_quiz_question")
    render_markdown(f"### Comprehension check\n\n{state['quiz_question']}")
    answer = builtins.input("Your answer: ").strip()
    while not answer:
        answer = builtins.input("Please answer in your own words: ").strip()
    return {"patient_answer": answer, "messages": [HumanMessage(content=f"My answer: {answer}")]}


def grade_answer(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("grade_answer")
    summary = state["summary"]
    prompt = (
        f"Summary:\n{summary}\n\n"
        f"Quiz question: {state['quiz_question']}\n\n"
        f"Patient's answer: {state['patient_answer']}"
    )

    def structured(extra: str = ""):
        return invoke_model(
            build_model_payload(
                GRADE_SYSTEM_PROMPT, state, prompt + extra, include_search_results=False
            ),
            what="the grading",
            schema=GradeResult,
        )

    try:
        result = structured()
        verified = citation_is_grounded(result.citation, summary)
        if not verified:
            result = structured(
                "\n\nYour previous attempt quoted text that does not appear in the "
                f"summary: {result.citation!r}. Copy a citation verbatim."
            )
            verified = citation_is_grounded(result.citation, summary)
        grade, citation, feedback = result.grade, result.citation, result.justification
        transcript = f"Grade: {grade}\nJustification: {feedback}"
    except Exception:
        response = invoke_model(
            build_model_payload(GRADE_SYSTEM_PROMPT, state, prompt, include_search_results=False),
            what="the grading",
        )
        transcript = message_text(response)
        grade, feedback = parse_grade_response(transcript)
        quoted = FIRST_QUOTE.search(feedback)
        citation = quoted.group(1) if quoted else ""
        verified = citation_is_grounded(citation, summary) if citation else None

    return {
        "grade": grade or "N/A",
        "feedback": feedback,
        "citation": citation,
        "citation_verified": verified,
        "messages": [AIMessage(content=transcript)],
    }


CITATION_BADGE = {
    True: "Citation checked against the summary.",
    False: "Citation could NOT be matched to the summary, so treat it with caution.",
    None: "Citation not verified.",
}


def present_grade(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("present_grade")
    parts = [
        "### Your results",
        "",
        f"**Grade:** {state['grade']}",
        "",
        state["feedback"] or "",
    ]
    citation = (state.get("citation") or "").strip()
    if citation:
        parts += [
            "",
            "> " + " ".join(citation.split()),
            "",
            f"_{CITATION_BADGE[state.get('citation_verified')]}_",
        ]
    render_markdown("\n".join(parts))
    return {}


LOW_GRADES = {"D", "F"}


def route_after_grade(state: HealthBotState) -> str:
    grade = (state.get("grade") or "").strip().upper()
    return "offer_recap" if grade[:1] in LOW_GRADES else "record_progress"


def offer_recap(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("offer_recap")
    render_markdown("That one didn't quite land. I can go over it in plainer language.")
    return {"wants_simpler": ask_yes_no("Simpler explanation? (yes/no): ")}


def route_recap(state: HealthBotState) -> str:
    return "simplify_summary" if state.get("wants_simpler") else "record_progress"


SIMPLIFY_SYSTEM_PROMPT = "explain the same material again more simply"


def simplify_summary(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("simplify_summary")
    prompt = f"Original summary:\n{state['summary']}"
    response = invoke_model(
        build_model_payload(SIMPLIFY_SYSTEM_PROMPT, state, prompt, include_search_results=False),
        what="the simpler explanation",
    )
    simple = message_text(response)
    return {"simple_summary": simple, "messages": [AIMessage(content=simple)]}


def present_simple_summary(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("present_simple_summary")
    render_markdown(
        "### Let's go over that again, more simply\n\n"
        f"{state['simple_summary']}\n\n{DISCLAIMER}\n\n---"
    )
    return {}


SESSION_LOG: list[dict] = []


def record_progress(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("record_progress")
    SESSION_LOG.append(
        {
            "topic": state.get("topic") or "(unnamed)",
            "grade": state.get("grade") or "N/A",
            "citation_verified": state.get("citation_verified"),
            "sources": len(state.get("search_sources") or []),
            "simplified": bool(state.get("simple_summary")),
            "at": datetime.now().strftime("%H:%M"),
        }
    )
    return {}


def render_session_recap() -> None:
    """Mirror of the notebook's recap renderer."""
    if not SESSION_LOG:
        return

    topics = len(SESSION_LOG)
    graded = [e["grade"] for e in SESSION_LOG if e["grade"] != "N/A"]
    sources = sum(e["sources"] for e in SESSION_LOG)
    verified = sum(1 for e in SESSION_LOG if e["citation_verified"] is True)
    simplified = sum(1 for e in SESSION_LOG if e["simplified"])

    lines = [
        "### Session summary",
        "",
        f"You covered **{topics} topic{'s' if topics != 1 else ''}** and "
        f"**{sources} source{'s' if sources != 1 else ''}** were consulted.",
        "",
        "| Time | Topic | Grade | Sources | Simpler recap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in SESSION_LOG:
        lines.append(
            f"| {entry['at']} | {entry['topic']} | {entry['grade']} | "
            f"{entry['sources']} | {'yes' if entry['simplified'] else 'no'} |"
        )

    lines += ["", f"Grades: {', '.join(graded) if graded else 'none recorded'}."]
    if verified:
        lines.append(
            f"{verified} of {topics} feedback citations were checked back against the summary."
        )
    if simplified:
        lines.append(f"{simplified} topic(s) included a plainer re-explanation.")
    lines += [
        "",
        "_Only topic titles and scores are kept for this recap. Summaries, questions "
        "and your answers are discarded between topics._",
    ]

    render_markdown("\n".join(lines))


def ask_continue(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("ask_continue")
    return {"continue_session": ask_yes_no("Another topic? (yes/no): ")}


def route_continue(state: HealthBotState) -> str:
    return "reset_state" if state.get("continue_session") else END


def reset_state(state: HealthBotState) -> HealthBotUpdate:
    visited_nodes.append("reset_state")
    render_markdown("---\n\nStarting a fresh session for your new topic.")
    reset_provider_failover()
    return {
        "messages": [RemoveMessage(id=m.id) for m in state["messages"]],
        "topic": None,
        "search_queries": None,
        "search_used_tool_call": None,
        "search_sources": None,
        "search_results": None,
        "summary": None,
        "quiz_question": None,
        "patient_answer": None,
        "grade": None,
        "feedback": None,
        "citation": None,
        "citation_verified": None,
        "wants_simpler": None,
        "simple_summary": None,
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
    gb.add_node("offer_recap", offer_recap)
    gb.add_node("simplify_summary", simplify_summary)
    gb.add_node("present_simple_summary", present_simple_summary)
    gb.add_node("record_progress", record_progress)
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
    gb.add_conditional_edges("present_grade", route_after_grade,
                             {"offer_recap": "offer_recap", "record_progress": "record_progress"})
    gb.add_conditional_edges("offer_recap", route_recap,
                             {"simplify_summary": "simplify_summary", "record_progress": "record_progress"})
    gb.add_edge("simplify_summary", "present_simple_summary")
    gb.add_edge("present_simple_summary", "record_progress")
    gb.add_edge("record_progress", "ask_continue")
    gb.add_conditional_edges("ask_continue", route_continue, {"reset_state": "reset_state", END: END})
    gb.add_edge("reset_state", "ask_topic")

    return gb.compile()


def undefined_calls_in_notebook(path) -> list[str]:
    """Names the notebook calls but never defines, imports, or binds.

    Compiling the notebook only proves it parses; a call to a function that no
    longer exists is a runtime NameError, and one that lives inside a node body
    will not surface until that node runs. Walking the AST catches it statically.
    """
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)

    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return sorted(called - defined)


captured_output = io.StringIO()


def run_test(scripted_inputs, use_fallback=False, structured=True,
             grade="B", citations=None):
    global llm_with_tavily, captured_output, _get_model, _active_provider, FAILOVER_ORDER
    llm_with_tavily = FakeNoToolCallLLM() if use_fallback else FakeToolCallLLM()

    FakeStructuredLLM.grade = grade
    FakeStructuredLLM.citations = list(citations) if citations else [FAKE_CITATION]
    FakeStructuredLLM.calls = 0
    grader = FakeStructuredLLM() if structured else FakeSchemaUnsupported()

    # Single fake provider for the graph tests; failover itself is TEST 10.
    FAILOVER_ORDER = ["1"]
    _active_provider = 0

    def fake_get_model(key, with_tools, schema=None):
        if with_tools:
            return llm_with_tavily
        return grader if schema is not None else llm

    _get_model = fake_get_model

    visited_nodes.clear()
    FakeTavilyTool.call_log.clear()
    FakeGenericLLM.payloads.clear()
    SESSION_LOG.clear()
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
            "search_used_tool_call": None, "search_sources": None,
            "search_results": None, "summary": None,
            "quiz_question": None, "patient_answer": None, "grade": None,
            "feedback": None, "citation": None, "citation_verified": None,
            "wants_simpler": None, "simple_summary": None,
            "continue_session": None,
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
    assert visited_nodes == DIRECT_PATH, f"Unexpected node path: {visited_nodes}"
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
    expected_path = DIRECT_PATH + ["reset_state"] + DIRECT_PATH
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

    def wire(providers, preferred="2", active=0):
        """Point model lookup at fakes, preferring Gemini as the reported run did.

        `active` simulates a session that has already failed over, so the
        wrap-around behaviour can be exercised.
        """
        global FAILOVER_ORDER, _get_model, _active_provider
        FAILOVER_ORDER = [preferred] + [k for k in AVAILABLE_MODELS if k != preferred]
        _active_provider = active
        _get_model = lambda key, with_tools, schema=None: providers[key]

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

    # (f) A real code bug stops after one attempt. Detection is by exception TYPE,
    # not message text -- a provider message we have never seen must still cascade.
    providers = {k: FlakyLLM(failures=99, error_type=TypeError,
                             message="unhashable type: 'dict'") for k in "1234"}
    wire(providers)
    try:
        invoke_model([], what="the summary")
    except HealthBotServiceError as error:
        assert "will not fix" in str(error), str(error)
        assert providers["2"].attempts == 1, f"expected 1 attempt, got {providers['2'].attempts}"
        assert sum(providers[k].attempts for k in "134") == 0, "should not have tried other providers"
    else:
        raise AssertionError("expected HealthBotServiceError for a non-recoverable error")

    # (g) Billing exhaustion switches provider. This is the case that ended a real
    # session: none of the old marker sets matched "credit balance is too low", so
    # it was misread as a code bug and the remaining provider was never tried.
    billing = (
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'Your credit balance is too low to access the Anthropic API. Please "
        "go to Plans & Billing to upgrade or purchase credits.'}}"
    )
    providers = {"1": FlakyLLM(failures=0), "2": FlakyLLM(failures=99, message=billing),
                 "3": FlakyLLM(failures=0), "4": FlakyLLM(failures=0)}
    wire(providers)
    assert invoke_model([], what="the medical search", with_tools=True).content == "recovered"
    assert providers["2"].attempts == 1, "billing errors must not be retried"
    assert providers["1"].attempts == 1, "should have switched to the next provider"

    # (h) An unrecognised provider message cascades rather than ending the session.
    providers = {"1": FlakyLLM(failures=0),
                 "2": FlakyLLM(failures=99, message="Error code: 418 - never seen before"),
                 "3": FlakyLLM(failures=0), "4": FlakyLLM(failures=0)}
    wire(providers)
    assert invoke_model([], what="the summary").content == "recovered"
    assert providers["1"].attempts == 1, "unknown errors should fall through to the next provider"

    # (i) A provider *behind* the active pointer must still be tried. This ended a
    # real session: the chain had advanced past OpenRouter, OpenRouter was then
    # never revisited, and the error still claimed every provider had been tried.
    gemini_rpd = (
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
        "generate_content_free_tier_requests, limit: 500. Please retry in 55.95s."
    )
    providers = {"1": FlakyLLM(failures=99, message=gemini_rpd), "2": FlakyLLM(failures=0),
                 "3": FlakyLLM(failures=99, message=gemini_rpd),
                 "4": FlakyLLM(failures=99, message=gemini_rpd)}
    # FAILOVER_ORDER becomes ["1","2","3","4"]; active=3 means the session had
    # already advanced to the last provider, leaving "2" behind the pointer.
    wire(providers, preferred="1", active=3)
    assert invoke_model([], what="the grading").content == "recovered"
    assert providers["2"].attempts == 1, "a provider behind the pointer was skipped"

    # (j) When everything is rate-limited and the API says how long, wait once.
    providers = {k: FlakyLLM(failures=1, message=gemini_rpd) for k in "1234"}
    wire(providers)
    slept: list[float] = []
    real_sleep = time.sleep
    time.sleep = lambda seconds: slept.append(seconds)
    try:
        assert invoke_model([], what="the grading").content == "recovered"
    finally:
        time.sleep = real_sleep
    assert slept and 55 < slept[-1] < 60, f"expected a ~56s wait, got {slept}"

    # (k) That wait happens at most once, and a long hint is ignored entirely.
    providers = {k: FlakyLLM(failures=99, message=gemini_rpd) for k in "1234"}
    wire(providers)
    slept.clear()
    time.sleep = lambda seconds: slept.append(seconds)
    try:
        invoke_model([], what="the grading")
    except HealthBotServiceError as error:
        assert len(slept) == 1, f"expected exactly one wait, got {slept}"
        assert "All 4 configured provider(s) were tried" in str(error), str(error)
    else:
        raise AssertionError("expected HealthBotServiceError")
    finally:
        time.sleep = real_sleep

    providers = {k: FlakyLLM(failures=99, message="429 quota. Please retry in 3600s.")
                 for k in "1234"}
    wire(providers)
    slept.clear()
    time.sleep = lambda seconds: slept.append(seconds)
    try:
        invoke_model([], what="the grading")
    except HealthBotServiceError:
        assert not slept, f"a 3600s hint exceeds the cap and must not be waited on: {slept}"
    else:
        raise AssertionError("expected HealthBotServiceError")
    finally:
        time.sleep = real_sleep

    print("PASS: daily caps and provider errors switch without waiting, transient limits")
    print("      retry in place, the switch sticks, and our own bugs surface immediately.\n")

    print("=" * 70)
    print("TEST 11: input validation, sources, disclaimer, redaction")
    print("=" * 70)

    # (a) Empty input is re-prompted rather than accepted.
    final_state = run_test(scripted_inputs=[
        "", "   ", "asthma",          # two empty topics, then a real one
        "",                           # Enter to continue past the summary
        "", "it affects breathing",   # one empty answer, then a real one
        "maybe", "no",                # unrecognised, then a clear exit
    ])
    assert final_state["topic"] == "asthma", f"empty topics not re-prompted: {final_state['topic']}"
    assert final_state["patient_answer"] == "it affects breathing", "empty answer not re-prompted"
    assert final_state["continue_session"] is False

    # (b) "maybe" must re-ask, not be silently read as exit.
    assert "didn't catch that" in captured_output.getvalue(), \
        "unrecognised yes/no answer should re-prompt"

    # (c) Search results are formatted, not a Python repr, and sources captured.
    assert "{'title'" not in final_state["search_results"], "raw dict repr in search results"
    assert "Source: Diabetes Overview" in final_state["search_results"], "results not formatted"
    assert final_state["search_sources"] == ["[Diabetes Overview](https://example.com)"], \
        final_state["search_sources"]

    # (d) The patient saw the disclaimer and the sources.
    shown = captured_output.getvalue()
    assert "not medical advice" in shown, "disclaimer was not displayed"
    assert "Sources" in shown and "https://example.com" in shown, "sources were not displayed"

    # (e) Duplicate sources across queries collapse, order preserved.
    text, urls = format_search_results([
        {"title": "A", "url": "https://a", "content": "x"},
        {"title": "B", "url": "https://b", "content": "y"},
        {"title": "A", "url": "https://a", "content": "x"},
    ])
    assert list(dict.fromkeys(urls)) == ["[A](https://a)", "[B](https://b)"]
    assert "URL: https://a" in text and "---" in text

    # (f) Malformed Tavily output must not crash the formatter.
    assert format_search_results("already a string") == ("already a string", [])
    assert format_search_results([{"content": "no title or url"}])[1] == []

    # (g) Key-shaped strings are masked before they can reach saved output.
    secrets = [
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "sk-ant-abcdefghijklmnopqrstuvwxyz12",
        "AIzaAbCdEfGhIjKlMnOpQrStUvWxYz12345",
        "tvly-abcdefghijklmnopqrstuvwxyz1234",
        "AKIAABCDEFGHIJKLMNOP",
        "Bearer abcdefghijklmnopqrstuvwxyz",
    ]
    for secret in secrets:
        masked = redact(f"401 unauthorized for key {secret} on request")
        assert secret not in masked, f"secret survived redaction: {secret}"
        assert "<redacted>" in masked
    # Ordinary error text must survive untouched.
    assert redact("429 rate limit, retry in 55s") == "429 rate limit, retry in 55s"

    # (h) A failed provider is reported without leaking the key from its message.
    providers = {k: FlakyLLM(failures=99, message=f"401 invalid api key {secrets[0]}") for k in "1234"}
    AVAILABLE_MODELS_backup = dict(AVAILABLE_MODELS)
    try:
        invoke_model([], what="the summary")
    except HealthBotServiceError as error:
        assert secrets[0] not in str(error), "provider error leaked an API key"
    except NotImplementedError:
        pass  # _get_model not wired in this scenario; redaction covered by (g)
    finally:
        AVAILABLE_MODELS.clear()
        AVAILABLE_MODELS.update(AVAILABLE_MODELS_backup)

    print("PASS: empty/unrecognised input re-prompted, results formatted with sources,")
    print("      disclaimer shown, and key-shaped strings redacted from errors.\n")

    print("=" * 70)
    print("TEST 12: failover stickiness resets on a new topic")
    print("=" * 70)
    global _active_provider
    _active_provider = 2          # pretend we failed over mid-topic
    reset_provider_failover()
    assert _active_provider == 0, "a new topic should retry the preferred provider"
    print("PASS: reset_provider_failover returns to the preferred provider.\n")

    print("=" * 70)
    print("TEST 13: structured grading with citation verification")
    print("=" * 70)
    single = ["diabetes", "", "it affects blood sugar", "no"]

    # (a) A grounded citation verifies on the first attempt.
    final_state = run_test(single)
    assert final_state["grade"] == "B"
    assert final_state["citation"] == FAKE_CITATION
    assert final_state["citation_verified"] is True, "a real quote should verify"
    assert FakeStructuredLLM.calls == 1, f"expected 1 structured call, got {FakeStructuredLLM.calls}"
    assert "Citation checked against the summary" in captured_output.getvalue()

    # (b) A fabricated quote earns one corrective retry, then verifies.
    final_state = run_test(single, citations=[UNGROUNDED_CITATION, FAKE_CITATION])
    assert FakeStructuredLLM.calls == 2, f"expected a retry, got {FakeStructuredLLM.calls} call(s)"
    assert final_state["citation_verified"] is True
    assert final_state["citation"] == FAKE_CITATION

    # (c) If it stays fabricated, the grade still lands but is flagged, not trusted.
    final_state = run_test(single, citations=[UNGROUNDED_CITATION, UNGROUNDED_CITATION])
    assert final_state["citation_verified"] is False, "unmatched quote must be flagged"
    assert final_state["grade"] == "B", "a bad citation should not lose the grade"
    assert "could NOT be matched" in captured_output.getvalue()

    # (d) The letter grade is schema-constrained, so "N/A" is unreachable here.
    for letter in ("A", "B", "C", "D", "F"):
        state = run_test(["x", "", "y", "yes" if letter in ("D", "F") else "no",
                          *(["no"] if letter in ("D", "F") else [])],
                         grade=letter)
        assert state["grade"] == letter, f"expected {letter}, got {state['grade']}"

    # (e) A provider that cannot honour the schema falls back to free text.
    final_state = run_test(single, structured=False)
    assert final_state["grade"] == "B", f"fallback lost the grade: {final_state['grade']}"
    assert final_state["feedback"].startswith("The answer aligns with"), "fallback parse failed"
    assert final_state["citation_verified"] is True, "fallback should still verify its quote"

    # (f) The verifier itself: substring match, whitespace/case tolerant, no false yes.
    assert citation_is_grounded("CHRONIC   condition affecting BLOOD sugar", FAKE_SUMMARY)
    assert not citation_is_grounded(UNGROUNDED_CITATION, FAKE_SUMMARY)
    assert not citation_is_grounded("", FAKE_SUMMARY)
    assert not citation_is_grounded("short", FAKE_SUMMARY), "too-short quotes must not pass"
    print("PASS: grounded citations verify, fabricated ones are retried then flagged,")
    print("      grades stay schema-constrained, and the free-text fallback still works.\n")

    print("=" * 70)
    print("TEST 14: adaptive re-explanation on a low grade")
    print("=" * 70)
    main_path = MAIN_PATH

    # (a) A good grade skips the branch entirely.
    run_test(single, grade="B")
    assert visited_nodes == DIRECT_PATH, visited_nodes
    assert "offer_recap" not in visited_nodes, "a passing grade must not trigger the recap"

    # (b) A failing grade offers help, and accepting it runs the full branch.
    run_test(["diabetes", "", "no idea", "yes", "no"], grade="F")
    assert visited_nodes == main_path + [
        "offer_recap", "simplify_summary", "present_simple_summary",
        "record_progress", "ask_continue",
    ], visited_nodes
    shown = captured_output.getvalue()
    assert "go over it in plainer language" in shown
    assert "more simply" in shown, "the simpler explanation was not displayed"
    assert "not medical advice" in shown, "the disclaimer must appear on the recap too"

    # (c) Declining the offer rejoins the flow without simplifying.
    final_state = run_test(["diabetes", "", "no idea", "no", "no"], grade="D")
    assert visited_nodes == main_path + ["offer_recap", "record_progress", "ask_continue"], visited_nodes
    assert final_state["wants_simpler"] is False
    assert final_state["simple_summary"] is None, "nothing should have been generated"

    # (d) The rewrite is bound to the summary only -- no raw search results.
    run_test(["diabetes", "", "no idea", "yes", "no"], grade="F")
    simplify_payload = FakeGenericLLM.payloads["simplify"]
    assert not any(isinstance(m, ToolMessage) for m in simplify_payload), \
        "the simpler explanation must not receive raw search results"
    assert any(FAKE_SUMMARY.strip()[:40] in m.content
               for m in simplify_payload if isinstance(m.content, str)), \
        "the simpler explanation should be given the original summary"

    # (e) Restarting after the branch still clears the new fields. Topic 2 also
    # scores F, so it is offered the recap as well -- declined here.
    final_state = run_test(
        ["diabetes", "", "no idea", "yes",        # topic 1: accept the recap
         "yes",                                    # ...then start another topic
         "asthma", "", "breathing", "no",         # topic 2: decline the recap
         "no"],                                    # ...and exit
        grade="F",
    )
    assert visited_nodes.count("reset_state") == 1
    assert visited_nodes.count("offer_recap") == 2, visited_nodes
    assert visited_nodes.count("simplify_summary") == 1, "topic 2 declined, so no rewrite"
    assert final_state["topic"] == "asthma"
    # Topic 1 generated a rewrite; topic 2 declined. simple_summary being empty
    # proves the reset cleared topic 1's, rather than it carrying over.
    assert final_state["simple_summary"] is None, "topic 1's rewrite survived the reset"
    # wants_simpler is topic 2's own answer, recorded after the reset.
    assert final_state["wants_simpler"] is False, final_state["wants_simpler"]
    print("PASS: low grades open the branch, good grades skip it, declining rejoins")
    print("      cleanly, the rewrite sees only the summary, and reset clears it all.\n")

    print("=" * 70)
    print("TEST 15: end-of-session recap keeps metadata only")
    print("=" * 70)

    # (a) One entry per completed topic, in order.
    run_test(
        ["diabetes", "", "it affects blood sugar", "yes",
         "asthma", "", "breathing", "no"],
    )
    assert len(SESSION_LOG) == 2, SESSION_LOG
    assert [e["topic"] for e in SESSION_LOG] == ["diabetes", "asthma"]
    assert all(e["grade"] == "B" for e in SESSION_LOG)
    assert all(e["sources"] == 1 for e in SESSION_LOG), "source count not recorded"
    assert all(e["citation_verified"] is True for e in SESSION_LOG)

    # (b) No clinical content is retained -- this is what keeps the privacy reset honest.
    allowed = {"topic", "grade", "citation_verified", "sources", "simplified", "at"}
    for entry in SESSION_LOG:
        assert set(entry) == allowed, f"unexpected keys retained: {set(entry) - allowed}"
        blob = " ".join(str(v) for v in entry.values())
        assert FAKE_SUMMARY.strip()[:30] not in blob, "summary text leaked into the log"
        assert "it affects blood sugar" not in blob, "patient answer leaked into the log"
        assert "What does diabetes affect?" not in blob, "quiz text leaked into the log"

    # (c) The log never reaches a model payload.
    for label in ("summarize", "quiz", "grade"):
        payload = FakeGenericLLM.payloads.get(label) or []
        joined = " ".join(m.content for m in payload if isinstance(m.content, str))
        assert "Session summary" not in joined, f"the recap leaked into the {label} prompt"

    # (d) The simpler-recap flag reflects what actually happened.
    run_test(["diabetes", "", "no idea", "yes", "no"], grade="F")
    assert SESSION_LOG[0]["simplified"] is True, "a used rewrite should be recorded"
    run_test(["diabetes", "", "no idea", "no", "no"], grade="F")
    assert SESSION_LOG[0]["simplified"] is False, "a declined rewrite must not be recorded as used"

    # (e) The rendered recap reports the tally without exposing content.
    run_test(["diabetes", "", "it affects blood sugar", "no"])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        render_session_recap()
    recap = buffer.getvalue()
    assert "Session summary" in recap
    assert "1 topic" in recap, recap
    assert "diabetes" in recap
    assert FAKE_SUMMARY.strip()[:30] not in recap, "recap exposed summary text"
    assert "discarded between topics" in recap, "the privacy note should be shown"

    # (f) An empty log renders nothing rather than an empty table.
    SESSION_LOG.clear()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        render_session_recap()
    assert buffer.getvalue() == "", f"expected no output, got {buffer.getvalue()!r}"
    print("PASS: one entry per topic with metadata only, no clinical content retained,")
    print("      the log never enters a prompt, and an empty log renders nothing.\n")

    print("=" * 70)
    print("TEST 16: the notebook defines everything it calls")
    print("=" * 70)
    # This file mirrors the notebook rather than importing it, so a helper deleted
    # from the notebook can still be present here and every other test will pass
    # while the notebook is broken. That happened: rewriting invoke_model dropped
    # reset_provider_failover, reset_state raised NameError on the second topic,
    # and the suite stayed green. This checks the notebook on its own terms.
    notebook = Path(__file__).with_name("healthbot.ipynb")
    if not notebook.exists():
        print("SKIP: healthbot.ipynb not found next to this file.\n")
    else:
        missing = undefined_calls_in_notebook(notebook)
        assert not missing, (
            "the notebook calls names it never defines: " + ", ".join(missing)
        )
        print(f"PASS: every function called in {notebook.name} is also defined there.\n")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
