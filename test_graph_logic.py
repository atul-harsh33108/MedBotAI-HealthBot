"""
Standalone logic test for the HealthBot LangGraph workflow.

This mirrors the graph structure defined in healthbot.ipynb but replaces the
LLM and Tavily tool with deterministic fakes, and replaces input() with a
scripted queue of patient responses. It verifies:

  1. The full linear path executes in the correct order.
  2. State fields are populated correctly by each node.
  3. The hybrid search fallback triggers when the LLM returns no tool call.
  4. Choosing "yes" at the end loops back through reset_state and clears
     topic-specific fields before ask_topic runs again.
  5. Choosing "no" ends the session.

Run with:
    .venv\\Scripts\\python.exe test_graph_logic.py
"""

import builtins
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
    search_results: Optional[str]
    summary: Optional[str]
    quiz_question: Optional[str]
    patient_answer: Optional[str]
    grade: Optional[str]
    feedback: Optional[str]
    continue_session: Optional[bool]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeToolCallLLM:
    """Simulates an LLM bound with forced tool choice that DOES call the tool."""

    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[{"name": "tavily_search_results_json", "args": {"query": "diabetes"}, "id": "call_1"}],
        )


class FakeNoToolCallLLM:
    """Simulates an LLM that fails to call the tool, to exercise the fallback path."""

    def invoke(self, messages):
        return AIMessage(content="I looked into it.", tool_calls=[])


class FakeTavilyTool:
    call_log = []

    def invoke(self, query):
        FakeTavilyTool.call_log.append(query)
        return [{"title": "Diabetes Overview", "url": "https://example.com", "content": "Diabetes is a condition...", "score": 0.9}]


class FakeGenericLLM:
    """Used for summarize / quiz / grade nodes -- returns canned, distinguishable text."""

    call_count = 0

    def invoke(self, messages):
        FakeGenericLLM.call_count += 1
        system_prompt = messages[0].content

        if "patient-friendly summary" in system_prompt:
            return AIMessage(content="FAKE SUMMARY: Diabetes is a chronic condition affecting blood sugar. " * 3)
        if "single comprehension-check question" in system_prompt:
            return AIMessage(content="What does diabetes affect?")
        if "grading a patient's answer" in system_prompt:
            return AIMessage(content='Grade: B\nJustification: The answer aligns with "Diabetes is a chronic condition affecting blood sugar."')
        return AIMessage(content="FAKE GENERIC RESPONSE")


# Toggle to test fallback path
USE_FALLBACK_LLM = False

llm_with_tavily = FakeNoToolCallLLM() if USE_FALLBACK_LLM else FakeToolCallLLM()
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
    return {"topic": topic, "messages": [HumanMessage(content=f"I would like to learn about: {topic}")]}


def search_topic(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("search_topic")
    topic = state["topic"]
    ai_message = llm_with_tavily.invoke([SystemMessage(content=SEARCH_SYSTEM_PROMPT), HumanMessage(content=f"Topic: {topic}")])

    tool_messages = []
    if ai_message.tool_calls:
        for call in ai_message.tool_calls:
            query = call["args"].get("query", topic)
            results = tavily_tool.invoke(query)
            tool_messages.append(ToolMessage(content=str(results), tool_call_id=call["id"]))
    else:
        results = tavily_tool.invoke(topic)
        tool_messages.append(ToolMessage(content=str(results), tool_call_id="fallback"))

    search_results_text = "\n\n".join(tm.content for tm in tool_messages)
    return {"search_results": search_results_text, "messages": [ai_message, *tool_messages]}


def summarize_results(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("summarize_results")
    response = llm.invoke([SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=state["search_results"])])
    return {"summary": response.content, "messages": [response]}


def present_summary(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("present_summary")
    builtins.input("Press Enter when ready...")
    return {}


def generate_quiz(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("generate_quiz")
    response = llm.invoke([SystemMessage(content=QUIZ_SYSTEM_PROMPT), HumanMessage(content=state["summary"])])
    return {"quiz_question": response.content, "messages": [response]}


def ask_quiz_question(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("ask_quiz_question")
    answer = builtins.input("Your answer: ").strip()
    return {"patient_answer": answer, "messages": [HumanMessage(content=f"My answer: {answer}")]}


def grade_answer(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("grade_answer")
    response = llm.invoke([SystemMessage(content=GRADE_SYSTEM_PROMPT), HumanMessage(content=state["patient_answer"])])
    result_text = response.content
    grade, feedback = "", result_text
    for line in result_text.splitlines():
        if line.lower().startswith("grade:"):
            grade = line.split(":", 1)[1].strip()
        elif line.lower().startswith("justification:"):
            feedback = line.split(":", 1)[1].strip()
    return {"grade": grade or "N/A", "feedback": feedback, "messages": [response]}


def present_grade(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("present_grade")
    return {}


def ask_continue(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("ask_continue")
    choice = builtins.input("Another topic? (yes/no): ").strip().lower()
    return {"continue_session": choice.startswith("y")}


def route_continue(state: HealthBotState) -> str:
    return "reset_state" if state.get("continue_session") else END


def reset_state(state: HealthBotState) -> HealthBotState:
    visited_nodes.append("reset_state")
    return {
        "messages": [RemoveMessage(id=m.id) for m in state["messages"]],
        "topic": None,
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


def run_test(scripted_inputs, use_fallback=False):
    global llm_with_tavily
    llm_with_tavily = FakeNoToolCallLLM() if use_fallback else FakeToolCallLLM()

    visited_nodes.clear()
    FakeTavilyTool.call_log.clear()

    input_queue = list(scripted_inputs)

    def fake_input(prompt=""):
        return input_queue.pop(0)

    original_input = builtins.input
    builtins.input = fake_input
    try:
        graph = build_graph()
        initial_state: HealthBotState = {
            "messages": [], "topic": None, "search_results": None, "summary": None,
            "quiz_question": None, "patient_answer": None, "grade": None,
            "feedback": None, "continue_session": None,
        }
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
    print("PASS: single-topic path, tool-call search, grading, and exit all correct.\n")

    print("=" * 70)
    print("TEST 2: Two topics via restart loop, then exit (verifies state reset)")
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
    print("PASS: reset_state correctly re-enters ask_topic and the graph completes cleanly for topic #2.\n")

    print("=" * 70)
    print("TEST 3: Hybrid search fallback (LLM returns no tool call)")
    print("=" * 70)
    final_state = run_test(
        scripted_inputs=["migraine", "", "head pain", "no"],
        use_fallback=True,
    )
    assert FakeTavilyTool.call_log == ["migraine"], f"Expected fallback direct search on topic, got {FakeTavilyTool.call_log}"
    print("PASS: fallback path correctly invokes Tavily directly with the topic when no tool call is returned.\n")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
