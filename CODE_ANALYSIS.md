# HealthBot-Accen — Code & Project Analysis

Analysis date: 2026-09-01
Scope: full workspace at `c:\Project\HealthBot-Accen`

## Status

Fixes were applied on 2026-09-01 for items 1, 3, 4, 5, 6 and the error-handling half
of 7. Findings below are tagged **RESOLVED** or **OPEN** accordingly. See
[section 8](#8-fixes-applied) for what changed.

**All rubric criteria are now met**, with a verified two-topic run saved in the
notebook covering both the restart and exit branches — see
[3.2](#32-no-successful-end-to-end-run-is-saved--resolved). Provider choice is *not*
a gap: the instructor confirmed any LLM key is acceptable, see
[3.3](#33-provider-choice--not-a-gap).

Remaining items are quality improvements only, none of them rubric-blocking. The
most visible is that library `UserWarning`s are interleaved into the patient-facing
output — 8 stderr blocks in the saved run.

Four follow-up fixes landed after the initial pass:

- [Section 9](#9-one-provider-failing-ended-the-session--resolved) — four provider keys
  were configured but only one was ever used, so Google's daily cap killed the session.
  Now fails over automatically.
- [Section 10](#10-responsecontent-is-not-always-a-string--resolved) — `response.content`
  is not always a string. This was the cause of raw `[{'type': 'text', ...}]` dicts
  appearing in the patient-facing summary.
- [Section 11](#11-grade-parsing-was-too-strict--resolved) — the grade parser silently
  reported `N/A` whenever the model decorated its labels.
- [Section 12](#12-output-formatting--resolved) — the plain-text output layer was
  replaced with rendered Markdown.

Verification: `test_graph_logic.py` was extended to cover the new behaviour and
passes all ten scenarios. The notebook validates under `nbformat`, every code
cell compiles, and the helper cells execute standalone.

---

## 1. What this project is

A LangGraph capstone ("HealthBot" for the fictional MediTech Solutions) that delivers
interactive patient education:

1. Ask the patient for a health topic.
2. Search Tavily for up-to-date medical information.
3. Summarize the results in patient-friendly language.
4. Present the summary and let the patient read it.
5. Generate and present one comprehension question drawn from the summary.
6. Collect the answer, grade it (letter grade + citation-backed justification).
7. Loop to a new topic with a state reset, or exit.

Requirements and the grading rubric live in `capstone_project_details.md`.

### File inventory

| File | Role |
| --- | --- |
| `healthbot.ipynb` | The entire implementation: state, nodes, prompts, graph, entry point |
| `capstone_project_details.md` | Requirements + rubric |
| `test_graph_logic.py` | Deterministic graph test using fake LLM/Tavily and scripted `input()` |
| `test_api_keys.py` | Connectivity checks: OpenAI, Google, Tavily |
| `test_api_keys_rest.py` | Connectivity checks: Anthropic, Cohere, HF, LangSmith, Firecrawl, Serper, Bedrock, Langfuse |
| `pyproject.toml` / `uv.lock` | Authoritative dependencies |
| `requirements.txt` | Subset of the above, unversioned |
| `main.py` | Untouched `uv init` stub — dead code |

Verified: `test_graph_logic.py` passes all three of its scenarios
(`.venv\Scripts\python.exe test_graph_logic.py` → `ALL TESTS PASSED`).

---

## 2. Architecture

Ten nodes, a linear chain with one cycle back to the start:

```
START → ask_topic → search_topic → summarize_results → present_summary
      → generate_quiz → ask_quiz_question → grade_answer → present_grade
      → ask_continue ─┬─(yes)→ reset_state → ask_topic
                      └─(no)─→ END
```

### State

A single `HealthBotState` TypedDict threaded through every node:

```python
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
```

`messages` accumulates via the `add_messages` reducer; the rest are last-write-wins.
Nodes return partial dicts rather than whole states, which is the correct LangGraph pattern.

### Node responsibilities

| Node | Responsibility |
| --- | --- |
| `ask_topic` | `input()` for the topic; writes `topic` + a `HumanMessage` |
| `search_topic` | Invokes the tool-bound LLM, executes each Tavily call, wraps results in `ToolMessage`; falls back to a direct search if no tool call is returned |
| `summarize_results` | 3-4 paragraph patient-friendly summary from the search results only |
| `present_summary` | Prints the summary, then gates on `input()` for "ready" |
| `generate_quiz` | One open-ended question answerable from the summary alone |
| `ask_quiz_question` | Prints the question, collects the answer |
| `grade_answer` | Letter grade + justification with a quoted citation; parses both out of the response |
| `present_grade` | Prints grade and feedback |
| `ask_continue` / `route_continue` | yes/no → sets `continue_session`; router picks `reset_state` or `END` |
| `reset_state` | Clears message history and all topic fields |

Single-responsibility separation is clean and matches the rubric's stated preference.

### Three choices that go beyond the spec

**Multi-provider model selector.** Rather than hardcoding OpenAI, the notebook builds
`AVAILABLE_MODELS` from whichever API keys are actually present, then dispatches through
`build_chat_model()` to `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatAnthropic`, or
`ChatOpenAI` pointed at OpenRouter's OpenAI-compatible base URL. Everything downstream
sees a `BaseChatModel`, so the nodes stay provider-agnostic.

**Forced tool call plus a deterministic fallback.**

```python
llm_with_tavily = llm.bind_tools([tavily_tool], tool_choice="any")
```

`tool_choice="any"` requires the model to search instead of answering from its own
knowledge. If it returns no tool call regardless, `search_topic` calls
`tavily_tool.invoke(topic)` directly with `tool_call_id="fallback"`. A search always
happens — this is what test 3 in `test_graph_logic.py` covers.

**Privacy reset via `RemoveMessage`.**

```python
"messages": [RemoveMessage(id=m.id) for m in state["messages"]]
```

Combined with nulling every topic field, this prevents one patient's summary, quiz, or
grade from leaking into the next topic. The rubric asks for this and it is done properly.

### Prompt quality

The system prompts are genuinely well-constrained, which matters for the rubric:

- Summary prompt forbids outside knowledge, bans unexplained jargon, and rules out
  personalized medical advice.
- Quiz prompt requires the question be answerable from the summary alone, open-ended,
  and introduces no new facts.
- Grading prompt fixes the output shape (`Grade: <letter>` / `Justification: ...`) and
  demands at least one direct quoted citation from the summary.

---

## 3. Rubric gaps

### 3.1 The model never saw the message history — RESOLVED

The rubric requires: *"The model has access to previous messages (tool calls, summary,
quiz question, etc.)"*

It did not. Every node constructed a fresh two-message payload:

```python
response = llm.invoke([SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                       HumanMessage(content=prompt)])
```

`state["messages"]` accumulated faithfully but was never passed to the LLM. The workflow
still functioned because each node hand-threaded the data it needed through explicit
state fields, but the criterion as written was unmet.

Fixed by a `build_model_payload()` helper that assembles system prompt + accumulated
history + the current request, now used by all four LLM-calling nodes. See
[8.1](#81-conversation-history-is-now-replayed-fix-1).

### 3.2 No successful end-to-end run is saved — RESOLVED

The final cell's stored output is a failure, not a success:

```
GoogleRateLimitError: RESOURCE_EXHAUSTED — free-tier quota, limit: 20,
model: gemini-3.5-flash-lite      (raised inside search_topic)
```

A clean two-topic run is now saved and verified. Cells executed sequentially 1 through 21
from a fresh kernel with **zero error outputs**, and the run cell captures both branches of
the conditional edge:

| Evidence in the saved output | Status |
| --- | --- |
| Two distinct topics (`headache`, then `Cold`) | present |
| `reset_state` fired between them | present |
| Model issued the Tavily tool call, both topics | present |
| Fallback path never used | confirmed |
| Summary length | 4 paragraphs each (315 / 321 words) — within the required 3-4 |
| Quiz question, both topics | present |
| Letter grade, both topics | `F` and `B` |
| Quoted citation from the summary in the feedback | present |
| Restart branch (`yes` → `reset_state` → `ask_topic`) | present |
| Exit branch (`no` → `END`) | present |
| Graph diagram | PNG rendered in cell 31 |

Two details worth noting as positive evidence. The model chose its own search queries and
steered them toward reputable sources unprompted by name —
`headache types symptoms treatments NIH CDC MedlinePlus` and
`common cold CDC overview symptoms treatment prevention` — which shows
`SEARCH_SYSTEM_PROMPT` is doing its job. And the `F` grade on topic 1 demonstrates the
grader handling a wrong answer: encouraging in tone, and still citing the summary
(*"called an aura—a group of nervous system symptoms that frequently affect vision"*).

Note that outputs on every cell modified during a fix pass are cleared, since they no
longer match the code — so a fresh run is needed after any future change.

### 3.3 Provider choice — NOT A GAP

An earlier version of this document flagged the rubric wording *"OpenAI successfully calls
Tavily for search"* as a gap, because the saved run used Gemini. The instructor has since
confirmed that **any** LLM provider key is acceptable, since not all keys work for
everyone. Running on Gemini, Anthropic, or OpenRouter is fine, and the automatic failover
added in section 9 means the workflow is not tied to any single provider.

### 3.5 Search provenance is now visible — RESOLVED

`search_topic` has two paths: a model-issued tool call, and a deterministic fallback that
calls Tavily directly. Nothing in the output distinguished them, so a reader could not tell
whether the *model* invoked the tool — which is exactly what the first rubric criterion
asks for.

Two additions fix this. `HealthBotState` gained `search_queries` and
`search_used_tool_call`, so the provenance is recorded data rather than a transient print.
And `describe_search()` renders one line naming the active provider, the tool it called,
and the queries it chose:

> **Google Gemini - gemini-3.5-flash-lite** issued a tool call to
> `tavily_search_results_json` with 1 query: `common cold symptoms treatment`

When the fallback runs instead, it says so plainly rather than implying the model made the
call. The run cell also restates the provenance from `final_state` at the end, so the
evidence survives in one place. Both branches are asserted in tests 1 and 5.

### 3.4 Minor deviation: `config.env` vs `.env` — OPEN (by design)

The spec names the file `config.env`; the project uses `.env`. The notebook markdown
calls this out explicitly, so it is likely acceptable, but it is a literal deviation.

---

## 4. Bugs and rough edges

**RESOLVED — `grade_answer` truncated multi-line justifications.** The parser kept only
the single line beginning with `justification:`, so a wrapped 2-4 sentence justification
lost everything after the first line. Now splits once on the label via regex and keeps
the remainder.

**RESOLVED — model selector labels were wrong.** Option 2 was labeled `gemini-3.1-flash`
but instantiated `gemini-3.5-flash-lite`. Option 3 hardcoded its label rather than
interpolating. Model ids are now declared once as constants and interpolated into the
labels, so they cannot drift again.

**RESOLVED — `RemoveMessage` was imported after the node that uses it.** Worked on a
sequential run, broke on out-of-order cell execution. Moved into the imports cell and the
redundant late-import cell removed.

**RESOLVED — no error handling around LLM or Tavily calls.** A single 429 aborted the
whole graph mid-session, exactly what the saved traceback demonstrated. Every model and
tool call now goes through `invoke_with_retry`.

**RESOLVED — `requirements.txt` was out of sync with `pyproject.toml`.** It was an
unversioned subset of 10 packages, so the capstone's own setup instruction
(`uv add -r requirements.txt`) would not have installed `jupyter`, `ipykernel`,
`anthropic`, `requests`, `boto3`, `cohere`, `huggingface-hub`, or `langfuse` — the
notebook kernel and both key-test scripts would have failed on a clean machine. It now
mirrors `pyproject.toml` exactly: 18 packages, matching version specifiers, verified
programmatically with no missing entries, no extras, and no version mismatches. Every
lower bound was confirmed against the versions actually installed in `.venv`. Grouped by
purpose with comments so it is obvious which packages the workflow needs versus which
exist only for the connectivity scripts.

**OPEN — `langchain-tavily` migration pending.** `TavilySearchResults` from
`langchain_community` emits a deprecation warning at runtime pointing to
`langchain_tavily.TavilySearch`. The spec asks for the community tool, so this is a
forward-looking note rather than a defect.

**OPEN — `test_graph_logic.py` duplicates rather than imports.** It copy-pastes the
notebook's state, nodes, and graph with fakes substituted. It was updated in step with the
fixes and now also asserts the history-replay behaviour, but the structural drift risk
remains: a notebook change still will not fail the test automatically.

**OPEN — `recursion_limit=100` caps the session at ~10 topics** (10 nodes per cycle). Fine
for a demo, but it is a silent ceiling.

**OPEN — docstring drift in `test_api_keys.py`.** The docstring says OpenAI is checked by
listing models ("lightweight, no completion cost"); the implementation performs a real
chat completion.

---

## 5. Dead code and unused surface

- `main.py` — the `uv init` stub. Either delete it or turn it into a CLI entry that runs
  the graph.
- `test_graph_logic.py`: module-level `USE_FALLBACK_LLM` and the initial
  `llm_with_tavily` assignment are unreachable — `run_test()` reassigns via `global` on
  every call.
- `langfuse`, `boto3`, `cohere`, `huggingface-hub` are in `pyproject.toml` but the
  HealthBot flow uses none of them. They exist only for the key-connectivity scripts.
  No tracing (Langfuse or LangSmith) is wired up despite the dependencies and keys.
- No checkpointer is configured, so there is no persistence or thread resumption. Not
  required by the spec.
- No test covers `OPENROUTER_API_KEY` even though OpenRouter is a selectable provider.

---

## 6. Secrets handling

Checked rather than assumed:

- `.env` is listed in `.gitignore`, and `git ls-files` confirms it is **not** tracked.
  Handled correctly.
- It holds a wide spread of live-looking keys across many providers (OpenAI, Anthropic,
  Google, Cohere, HuggingFace, LangSmith, Langfuse, Bedrock, OpenRouter, Firecrawl,
  Serper, Azure). Most are unused by the HealthBot flow itself.
- The three `LANGFUSE_*` variables are each defined twice; the later block wins. Worth
  cleaning up to avoid confusion about which value is live.

---

## 7. Recommended fixes, in order of payoff

| # | Fix | Status |
| --- | --- | --- |
| 1 | Thread `state["messages"]` into the LLM calls — closes the only substantive rubric gap | Done |
| 2 | Re-run end-to-end and commit the saved outputs | Done for one topic; needs a two-topic run to show the restart branch |
| 3 | Fix the `grade_answer` justification parser so multi-line feedback survives | Done |
| 4 | Sync `requirements.txt` with `pyproject.toml` and add versions | Done |
| 5 | Move the `RemoveMessage` import into the imports cell | Done |
| 6 | Correct the two model selector labels | Done |
| 7 | Retry/error handling around LLM and Tavily calls | Done |
| 7b | Delete `main.py` or promote it to a real CLI entry point | Deferred — needs a decision |

On 7b: building a CLI would mean a second copy of the graph in Python, reintroducing the
same drift problem `test_graph_logic.py` already has. Deleting the stub is the cheaper
option. Left untouched pending a call on which way to go.

---

## 8. Fixes applied

All changes are in `healthbot.ipynb` unless noted. Outputs were cleared on every modified
cell, since the stored outputs no longer matched the code.

### 8.1 Conversation history is now replayed (fix 1)

A new cell after the state schema adds three helpers:

- `_is_plain_text(message)` — identifies ordinary text turns, i.e. not a tool call and not
  a tool result.
- `_merge_consecutive(messages)` — collapses consecutive same-role plain-text messages.
  Needed because replaying history can produce two assistant turns back to back (the
  summary followed by the quiz question), which stricter providers reject. Tool calls and
  tool results are never merged, so their pairing stays intact.
- `build_model_payload(system_prompt, state, user_prompt, include_search_results=True)` —
  assembles system prompt + accumulated history + this turn's request.

All four LLM-calling nodes now use it. The `include_search_results` flag resolves a real
tension between two rubric criteria — "the model has access to previous messages" versus
"grade using only the summary as a data source":

| Node | `include_search_results` | Rationale |
| --- | --- | --- |
| `search_topic` | `True` | History is empty or near-empty at this point |
| `summarize_results` | `True` | The search results are its source material |
| `generate_quiz` | `False` | The summary must be the only source for the question |
| `grade_answer` | `False` | The summary must be the only source of truth for the grade |

When the flag is `False`, both the `ToolMessage` and the `AIMessage` carrying the
`tool_calls` are dropped together — a tool call without its matching result is an invalid
sequence for most providers.

### 8.2 Justification parsing (fix 3)

The grade is still read by line scanning, since it is always one line. The justification
is now located with `re.search(r"justification\s*:", ..., re.IGNORECASE)` and everything
after the label is kept, so multi-sentence feedback that wraps across lines survives
intact.

### 8.3 Import ordering (fix 5)

`RemoveMessage` moved into the section 2 imports cell alongside `add_messages`. `re` and
`time` added there too. The standalone late-import cell and its markdown lead-in were
deleted.

### 8.4 Model labels (fix 6)

`OPENAI_MODEL`, `GOOGLE_MODEL`, `ANTHROPIC_MODEL`, and `OPENROUTER_MODEL` are declared
once at the top of the selector cell and interpolated into the menu labels with f-strings.

### 8.5 Resilient invocation (fix 7)

- `HealthBotServiceError(RuntimeError)` — raised when an upstream call cannot be recovered.
- `invoke_with_retry(runnable, payload, what)` — wraps every model and tool call. Retries
  up to 3 attempts with linear backoff, but only when the error text matches a transient
  marker (rate limit, 429, `resource_exhausted`, quota, overloaded, timeout, 502/503/504).
  Non-transient failures such as a bad API key fail immediately instead of burning
  retries. On exhaustion it raises `HealthBotServiceError` with a readable message rather
  than surfacing a raw provider traceback.
- The section 8 run cell catches `HealthBotServiceError` and prints a friendly message
  suggesting a re-run or a different model.

A hard daily quota will not be rescued by retrying — that case now produces a clear
message instead of a traceback.

### 8.6 Test coverage

`test_graph_logic.py` was extended from 3 to 6 scenarios. New coverage:

- **Test 2** — asserts every payload carries replayed history beyond `[system, user]`,
  that `summarize` receives the `ToolMessage` while `quiz`/`grade` receive neither the
  tool result nor an unmatched tool call, that `quiz` sees the summary and `grade` sees the
  quiz question, and that no payload contains consecutive same-role messages.
- **Test 3** — a deliberately three-line justification must survive parsing with the
  `Grade:` line excluded.
- **Test 4** — additionally asserts that after `reset_state`, topic 1's content does not
  appear in topic 2's replayed payloads.
- **Test 6** — `invoke_with_retry` recovers after two transient failures, raises
  `HealthBotServiceError` after exhausting attempts, and fails fast on a non-transient
  error.
- **Test 7** (added with the display work, see section 9) — `render_markdown` falls back
  to cleaned plain text off-notebook, and a real run displays the quiz and grade blocks
  with no raw markup reaching the patient.

Result: all seven pass.

---

## 9. One provider failing ended the session — RESOLVED

Four LLM provider keys are configured (OpenAI, Google, Anthropic, OpenRouter), but the
notebook built exactly one model from the menu choice and used it for everything. When
Google's free tier hit its **daily** cap of 20 requests, the session died — with three
working keys sitting unused in `.env`.

The retry logic added earlier made this worse rather than better in that specific case: a
daily cap does not clear in seconds, so retrying with backoff just delayed the same
failure three times over.

### Fix

`select_model()` now returns the *preferred* provider key rather than a built model, and
prints the fallback order. `invoke_model()` walks `FAILOVER_ORDER` — the preferred
provider first, then every other provider with a key — and classifies errors into three
buckets:

| Error kind | Markers | Behaviour |
| --- | --- | --- |
| Transient | rate limit, 429, overloaded, timeout, 502/503/504 | Retry same provider, up to 3 attempts with linear backoff |
| Provider-level | daily/per-day quota, `resource_exhausted`, bad key, 401/403, model not found, unsupported tool choice | Switch provider immediately, no waiting |
| Anything else | — | Raise at once; trying four providers would only obscure a bug in this notebook |

Once it switches it stays switched for the rest of the session, so later nodes do not keep
probing a provider already known to be exhausted. If every provider fails, a single
`HealthBotServiceError` reports what each one said.

Two supporting changes: tool binding moved out of section 4 into `_get_model()`, since any
provider may end up running the search and each needs its own bound instance; and the
Tavily path uses a separate `invoke_tool()` that retries but never fails over, because
there is only one search backend.

Marker matching normalises away spaces and underscores, so `RESOURCE_EXHAUSTED`,
`resource exhausted`, and `GenerateRequestsPerDayPerProjectPerModel` all match.

### Verified

Against the exact error text from the reported Gemini run: the daily cap is tried once and
switches immediately (no retries burned), the switch survives three subsequent node calls,
a per-minute `429` still retries in place without failing over, a cascade of bad key →
daily cap → missing model lands on the fourth provider, all-four-down produces one error
naming each, and a `TypeError`-shaped error surfaces after a single attempt.

---

## 10. `response.content` is not always a string — RESOLVED

This was the cause of raw dicts appearing in the patient-facing summary:

```
[{'type': 'text', 'text': 'The common cold is an acute...\n\nSymptoms typically
begin...', 'extras': {'signature': 'El4KXAERTTIPGEyN5yBWlBcGHX4GS5HfV6/81rif...
```

Every node read `response.content` and assumed a `str`. Providers that return
reasoning or thinking blocks — Anthropic, and Gemini with thought signatures — return
`content` as a **list of content blocks** instead:

```python
[{"type": "text", "text": "...", "extras": {"signature": "..."}},
 {"type": "thinking", "thinking": "..."}]
```

Interpolating that list into a display string renders its `repr`, so the patient saw
the dict structure, the base64 signature, and literal `\n\n` escapes rather than real
paragraph breaks. Three further consequences:

- `state["summary"]` held a list, not text, so the quiz and grading prompts were fed a
  stringified list as their source material.
- `grade_answer` called `result_text.splitlines()`, which raises `AttributeError` on a
  list — the run would have died at the grading node.
- `_is_plain_text()` tests `isinstance(message.content, str)`, so list-content messages
  were never merged, reintroducing the consecutive-same-role risk that
  `_merge_consecutive` exists to prevent.

### Fix

A `message_text()` helper returns the plain text of a response regardless of shape. It
prefers `.text` (langchain-core 1.x, which concatenates text blocks and skips
reasoning ones), coerces the returned `TextAccessor` to a plain `str`, and falls back to
manual block extraction on older core versions.

`summarize_results`, `generate_quiz`, and `grade_answer` now store the extracted text and
append a normalised `AIMessage(content=text)` to state rather than the raw response. That
keeps provider-specific blocks out of both the display and the replayed history — which
also means signatures are never echoed back to the provider. `search_topic` still stores
its raw response, because that one carries the `tool_calls`.

## 11. Grade parsing was too strict — RESOLVED

Related, and confirmed by testing the notebook's own logic against realistic responses.
The old check only matched a bare label at line start:

```python
if line.strip().lower().startswith("grade:"):
```

Models routinely decorate labels. Three of five realistic shapes lost the grade entirely
and fell through to `"N/A"`, silently — with a perfectly good justification displayed
underneath. The bolded case was worse: `re.search(r"justification\s*:")` matched *inside*
`**Justification:**`, leaving a stray `**` at the front of the feedback, which then
collided with the Markdown display layer and broke the bold rendering.

### Fix

`parse_grade_response()` matches both labels tolerantly — a leading character class
absorbs list, quote, heading and emphasis markers, and a trailing `\**` absorbs a closing
`**`. It strips a wrapping code fence, extracts a letter grade with an optional `+`/`-`
modifier, keeps the raw value rather than discarding it when no letter is found, and falls
back to the text after the grade line when there is no justification label.

One subtlety the tests caught: `^([A-F][+-]?)\b` parsed `A-` as `A`, because `\b` cannot
match between a trailing `-` and end of string, so the regex backtracked and dropped the
modifier. Replaced with a negative lookahead, `^([A-F][+-]?)(?![A-Za-z0-9])`.

Now verified across nine formats — plain, bolded, heading-prefixed, bullet-prefixed,
blockquoted with a modifier, code-fenced, lowercase, missing justification label, and a
trailing parenthetical — plus an ungradeable response that correctly reports `N/A`.

---

## 13. Beyond-the-rubric improvements — RESOLVED

Applied as one batch so the notebook only needed re-running once. Covered by tests 11
and 12.

### Correctness

**Library warnings reached the patient.** `warnings.filterwarnings` covered only
`DeprecationWarning`, so a `UserWarning` about `temperature` being discarded appeared
between every section — 8 stderr blocks in one run. Fixed at the cause: some Gemini models
use fixed sampling defaults, and `ignores_sampling_settings()` reads
langchain-google-genai's own list of them (rather than keeping a copy that would go stale)
so `temperature` is simply not sent where it has no effect. `select_model()` states the
tradeoff once at setup instead. A separate `logging.Filter` drops google-genai's
automatic-function-calling advisory, which is `logging` rather than `warnings` and so was
never catchable by a warnings filter. Both filters match on message text, so unrelated
warnings still surface.

**`GraphRecursionError` escaped the handler.** `recursion_limit=100` capped a session at
roughly 10 topics, and the resulting error subclasses `RecursionError` — not
`HealthBotServiceError` — so it bypassed the run cell's `except` and produced a raw
traceback. The limit is now derived (`NODES_PER_TOPIC * (MAX_TOPICS + 1)`) and both
`GraphRecursionError` and `KeyboardInterrupt` are handled with a readable message.

**Empty and unrecognised input was accepted.** A bare Enter at the topic prompt searched
for `""`. `ask_continue` used `choice.startswith("y")`, so "sure", "ok" and a bare Enter
all silently ended the session. All three prompts now re-ask, and yes/no is matched
against explicit vocabularies.

**Node annotations were wrong.** All ten nodes declared `-> HealthBotState`, but they
return *partial* updates; `HealthBotState` is total, so a type checker rejects
`{"topic": ...}` for missing the other keys. A `HealthBotUpdate` TypedDict with
`total=False` sits directly beneath the state class and is now used by every node.

### Safety

**Prompt injection via search results.** Tavily returns untrusted web text that went
straight into the summarize prompt — a live risk in a health context. The results are now
fenced in an explicit `<search_results>` block, and the system prompt states that anything
inside reading like an instruction is quoted text to disregard, never obey.

**No disclaimer reached the patient.** The prompts forbid personalised advice, but nothing
on screen said so. `present_summary` now shows one with every summary.

**Provider errors were echoed verbatim into committed output.** Notebook outputs are
committed, and provider errors sometimes echo request details. `redact()` masks anything
key-shaped (OpenAI, Anthropic, Google, Tavily, AWS, bearer tokens) before it can reach a
saved transcript. Ordinary error text passes through unchanged.

### Quality

**`str(results)` discarded structure.** The Tavily result list was stringified into a
Python repr — noisy, token-hungry, and it buried the URLs. `format_search_results()`
returns readable titled sections plus a source list, and tolerates malformed output.

**Sources are now shown.** The URLs Tavily returned are displayed under each summary,
deduplicated with order preserved. This lets patients verify the material and makes the
"reputable sources" intent visible.

**Failover stickiness never reset.** Once switched, the session stayed switched forever.
`reset_provider_failover()` is called from `reset_state`, so each new topic gets a fresh
attempt at the preferred provider — any short-lived limit has long since cleared.

**Redundant import.** Cell 31 re-imported `display`, already imported in section 2.

---

## 12. Output formatting — RESOLVED

### The problem

There was no formatting layer. All 26 patient-facing outputs went through bare `print()`,
which writes a plain stdout stream. Three consequences:

**Nothing wrapped.** An LLM returns each paragraph as one long line — newlines appear only
*between* paragraphs, never inside them. `print()` emitted that verbatim, so
`present_summary` framed the text with `"=" * 70` rulers while the body itself was
unbounded. Where lines broke was decided by the render width, not the code, and the rulers
never lined up with the text.

**Markdown was never rendered.** Models routinely emit `**bold**`, emphasis, and lists even
when told to write plain prose, and the grading prompt actively asks for quoted citations.
stdout is not Markdown, so those arrived as literal asterisks.

**Labels ran into their content.** `print(f"Feedback: {state['feedback']}")` put a 2-4
sentence justification on the same logical line as its label, with no hanging indent.

Minor: frame widths were inconsistent — `"=" * 70` in `present_summary` versus
`--- Comprehension Check ---` (27 chars) and `--- Your Results ---` (20).

### The fix

A new cell adds a `render_markdown()` helper, and every patient-facing node uses it:

| Node | Output |
| --- | --- |
| `ask_topic` | `Got it -- let's learn about **{topic}**.` |
| `present_summary` | `### Here's what we found about: {topic}` + summary + `---` |
| `ask_quiz_question` | `### Comprehension check` + question |
| `present_grade` | `### Your results` + `**Grade:**` + `**Feedback:**` |
| `reset_state` | `---` + fresh-session notice |
| section 8 | closing thanks |

The fixed-width rulers are gone; wrapping is the notebook's job now.

Two details worth noting:

`render_markdown()` degrades gracefully. `_in_notebook()` checks for a
`ZMQInteractiveShell`, and when there isn't one — plain Python, or the logic tests —
`_to_plain_text()` strips the heading and bold markers and turns `---` into a ruler line
before printing. Without this the tests and any non-notebook use would show raw markup.

It flushes stdout first. `display()` and `input()` travel over different kernel channels,
so flushing keeps earlier plain-text output from landing after a prompt it was meant to
precede.

The error path in section 8 deliberately stays on `print()`: if the display layer itself
misbehaves, the error still needs to surface.

### Trade-off accepted

The project brief says *"For displaying output to the patient, you can use the print
function."* Rendering Markdown is a deliberate deviation from that wording, chosen for
legibility. Worth knowing if a grader reads the instruction strictly. The plain-text
alternative would have been `textwrap.fill()` on each paragraph, which keeps `print()` but
cannot render the model's emphasis or citation quoting.
