"""Audit the saved healthbot.ipynb against the project rubric and feature list.

Run this after an interactive session, before submitting:

    .venv\\Scripts\\python.exe verify_submission.py

It reads only the *saved* notebook, so it checks exactly what a reviewer would see.
Nothing here calls an API, so it costs no quota and can be run as often as you like.

Exit code 0 = everything a reviewer needs is present. 1 = something is missing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).with_name("healthbot.ipynb")

# Key-shaped strings that must never appear in committed output.
SECRET_PATTERNS = {
    "OpenAI": r"sk-[A-Za-z0-9_\-]{20,}",
    "Anthropic": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "Google": r"AIza[A-Za-z0-9_\-]{30,}",
    "Tavily": r"tvly-[A-Za-z0-9_\-]{20,}",
    "AWS": r"AKIA[A-Z0-9]{16}",
}

REPUTABLE = (
    "nih.gov", "ncbi.nlm", "cdc.gov", "who.int", "mayoclinic.org",
    "medlineplus.gov", "nhs.uk", "clevelandclinic.org", "hopkinsmedicine.org",
)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        self.rows.append((label, bool(ok), detail))

    def section(self, title: str) -> None:
        self.rows.append((title, None, ""))

    def render(self) -> bool:
        width = max(len(label) for label, _, _ in self.rows) + 2
        failures = 0
        for label, ok, detail in self.rows:
            if ok is None:
                print(f"\n{label}")
                print("-" * (width + 12))
                continue
            mark = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"  [{mark}] {label:<{width}} {detail}")
        print()
        if failures:
            print(f"{failures} check(s) failed -- see above.")
        else:
            print("All checks passed. The saved notebook shows everything a reviewer needs.")
        return failures == 0


def markdown_outputs(cell) -> list[str]:
    out = []
    for o in cell.get("outputs", []):
        data = o.get("data") or {}
        if "text/markdown" in data:
            value = data["text/markdown"]
            out.append(value if isinstance(value, str) else "".join(value))
    return out


def stream_text(cell) -> str:
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            text = o.get("text", "")
            parts.append(text if isinstance(text, str) else "".join(text))
    return "\n".join(parts)


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"Cannot find {NOTEBOOK.name}")
        return 1

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    source = "\n".join("".join(c["source"]) for c in code)

    run_cell = next(
        (c for c in code if "healthbot_graph.invoke" in "".join(c["source"])), None
    )
    if run_cell is None:
        print("Could not find the cell that runs the graph.")
        return 1

    blocks = markdown_outputs(run_cell)
    transcript = "\n".join(blocks)
    r = Report()

    # ---------------------------------------------------------------- execution
    r.section("Notebook execution state")

    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    r.check("No exceptions in any cell", not errors,
            errors[0].get("ename", "") if errors else "")

    stderr_cells = [
        i for i, c in enumerate(code)
        for o in c.get("outputs", [])
        if o.get("output_type") == "stream" and o.get("name") == "stderr"
    ]
    r.check("No warnings or stderr noise", not stderr_cells,
            f"cells {sorted(set(stderr_cells))}" if stderr_cells else "")

    counts = [c.get("execution_count") for c in code]
    unrun = [i + 1 for i, n in enumerate(counts) if n is None]
    r.check("Every code cell has been run", not unrun,
            f"unrun: {unrun}" if unrun else f"{len(code)} cells")

    ran = [n for n in counts if n is not None]
    sequential = ran == sorted(ran) and ran == list(range(1, len(ran) + 1))
    r.check("Ran top-to-bottom from a fresh kernel", sequential and not unrun,
            "" if sequential and not unrun else f"counts: {counts}")

    # ------------------------------------------------------------------- secrets
    r.section("Secrets")
    blob = json.dumps(nb)
    leaked = [name for name, pat in SECRET_PATTERNS.items() if re.search(pat, blob)]
    r.check("No API keys anywhere in the notebook", not leaked,
            ", ".join(leaked) if leaked else "")

    # -------------------------------------------------------------------- rubric
    r.section("Rubric evidence in the saved run")

    topics = transcript.count("Got it -- let's learn about")
    r.check("At least two topics covered", topics >= 2, f"{topics} found")

    r.check("Restart branch used (reset_state fired)",
            "Starting a fresh session" in transcript)
    r.check("Exit branch used (session ended)",
            "Thanks for using HealthBot" in transcript)

    tool_calls = transcript.count("issued a tool call")
    r.check("Model issued the Tavily tool call for every topic",
            topics > 0 and tool_calls >= topics, f"{tool_calls}/{topics}")
    r.check("Direct-search fallback not needed",
            "returned no tool call" not in transcript)

    summaries = [b for b in blocks if b.startswith("### Here's what we found")]
    para_counts = []
    for block in summaries:
        prose = block.split("\n", 1)[1].split("_General health education")[0].strip()
        para_counts.append(len([p for p in re.split(r"\n\s*\n", prose) if p.strip()]))
    r.check("Every summary is 3-4 paragraphs",
            bool(para_counts) and all(3 <= n <= 4 for n in para_counts),
            f"paragraphs: {para_counts}")

    grades = re.findall(r"\*\*Grade:\*\* ([A-F][+-]?)", transcript)
    r.check("A letter grade per topic", len(grades) >= topics, f"grades: {grades}")

    verified = transcript.count("Citation checked against the summary")
    r.check("Every citation verified against the summary",
            topics > 0 and verified >= topics, f"{verified}/{topics}")
    r.check("No unverifiable citation shown",
            "could NOT be matched" not in transcript)

    # ------------------------------------------------------------------ features
    r.section("Feature evidence")

    r.check("Sources listed for every topic",
            transcript.count("**Sources**") >= topics,
            f"{transcript.count('**Sources**')}/{topics}")

    urls = re.findall(r"\((https?://[^)]+)\)", transcript)
    good = [u for u in urls if any(d in u for d in REPUTABLE)]
    r.check("Most sources are reputable medical domains",
            bool(urls) and len(good) >= len(urls) / 2,
            f"{len(good)}/{len(urls)}")

    r.check("Medical disclaimer shown per topic",
            transcript.count("not medical advice") >= topics,
            f"{transcript.count('not medical advice')}/{topics}")

    r.check("Adaptive branch offered on a low grade",
            "didn't quite land" in transcript,
            "needs a D or F to appear")
    r.check("Adaptive rewrite actually displayed",
            "more simply" in transcript,
            "accept the offer to capture this")

    r.check("Session recap rendered", "Session summary" in transcript)
    r.check("Recap states what is discarded",
            "discarded between topics" in transcript)
    r.check("Search provenance restated at the end",
            "Search provenance" in transcript)

    r.check("No raw content blocks leaked", "{'type'" not in transcript)
    r.check("No stringified search results leaked", "{'title'" not in transcript)

    # ------------------------------------------------------------------- diagram
    r.section("Supporting output")
    diagram = any(
        "image/png" in (o.get("data") or {})
        for c in code for o in c.get("outputs", [])
    )
    r.check("Graph diagram rendered", diagram)

    selector = "\n".join(stream_text(c) for c in code)
    r.check("Model menu and fallback order shown",
            "Automatic fallbacks" in selector or "no fallback" in selector)

    print()
    print("=" * 72)
    print(f"  Submission audit: {NOTEBOOK.name}")
    print("=" * 72)
    ok = r.render()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
