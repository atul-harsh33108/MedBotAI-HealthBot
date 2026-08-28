"""
Quick connectivity test for the API keys required by the HealthBot project.

Tests, in order:
1. OpenAI API key  -> lists models (lightweight, no completion cost)
2. Google API key  -> lists Gemini models
3. Tavily API key  -> runs a trivial search query

Run with:
    .venv\\Scripts\\python.exe test_api_keys.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv(".env")

results = {}


def test_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not found in .env"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            max_tokens=5,
        )
        content = resp.choices[0].message.content
        return True, f"chat completion succeeded, model replied: {content!r}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_google():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False, "GOOGLE_API_KEY not found in .env"
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        if not models:
            return False, "No models returned (key may be invalid or has no access)"
        sample = models[0].name
        return True, f"listed {len(models)} models, e.g. {sample!r}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_tavily():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return False, "TAVILY_API_KEY not found in .env"
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        resp = client.search(query="What is type 2 diabetes?", max_results=1)
        n = len(resp.get("results", []))
        return True, f"search succeeded, {n} result(s) returned"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    tests = [
        ("OpenAI", test_openai),
        ("Google", test_google),
        ("Tavily", test_tavily),
    ]

    print("=" * 60)
    print("HealthBot API Key Connectivity Test")
    print("=" * 60)

    all_ok = True
    for name, fn in tests:
        print(f"\n[{name}] Testing...")
        ok, message = fn()
        results[name] = ok
        status = "PASS" if ok else "FAIL"
        print(f"[{name}] {status}: {message}")
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
