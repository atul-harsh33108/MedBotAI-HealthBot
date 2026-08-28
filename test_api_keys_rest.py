"""
Connectivity test for the remaining API keys found in .env (batch 2):
Anthropic, Cohere, HuggingFace, LangSmith, Firecrawl, Serper, AWS Bedrock, Langfuse.

Run with:
    .venv\\Scripts\\python.exe test_api_keys_rest.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(".env")

results = {}


def test_anthropic():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return False, "ANTHROPIC_API_KEY not found in .env"
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=5,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
        )
        text = resp.content[0].text if resp.content else ""
        return True, f"message succeeded, reply: {text!r}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_cohere():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return False, "COHERE_API_KEY not found in .env"
    try:
        import cohere

        client = cohere.ClientV2(api_key=api_key)
        resp = client.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Reply with just: OK"}],
        )
        return True, "chat call succeeded"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_huggingface():
    token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    if not token:
        return False, "HUGGINGFACEHUB_ACCESS_TOKEN not found in .env"
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        info = api.whoami()
        return True, f"whoami succeeded, user: {info.get('name', info)!r}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_langsmith():
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return False, "LANGSMITH_API_KEY not found in .env"
    try:
        resp = requests.get(
            "https://api.smith.langchain.com/info",
            headers={"x-api-key": api_key},
            timeout=15,
        )
        if resp.status_code == 200:
            return True, f"info endpoint reachable, status {resp.status_code}"
        return False, f"status {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_firecrawl():
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return False, "FIRECRAWL_API_KEY not found in .env"
    try:
        resp = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"url": "https://example.com"},
            timeout=30,
        )
        if resp.status_code == 200:
            return True, f"scrape succeeded, status {resp.status_code}"
        return False, f"status {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_serper():
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return False, "SERPER_API_KEY not found in .env"
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": "test"},
            timeout=15,
        )
        if resp.status_code == 200:
            return True, f"search succeeded, status {resp.status_code}"
        return False, f"status {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_bedrock():
    profile = os.getenv("BEDROCK_AWS_PROFILE")
    region = os.getenv("BEDROCK_AWS_REGION")
    access_key = os.getenv("BEDROCK_AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("BEDROCK_AWS_SESSION_TOKEN")
    if not access_key or not secret_key:
        return False, "BEDROCK_AWS_ACCESS_KEY_ID / SECRET not found in .env"
    try:
        import boto3

        client = boto3.client(
            "bedrock",
            region_name=region or "us-east-1",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
        )
        resp = client.list_foundation_models()
        n = len(resp.get("modelSummaries", []))
        return True, f"list_foundation_models succeeded, {n} models"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_langfuse():
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    if not secret_key or not public_key:
        return False, "LANGFUSE_SECRET_KEY / PUBLIC_KEY not found in .env"
    try:
        resp = requests.get(
            f"{base_url}/api/public/health",
            auth=(public_key, secret_key),
            timeout=15,
        )
        if resp.status_code == 200:
            return True, f"health endpoint reachable, status {resp.status_code}"
        return False, f"status {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    tests = [
        ("Anthropic", test_anthropic),
        ("Cohere", test_cohere),
        ("HuggingFace", test_huggingface),
        ("LangSmith", test_langsmith),
        ("Firecrawl", test_firecrawl),
        ("Serper", test_serper),
        ("AWS Bedrock", test_bedrock),
        ("Langfuse", test_langfuse),
    ]

    print("=" * 60)
    print("Remaining API Key Connectivity Test (batch 2)")
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
