#!/usr/bin/env python3
"""Demonstrate gemini-embedding-2-preview failures via OpenRouter.

Run from repo root (needs OPENROUTER_API_KEY in env):

    python scripts/demo_openrouter_embedding_error.py

On sliver:

    docker exec letta-vision python3 /app/scripts/demo_openrouter_embedding_error.py

This script does NOT call letta-vision server code. It hits OpenRouter directly
so you can see whether the error originates upstream of our enrichment pipeline.

What it proves when embedding-2-preview fails but gemini-embedding-2 (GA) succeeds:
  - The failure is specific to google/gemini-embedding-2-preview on OpenRouter.
  - google/gemini-embedding-2 (GA) may work on the same key in the same run.
  - OpenRouter returns HTTP 200 with a JSON body where ``data`` is null and
    ``error.message`` contains Google's upstream text (often misleadingly
    mentioning "monthly spending cap").
  - That is not the same as your AI Studio Spend page showing no project cap;
    requests go OpenRouter → Google provider, not your browser session.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

import httpx

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcp"
    "LDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAA"
    "AAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA//2Q=="
)


@dataclass
class CaseResult:
    name: str
    endpoint: str
    model: str
    http_status: int
    ok: bool
    summary: str
    raw_body: dict[str, Any]


def _headers() -> dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_REFERER", "").strip()
    title = os.environ.get("OPENROUTER_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _summarize_embedding_body(body: dict[str, Any]) -> tuple[bool, str]:
    if body.get("data"):
        dim = len(body["data"][0].get("embedding") or [])
        return True, f"embedding returned (dim={dim}, response model={body.get('model')!r})"
    err = body.get("error") or {}
    msg = (err.get("message") or json.dumps(err))[:500]
    code = err.get("code")
    return False, f"OpenRouter error payload (code={code}): {msg}"


def _summarize_chat_body(body: dict[str, Any]) -> tuple[bool, str]:
    choices = body.get("choices") or []
    if choices and choices[0].get("message", {}).get("content"):
        text = choices[0]["message"]["content"]
        return True, f"chat returned: {text[:80]!r}"
    err = body.get("error") or {}
    if err:
        return False, f"OpenRouter error: {err.get('message', err)[:500]}"
    return False, f"unexpected chat body: {json.dumps(body)[:300]}"


def _post(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    url = f"{OPENROUTER_BASE}{path}"
    with httpx.Client(timeout=60.0) as client:
        res = client.post(url, headers=_headers(), json=payload)
    try:
        body = res.json()
    except json.JSONDecodeError:
        body = {"_non_json_body": res.text[:2000]}
    return res.status_code, body


def _run_cases() -> list[CaseResult]:
    cases: list[tuple[str, str, dict[str, Any], str]] = [
        (
            "chat control (google/gemini-2.5-flash)",
            "/chat/completions",
            {
                "model": "google/gemini-2.5-flash",
                "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
                "max_tokens": 16,
            },
            "chat",
        ),
        (
            "embed preview — text (google/gemini-embedding-2-preview @768)",
            "/embeddings",
            {
                "model": "google/gemini-embedding-2-preview",
                "input": ["letta-vision image caption fallback probe"],
                "encoding_format": "float",
                "dimensions": 768,
            },
            "embed",
        ),
        (
            "embed preview — multimodal image (google/gemini-embedding-2-preview)",
            "/embeddings",
            {
                "model": "google/gemini-embedding-2-preview",
                "input": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{TINY_JPEG_B64}"},
                            }
                        ]
                    }
                ],
                "encoding_format": "float",
                "dimensions": 768,
            },
            "embed",
        ),
        (
            "embed GA — text (google/gemini-embedding-2 @768)",
            "/embeddings",
            {
                "model": "google/gemini-embedding-2",
                "input": ["letta-vision image caption fallback probe"],
                "encoding_format": "float",
                "dimensions": 768,
            },
            "embed",
        ),
        (
            "embed GA — multimodal image (google/gemini-embedding-2)",
            "/embeddings",
            {
                "model": "google/gemini-embedding-2",
                "input": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{TINY_JPEG_B64}"},
                            }
                        ]
                    }
                ],
                "encoding_format": "float",
                "dimensions": 768,
            },
            "embed",
        ),
        (
            "embed alternate — text (google/gemini-embedding-001 @768)",
            "/embeddings",
            {
                "model": "google/gemini-embedding-001",
                "input": ["letta-vision image caption fallback probe"],
                "encoding_format": "float",
                "dimensions": 768,
            },
            "embed",
        ),
    ]

    results: list[CaseResult] = []
    for name, path, payload, kind in cases:
        status, body = _post(path, payload)
        if kind == "chat":
            ok, summary = _summarize_chat_body(body)
        else:
            ok, summary = _summarize_embedding_body(body)
        results.append(
            CaseResult(
                name=name,
                endpoint=path,
                model=str(payload.get("model")),
                http_status=status,
                ok=ok,
                summary=summary,
                raw_body=body,
            )
        )
    return results


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_result(r: CaseResult) -> None:
    status = "PASS" if r.ok else "FAIL"
    print(f"\n[{status}] {r.name}")
    print(f"  POST {OPENROUTER_BASE}{r.endpoint}")
    print(f"  model: {r.model}")
    print(f"  HTTP {r.http_status}")
    print(f"  {r.summary}")
    if not r.ok:
        err = r.raw_body.get("error")
        if err:
            print("  --- OpenRouter error object ---")
            print(textwrap.indent(json.dumps(err, indent=2), "  "))


def _google_upstream_text(results: list[CaseResult]) -> Optional[str]:
    for r in results:
        err = r.raw_body.get("error") or {}
        msg = str(err.get("message") or "")
        if "RESOURCE_EXHAUSTED" in msg or "spending cap" in msg.lower():
            return msg
    return None


def _print_conclusion(results: list[CaseResult]) -> None:
    def _models_ok(*models: str) -> bool:
        matched = [r for r in results if r.model in models]
        return bool(matched) and any(r.ok for r in matched)

    embed2_preview_ok = _models_ok("google/gemini-embedding-2-preview")
    embed2_ga_ok = _models_ok("google/gemini-embedding-2")
    chat_ok = any(r.ok and "chat" in r.endpoint for r in results)
    embed1_ok = any(r.ok and "gemini-embedding-001" in r.model for r in results)
    google_msg = _google_upstream_text(results)

    _print_section("Interpretation")
    print(
        textwrap.dedent(
            """
            This script calls OpenRouter directly (no letta-vision code).

            OpenRouter often returns HTTP 200 for /embeddings even when upstream failed.
            Success: ``data: [{ "embedding": [...] }]``
            Failure: ``data`` absent/null and ``error.message`` set.

            For gemini-embedding-2-preview failures, OpenRouter wraps Google's JSON
            inside error.message (look for RESOURCE_EXHAUSTED / spending cap text).
            That proves the rejection is upstream of letta-vision — OpenRouter relayed
            it from Google's embedding API, not from our enrichment pipeline.

            Compare models on the same API key in one run: if gemini-embedding-001
            succeeds while gemini-embedding-2-preview fails, the block is model-specific
            on the OpenRouter → Google path, not a universal "your AI Studio has no cap"
            misconfiguration in isolation.
            """
        ).strip()
    )
    print()
    print(f"  google/gemini-2.5-flash chat OK:           {chat_ok}")
    print(f"  google/gemini-embedding-001 OK:              {embed1_ok}")
    print(f"  google/gemini-embedding-2-preview OK:        {embed2_preview_ok}")
    print(f"  google/gemini-embedding-2 (GA) OK:           {embed2_ga_ok}")
    if google_msg:
        print()
        print("  Google upstream text (via OpenRouter error.message):")
        print(textwrap.indent(google_msg[:400] + ("..." if len(google_msg) > 400 else ""), "    "))
    if not embed2_preview_ok or not embed2_ga_ok:
        print()
        print("  => letta-vision deploy uses gemini-embedding-2-preview (LETTA_DEFAULT_EMBEDDING_HANDLE).")
        if embed2_ga_ok and not embed2_preview_ok:
            print("  => GA model works; preview may be the broken handle — consider switching handle.")
        elif not embed2_ga_ok and embed1_ok:
            print("  => embedding-001 works but embedding-2 family fails on this OpenRouter/Google path.")
        elif not embed2_ga_ok:
            print("  => Both preview and GA embedding-2 fail — broader Google embedding outage/quota.")


def main() -> None:
    _print_section("OpenRouter embedding diagnostic (direct API, no letta-vision)")
    print(f"Base URL: {OPENROUTER_BASE}")
    print(f"OPENROUTER_REFERER: {os.environ.get('OPENROUTER_REFERER', '(not set)')}")
    print(f"OPENROUTER_TITLE:   {os.environ.get('OPENROUTER_TITLE', '(not set)')}")

    results = _run_cases()
    for r in results:
        _print_result(r)

    _print_section("Raw JSON bodies")
    for r in results:
        print(f"\n--- {r.name} ---")
        print(json.dumps(r.raw_body, indent=2)[:4000])

    _print_conclusion(results)


if __name__ == "__main__":
    main()
