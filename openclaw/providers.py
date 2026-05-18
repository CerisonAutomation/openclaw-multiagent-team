"""Universal LLM provider — Anthropic + any OpenAI-compatible endpoint.

OpenAI-compatible includes: OpenAI, NVIDIA NIM, OpenRouter, DeepSeek, Together,
Groq, Fireworks, vLLM, Ollama (with /v1), and most self-hosted gateways. Set
`OPENCLAW_PROVIDER` and the corresponding env vars below.

    export OPENCLAW_PROVIDER=anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENCLAW_MODEL=claude-sonnet-4-20250514          # optional

    export OPENCLAW_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    export OPENAI_BASE_URL=https://api.openai.com/v1        # or any compatible URL
    export OPENCLAW_MODEL=gpt-4o-mini                       # optional

    export OPENCLAW_PROVIDER=mock                           # dry-run, no API key needed

Routing OpenAI-compatible providers:
    export OPENCLAW_PROVIDER=openai
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1
    export OPENAI_API_KEY=$OPENROUTER_API_KEY
    export OPENCLAW_MODEL=anthropic/claude-3.5-sonnet
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


class Provider(Protocol):
    name: str
    model: str

    def chat(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.7) -> str: ...

    def json(self, system: str, user: str, max_tokens: int = 1500) -> dict: ...


def _strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        # remove leading ```json or ```
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


@dataclass
class AnthropicProvider:
    name: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    _client: Any = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def chat(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    def json(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        raw = self.chat(system + "\n\nRespond ONLY with valid JSON. No prose, no markdown.", user, max_tokens, 0.2)
        try:
            return json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError:
            return {"_parse_error": True, "_raw": raw}


@dataclass
class OpenAIProvider:
    """Works with OpenAI and any OpenAI-compatible endpoint."""

    name: str = "openai"
    model: str = field(default_factory=lambda: os.environ.get("OPENCLAW_MODEL", "gpt-4o-mini"))
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    _client: Any = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai SDK not installed. Run: pip install openai") from e
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def json(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        raw = self.chat(system + "\n\nRespond ONLY with valid JSON. No prose, no markdown.", user, max_tokens, 0.2)
        try:
            return json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError:
            return {"_parse_error": True, "_raw": raw}


@dataclass
class MockProvider:
    """Deterministic offline provider — no network. For tests and dry-runs."""

    name: str = "mock"
    model: str = "mock-1"

    def chat(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        head = user.splitlines()[0][:80] if user else ""
        return (
            f"[MOCK] Task: {head}\n\n"
            "1. Outlined the requested change.\n"
            "2. Drafted the implementation.\n"
            "3. Self-checked against constraints.\n\n"
            "SEAL: mock output produced. Next step: replace MockProvider with a real LLM."
        )

    def json(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        # Order matters — match the most specific role first.
        sl = system.lower()
        if "senior software architect" in sl or "files_to_create" in sl:
            return {
                "framework": "nextjs",
                "language": "typescript",
                "package_manager": "npm",
                "files_to_create": [
                    {"path": "package.json", "purpose": "deps + scripts"},
                    {"path": "app/page.tsx", "purpose": "landing page"},
                    {"path": "vercel.json", "purpose": "deploy config"},
                ],
                "commands": ["echo 'mock-install ok'", "echo 'mock-build ok'"],
                "deploy_target": "vercel",
                "env_vars_required": [],
            }
        if "ruthless senior reviewer" in sl or "score the artifact" in sl:
            return {
                "clarity": 8.5, "structure": 8.0, "completeness": 8.2, "accuracy": 8.0,
                "applicability": 8.4, "expertise": 8.0, "originality": 7.5, "compliance": 9.0,
                "efficiency": 8.0, "resonance": 7.8,
                "lowest_dimension": "originality",
                "critical_fix": "Add a concrete example to ground the abstractions.",
            }
        if "debug engineer" in sl or "diagnose" in sl:
            return {
                "diagnosis": "missing dependency or path mismatch",
                "fix_kind": "change_command",
                "target_path": "",
                "patch": "echo 'mock-fix applied'",
                "confidence": "low",
            }
        if "test engineer" in sl or "verification_commands" in sl:
            return {
                "verification_commands": ["echo 'mock-verify ok'"],
                "expected_exit_code": 0,
                "skip_if": "",
            }
        if "deployment engineer" in sl or "deploy_commands" in sl:
            return {
                "preflight": ["echo 'mock-preflight ok'"],
                "deploy_commands": ["echo 'mock-deploy ok'"],
                "expected_output_pattern": r"https?://\S+",
                "env_vars_to_set": [],
            }
        if "classify the user's request" in sl or "primary_intent" in sl:
            return {
                "primary_intent": "build",
                "secondary_intent": None,
                "complexity": "moderate",
                "domain": "fullstack",
                "stated_goal": (user.splitlines()[0][:120] if user else "build something"),
                "unstated_goal": "production-ready output with sensible defaults",
                "input_rewrite": (user.splitlines()[0][:200] if user else "build something"),
                "key_constraints": [],
                "needs_repo_context": False,
                "needs_deployment": False,
            }
        return {"_mock": True}


def get_provider(name: str | None = None, model: str | None = None) -> Provider:
    """Resolve a provider from env or explicit args.

    Precedence: explicit `name` arg > $OPENCLAW_PROVIDER > anthropic (if key set) >
    openai (if key set) > mock.
    """
    chosen = (name or os.environ.get("OPENCLAW_PROVIDER", "")).lower().strip()
    if not chosen:
        if os.environ.get("ANTHROPIC_API_KEY"):
            chosen = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            chosen = "openai"
        else:
            chosen = "mock"

    explicit_model = model or os.environ.get("OPENCLAW_MODEL")

    if chosen == "anthropic":
        p = AnthropicProvider()
        if explicit_model:
            p.model = explicit_model
        return p
    if chosen == "openai":
        p = OpenAIProvider()
        if explicit_model:
            p.model = explicit_model
        return p
    if chosen == "mock":
        return MockProvider()
    raise ValueError(f"Unknown OPENCLAW_PROVIDER: {chosen!r}. Use anthropic|openai|mock.")
