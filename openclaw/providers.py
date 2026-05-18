"""Universal LLM provider — Anthropic + any OpenAI-compatible endpoint, with
per-role model routing.

    # Pick a provider by name; the preset knows the URL, key var, and which
    # model to use for which agent role (intent/critic on cheap, coder on smart).
    export OPENCLAW_PROVIDER=groq
    export GROQ_API_KEY=...
    openclaw build "..."

    # Override a single role's model:
    export OPENCLAW_MODEL_CRITIC=llama-3.1-8b-instant
    export OPENCLAW_MODEL_CODER=llama-3.3-70b-versatile

    # Or pin a single model for every role:
    export OPENCLAW_MODEL=llama-3.3-70b-versatile

    # Dry-run with zero credentials:
    export OPENCLAW_PROVIDER=mock
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from openclaw.models import PRESETS, ProviderPreset


class Provider(Protocol):
    name: str
    model: str

    def chat(self, system: str, user: str, max_tokens: int = 2000,
             temperature: float = 0.7, role: str | None = None) -> str: ...

    def json(self, system: str, user: str, max_tokens: int = 1500,
             role: str | None = None) -> dict: ...

    def model_for(self, role: str | None) -> str: ...


def _strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def _role_env_override(role: str | None) -> str | None:
    """OPENCLAW_MODEL_CRITIC, OPENCLAW_MODEL_CODER, etc."""
    if not role:
        return None
    return os.environ.get(f"OPENCLAW_MODEL_{role.upper()}")


@dataclass
class AnthropicProvider:
    name: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    role_models: dict[str, str] = field(default_factory=dict)
    _client: Any = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def model_for(self, role: str | None) -> str:
        return _role_env_override(role) or self.role_models.get(role or "", self.model)

    def chat(self, system: str, user: str, max_tokens: int = 2000,
             temperature: float = 0.7, role: str | None = None) -> str:
        msg = self._client.messages.create(
            model=self.model_for(role),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    def json(self, system: str, user: str, max_tokens: int = 1500, role: str | None = None) -> dict:
        raw = self.chat(
            system + "\n\nRespond ONLY with valid JSON. No prose, no markdown.",
            user, max_tokens, 0.2, role=role,
        )
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
    role_models: dict[str, str] = field(default_factory=dict)
    _client: Any = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError(f"API key not set (expected env var for provider '{self.name}')")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai SDK not installed. Run: pip install openai") from e
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def model_for(self, role: str | None) -> str:
        return _role_env_override(role) or self.role_models.get(role or "", self.model)

    def chat(self, system: str, user: str, max_tokens: int = 2000,
             temperature: float = 0.7, role: str | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_for(role),
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def json(self, system: str, user: str, max_tokens: int = 1500, role: str | None = None) -> dict:
        raw = self.chat(
            system + "\n\nRespond ONLY with valid JSON. No prose, no markdown.",
            user, max_tokens, 0.2, role=role,
        )
        try:
            return json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError:
            return {"_parse_error": True, "_raw": raw}


@dataclass
class MockProvider:
    """Deterministic offline provider — no network. For tests and dry-runs."""

    name: str = "mock"
    model: str = "mock-1"
    role_models: dict[str, str] = field(default_factory=dict)

    def model_for(self, role: str | None) -> str:
        return self.role_models.get(role or "", self.model)

    def chat(self, system: str, user: str, max_tokens: int = 2000,
             temperature: float = 0.7, role: str | None = None) -> str:
        head = user.splitlines()[0][:80] if user else ""
        return (
            f"[MOCK] Task: {head}\n\n"
            "1. Outlined the requested change.\n"
            "2. Drafted the implementation.\n"
            "3. Self-checked against constraints.\n\n"
            "SEAL: mock output produced. Next step: replace MockProvider with a real LLM."
        )

    def json(self, system: str, user: str, max_tokens: int = 1500, role: str | None = None) -> dict:
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


def _from_preset(name: str, preset: ProviderPreset, model_override: str | None) -> Provider:
    """Construct the right provider class from a preset entry."""
    if preset.kind == "mock":
        p = MockProvider()
    elif preset.kind == "anthropic":
        key = os.environ.get(preset.key_env, "")
        if not key:
            raise RuntimeError(f"{preset.key_env} is not set (required for provider '{name}')")
        p = AnthropicProvider(name=name, api_key=key, model=preset.default_model,
                              role_models=dict(preset.role_models))
    elif preset.kind == "openai_compatible":
        key = os.environ.get(preset.key_env, "") if preset.key_env else ""
        if not key:
            # Local providers (ollama, jan) need no real API key
            if name in ("ollama", "jan"):
                key = name
            else:
                raise RuntimeError(f"{preset.key_env} is not set (required for provider '{name}')")
        p = OpenAIProvider(
            name=name, api_key=key, base_url=preset.base_url or "https://api.openai.com/v1",
            model=preset.default_model, role_models=dict(preset.role_models),
        )
    else:
        raise ValueError(f"Unknown preset kind: {preset.kind!r}")

    # Apply explicit model override (pins all roles to the same model)
    if model_override:
        p.model = model_override
        p.role_models = {}   # explicit override beats role routing
    return p


def get_provider(name: str | None = None, model: str | None = None) -> Provider:
    """Resolve a provider from a preset name or auto-detect.

    Precedence: explicit `name` arg > $OPENCLAW_PROVIDER > first preset whose
    key env var is set > mock.
    """
    chosen = (name or os.environ.get("OPENCLAW_PROVIDER", "")).lower().strip()
    explicit_model = model or os.environ.get("OPENCLAW_MODEL")

    if not chosen:
        # Auto-detect: pick the first preset whose key env var is populated
        for preset_name, preset in PRESETS.items():
            if preset.kind == "mock":
                continue
            if preset.key_env and os.environ.get(preset.key_env):
                chosen = preset_name
                break
        if not chosen:
            chosen = "mock"

    if chosen not in PRESETS:
        raise ValueError(
            f"Unknown provider: {chosen!r}. "
            f"Available: {', '.join(PRESETS.keys())}"
        )
    return _from_preset(chosen, PRESETS[chosen], explicit_model)
