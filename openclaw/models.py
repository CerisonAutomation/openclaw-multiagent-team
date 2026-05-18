"""Provider presets + per-role model routing.

Inspired by ruvnet/agentic-flow's cost-aware Sonnet↔Haiku routing. The idea:
expensive smart models for code writing and bug fixing, cheap fast models for
intent classification and quality scoring. Same provider, different model per
agent role.

Usage:
    get_provider("groq")                # uses presets[groq]
    get_provider("openrouter")          # cheap routing across many models
    get_provider("openai", model="gpt-4o-mini")  # override default

Add a new provider by appending to PRESETS — no other code change needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProviderPreset:
    """Knows how to construct a provider client and which model to use per role."""

    kind: str                        # 'anthropic' | 'openai_compatible'
    key_env: str                     # env var holding the API key
    default_model: str               # model used when no role override applies
    base_url: str | None = None      # only for openai_compatible providers
    role_models: dict[str, str] = field(default_factory=dict)
    # Documentation surface — shown by `openclaw providers list`.
    description: str = ""
    docs_url: str = ""


# ── The presets ─────────────────────────────────────────────────────────────
# Models chosen to be widely available + cheap variants on the same provider.
# Override at runtime with --model or OPENCLAW_MODEL=... ; override per role
# with OPENCLAW_MODEL_<ROLE>=... (e.g. OPENCLAW_MODEL_CRITIC=...).

PRESETS: dict[str, ProviderPreset] = {
    # Priority defaults — openrouter (free) and nvidia (high-end) are tried
    # first by get_provider() auto-detect.
    "openrouter": ProviderPreset(
        kind="openai_compatible",
        key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        # Free tier, top-tier coding model.
        default_model="deepseek/deepseek-chat-v3-0324:free",
        role_models={
            # All roles stay on high-end free 70B+ models — no cheap fallback.
            "intent":    "meta-llama/llama-3.3-70b-instruct:free",
            "critic":    "meta-llama/llama-3.3-70b-instruct:free",
            "tester":    "meta-llama/llama-3.3-70b-instruct:free",
            "deployer":  "meta-llama/llama-3.3-70b-instruct:free",
            "architect": "deepseek/deepseek-chat-v3-0324:free",
            "coder":     "deepseek/deepseek-chat-v3-0324:free",
            "fixer":     "deepseek/deepseek-chat-v3-0324:free",
        },
        description="OpenRouter — free tier, high-end models (DeepSeek V3 free, Llama 3.3 70B free).",
        docs_url="https://openrouter.ai/docs",
    ),
    "nvidia": ProviderPreset(
        kind="openai_compatible",
        key_env="NVIDIA_NIM_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
        # NVIDIA's strongest hosted model.
        default_model="nvidia/llama-3.1-nemotron-70b-instruct",
        role_models={
            # All high-end NIM models, no cheap fallback.
            "intent":    "meta/llama-3.3-70b-instruct",
            "critic":    "meta/llama-3.3-70b-instruct",
            "tester":    "meta/llama-3.3-70b-instruct",
            "deployer":  "meta/llama-3.3-70b-instruct",
            "architect": "nvidia/llama-3.1-nemotron-70b-instruct",
            "coder":     "qwen/qwen2.5-coder-32b-instruct",
            "fixer":     "qwen/qwen2.5-coder-32b-instruct",
        },
        description="NVIDIA NIM — high-end hosted (Nemotron 70B, Llama 3.3 70B, Qwen Coder 32B).",
        docs_url="https://docs.nvidia.com/nim",
    ),
    "anthropic": ProviderPreset(
        kind="anthropic",
        key_env="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-20250514",
        role_models={
            "intent":    "claude-haiku-4-5-20251001",
            "critic":    "claude-haiku-4-5-20251001",
            "tester":    "claude-haiku-4-5-20251001",
            "deployer":  "claude-haiku-4-5-20251001",
            "architect": "claude-sonnet-4-20250514",
            "coder":     "claude-sonnet-4-20250514",
            "fixer":     "claude-sonnet-4-20250514",
        },
        description="Anthropic direct. Sonnet for code, Haiku for cheap roles.",
        docs_url="https://docs.anthropic.com",
    ),
    "openai": ProviderPreset(
        kind="openai_compatible",
        key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        role_models={
            "intent": "gpt-4o-mini",
            "critic": "gpt-4o-mini",
            "tester": "gpt-4o-mini",
            "coder":  "gpt-4o",
            "fixer":  "gpt-4o",
        },
        description="OpenAI direct.",
        docs_url="https://platform.openai.com/docs",
    ),
    "groq": ProviderPreset(
        kind="openai_compatible",
        key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        role_models={
            "intent": "llama-3.1-8b-instant",
            "critic": "llama-3.1-8b-instant",
            "tester": "llama-3.1-8b-instant",
        },
        description="Groq — fastest tokens/sec. Llama 3.3 70B for code, 8B for cheap roles.",
        docs_url="https://console.groq.com/docs",
    ),
    "deepseek": ProviderPreset(
        kind="openai_compatible",
        key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        role_models={},  # single model is already cheap
        description="DeepSeek — very cheap, strong on code.",
        docs_url="https://api-docs.deepseek.com",
    ),
    "ollama": ProviderPreset(
        kind="openai_compatible",
        key_env="OLLAMA_API_KEY",        # ollama doesn't need a key; any non-empty value works
        base_url="http://localhost:11434/v1",
        default_model="mistral:latest",  # fastest installed model for cheap roles
        role_models={
            # deepseek-r1:8b leads on reasoning: architect, critic, fixer (CoT)
            "architect": "deepseek-r1:8b",
            "critic":    "deepseek-r1:8b",
            "fixer":     "deepseek-r1:8b",
            # qwen2.5-coder:7b is the code specialist
            "coder":     "qwen2.5-coder:7b",
            "reviewer":  "qwen2.5-coder:7b",
            "tester":    "qwen2.5-coder:7b",
            # mistral:latest is fastest — use for routing and cheap classification
            "intent":    "mistral:latest",
            "deployer":  "mistral:latest",
        },
        description="Ollama — local models, no cost, no network. (mistral/deepseek-r1/qwen2.5-coder/qwen3/llama3)",
        docs_url="https://github.com/ollama/ollama",
    ),
    "jan": ProviderPreset(
        kind="openai_compatible",
        key_env="",                      # Jan doesn't require an API key
        base_url="http://localhost:1337/v1",
        default_model="mistral-ins-7b-q4",  # typical Jan default; overridden at runtime
        description="Jan — local OpenAI-compatible server, fallback when Ollama is offline.",
        docs_url="https://jan.ai/docs",
    ),
    "mock": ProviderPreset(
        kind="mock",
        key_env="",
        default_model="mock-1",
        description="Deterministic offline provider for tests and dry-runs.",
        docs_url="",
    ),
}


def list_presets() -> list[tuple[str, str]]:
    return [(name, p.description) for name, p in PRESETS.items()]
