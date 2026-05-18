"""Model registry, skill definitions, runtime config.

Tuned for the user's actual Ollama install:
  mistral:latest       fast routing / summarisation
  deepseek-r1:8b       chain-of-thought reasoning (critic, fix, arch)
  qwen:latest          general-purpose fallback
  qwen3:8b             strong reasoning + instruction following
  qwen2.5-coder:7b     code specialist (coder, reviewer, tester)
  llama3:latest        solid general baseline
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


OLLAMA_BASE    = os.environ.get("OLLAMA_BASE",    "http://localhost:11434")
JAN_BASE       = os.environ.get("JAN_BASE",       "http://localhost:1337/v1")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
SERVER_HOST    = os.environ.get("INSPECTOR_HOST",  "0.0.0.0")
SERVER_PORT    = int(os.environ.get("INSPECTOR_PORT", "8765"))


@dataclass(frozen=True)
class ModelPreset:
    name: str
    ctx_window: int
    roles: tuple[str, ...]
    description: str


# ── User's installed models (from `ollama list`) ─────────────────────────────

OLLAMA_MODEL_PRESETS: dict[str, ModelPreset] = {
    "mistral":       ModelPreset("mistral",       32_768, ("intent", "summarizer"),
                                 "Mistral 7B — fastest, lowest RAM, ideal for routing and classification"),
    "deepseek-r1":   ModelPreset("deepseek-r1",  131_072, ("architect", "critic", "fixer"),
                                 "DeepSeek R1 8B — chain-of-thought reasoning, best for analysis and debugging"),
    "qwen":          ModelPreset("qwen",           32_768, ("intent", "summarizer"),
                                 "Qwen — general-purpose, good multilingual fallback"),
    "qwen3":         ModelPreset("qwen3",         131_072, ("architect", "critic", "intent", "summarizer"),
                                 "Qwen 3 8B — strong reasoning and instruction following"),
    "qwen2.5-coder": ModelPreset("qwen2.5-coder", 131_072, ("coder", "reviewer", "fixer", "tester"),
                                 "Qwen 2.5 Coder 7B — best code model in this set"),
    "llama3":        ModelPreset("llama3",          8_192, ("intent", "architect", "critic"),
                                 "Llama 3 — solid general-purpose baseline"),
}

# Ordered preference list — first match found in the running Ollama instance wins.
# deepseek-r1 leads on reasoning roles; qwen2.5-coder leads on code roles; mistral
# leads on cheap/fast roles to avoid burning the heavy models on classification.
ROLE_MODEL_PREFERENCE: dict[str, list[str]] = {
    "intent":     ["mistral",       "llama3",       "qwen3",         "qwen"],
    "architect":  ["deepseek-r1",   "qwen3",        "llama3"],
    "coder":      ["qwen2.5-coder", "qwen3",        "deepseek-r1"],
    "reviewer":   ["qwen2.5-coder", "deepseek-r1",  "qwen3"],
    "fixer":      ["deepseek-r1",   "qwen2.5-coder","mistral"],
    "tester":     ["qwen2.5-coder", "qwen3",        "deepseek-r1"],
    "critic":     ["deepseek-r1",   "qwen3",        "llama3"],
    "summarizer": ["mistral",       "qwen",         "llama3"],
}


@dataclass(frozen=True)
class SkillDef:
    name: str
    description: str
    role: str           # key into ROLE_MODEL_PREFERENCE
    output_format: str  # "json" | "markdown" | "text"
    system_prompt: str
    temperature: float = 0.2
    max_tokens: int = 2048


# ── Skill index — 9 built-in skills ─────────────────────────────────────────
# Prompt design principles applied throughout:
#   1. Role mandate first — who you are and what your job is.
#   2. Systematic approach — numbered steps for reasoning-heavy skills.
#   3. Output enforcement — "Start with `{`", exact schema, no preamble.
#   4. Calibration anchors — concrete examples of what scores mean.
#   5. Negative constraints — what NOT to do (avoids verbose hedging).

SKILL_INDEX: dict[str, SkillDef] = {

    "intent": SkillDef(
        name="intent",
        description="Classify intent, domain, language, complexity, and recommended next action",
        role="intent",
        output_format="json",
        system_prompt=(
            "You are an intent classifier for a multi-agent code analysis system.\n"
            "Your job: determine what the user ACTUALLY needs, not just what they said.\n\n"
            "Analyse in this order:\n"
            "1. What language/stack is present? Look for syntax, keywords, file patterns.\n"
            "2. What is the primary intent? Choose the single best label.\n"
            "3. What complexity level? Count branches, abstractions, external deps.\n"
            "4. What obvious issues can you spot without deep analysis?\n"
            "5. What is the single highest-value next action for the user?\n\n"
            "Start your response with `{` — no preamble, no explanation, no markdown.\n"
            '{"primary_intent":"<build|fix|test|deploy|review|explain|data|unknown>",'
            '"domain":"<frontend|backend|fullstack|cli|api|data|ml|devops|config|other>",'
            '"language":"<detected language or mixed>",'
            '"complexity":"<trivial|low|moderate|high|very_high>",'
            '"stated_goal":"<one sentence: what they literally asked>",'
            '"key_patterns":["<notable code pattern or library>"],'
            '"issues_spotted":["<obvious problem visible without running the code>"],'
            '"recommended_next":"<the single highest-value action to take next>"}'
        ),
        temperature=0.1,
    ),

    "arch_review": SkillDef(
        name="arch_review",
        description="Rate architecture 1-10, find coupling, missing abstractions, SOLID violations",
        role="architect",
        output_format="json",
        system_prompt=(
            "You are a senior software architect doing a structural review.\n"
            "Apply SOLID principles. Look for:\n"
            "  · Single Responsibility violations (class/module doing too much)\n"
            "  · Open/Closed violations (must modify internals to extend)\n"
            "  · Tight coupling (concrete deps instead of interfaces/protocols)\n"
            "  · Missing abstractions (copy-paste that should be a shared module)\n"
            "  · Layering violations (UI logic in data layer, etc.)\n\n"
            "Think through each concern, then start your response with `{`:\n"
            '{"overall_rating":<1.0-10.0>,'
            '"architecture_style":"<MVC|layered|event-driven|microservice|monolith|functional|other>",'
            '"strengths":["<concrete strength>"],'
            '"weaknesses":["<concrete weakness>"],'
            '"coupling_issues":["<specific tight coupling>"],'
            '"missing_abstractions":["<what should be extracted>"],'
            '"top_priority_fix":"<the single most impactful improvement>",'
            '"estimated_refactor_effort":"<e.g. 4 hours | 2 days>"}'
        ),
    ),

    "security": SkillDef(
        name="security",
        description="OWASP vulnerability scan with severity, location, and fix for each finding",
        role="reviewer",
        output_format="json",
        system_prompt=(
            "You are a security engineer conducting an OWASP Top 10 review.\n"
            "Check systematically for:\n"
            "  A01 Broken Access Control · A02 Cryptographic Failures\n"
            "  A03 Injection (SQL/cmd/LDAP/XSS) · A04 Insecure Design\n"
            "  A05 Security Misconfiguration · A06 Vulnerable Components\n"
            "  A07 Auth Failures · A08 Software Integrity Failures\n"
            "  A09 Logging Failures · A10 SSRF\n"
            "Also flag: hardcoded secrets, eval/exec, unsafe deserialization, path traversal.\n\n"
            "Start your response with `{`:\n"
            '{"risk_level":"<critical|high|medium|low|none>",'
            '"vulnerabilities":[{"type":"<OWASP category or pattern>",'
            '"location":"<file:line or function name>",'
            '"severity":"<critical|high|medium|low>",'
            '"description":"<what is wrong and why it matters>",'
            '"fix":"<concrete remediation>"}],'
            '"hardening_suggestions":["<defence-in-depth suggestion>"],'
            '"owasp_categories":["<e.g. A03:2021-Injection>"]}'
        ),
    ),

    "test_gen": SkillDef(
        name="test_gen",
        description="Generate a complete test file with happy path, edge cases, and error handling",
        role="tester",
        output_format="markdown",
        system_prompt=(
            "You are a test engineer. Generate a production-ready test file.\n"
            "Coverage strategy:\n"
            "  · Happy path — the intended inputs and expected outputs\n"
            "  · Boundary values — off-by-one, empty, max, min\n"
            "  · Error cases — invalid input, missing deps, network failure\n"
            "  · Concurrency/state — if relevant\n\n"
            "Rules:\n"
            "  · Match the language and test framework of the source code.\n"
            "  · Every test must be independently runnable.\n"
            "  · No mocks unless the code calls external services.\n"
            "  · No preamble — output a single fenced code block, nothing else."
        ),
        temperature=0.3,
        max_tokens=4096,
    ),

    "summarize": SkillDef(
        name="summarize",
        description="Terse technical summary: what it does, public API, dependencies, patterns",
        role="summarizer",
        output_format="markdown",
        system_prompt=(
            "You are a technical writer who values extreme brevity.\n"
            "Produce a summary with exactly these sections (omit if not applicable):\n"
            "**What it does** — 1-2 sentences, no fluff.\n"
            "**Public API** — function/class signatures that callers use.\n"
            "**Dependencies** — external packages or services required.\n"
            "**Patterns** — notable design patterns, idioms, or non-obvious choices.\n"
            "**Watch out** — anything surprising, deprecated, or dangerous.\n\n"
            "Be terse. Start directly with **What it does**. No intro sentence."
        ),
    ),

    "fix": SkillDef(
        name="fix",
        description="Diagnose root cause and propose the smallest correct patch",
        role="fixer",
        output_format="json",
        system_prompt=(
            "You are a debug engineer. Find the ROOT CAUSE, not the symptom.\n\n"
            "Reasoning approach:\n"
            "1. What is the exact failure? (error message, wrong output, crash)\n"
            "2. What is the simplest explanation? (Occam's razor — eliminate complex causes first)\n"
            "3. What is the SMALLEST change that fixes the root cause?\n"
            "4. How confident are you? (high = reproducible + one clear cause)\n\n"
            "Never over-engineer the patch. If you are unsure, say so.\n"
            "Start your response with `{`:\n"
            '{"diagnosis":"<root cause in one sentence, not a description of symptoms>",'
            '"fix_kind":"<rewrite|patch|config|dependency|environment>",'
            '"patch":"<corrected code snippet or unified diff>",'
            '"explanation":"<why this fixes the root cause, not just what it does>",'
            '"confidence":"<low|medium|high>"}'
        ),
    ),

    "doc": SkillDef(
        name="doc",
        description="Add or improve docstrings using native format (Google-style / JSDoc)",
        role="coder",
        output_format="markdown",
        system_prompt=(
            "You are a technical writer adding documentation to code.\n"
            "Format rules:\n"
            "  · Python: Google-style docstrings (Args:, Returns:, Raises:, Example:)\n"
            "  · JavaScript/TypeScript: JSDoc (@param, @returns, @throws, @example)\n"
            "  · Other: use the language's idiomatic doc format\n\n"
            "Document every public function, class, and module-level constant.\n"
            "Skip private helpers (prefixed _) unless they are complex.\n"
            "No introductory sentence — output the complete documented file, nothing else."
        ),
        temperature=0.3,
        max_tokens=4096,
    ),

    "critic": SkillDef(
        name="critic",
        description="Score 8 quality dimensions (0-10) with calibrated anchors and critical fix",
        role="critic",
        output_format="json",
        system_prompt=(
            "You are a ruthless senior engineer doing a code review. Be accurate, not kind.\n\n"
            "Score 8 dimensions on 0.0–10.0:\n"
            "  correctness   — logic errors, off-by-ones, wrong assumptions\n"
            "  clarity       — readable without context, good names, no magic numbers\n"
            "  structure     — separation of concerns, single responsibility, file layout\n"
            "  testability   — can each unit be tested in isolation?\n"
            "  security      — no obvious vulnerabilities, secrets, injection points\n"
            "  performance   — no O(n²) where O(n) works, no unnecessary allocations\n"
            "  maintainability — next developer won't curse you, low bus factor\n"
            "  completeness  — no TODOs, stubs, or unimplemented branches\n\n"
            "Calibration: 5=broken/dangerous, 6=needs work, 7=acceptable, 8=good, 9=excellent.\n"
            "Never give 10. Rarely give 9. Give 6 or below for anything production-unsafe.\n\n"
            "Think through each dimension, then start your response with `{`:\n"
            '{"correctness":<float>,"clarity":<float>,"structure":<float>,'
            '"testability":<float>,"security":<float>,"performance":<float>,'
            '"maintainability":<float>,"completeness":<float>,'
            '"lowest_dimension":"<name of worst-scoring dimension>",'
            '"critical_fix":"<one sentence: the single most important thing to fix>",'
            '"summary":"<2-3 sentence honest verdict>"}'
        ),
        temperature=0.1,
    ),

    "loop_runner": SkillDef(
        name="loop_runner",
        description="Decompose a task into a bounded autonomous loop spec with verifiable acceptance criteria",
        role="architect",
        output_format="json",
        system_prompt=(
            "You are a task planner for bounded autonomous agent loops (Ralph Wiggum pattern).\n"
            "Your job: turn a vague task into a precise, machine-verifiable loop specification.\n\n"
            "Rules:\n"
            "  · Acceptance criteria must be objectively checkable (tests pass, file exists, etc.)\n"
            "  · The completion_token must be unique and SCREAMING_SNAKE_CASE\n"
            "  · recommended_max_iter: use 5 for tiny tasks, 15 for moderate, 25 for complex\n"
            "  · verification_commands must be real shell commands that return 0 on success\n"
            "  · risk_level: high if the task modifies shared state, databases, or CI/CD\n\n"
            "Start your response with `{`:\n"
            '{"task_summary":"<one sentence>","acceptance_criteria":["<verifiable criterion>"],'
            '"completion_token":"<SCREAMING_SNAKE_CASE>",'
            '"recommended_max_iter":<5-25>,'
            '"sub_tasks":["<concrete ordered step>"],'
            '"verification_commands":["<shell command that exits 0 on success>"],'
            '"risk_level":"<low|medium|high>",'
            '"notes":"<edge cases or watch-outs for the agent>"}'
        ),
        temperature=0.2,
        max_tokens=1024,
    ),
}


@dataclass
class InspectorConfig:
    ollama_base: str = OLLAMA_BASE
    jan_base: str = JAN_BASE
    ollama_timeout: int = OLLAMA_TIMEOUT
    default_model: str = "mistral"      # fastest of user's installed models
    heartbeat_interval_s: int = 30
    server_host: str = SERVER_HOST
    server_port: int = SERVER_PORT
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /", "mkfs", ":(){ :|:& };:", "dd if=/dev/zero",
        "chmod -R 777 /", "> /dev/sda",
    ])


_config = InspectorConfig()


def get_config() -> InspectorConfig:
    return _config
