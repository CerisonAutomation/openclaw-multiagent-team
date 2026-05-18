"""Model registry, skill definitions, and runtime configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
SERVER_HOST = os.environ.get("INSPECTOR_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("INSPECTOR_PORT", "8765"))


@dataclass(frozen=True)
class ModelPreset:
    name: str
    ctx_window: int
    roles: tuple[str, ...]
    description: str


# Models that are commonly available via `ollama pull <name>`
OLLAMA_MODEL_PRESETS: dict[str, ModelPreset] = {
    "llama3.2":           ModelPreset("llama3.2",          128_000, ("intent", "architect", "critic", "summarizer"), "Meta Llama 3.2 — fast general-purpose"),
    "llama3.1":           ModelPreset("llama3.1",          128_000, ("intent", "architect", "critic"),               "Meta Llama 3.1 — strong reasoning"),
    "codellama":          ModelPreset("codellama",          16_384, ("coder", "reviewer", "fixer", "tester"),         "Meta CodeLlama — code specialist"),
    "deepseek-coder-v2":  ModelPreset("deepseek-coder-v2", 163_840, ("coder", "reviewer", "tester"),                  "DeepSeek Coder V2 — top code benchmark"),
    "qwen2.5-coder":      ModelPreset("qwen2.5-coder",    131_072, ("coder", "reviewer", "fixer"),                   "Qwen 2.5 Coder — strong instruction+code"),
    "mistral":            ModelPreset("mistral",            32_768, ("intent", "fixer", "summarizer"),                "Mistral 7B — fast, low RAM"),
    "phi4":               ModelPreset("phi4",               16_384, ("intent", "architect", "critic"),                "Microsoft Phi-4 — punches above its weight"),
    "phi3":               ModelPreset("phi3",                4_096, ("intent", "critic"),                             "Microsoft Phi-3 — smallest capable"),
    "gemma2":             ModelPreset("gemma2",              8_192, ("intent", "summarizer"),                         "Google Gemma 2 — good multilingual"),
    "yi-coder":           ModelPreset("yi-coder",           65_536, ("coder", "reviewer", "tester"),                  "01.AI Yi-Coder — strong code"),
    "starcoder2":         ModelPreset("starcoder2",         16_384, ("coder", "reviewer"),                            "BigCode StarCoder2 — trained on code exclusively"),
}

# Ordered preference list for each role — first match found via /api/tags wins
ROLE_MODEL_PREFERENCE: dict[str, list[str]] = {
    "intent":     ["llama3.2", "llama3.1", "phi4", "mistral", "phi3"],
    "architect":  ["llama3.1", "llama3.2", "phi4", "mistral"],
    "coder":      ["qwen2.5-coder", "deepseek-coder-v2", "codellama", "yi-coder", "starcoder2"],
    "reviewer":   ["deepseek-coder-v2", "qwen2.5-coder", "codellama", "llama3.1"],
    "fixer":      ["qwen2.5-coder", "codellama", "mistral", "llama3.2"],
    "tester":     ["deepseek-coder-v2", "qwen2.5-coder", "codellama", "yi-coder"],
    "critic":     ["llama3.1", "llama3.2", "phi4"],
    "summarizer": ["llama3.2", "mistral", "gemma2"],
}


@dataclass(frozen=True)
class SkillDef:
    name: str
    description: str
    role: str              # which ROLE_MODEL_PREFERENCE entry to use
    output_format: str     # "json" | "text" | "markdown"
    system_prompt: str
    temperature: float = 0.2
    max_tokens: int = 2048


SKILL_INDEX: dict[str, SkillDef] = {
    "intent": SkillDef(
        name="intent",
        description="Classify intent, domain, language, complexity, and next action",
        role="intent",
        output_format="json",
        system_prompt=(
            "Classify the code or text. Respond ONLY with valid JSON:\n"
            '{\n'
            '  "primary_intent": "<build|fix|test|deploy|review|explain|data|unknown>",\n'
            '  "domain": "<frontend|backend|fullstack|cli|api|data|ml|devops|config|other>",\n'
            '  "language": "<detected language or mixed>",\n'
            '  "complexity": "<trivial|low|moderate|high|very_high>",\n'
            '  "stated_goal": "<one sentence>",\n'
            '  "key_patterns": ["<pattern>"],\n'
            '  "issues_spotted": ["<issue>"],\n'
            '  "recommended_next": "<highest-value next action>"\n'
            "}"
        ),
        temperature=0.1,
    ),
    "arch_review": SkillDef(
        name="arch_review",
        description="Architecture review: coupling, abstractions, structural issues",
        role="architect",
        output_format="json",
        system_prompt=(
            "You are a senior software architect. Analyze the code. Respond ONLY with valid JSON:\n"
            '{\n'
            '  "overall_rating": <1.0-10.0>,\n'
            '  "architecture_style": "<MVC|layered|event-driven|microservice|monolith|functional|other>",\n'
            '  "strengths": ["<strength>"],\n'
            '  "weaknesses": ["<weakness>"],\n'
            '  "coupling_issues": ["<issue>"],\n'
            '  "missing_abstractions": ["<abstraction>"],\n'
            '  "top_priority_fix": "<most important improvement>",\n'
            '  "estimated_refactor_effort": "<e.g. 2 hours>"\n'
            "}"
        ),
    ),
    "security": SkillDef(
        name="security",
        description="Identify security vulnerabilities (OWASP, secrets, injection)",
        role="reviewer",
        output_format="json",
        system_prompt=(
            "You are a security engineer. Find vulnerabilities. Respond ONLY with valid JSON:\n"
            '{\n'
            '  "risk_level": "<critical|high|medium|low|none>",\n'
            '  "vulnerabilities": [\n'
            '    {"type": "<type>", "location": "<where>", "severity": "<critical|high|medium|low>",\n'
            '     "description": "<what>", "fix": "<how>"}\n'
            '  ],\n'
            '  "hardening_suggestions": ["<suggestion>"],\n'
            '  "owasp_categories": ["<category>"]\n'
            "}"
        ),
    ),
    "test_gen": SkillDef(
        name="test_gen",
        description="Generate a complete test file for the given code",
        role="tester",
        output_format="markdown",
        system_prompt=(
            "Generate comprehensive tests for the provided code.\n"
            "Output a single markdown code block with the complete test file.\n"
            "Cover: happy path, edge cases, exceptions, boundary conditions.\n"
            "Use the same language/framework as the input. No preamble — just the code block."
        ),
        temperature=0.3,
        max_tokens=4096,
    ),
    "summarize": SkillDef(
        name="summarize",
        description="Concise technical summary: what it does, API surface, notable patterns",
        role="summarizer",
        output_format="markdown",
        system_prompt=(
            "Produce a concise technical summary. Include:\n"
            "- What it does (1-2 sentences)\n"
            "- Key dependencies\n"
            "- Entry points / public API\n"
            "- Notable patterns\n"
            "- Anything unusual worth flagging\n"
            "Be terse. Start directly with substance."
        ),
    ),
    "fix": SkillDef(
        name="fix",
        description="Diagnose errors and propose the smallest correct patch",
        role="fixer",
        output_format="json",
        system_prompt=(
            "Diagnose the error and propose the minimal fix. Respond ONLY with valid JSON:\n"
            '{\n'
            '  "diagnosis": "<root cause in one sentence>",\n'
            '  "fix_kind": "<rewrite|patch|config|dependency|environment>",\n'
            '  "patch": "<corrected code or unified diff>",\n'
            '  "explanation": "<why this fixes it>",\n'
            '  "confidence": "<low|medium|high>"\n'
            "}"
        ),
    ),
    "doc": SkillDef(
        name="doc",
        description="Add or improve docstrings and inline documentation",
        role="coder",
        output_format="markdown",
        system_prompt=(
            "Add or improve docstrings/comments for the provided code.\n"
            "Return the complete file with documentation added.\n"
            "Use the language's native doc format (Python: Google-style, JS/TS: JSDoc).\n"
            "No preamble — just the documented code."
        ),
        temperature=0.3,
        max_tokens=4096,
    ),
    "critic": SkillDef(
        name="critic",
        description="Score code on 8 quality dimensions with actionable critique",
        role="critic",
        output_format="json",
        system_prompt=(
            "Score the code on 8 quality dimensions (0.0–10.0 each). Be accurate, not generous.\n"
            "Anchors: <6=needs work, 6-8=acceptable, >8=excellent. Rarely give >9.\n"
            "Respond ONLY with valid JSON:\n"
            '{\n'
            '  "correctness": <float>,\n'
            '  "clarity": <float>,\n'
            '  "structure": <float>,\n'
            '  "testability": <float>,\n'
            '  "security": <float>,\n'
            '  "performance": <float>,\n'
            '  "maintainability": <float>,\n'
            '  "completeness": <float>,\n'
            '  "lowest_dimension": "<name>",\n'
            '  "critical_fix": "<most important improvement>",\n'
            '  "summary": "<2-3 sentence verdict>"\n'
            "}"
        ),
        temperature=0.1,
    ),
    "loop_runner": SkillDef(
        name="loop_runner",
        description="Plan a bounded autonomous loop: decompose task, write acceptance criteria and completion token",
        role="architect",
        output_format="json",
        system_prompt=(
            "You are a task planner for a bounded autonomous agent loop (Ralph Wiggum pattern).\n"
            "Given a task description, produce a precise loop specification. Respond ONLY with valid JSON:\n"
            '{\n'
            '  "task_summary": "<one sentence>",\n'
            '  "acceptance_criteria": ["<verifiable criterion>"],\n'
            '  "completion_token": "<SCREAMING_SNAKE_CASE token Claude writes when done>",\n'
            '  "recommended_max_iter": <int between 5 and 30>,\n'
            '  "sub_tasks": ["<concrete step>"],\n'
            '  "verification_commands": ["<shell command to verify success>"],\n'
            '  "risk_level": "<low|medium|high>",\n'
            '  "notes": "<anything the agent should watch out for>"\n'
            "}"
        ),
        temperature=0.2,
        max_tokens=1024,
    ),
}


@dataclass
class InspectorConfig:
    ollama_base: str = OLLAMA_BASE
    ollama_timeout: int = OLLAMA_TIMEOUT
    default_model: str = "llama3.2"
    heartbeat_interval_s: int = 30
    server_host: str = SERVER_HOST
    server_port: int = SERVER_PORT
    workspace: str = "./inspector_workspace"
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /", "mkfs", ":(){ :|:& };:", "dd if=/dev/zero",
        "chmod -R 777 /", "> /dev/sda",
    ])


_config = InspectorConfig()


def get_config() -> InspectorConfig:
    return _config
