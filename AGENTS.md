# openclaw — Agent Framework & Trait Reference

## Architecture

```
User prompt / code input
        │
        ▼
┌───────────────────┐
│  SkillIndex        │  routes to best available Ollama or Jan model
│  (agents.py)       │  per role-preference list
└────────┬──────────┘
         │  fires skills in sequence or pipeline
         ▼
┌───────────────────┐     ┌──────────────────────┐
│  OllamaClient      │ ←── │  Jan fallback client  │
│  localhost:11434   │     │  localhost:1337/v1    │
└────────┬──────────┘     └──────────────────────┘
         │  structured JSON / markdown output
         ▼
┌───────────────────┐
│  EventBus          │  async pub/sub
│  (scheduler.py)    │
└────────┬──────────┘
         │
   ┌─────┴──────────────────────────────────────────┐
   │                                                  │
   ▼                                                  ▼
HeartbeatMonitor                              CronScheduler
30-second Ollama ping                         interval-based job runner
emits ollama_up / ollama_down                 fires skills on a schedule
         │                                            │
         └─────────────────────────────┬─────────────┘
                                        ▼
                              FastAPI server (:8765)
                              WebSocket broadcast
                              inspector.html UI
```

---

## Skill Index — all 8 built-in skills

| Skill | Role | Output | What it does |
|---|---|---|---|
| `intent` | intent | JSON | Classify language, domain, complexity, next action |
| `arch_review` | architect | JSON | Rate architecture 1-10, find coupling/missing abstractions |
| `security` | reviewer | JSON | OWASP vulnerability scan with severity + fix |
| `test_gen` | tester | markdown | Generate full test file (happy path + edge + error) |
| `summarize` | summarizer | markdown | Terse technical summary: what it does, API, patterns |
| `fix` | fixer | JSON | Diagnose error + smallest correct patch |
| `doc` | coder | markdown | Add/improve docstrings in native format (Google/JSDoc) |
| `critic` | critic | JSON | Score 8 quality dimensions + critical fix sentence |

### Critic dimensions
`correctness · clarity · structure · testability · security · performance · maintainability · completeness`

Anchor: `<6 = needs work`, `6–8 = acceptable`, `>8 = excellent`. Model is instructed never to exceed 9.0 without strong justification.

---

## Model Trait Matrix

### Ollama model presets

| Model | Context | Best roles | Notes |
|---|---|---|---|
| `llama3.2` | 128k | intent, architect, critic, summarizer | Fast, general-purpose default |
| `llama3.1` | 128k | intent, architect, critic | Stronger reasoning than 3.2 |
| `codellama` | 16k | coder, reviewer, fixer, tester | Meta code specialist |
| `deepseek-coder-v2` | 164k | coder, reviewer, tester | Top code benchmark, long context |
| `qwen2.5-coder` | 131k | coder, reviewer, fixer | Strong instruction-following + code |
| `mistral` | 32k | intent, fixer, summarizer | Fast, low RAM |
| `phi4` | 16k | intent, architect, critic | Punches above its weight class |
| `phi3` | 4k | intent, critic | Smallest capable model |
| `gemma2` | 8k | intent, summarizer | Good multilingual |
| `yi-coder` | 65k | coder, reviewer, tester | 01.AI code model |
| `starcoder2` | 16k | coder, reviewer | Trained exclusively on code |

### Role → model preference order

Models are tried in order; the first one present in your running Ollama instance wins.

```
intent:     llama3.2 → llama3.1 → phi4 → mistral → phi3
architect:  llama3.1 → llama3.2 → phi4 → mistral
coder:      qwen2.5-coder → deepseek-coder-v2 → codellama → yi-coder → starcoder2
reviewer:   deepseek-coder-v2 → qwen2.5-coder → codellama → llama3.1
fixer:      qwen2.5-coder → codellama → mistral → llama3.2
tester:     deepseek-coder-v2 → qwen2.5-coder → codellama → yi-coder
critic:     llama3.1 → llama3.2 → phi4
summarizer: llama3.2 → mistral → gemma2
```

Provider fallback: **Ollama** is tried first (`:11434`). If unavailable, **Jan** (`:1337/v1`) is used automatically via OpenAI-compatible API.

---

## Execution Phases

The orchestrator runs skills in these conceptual phases. Each maps to one or more skills:

| Phase | Skills fired | What happens |
|---|---|---|
| `contextual_analysis` | `intent` | Classify input, detect language/domain/complexity |
| `architecture` | `arch_review` | Structural review, coupling, abstraction gaps |
| `security` | `security` | Vulnerability scan, OWASP categorisation |
| `verification` | `critic` | 8-dimension quality scoring |
| `generation` | `test_gen`, `doc` | Produce test file or documentation |
| `repair` | `fix` | Diagnose error + propose patch |
| `synthesis` | `summarize` | Produce terse technical summary |

Run a single skill: `POST /api/skills/run`
Run a full pipeline: `POST /api/skills/pipeline` (all 8 skills in sequence)

---

## Tools Layer

Real tools available to agents and via the API:

| Tool | What it does |
|---|---|
| `run_shell(cmd)` | Safe subprocess with blocklist (`rm -rf /`, `mkfs`, …) |
| `analyze_python_ast(src)` | stdlib `ast`: functions, classes, decorators, cyclomatic complexity, imports |
| `compute_metrics(src, lang)` | bytes, line counts, comment ratio, cyclomatic-avg, SHA-256, longest line, TODOs |
| `scan_secrets(src)` | 13 regex patterns: AWS keys, JWT, GitHub tokens, Anthropic keys, Stripe, DB URLs, private keys, … |
| `lint_python(src)` | Calls `ruff check --output-format=json` → structured issue list |
| `git_summary(path)` | `git log`, `shortlog`, `diff --stat`, stash count, tags, uncommitted files |
| `analyze_repo(path)` | File walk, language detection, ext breakdown, top-level entries |

---

## Scheduler & Event Bus

### Heartbeat
`HeartbeatMonitor` pings Ollama every 30 seconds and emits:
- `heartbeat` — always: `{ts, ollama: {available, models, error}, system?: {cpu, mem, disk}}`
- `ollama_up` — when Ollama comes back online
- `ollama_down` — when Ollama goes offline

### CronScheduler
Add a recurring skill run via the API or UI:

```bash
curl -X POST http://localhost:8765/api/schedule/add \
  -H "Content-Type: application/json" \
  -d '{"name": "nightly-review", "skill": "critic", "source": "<code>", "interval_s": 3600}'
```

Results are broadcast over WebSocket to all connected clients.

### FileWatcher
Watch a directory for changes (mtime polling, 5s default):

```bash
curl -X POST http://localhost:8765/api/watch \
  -d '{"path": "./src", "interval_s": 5}'
```

Emits `file_changed` events to WebSocket clients.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthz` | Server health + heartbeat status |
| `GET` | `/api/providers` | Ollama + Jan availability + model lists |
| `GET` | `/api/providers/models` | Full model list from both providers |
| `POST` | `/api/analyze/code` | `{source, language?}` → metrics, secrets, lint |
| `POST` | `/api/analyze/repo` | `{path}` → file walk + git summary |
| `POST` | `/api/analyze/shell` | `{cmd, cwd?, timeout?}` → run command safely |
| `GET` | `/api/skills` | List all registered skills |
| `POST` | `/api/skills/run` | `{skill, source, model?, provider?}` → run one skill |
| `POST` | `/api/skills/pipeline` | `{source, skills?, model?}` → run all skills |
| `POST` | `/api/chat` | `{messages, model?, provider?}` → direct LLM chat |
| `GET` | `/api/schedule` | List scheduled jobs |
| `POST` | `/api/schedule/add` | `{name, skill, source, interval_s}` |
| `DELETE` | `/api/schedule/{name}` | Remove a job |
| `POST` | `/api/watch` | `{path, interval_s?}` → start watching |
| `DELETE` | `/api/watch` | `{path}` → stop watching |
| `WS` | `/ws` | Real-time events + inline skill runs |

### WebSocket message types (client → server)
```json
{"type": "ping"}
{"type": "run_skill", "skill": "critic", "source": "...", "model": null}
```

---

## Adding a New Skill

1. Add a `SkillDef` entry to `SKILL_INDEX` in `config.py` (or `tools/inspector.py`):

```python
SKILL_INDEX["my_skill"] = SkillDef(
    name="my_skill",
    description="One line: what it does",
    role="coder",                 # which role-model-preference list to use
    output_format="json",         # "json" | "text" | "markdown"
    system_prompt="...",          # actual LLM system prompt — be specific
    temperature=0.2,
    max_tokens=2048,
)
```

2. That's it — the skill appears automatically in `/api/skills`, the UI Skills tab, and the scheduler.

3. For a custom executor (tool calls, multi-step), subclass or extend `SkillIndex.run()` to intercept by name.

---

## Running

```bash
# single-file mode (easiest)
python tools/inspector.py

# package mode
python -m tools.inspector

# then open in browser
open tools/inspector.html        # or http://localhost:8765/
```

### Prerequisites
- Python 3.10+, `fastapi`, `uvicorn`, `httpx`, `pydantic`, `websockets` (all in `pyproject.toml`)
- [Ollama](https://ollama.com) running: `ollama serve` + `ollama pull llama3.2`
- [Jan](https://jan.ai) (optional fallback): start Jan app → models auto-detected

---

## Glossary

| Term | Meaning |
|---|---|
| **Skill** | A named, prompt-driven capability with a fixed output format |
| **Role** | A functional category (`coder`, `critic`, …) used to select the best model |
| **Trait** | A property of a model or skill that influences behaviour (temperature, ctx window, output format) |
| **Phase** | A logical stage in a multi-step analysis pipeline |
| **EventBus** | Async pub/sub that connects heartbeat, scheduler, watcher, and WebSocket |
| **Provider** | A local LLM server (Ollama or Jan) — inspector auto-detects and routes |
