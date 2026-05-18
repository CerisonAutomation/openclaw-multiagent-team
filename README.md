# openclaw

**Drop in a prompt or a repo — get a built, tested, deployable app.**
A multi-agent orchestrator that wires the SINGULARITY 7-phase critique loop
(`omni.py`) to real tools (shell, fs, Vercel, image-gen) and runs them with
specialized agents — not a single GPT call.

```bash
pip install -e .
openclaw build "a Next.js todo app with dark mode and a /health endpoint"
openclaw fix ./my-repo --task "the tailwind build is broken"
openclaw deploy ./my-repo --prod
openclaw serve --port 8000
```

No API key? Everything still runs end-to-end via the **mock provider** so you can
verify the pipeline before paying for tokens.

```bash
OPENCLAW_PROVIDER=mock openclaw build "anything you want"
```

---

## How it works

```
                  ┌────────── intent ──────────┐
PROMPT  ──────►   │  classify → route agents   │
                  └────────────┬───────────────┘
   or                          ▼
REPO PATH  ───►   architect → coder → tester → fixer → (deployer)
                              │
                              ▼
                  ┌─── critic loop (10 dims) ──┐
                  │  worst < 7.5 → improve     │
                  └────────────┬───────────────┘
                               ▼
                     reality gates  →  SEAL
                          │
                          ▼
                  4-block report + audit.json
```

| Phase | Who runs | What happens |
|-------|----------|--------------|
| 1 FREEZE      | `intent`     | Parse the request; resolve unstated goal; pick route |
| 2 DOMAIN      | `architect`  | Choose the smallest stack; list files + commands |
| 3 ARCHITECT   | gates        | `gate_intent_clarity`, `gate_architecture_plausible` |
| 4 EXECUTE     | `coder`, `tester`, `fixer` | Write files; run commands; patch on failure |
| 5 CRITIQUE    | `critic`     | 10-dim scoring; loop until threshold or max iterations |
| 6 REALITY     | gates        | `gate_no_placeholders`, `gate_quality_threshold` |
| 6.5 DEPLOY    | `deployer`   | Optional: `vercel deploy --token …` |
| 7 SEAL        | orchestrator | 4-block output + audit log (HORUS schema) |

## CLI

```bash
openclaw build [prompt]            # build from a prompt
openclaw fix    ./repo --task ...  # analyze + fix an existing repo
openclaw deploy ./repo --prod      # deploy to Vercel
openclaw analyze ./repo            # repo summary only, no LLM calls
openclaw serve  --port 8000        # start FastAPI server
```

Common flags: `--provider {anthropic|openai|mock}`, `--model <name>`,
`--iterations 3`, `--threshold 7.5`, `--audit-out path.json`, `--quiet`.

## HTTP API (FastAPI)

```bash
openclaw serve --port 8000
```

| Method | Path | Body |
|--------|------|------|
| POST | `/api/build`   | `{"task": "...", "deploy": false, "provider": "anthropic"}` |
| POST | `/api/fix`     | multipart: zip upload + form field `task`, OR `repo_path` |
| POST | `/api/analyze` | `{"repo_path": "/path/to/repo"}` |
| GET  | `/api/audit/{session_id}` | fetch a past run |
| GET  | `/healthz`     | health check |

Example:

```bash
curl -X POST http://localhost:8000/api/build \
  -H 'content-type: application/json' \
  -d '{"task":"build a Next.js todo app","provider":"mock"}'
```

## Universal LLM provider

Set `OPENCLAW_PROVIDER` and the matching env vars. OpenAI mode works with any
OpenAI-compatible endpoint (NVIDIA NIM, OpenRouter, DeepSeek, Together, Groq,
Fireworks, vLLM, Ollama via `/v1`).

```bash
# Anthropic
export OPENCLAW_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export OPENCLAW_MODEL=claude-sonnet-4-20250514   # optional

# OpenAI
export OPENCLAW_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENCLAW_MODEL=gpt-4o-mini

# OpenRouter (or any OpenAI-compatible gateway)
export OPENCLAW_PROVIDER=openai
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export OPENCLAW_MODEL=anthropic/claude-3.5-sonnet

# Local / no key
export OPENCLAW_PROVIDER=mock
```

## Audit log (HORUS schema)

Every run writes `openclaw_audit.json`:

```json
{
  "session_id": "2026-05-18T07:30:00+00:00",
  "task": "...",
  "input_rewrite": "...",
  "intent": { "primary_intent": "build", "complexity": "moderate", ... },
  "architecture": { "framework": "nextjs", "files_to_create": [...] },
  "active_agents": ["intent", "architect", "coder", "tester", "critic"],
  "tool_calls": [
    {"tool": "write_file", "args": {"path": "..."}, "ok": true, "at": "..."}
  ],
  "critique_history": [
    {"iteration": 1, "avg": 7.9, "worst": "originality", "worst_score": 6.8}
  ],
  "final_scores": { "clarity": 9.1, ... },
  "validation": {
    "real_world_deployable": true,
    "domain_aligned": true,
    "structure_locked": true
  },
  "seal": "SEAL: deployed 8 files → live at https://x.vercel.app"
}
```

## Architecture in one diagram

```
openclaw/
├── providers.py    # Anthropic + OpenAI-compatible + Mock
├── prompts.py      # System prompts for every agent role
├── audit.py        # HORUS-schema audit log
├── gates.py        # Three-stage (pre/mid/post) gate functions
├── tools.py        # shell, fs, vercel, repo-analysis, image-gen
├── agents.py       # Per-agent specs + invocations + routing table
├── orchestrator.py # 7-phase pipeline tying it all together
├── cli.py          # `openclaw build|fix|deploy|analyze|serve`
└── server.py       # FastAPI endpoints
```

## What was distilled from the source archives

| Source | What it contributed |
|--------|---------------------|
| `omni.py` (your own) | 7-phase execution, 10-dimension critique, intent classifier |
| `OMNIPERFECTMETAWEAVER_FINAL` | Execution-freeze concept, repair triggers |
| `omnifinisher_supreme`        | 10 quality dimensions, loop-until-finality |
| `Omega_Intent_Mesh`           | Three-stage gates (pre/mid/post), weighted intent routing |
| `HORUS_OMNIFUSION_ONEFLOW`    | Per-agent `{purpose, traits, checks, on_fail}` schema |
| `HORUS_XTRILLINITY` rewrite    | Audit schema: `input_rewrite`/`primary_output`/`optional_alt` |
| `HORUS_JSON_BROWSER`          | `AgentLimits` (max time, restricted commands, allowed managers) |
| `omnifinisher_brutalcore`     | 4-block output contract (Executive / Developer / User / Meta) |
| `ΩCreateGPT_Gizmo_CORRECTED`  | Self-repair trigger table (drift / entropy / audit failure) |

Everything theatrical was dropped. Every pattern kept was implemented as
runnable Python with tests.

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

The whole suite runs against `MockProvider` — no API key required, no network.

## License

MIT.
