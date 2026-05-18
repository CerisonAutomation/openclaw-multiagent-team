# openclaw

**Drop in a prompt or a repo — get a built, tested, deployable app.**
A multi-agent orchestrator that wires the SINGULARITY 7-phase critique loop
to real tools (shell, fs, Vercel, image-gen) with **specialized agents** and
**per-role model routing** across any LLM provider.

## 60-second quickstart

```bash
# 1. install (one option)
pipx install openclaw                                      # recommended
pip install -e .                                           # from source
docker run --rm -p 8000:8000 ghcr.io/cerisonautomation/openclaw  # zero-install

# 2. point at a provider (any one — auto-detects from env)
export OPENROUTER_API_KEY=sk-or-...      # FREE high-end models, default
# or
export NVIDIA_NIM_API_KEY=nvapi-...      # high-end Nemotron / Qwen / Llama

# 3. go
openclaw build "a Next.js todo app with dark mode"
openclaw fix ./my-repo --task "the tailwind build is broken"
openclaw deploy ./my-repo --prod
openclaw serve --port 8000
```

**No keys?** Everything still runs end-to-end through the mock provider so you
can verify the pipeline before paying for tokens:

```bash
OPENCLAW_PROVIDER=mock openclaw build "anything"
```

## Provider presets — `openclaw providers`

Each preset bundles a base URL, an env-var name, and per-role model defaults
(cost-aware: cheap fast model for intent/critic, smart model for coder/fixer —
inspired by [agentic-flow's](https://github.com/ruvnet/agentic-flow)
Sonnet↔Haiku routing).

| Preset       | Default model                                | Notes |
|--------------|----------------------------------------------|-------|
| `openrouter` | `deepseek/deepseek-chat-v3-0324:free`        | **Default.** Free tier, high-end models across all roles (DeepSeek V3 free + Llama 3.3 70B free). |
| `nvidia`     | `nvidia/llama-3.1-nemotron-70b-instruct`     | **Default.** High-end NIM hosted (Nemotron 70B, Llama 3.3 70B, Qwen Coder 32B). |
| `anthropic`  | `claude-sonnet-4-20250514`                   | Sonnet for code, Haiku for cheap roles. |
| `openai`     | `gpt-4o-mini`                                | OpenAI direct. |
| `groq`       | `llama-3.3-70b-versatile`                    | Fastest tokens/sec. |
| `deepseek`   | `deepseek-chat`                              | Cheapest. |
| `ollama`     | `llama3.1:latest`                            | Local. No cost. No network. |
| `mock`       | `mock-1`                                     | Offline dry-run for CI / smoke tests. |

Auto-detect tries them in the order shown (so `openrouter` and `nvidia` win
when both are set). Override anything:

```bash
openclaw build "..."  --provider groq  --model llama-3.3-70b-versatile
export OPENCLAW_MODEL_CRITIC=anthropic/claude-3.5-haiku   # per-role override
export OPENCLAW_PROVIDER=openrouter                        # pin a provider globally
```

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
openclaw providers                 # list provider presets and their models
```

Common flags: `--provider <preset>`, `--model <name>`, `--iterations 3`,
`--threshold 7.5`, `--audit-out path.json`, `--quiet`.

## HTTP API (FastAPI)

```bash
openclaw serve --port 8000
```

| Method | Path | Body |
|--------|------|------|
| POST | `/api/build`   | `{"task": "...", "deploy": false, "provider": "openrouter"}` |
| POST | `/api/fix`     | multipart: zip upload + form field `task`, OR `repo_path` |
| POST | `/api/analyze` | `{"repo_path": "/path/to/repo"}` |
| GET  | `/api/audit/{session_id}` | fetch a past run |
| GET  | `/healthz`     | health check |

Example:

```bash
curl -X POST http://localhost:8000/api/build \
  -H 'content-type: application/json' \
  -d '{"task":"build a Next.js todo app","provider":"openrouter"}'
```

## Docker

```bash
# One-shot run (build the image once, use it forever)
docker build -t openclaw .
docker run --rm -p 8000:8000 \
  -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  -e OPENCLAW_PROVIDER=openrouter \
  openclaw

# Or use as a CLI inside the container:
docker run --rm -it \
  -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  -v "$PWD:/work" -w /work \
  openclaw build "a static landing page"
```

## Audit log (HORUS schema)

Every run writes `openclaw_audit.json`:

```json
{
  "session_id": "2026-05-18T...",
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
├── models.py       # Provider presets registry + per-role model routing
├── providers.py    # Anthropic + OpenAI-compatible + Mock
├── prompts.py      # System prompts for every agent role
├── audit.py        # HORUS-schema audit log
├── gates.py        # Three-stage (pre/mid/post) gate functions
├── tools.py        # shell, fs, vercel, repo-analysis, image-gen
├── agents.py       # Per-agent specs + invocations + routing table
├── orchestrator.py # 7-phase pipeline tying it all together
├── cli.py          # `openclaw build|fix|deploy|analyze|serve|providers`
└── server.py       # FastAPI endpoints
```

## What was distilled from the source archives

| Source | Contribution |
|--------|-------------|
| `omni.py` (your own)             | 7-phase execution, 10-dimension critique, intent classifier |
| `OMNIPERFECTMETAWEAVER_FINAL`    | Execution-freeze concept, repair triggers |
| `omnifinisher_supreme`           | 10 quality dimensions, loop-until-finality |
| `Omega_Intent_Mesh`              | Three-stage gates (pre/mid/post), weighted intent routing |
| `HORUS_OMNIFUSION_ONEFLOW`       | Per-agent `{purpose, traits, checks, on_fail}` schema |
| `HORUS_XTRILLINITY` rewrite      | Audit schema: `input_rewrite`/`primary_output`/`optional_alt` |
| `HORUS_JSON_BROWSER`             | `AgentLimits` (max time, restricted commands) |
| `omnifinisher_brutalcore`        | 4-block output contract (Executive / Developer / User / Meta) |
| `ΩCreateGPT_Gizmo_CORRECTED`     | Self-repair trigger table (drift / entropy / audit failure) |
| ruvnet `agentic-flow`            | Per-role cost-aware model routing (Sonnet↔Haiku pattern) |

Everything theatrical was dropped. Every pattern kept was implemented as
runnable Python with tests.

## Testing

```bash
pip install -e ".[dev]"
pytest -q
# 28 passed
```

The whole suite runs against `MockProvider` — no API key required, no network.

## License

MIT.
