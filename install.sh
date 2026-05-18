#!/usr/bin/env bash
# openclaw one-shot installer
# Usage:  curl -fsSL https://raw.githubusercontent.com/cerisonautomation/openclaw-multiagent-team/main/install.sh | bash
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

echo -e "${BOLD}openclaw installer${RESET}"
echo ""

# ── 1. Ensure Python 3.10+ ───────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "error: python3 not found. Install Python 3.10+ first." >&2
    exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    echo "error: Python 3.10+ required (found $PY_VER)" >&2
    exit 1
fi
echo -e "${GREEN}✓${RESET} Python $PY_VER"

# ── 2. Ensure pipx ───────────────────────────────────────────────────────────
if ! command -v pipx &>/dev/null; then
    echo -e "${YELLOW}→${RESET} Installing pipx..."
    python3 -m pip install --user --quiet pipx
    python3 -m pipx ensurepath
    export PATH="$HOME/.local/bin:$PATH"
fi
echo -e "${GREEN}✓${RESET} pipx $(pipx --version)"

# ── 3. Install openclaw ──────────────────────────────────────────────────────
echo -e "${YELLOW}→${RESET} Installing openclaw[all]..."
pipx install --force "openclaw[all]" 2>/dev/null || \
    pip install --user --quiet "openclaw[all]"
echo -e "${GREEN}✓${RESET} openclaw installed"

# ── 4. Detect existing keys ──────────────────────────────────────────────────
echo ""
echo "API key check:"
FOUND=0
for VAR in OPENROUTER_API_KEY NVIDIA_NIM_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY GROQ_API_KEY DEEPSEEK_API_KEY; do
    if [[ -n "${!VAR:-}" ]]; then
        echo -e "  ${GREEN}✓${RESET} $VAR is set"
        FOUND=1
    fi
done
if [[ $FOUND -eq 0 ]]; then
    echo "  No API keys found. Add one to your shell profile, e.g.:"
    echo "    export OPENROUTER_API_KEY=sk-or-..."
    echo "    export NVIDIA_NIM_API_KEY=nvapi-..."
    echo ""
    echo "  Or test with the offline mock:"
    echo "    OPENCLAW_PROVIDER=mock openclaw build 'a todo app'"
fi

# ── 5. Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Ready. Try:${RESET}"
echo "  openclaw setup                       # configure + create .openclaw.toml"
echo "  openclaw providers                   # list all providers"
echo "  openclaw build 'a Next.js todo app'  # build from prompt"
echo "  openclaw auto                        # detect + run from current directory"
