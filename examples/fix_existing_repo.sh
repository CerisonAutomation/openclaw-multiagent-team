#!/usr/bin/env bash
# Example: point openclaw at an existing repo and ask it to fix something.
set -euo pipefail
: "${OPENCLAW_PROVIDER:=mock}"
export OPENCLAW_PROVIDER

REPO="${1:-./my-repo}"
openclaw fix "$REPO" \
  --task "Identify the most important bug, then fix it with the smallest possible change." \
  --iterations 3 \
  --audit-out ./fix_audit.json
