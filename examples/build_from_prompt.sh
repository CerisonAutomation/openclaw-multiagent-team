#!/usr/bin/env bash
# Example: build a small Next.js app from a prompt.
# Uses mock provider by default so you can verify the pipeline without API keys.
#
# To use a real model:
#   export OPENCLAW_PROVIDER=anthropic; export ANTHROPIC_API_KEY=sk-...
# or
#   export OPENCLAW_PROVIDER=openai;    export OPENAI_API_KEY=sk-...
set -euo pipefail
: "${OPENCLAW_PROVIDER:=mock}"
export OPENCLAW_PROVIDER

openclaw build \
  --workspace ./build_out \
  --iterations 2 \
  --threshold 7.5 \
  --audit-out ./build_audit.json \
  "Build a Next.js todo app with a landing page, dark-mode toggle, and a /health endpoint."
