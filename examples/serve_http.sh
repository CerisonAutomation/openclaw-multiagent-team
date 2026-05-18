#!/usr/bin/env bash
# Start the FastAPI server. Hit it with:
#   curl -X POST http://localhost:8000/api/build \
#     -H 'content-type: application/json' \
#     -d '{"task":"build a Next.js todo app","provider":"mock"}'
set -euo pipefail
: "${OPENCLAW_PROVIDER:=mock}"
export OPENCLAW_PROVIDER

openclaw serve --host 0.0.0.0 --port 8000 --reload
