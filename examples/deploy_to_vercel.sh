#!/usr/bin/env bash
# Example: deploy a repo to Vercel.
# Requires the vercel CLI (`npm i -g vercel`) and a VERCEL_TOKEN.
set -euo pipefail
: "${OPENCLAW_PROVIDER:=mock}"
export OPENCLAW_PROVIDER

REPO="${1:-./my-repo}"
openclaw deploy "$REPO" --prod --audit-out ./deploy_audit.json
