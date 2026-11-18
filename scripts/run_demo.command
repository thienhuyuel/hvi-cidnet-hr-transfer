#!/bin/zsh
set -e
TASK_REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TASK_REPO_DIR"
exec .venv/bin/python -m upstream.app --cpu
