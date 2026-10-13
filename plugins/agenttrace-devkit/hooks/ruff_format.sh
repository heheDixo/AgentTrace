#!/usr/bin/env bash
# PostToolUse hook: run `ruff format` on the file an Edit/Write just touched, if it is Python.
# Reads the hook payload (JSON) on stdin; a non-.py path is a no-op. Never fails the tool call.
# Runs from the project working directory, so it prefers that project's own .venv ruff.
set -euo pipefail

f="$(python3 -c 'import json,sys; print((json.load(sys.stdin).get("tool_input") or {}).get("file_path",""))' 2>/dev/null || true)"

case "$f" in
  *.py)
    if [ -x ".venv/bin/ruff" ]; then
      ruff=".venv/bin/ruff"
    else
      ruff="ruff"
    fi
    "$ruff" format "$f" >/dev/null 2>&1 || true
    ;;
esac
