#!/usr/bin/env bash
# PreToolUse hook: require confirmation before editing the load-bearing schema.py.
# Reads the hook payload (JSON) on stdin. For an edit to src/agenttrace/schema.py it emits an
# "ask" permission decision so Claude Code prompts the user; every other path is left untouched
# (no output => the normal permission flow proceeds).
set -euo pipefail

f="$(python3 -c 'import json,sys; print((json.load(sys.stdin).get("tool_input") or {}).get("file_path",""))' 2>/dev/null || true)"

case "$f" in
  */src/agenttrace/schema.py|src/agenttrace/schema.py)
    printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"schema.py is the load-bearing boundary (RunConfig/Trajectory/Step/Outcome/EvalResult) that every evaluator, the judge, benchmarks, and the report depend on. Confirm before editing."}}'
    ;;
esac
