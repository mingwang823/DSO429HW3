#!/usr/bin/env bash
# One backup run of the quiz bot, started by launchd (see install.sh).
# Works in a dedicated clone, so `git reset --hard` can never touch your own working folder.
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export QUIZBOT_ALLOW_RESET=1

BASE="${QUIZBOT_BASE:-$HOME/Library/Application Support/dso429-quizbot}"
CLONE="${QUIZBOT_CLONE:-$BASE/repo}"
LOCK="$BASE/run.lock"

mkdir -p "$BASE"

# Skip if another run is still going; a lock older than 20 minutes is treated as stale.
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +20 2>/dev/null)" ]; then
  rmdir "$LOCK" 2>/dev/null || true
fi
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) another run is in progress; skipping"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

echo "=== $(date -u +%FT%TZ) quizbot local run ==="
if [ ! -f "$CLONE/run.py" ]; then
  echo "no run.py in $CLONE yet (has the repo been pushed to GitHub?)"
  exit 0
fi
cd "$CLONE"
python3 run.py --git
