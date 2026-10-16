#!/usr/bin/env bash
# Remove the launchd backup runner.  Keeps the clone and the log unless you pass --purge.
# Safe to run when nothing is installed (idempotent).
set -euo pipefail

LABEL="com.dso429.quizbot"
BASE="$HOME/Library/Application Support/dso429-quizbot"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/dso429-quizbot.log"
UID_NUM="$(id -u)"

launchctl bootout "gui/$UID_NUM/$LABEL" >/dev/null 2>&1 || true
rm -f "$PLIST"
echo "Removed $LABEL."

if [ "${1:-}" = "--purge" ]; then
  rm -rf "$BASE"
  rm -f "$LOG"
  echo "Deleted $BASE and $LOG."
else
  echo "Kept $BASE and $LOG (use --purge to delete them)."
fi
