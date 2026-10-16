#!/usr/bin/env bash
# Install the optional macOS launchd backup runner for the quiz bot.
#
#   * clones the repo into ~/Library/Application Support/dso429-quizbot/repo (a SEPARATE clone,
#     so `git reset --hard` can never touch your working folder, and macOS does not ask for
#     Documents-folder permission)
#   * installs ~/Library/LaunchAgents/com.dso429.quizbot.plist: every 10 minutes and at login
#   * logs to ~/Library/Logs/dso429-quizbot.log
#
# Safe to run again (idempotent).  Undo with uninstall.sh.
set -euo pipefail

LABEL="com.dso429.quizbot"
REPO_URL="${QUIZBOT_REPO_URL:-https://github.com/mingwang823/DSO429HW3.git}"
BASE="$HOME/Library/Application Support/dso429-quizbot"
CLONE="$BASE/repo"
RUNNER="$BASE/run_local.sh"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/dso429-quizbot.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UID_NUM="$(id -u)"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This installer is for macOS (launchd)." >&2
  exit 1
fi
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }

mkdir -p "$BASE" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

# 1) dedicated clone
if [ -d "$CLONE/.git" ]; then
  echo "Updating existing clone in $CLONE"
  git -C "$CLONE" fetch origin main
  git -C "$CLONE" reset --hard origin/main
else
  echo "Cloning $REPO_URL into $CLONE"
  git clone "$REPO_URL" "$CLONE"
fi
if [ ! -f "$CLONE/run.py" ]; then
  echo "WARNING: run.py is not in the clone yet - push the bot to GitHub first." >&2
  echo "         The agent is installed anyway and will start working once it is there." >&2
fi

# 2) runner script (a stable copy outside the clone; the clone is hard-reset on every run)
install -m 755 "$SCRIPT_DIR/run_local.sh" "$RUNNER"

# 3) launchd agent
cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$RUNNER</string>
  </array>
  <key>StartInterval</key>
  <integer>600</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>QUIZBOT_ALLOW_RESET</key>
    <string>1</string>
  </dict>
  <key>StandardOutPath</key>
  <string>$LOG</string>
  <key>StandardErrorPath</key>
  <string>$LOG</string>
</dict>
</plist>
PLISTEOF
plutil -lint "$PLIST" >/dev/null

# 4) (re)load it
launchctl bootout "gui/$UID_NUM/$LABEL" >/dev/null 2>&1 || true
sleep 1
launchctl bootstrap "gui/$UID_NUM" "$PLIST"

echo
echo "Installed $LABEL (runs every 10 minutes and at login)."
echo "  log:   $LOG"
echo "  clone: $CLONE"
echo "Pushing needs git credentials that work without a prompt (e.g. 'gh auth login' then 'gh auth setup-git')."
echo "Check:  launchctl print gui/$UID_NUM/$LABEL | head"
