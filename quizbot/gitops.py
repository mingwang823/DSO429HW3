"""Commit/push flow for ``run.py --git`` and GitHub issue notification.

Two phases, so that LLM latency can never delay a rule answer:

* phase 1 - rule answers (``guess`` placeholders for what the rules cannot solve) are
  written, committed ("Answer round 49") and pushed first;
* phase 2 - the LLM upgrade pass runs on top and, if it improved anything, a second
  commit ("Update round 49") is pushed.

Each phase is up to 5 attempts (5/10/20/30 s backoff between attempts):

1. sync with ``origin/main``: ``fetch`` + ``reset --hard`` when ``QUIZBOT_ALLOW_RESET=1``
   (CI and the dedicated local backup clone), else ``pull --rebase``
   (phase 2 skips this on its first attempt: the tree was just pushed)
2. run the pipeline (phase 1: rules only; phase 2: with the LLM)
3. open GitHub issues for rounds with non-rule answers (Actions only, never fatal;
   after phase 2 when there is one, so they show the final answers)
4. nothing changed under ``answers state STATUS.md`` -> done
5. commit + push; if the push is rejected, start over at 1
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import pipeline
from .source import SourceError, ssl_context

log = logging.getLogger("quizbot.gitops")

BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"
MAX_ATTEMPTS = 5
BACKOFFS = (5, 10, 20, 30)
TRACKED_PATHS = ("answers", "state", "STATUS.md")
GIT_TIMEOUT = 120
ISSUE_TIMEOUT = 20
MAX_ISSUES_PER_RUN = 10


# --------------------------------------------------------------------------
# commit message
# --------------------------------------------------------------------------
def format_round_list(ids: Sequence[int]) -> str:
    """[49, 50] -> "49, 50"; runs of 3+ consecutive ids collapse ("1-48")."""
    ids = sorted(set(ids))
    parts: List[str] = []
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append("%d-%d" % (ids[i], ids[j]))
        else:
            parts.extend(str(x) for x in ids[i:j + 1])
        i = j + 1
    return ", ".join(parts)


def _noun(ids: Sequence[int]) -> str:
    return "round" if len(set(ids)) == 1 else "rounds"


def commit_message(new_ids: Sequence[int], updated_ids: Sequence[int]) -> str:
    """"Answer round 49" / "Answer rounds 49, 50" / "Update round 49" (both: joined by "; ")."""
    parts = []
    if new_ids:
        parts.append("Answer %s %s" % (_noun(new_ids), format_round_list(new_ids)))
    if updated_ids:
        parts.append("Update %s %s" % (_noun(updated_ids), format_round_list(updated_ids)))
    if not parts:
        return "Update quiz state"
    return "; ".join(parts)


# --------------------------------------------------------------------------
# git wrapper
# --------------------------------------------------------------------------
class Git:
    def __init__(self, root: str, runner: Optional[Callable] = None):
        self.root = root
        self._runner = runner

    def run(self, *args: str, timeout: int = GIT_TIMEOUT) -> "subprocess.CompletedProcess":
        runner = self._runner or subprocess.run
        cmd = ["git"] + list(args)
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            proc = runner(cmd, cwd=self.root, capture_output=True, text=True, timeout=timeout, env=env)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("git %s failed to run: %s", " ".join(args), exc)
            return subprocess.CompletedProcess(cmd, 127, "", str(exc))
        return proc


def _identity_args(git: Git, env: Mapping[str, str]) -> List[str]:
    bot = ["-c", "user.name=%s" % BOT_NAME, "-c", "user.email=%s" % BOT_EMAIL]
    if env.get("GITHUB_ACTIONS") == "true":
        return bot
    name = git.run("config", "user.name").stdout.strip()
    email = git.run("config", "user.email").stdout.strip()
    if name and email:
        return []
    return bot


def _sync(git: Git, allow_reset: bool) -> bool:
    if allow_reset:
        steps = [("fetch", "origin", "main"), ("reset", "--hard", "origin/main")]
        for step in steps:
            proc = git.run(*step)
            if proc.returncode != 0:
                log.warning("git %s failed: %s", " ".join(step), (proc.stderr or "").strip()[:300])
                return False
        return True
    log.info("QUIZBOT_ALLOW_RESET is not 1: refusing to reset, using pull --rebase")
    proc = git.run("pull", "--rebase", "--autostash", "origin", "main")
    if proc.returncode != 0:
        log.warning("git pull --rebase failed: %s", (proc.stderr or "").strip()[:300])
        git.run("rebase", "--abort")
        return False
    return True


def _unpushed_commits(git: Git) -> int:
    proc = git.run("rev-list", "--count", "origin/main..HEAD")
    try:
        return int((proc.stdout or "0").strip() or 0) if proc.returncode == 0 else 0
    except ValueError:
        return 0


# --------------------------------------------------------------------------
# GitHub issues
# --------------------------------------------------------------------------
def _github_request(method: str, url: str, token: str, body: Optional[dict] = None, timeout: int = ISSUE_TIMEOUT):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer %s" % token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "dso429-quizbot",
        **({"Content-Type": "application/json"} if data is not None else {}),
    })
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def issue_candidates(state: dict, rounds: Mapping[int, object]) -> Dict[int, List[int]]:
    """{round_id: [q_index,...]} for rounds with non-rule answers and no issue yet."""
    out: Dict[int, List[int]] = {}
    for key, entry in state.items():
        if not str(key).isdigit() or entry.get("issue_opened"):
            continue
        rid = int(key)
        if rid not in rounds:
            continue
        methods = entry.get("methods") or {}
        bad = []
        for i in pipeline.ALL_INDEXES:
            m = str(methods.get("q%d" % i, ""))
            if m.startswith("rule") or m.startswith("guess:missing"):
                continue
            bad.append(i)
        if bad:
            out[rid] = bad
    return out


def issue_title(round_id: int, count: int) -> str:
    return "Round %d: %d question(s) not solved by rules" % (round_id, count)


def issue_body(rnd, entry: dict, indexes: Sequence[int], answers: Mapping[str, str]) -> str:
    methods = entry.get("methods") or {}
    details = entry.get("details") or {}
    lines = [
        "The bot answered these questions of round %d without a deterministic rule. "
        "Answers were already pushed; please double-check them before the 3 h window closes." % rnd.round_id,
        "",
    ]
    for i in indexes:
        q = rnd.questions.get(i)
        lines.append("### q%d" % i)
        if q is None:
            lines.append("(question missing from the source CSV)")
        else:
            lines.append(q.text)
            lines.append("")
            for letter in pipeline.LETTERS:
                lines.append("- %s) %s" % (letter, q.options.get(letter, "")))
        lines.append("")
        lines.append("Chosen: **%s**  |  method: `%s`" % (answers.get("q%d" % i, "?"), methods.get("q%d" % i, "?")))
        if details.get("q%d" % i):
            lines.append("")
            lines.append("Detail: %s" % details["q%d" % i])
        lines.append("")
    return "\n".join(lines)


def notify_issues(root: str, rounds: Mapping[int, object], env: Mapping[str, str],
                  request: Optional[Callable] = None) -> List[int]:
    """Open one issue per round that has non-rule answers.  Never raises.

    Only in GitHub Actions with a token.  A round is never notified twice:
    ``issue_opened`` in state/rounds.json, plus a title check against existing issues
    (covers a run whose state commit never made it to the remote).
    """
    try:
        if env.get("GITHUB_ACTIONS") != "true":
            return []
        token = (env.get("GITHUB_TOKEN") or env.get("GH_TOKEN") or "").strip()
        repo = (env.get("GITHUB_REPOSITORY") or "").strip()
        if not token or not repo:
            return []
        request = request or _github_request
        state = pipeline.load_state(root)
        todo = issue_candidates(state, rounds)
        if not todo:
            return []
        try:
            listing = request("GET", "https://api.github.com/repos/%s/issues?state=all&per_page=100" % repo, token)
            existing_titles = [i.get("title", "") for i in listing if isinstance(i, dict)]
        except Exception as exc:
            log.warning("could not list existing issues (%s); relying on state only", exc)
            existing_titles = []
        opened: List[int] = []
        dirty = False
        for rid in sorted(todo)[:MAX_ISSUES_PER_RUN]:
            entry = state[str(rid)]
            if any(t.startswith("Round %d: " % rid) for t in existing_titles):
                log.info("round %d: an issue already exists; marking it as opened", rid)
                entry["issue_opened"] = True
                dirty = True
                continue
            letters = _letters_for(root, rid)
            title = issue_title(rid, len(todo[rid]))
            body = issue_body(rounds[rid], entry, todo[rid], letters)
            try:
                request("POST", "https://api.github.com/repos/%s/issues" % repo, token, {"title": title, "body": body})
            except Exception as exc:
                log.warning("round %d: could not open issue (%s)", rid, exc)
                continue
            log.info("round %d: opened issue %r", rid, title)
            entry["issue_opened"] = True
            dirty = True
            opened.append(rid)
        if dirty:
            pipeline.save_state(root, state)
        return opened
    except Exception as exc:
        log.warning("issue notification failed: %s: %s", type(exc).__name__, exc)
        return []


def _letters_for(root: str, rid: int) -> Dict[str, str]:
    try:
        with open(pipeline.answer_path(root, rid), "r", encoding="utf-8") as fh:
            return dict(json.load(fh).get("answers", {}))
    except (OSError, ValueError, AttributeError):
        return {}


# --------------------------------------------------------------------------
# the flow
# --------------------------------------------------------------------------
def _commit_and_push(git: Git, result: "pipeline.PipelineResult", env: Mapping[str, str],
                     attempt: int) -> Optional[bool]:
    """Commit what changed under TRACKED_PATHS and push.

    Returns True when everything is pushed (or there was nothing to push), False when the
    push was rejected/failed (the caller re-syncs and retries), None on a local git failure.
    """
    status = git.run("status", "--porcelain", "--", *TRACKED_PATHS)
    if status.returncode != 0:
        log.error("git status failed: %s", (status.stderr or "").strip()[:300])
        return None
    dirty = bool((status.stdout or "").strip())
    if not dirty and _unpushed_commits(git) == 0:
        log.info("nothing to commit")
        return True
    if dirty:
        message = commit_message(result.new_rounds, result.updated_rounds)
        add = git.run("add", "-A", "--", *TRACKED_PATHS)
        if add.returncode != 0:
            log.error("git add failed: %s", (add.stderr or "").strip()[:300])
            return None
        commit = git.run(*(_identity_args(git, env) + ["commit", "-m", message]))
        if commit.returncode != 0:
            log.error("git commit failed: %s", ((commit.stderr or "") + (commit.stdout or "")).strip()[:300])
            return None
        log.info("committed: %s", message)
    push = git.run("push", "origin", "HEAD:main")
    if push.returncode == 0:
        log.info("pushed to origin/main")
        return True
    log.warning("push rejected/failed (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, (push.stderr or "").strip()[:300])
    return False


def _push_loop(git: Git, step: Callable[[], "pipeline.PipelineResult"], *, env: Mapping[str, str],
               allow_reset: bool, sleep: Callable[[float], None], issue_request: Optional[Callable],
               open_issues: bool, sync_first: bool, root: str) -> Tuple[int, Optional["pipeline.PipelineResult"]]:
    """Up to MAX_ATTEMPTS x (sync, ``step()``, issues, commit, push).  Returns (exit code, last result)."""
    result = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            delay = BACKOFFS[min(attempt - 2, len(BACKOFFS) - 1)]
            log.info("attempt %d/%d in %d s", attempt, MAX_ATTEMPTS, delay)
            sleep(delay)
        if (attempt > 1 or sync_first) and not _sync(git, allow_reset):
            continue
        try:
            result = step()
        except SourceError as exc:
            log.error("cannot get the questions: %s", exc)
            return 1, result
        if open_issues:
            notify_issues(root, result.rounds, env, issue_request)
        outcome = _commit_and_push(git, result, env, attempt)
        if outcome:
            return 0, result
    log.error("giving up after %d attempts", MAX_ATTEMPTS)
    return 1, result


def run_git_flow(pipeline_fn: Callable[[], "pipeline.PipelineResult"], *, root: str,
                 env: Optional[Mapping[str, str]] = None, runner: Optional[Callable] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 issue_request: Optional[Callable] = None,
                 llm_pass_fn: Optional[Callable[["pipeline.PipelineResult"], "pipeline.PipelineResult"]] = None) -> int:
    """sync -> ``pipeline_fn()`` -> commit -> push, then (optionally) the slow LLM upgrade pass.

    Phase 1 must contain nothing slow: ``pipeline_fn`` writes the rule answers (with ``guess``
    placeholders for questions the rules could not solve) and they are pushed straight away.
    Phase 2, only when ``llm_pass_fn`` is given, calls ``llm_pass_fn(phase_1_result)`` - the
    pipeline again, now with the LLM - and pushes a second commit ("Update round N") if it
    changed anything.  An LLM that is slow or broken therefore never delays the rule answers.
    Without ``llm_pass_fn`` the whole job is phase 1 (``pipeline_fn`` may use the LLM itself).
    """
    env = os.environ if env is None else env
    allow_reset = env.get("QUIZBOT_ALLOW_RESET") == "1"
    git = Git(root, runner)
    common = dict(env=env, allow_reset=allow_reset, sleep=sleep, issue_request=issue_request, root=root)

    # Issues list the final letters and methods, so with an LLM pass they wait for it.
    rc, first = _push_loop(git, pipeline_fn, open_issues=llm_pass_fn is None, sync_first=True, **common)
    if rc != 0 or llm_pass_fn is None or first is None:
        return rc

    def llm_step() -> "pipeline.PipelineResult":
        return llm_pass_fn(first)

    try:
        # The tree is in sync with origin right after phase 1, so the first attempt skips the sync.
        rc2, _second = _push_loop(git, llm_step, open_issues=True, sync_first=False, **common)
    except Exception as exc:  # the rule answers are already pushed; report the crash but stay loud
        log.exception("LLM upgrade pass crashed: %s: %s", type(exc).__name__, exc)
        return 1
    return rc2
