"""Decide which rounds to (re)answer, solve them, and write answers/state/STATUS.md.

Decision rules per round present in the source CSV (``now`` is injectable):

* no answer file                       -> solve everything and write it (even when late)
* file + state entry + still editable  -> re-solve only questions whose recorded
                                          method ranks below "rule" (or everything
                                          if the question content hash changed);
                                          a method is never downgraded
* file but no state entry              -> untouched (could be a manual edit)
* file + state entry, not editable     -> untouched
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import quizbot
from .llm import LLMAnswer, LLMItem
from .source import (
    NUM_QUESTIONS,
    Round,
    format_timestamp,
    parse_questions_csv,
    parse_timestamp,
)

log = logging.getLogger("quizbot.pipeline")

WINDOW = timedelta(hours=3)
EDIT_MARGIN = timedelta(minutes=10)
EDIT_WINDOW = WINDOW - EDIT_MARGIN  # editable while now < released_at + 2h50m
LETTERS = ("A", "B", "C", "D")
RULE_RANK = 3
METHOD_RANK = {"rule": 3, "llm": 2, "llm-disagree": 2, "guess": 1}
QKEYS = ["q%d" % i for i in range(1, NUM_QUESTIONS + 1)]
ALL_INDEXES = list(range(1, NUM_QUESTIONS + 1))
DETAIL_LIMIT = 300
STATUS_ROWS = 15
TMP_PREFIX = ".tmp-"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def method_rank(method: Optional[str]) -> int:
    """rule=3, llm/llm-disagree=2, guess=1, anything else (or empty) 0."""
    if not method:
        return 0
    return METHOD_RANK.get(method.split(":", 1)[0], 0)


def is_editable(now: datetime, released_at: datetime) -> bool:
    return now < released_at + EDIT_WINDOW


@dataclass(frozen=True)
class Answer:
    letter: str
    method: str
    detail: str = ""

    @property
    def rank(self) -> int:
        return method_rank(self.method)


@dataclass(frozen=True)
class _Solution:
    letter: Optional[str]
    method: str
    detail: str = ""


def default_solver() -> Callable:
    """The real rule solver, imported lazily; abstains on everything if missing."""
    try:
        from .solvers import solve_question
        return solve_question
    except Exception as exc:  # ImportError or a bug in the solver module
        reason = "%s: %s" % (type(exc).__name__, exc)
        log.error("rule solver unavailable (%s); everything falls to LLM/guess", reason)

        def _missing(question_text, options):
            return _Solution(None, "abstain:solver_unavailable", reason)

        return _missing


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= DETAIL_LIMIT else text[: DETAIL_LIMIT - 3] + "..."


def solve_one(solver: Callable, question) -> Tuple[Optional[Answer], str]:
    """Run the rule solver.  Returns (Answer, "") or (None, abstain-note)."""
    try:
        sol = solver(question.text, dict(question.options))
    except Exception as exc:
        log.warning("solver raised %s: %s", type(exc).__name__, exc)
        return None, "solver error: %s" % type(exc).__name__
    letter = getattr(sol, "letter", None)
    method = str(getattr(sol, "method", "") or "")
    detail = str(getattr(sol, "detail", "") or "")
    if isinstance(letter, str) and letter.strip().upper() in LETTERS and not method.startswith("abstain"):
        if not method.startswith("rule"):
            method = "rule:" + (method or "unknown")
        return Answer(letter.strip().upper(), method, _clip(detail)), ""
    return None, method or "abstain"


# --------------------------------------------------------------------------
# paths + JSON IO
# --------------------------------------------------------------------------
def answer_path(root: str, round_id: int) -> str:
    return os.path.join(root, "answers", "round-%04d.json" % round_id)


def state_path(root: str) -> str:
    return os.path.join(root, "state", "rounds.json")


def status_path(root: str) -> str:
    return os.path.join(root, "STATUS.md")


def validate_answers_obj(obj, round_id: int) -> None:
    """Raise ValueError unless obj is exactly {"round_id": int, "answers": {q1..q25: A-D}}."""
    if not isinstance(obj, dict) or list(obj.keys()) != ["round_id", "answers"]:
        raise ValueError("answer object must have exactly the keys round_id, answers")
    rid = obj["round_id"]
    if isinstance(rid, bool) or not isinstance(rid, int) or rid != round_id:
        raise ValueError("round_id must be the int %d" % round_id)
    answers = obj["answers"]
    if not isinstance(answers, dict) or list(answers.keys()) != QKEYS:
        raise ValueError("answers must have exactly q1..q%d in order" % NUM_QUESTIONS)
    for key, value in answers.items():
        if value not in LETTERS:
            raise ValueError("%s has invalid answer %r" % (key, value))


def build_answers_obj(round_id: int, letters: Dict[int, str]) -> dict:
    return {
        "round_id": round_id,
        "answers": {"q%d" % i: letters[i] for i in ALL_INDEXES},
    }


def write_json_atomic(path: str, obj, sort_keys: bool = False) -> None:
    """Write ``json.dumps(obj, indent=2) + "\\n"`` via a temp file + rename."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=sort_keys) + "\n"
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=TMP_PREFIX, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_text_atomic(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=TMP_PREFIX, suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_existing_answers(path: str, round_id: int) -> Optional[Dict[int, str]]:
    """Letters {q_index: letter} from a valid answer file, else None."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        validate_answers_obj(obj, round_id)
    except (OSError, ValueError) as exc:
        log.warning("answer file %s is invalid: %s", path, exc)
        return None
    return {i: obj["answers"]["q%d" % i] for i in ALL_INDEXES}


def load_state(root: str) -> dict:
    path = state_path(root)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.error("state file %s unreadable (%s); treating as empty", path, exc)
        return {}
    if not isinstance(data, dict):
        log.error("state file %s is not an object; treating as empty", path)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def save_state(root: str, state: dict) -> None:
    write_json_atomic(state_path(root), state, sort_keys=True)


def cleanup_stray_tmp(root: str) -> None:
    """Remove temp files left by a killed run so they never get committed."""
    for sub in ("answers", "state"):
        for path in glob.glob(os.path.join(root, sub, TMP_PREFIX + "*")):
            try:
                os.unlink(path)
            except OSError:
                pass


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
@dataclass
class Plan:
    round: Round
    action: str                      # new | resolve_all | upgrade | skip
    reason: str
    targets: List[int] = field(default_factory=list)
    sha: str = ""
    entry: Optional[dict] = None     # existing state entry
    existing: Optional[Dict[int, str]] = None  # letters in the existing file


def plan_round(rnd: Round, root: str, state: dict, now: datetime) -> Plan:
    rid = rnd.round_id
    path = answer_path(root, rid)
    entry = state.get(str(rid))
    sha = rnd.content_hash()
    base = dict(round=rnd, sha=sha, entry=entry)
    if not os.path.exists(path):
        return Plan(action="new", reason="no answer file", targets=list(ALL_INDEXES), **base)
    existing = load_existing_answers(path, rid)
    if existing is None:
        if entry is not None:
            return Plan(action="new", reason="existing answer file is invalid; regenerating",
                        targets=list(ALL_INDEXES), **base)
        return Plan(action="skip", reason="answer file invalid but has no state entry; left untouched", **base)
    if entry is None:
        return Plan(action="skip", reason="answer file has no state entry (manual edit?); left untouched",
                    existing=existing, **base)
    if not is_editable(now, rnd.released_at):
        return Plan(action="skip", reason="past the edit window; left untouched", existing=existing, **base)
    if entry.get("questions_sha256") != sha:
        return Plan(action="resolve_all", reason="question content changed", targets=list(ALL_INDEXES),
                    existing=existing, **base)
    methods = entry.get("methods") if isinstance(entry.get("methods"), dict) else {}
    targets = [i for i in ALL_INDEXES if method_rank(methods.get("q%d" % i)) < RULE_RANK]
    if not targets:
        return Plan(action="skip", reason="all questions already rule-solved", existing=existing, **base)
    return Plan(action="upgrade", reason="%d question(s) below rule rank" % len(targets), targets=targets,
                existing=existing, **base)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass
class RoundResult:
    round_id: int
    action: str
    reason: str = ""
    is_new: bool = False             # the answer file did not exist (or was invalid)
    file_changed: bool = False
    state_changed: bool = False
    answers: Dict[int, Answer] = field(default_factory=dict)
    changed_questions: List[int] = field(default_factory=list)


@dataclass
class PipelineResult:
    results: List[RoundResult]
    state: dict
    rounds: Dict[int, Round]
    dry_run: bool = False
    written: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(r.file_changed or r.state_changed for r in self.results)

    @property
    def new_rounds(self) -> List[int]:
        return sorted(r.round_id for r in self.results if r.file_changed and r.is_new)

    @property
    def updated_rounds(self) -> List[int]:
        return sorted(r.round_id for r in self.results if r.file_changed and not r.is_new)


def _entry_answers(plan: Plan) -> Dict[int, Answer]:
    methods = (plan.entry or {}).get("methods") or {}
    details = (plan.entry or {}).get("details") or {}
    out = {}
    for i in ALL_INDEXES:
        out[i] = Answer((plan.existing or {}).get(i, "A"), str(methods.get("q%d" % i, "")),
                        str(details.get("q%d" % i, "")))
    return out


def run_pipeline(rounds: Sequence[Round], *, root: str, now: datetime,
                 solver: Optional[Callable] = None,
                 llm_fn: Optional[Callable[[List[LLMItem]], Dict[str, LLMAnswer]]] = None,
                 dry_run: bool = False, min_round: int = 1) -> PipelineResult:
    now = now.astimezone(timezone.utc)
    solver = solver or default_solver()
    state = load_state(root)
    if not dry_run:
        cleanup_stray_tmp(root)

    plans = [plan_round(r, root, state, now) for r in sorted(rounds, key=lambda r: r.round_id)
             if r.round_id >= min_round]
    rounds_by_id = {r.round_id: r for r in rounds}

    # ---- solve with rules ------------------------------------------------
    candidates: Dict[Tuple[int, int], Answer] = {}
    notes: Dict[Tuple[int, int], str] = {}
    llm_items: List[LLMItem] = []
    for plan in plans:
        if plan.action == "skip":
            continue
        rid = plan.round.round_id
        old = _entry_answers(plan) if plan.action == "upgrade" else {}
        for qi in plan.targets:
            question = plan.round.questions.get(qi)
            if question is None:
                candidates[(rid, qi)] = Answer("A", "guess:missing", "question not present in the source CSV")
                continue
            ans, note = solve_one(solver, question)
            if ans is not None:
                candidates[(rid, qi)] = ans
                continue
            notes[(rid, qi)] = note
            # An existing LLM answer is not re-asked (it can only be upgraded by a rule).
            if plan.action == "upgrade" and old[qi].rank >= 2:
                continue
            llm_items.append(LLMItem("r%dq%d" % (rid, qi), question.text, dict(question.options)))

    # ---- LLM fallback (one batched call for the whole run) ----------------
    llm_answers: Dict[str, LLMAnswer] = {}
    if llm_items and llm_fn is not None:
        log.info("asking the LLM fallback about %d question(s)", len(llm_items))
        try:
            llm_answers = llm_fn(llm_items) or {}
        except Exception as exc:
            log.warning("LLM fallback raised %s: %s", type(exc).__name__, exc)
            llm_answers = {}
    for item in llm_items:
        rid, qi = (int(x) for x in item.id[1:].split("q"))
        got = llm_answers.get(item.id)
        if got is not None and got.letter in LETTERS:
            candidates[(rid, qi)] = Answer(got.letter, got.method, _clip(got.detail))
        else:
            candidates[(rid, qi)] = Answer("A", "guess", _clip("no rule or LLM answer (%s)" % notes.get((rid, qi), "abstain")))
    # upgrade-plan questions that stayed at rank 2 and abstained produced no candidate: fine.

    # ---- assemble per round ------------------------------------------------
    results: List[RoundResult] = []
    new_state = dict(state)
    answer_writes: List[Tuple[str, dict]] = []
    for plan in plans:
        rnd = plan.round
        rid = rnd.round_id
        if plan.action == "skip":
            log.debug("round %d: %s", rid, plan.reason)
            results.append(RoundResult(rid, "skip", plan.reason))
            continue
        res = RoundResult(rid, plan.action, plan.reason, is_new=(plan.action == "new"))
        if plan.action == "upgrade":
            final = _entry_answers(plan)
            for qi in plan.targets:
                cand = candidates.get((rid, qi))
                if cand is not None and cand.rank > final[qi].rank:
                    final[qi] = cand
                    res.changed_questions.append(qi)
            if not res.changed_questions:
                log.debug("round %d: nothing improved", rid)
                res.action, res.reason = "skip", "no improvement"
                results.append(res)
                continue
            res.state_changed = True
            res.file_changed = any(final[i].letter != plan.existing[i] for i in ALL_INDEXES)
        else:
            final = {i: candidates[(rid, i)] for i in ALL_INDEXES}
            res.state_changed = True
            res.file_changed = plan.existing is None or any(final[i].letter != plan.existing[i] for i in ALL_INDEXES)
            res.changed_questions = list(ALL_INDEXES)
        res.answers = final

        old_entry = plan.entry or {}
        new_state[str(rid)] = {
            "released_at": format_timestamp(rnd.released_at),
            "questions_sha256": plan.sha,
            "answered_at": old_entry.get("answered_at") if plan.action != "new" and old_entry.get("answered_at")
            else format_timestamp(now),
            "updated_at": format_timestamp(now),
            "solver_version": quizbot.SOLVER_VERSION,
            "methods": {"q%d" % i: final[i].method for i in ALL_INDEXES},
            "details": {"q%d" % i: final[i].detail for i in ALL_INDEXES},
            "issue_opened": bool(old_entry.get("issue_opened", False)),
        }
        if res.file_changed:
            obj = build_answers_obj(rid, {i: final[i].letter for i in ALL_INDEXES})
            validate_answers_obj(obj, rid)
            answer_writes.append((answer_path(root, rid), obj))
        counts = _method_counts(final)
        log.info("round %d: %s (%s) -> rule=%d llm=%d guess=%d%s", rid, res.action, res.reason, counts[0], counts[1],
                 counts[2], "" if res.file_changed else " [file unchanged]")
        results.append(res)

    result = PipelineResult(results=results, state=new_state, rounds=rounds_by_id, dry_run=dry_run)
    skipped = sum(1 for r in results if r.action == "skip")
    log.info("%d round(s) checked: %d processed, %d left as they are", len(results), len(results) - skipped, skipped)

    # ---- persist: state first (a crash after it self-heals as "no file"), then
    # ---- answer files, then STATUS.md ----------------------------------------
    if result.changed and not dry_run:
        save_state(root, new_state)
        result.written.append("state/rounds.json")
        for path, obj in answer_writes:
            write_json_atomic(path, obj)
            result.written.append(os.path.relpath(path, root).replace(os.sep, "/"))
        write_text_atomic(status_path(root), render_status(new_state, answered_round_ids(root, new_state), now))
        result.written.append("STATUS.md")
    return result


def _method_counts(answers: Dict[int, Answer]) -> Tuple[int, int, int]:
    rule = sum(1 for a in answers.values() if a.rank == 3)
    llm = sum(1 for a in answers.values() if a.rank == 2)
    guess = len(answers) - rule - llm
    return rule, llm, guess


# --------------------------------------------------------------------------
# STATUS.md
# --------------------------------------------------------------------------
def answered_round_ids(root: str, state: dict) -> List[int]:
    ids = set()
    for key in state:
        if str(key).isdigit():
            ids.add(int(key))
    for path in glob.glob(os.path.join(root, "answers", "round-*.json")):
        name = os.path.basename(path)[len("round-"):-len(".json")]
        if name.isdigit():
            ids.add(int(name))
    return sorted(ids)


def to_pacific(dt: datetime) -> Tuple[datetime, str]:
    """Convert to America/Los_Angeles (zoneinfo; hand-rolled US DST rules as fallback)."""
    dt = dt.astimezone(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        local = dt.astimezone(ZoneInfo("America/Los_Angeles"))
        return local, local.tzname() or "PT"
    except Exception:
        pass

    def nth_sunday(year, month, n):
        d = datetime(year, month, 1, tzinfo=timezone.utc)
        d += timedelta(days=(6 - d.weekday()) % 7)  # first Sunday
        return d + timedelta(weeks=n - 1)

    year = dt.year
    start = nth_sunday(year, 3, 2).replace(hour=10)   # 2am PST == 10:00 UTC
    end = nth_sunday(year, 11, 1).replace(hour=9)     # 2am PDT == 09:00 UTC
    if start <= dt < end:
        return dt.astimezone(timezone(timedelta(hours=-7))), "PDT"
    return dt.astimezone(timezone(timedelta(hours=-8))), "PST"


def render_status(state: dict, answered_ids: Sequence[int], now: datetime) -> str:
    now = now.astimezone(timezone.utc)
    pac, pac_name = to_pacific(now)
    lines = [
        "# Hourly quiz bot status",
        "",
        "- Last update: %s UTC (%s %s)" % (now.strftime("%Y-%m-%d %H:%M:%S"), pac.strftime("%Y-%m-%d %H:%M:%S"), pac_name),
        "- Answered rounds: %d" % len(answered_ids),
        "- Solver version: `%s`" % quizbot.SOLVER_VERSION,
        "",
        "Latest %d rounds:" % STATUS_ROWS,
        "",
        "| Round | Released (UTC) | Answered (UTC) | Minutes after release | rule | llm | guess |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    ids = sorted((int(k) for k in state if str(k).isdigit()), reverse=True)[:STATUS_ROWS]
    for rid in ids:
        entry = state[str(rid)]
        methods = entry.get("methods") or {}
        ranks = [method_rank(m) for m in methods.values()]
        rule = sum(1 for r in ranks if r == 3)
        llm = sum(1 for r in ranks if r == 2)
        guess = sum(1 for r in ranks if r < 2)
        released = entry.get("released_at", "")
        answered = entry.get("answered_at", "")
        try:
            minutes = int((parse_timestamp(answered) - parse_timestamp(released)).total_seconds() // 60)
            minutes_s = str(minutes)
        except (ValueError, TypeError):
            minutes_s = "?"
        lines.append("| %d | %s | %s | %s | %d | %d | %d |" % (rid, released.replace("T", " ").rstrip("Z"),
                                                            answered.replace("T", " ").rstrip("Z"), minutes_s,
                                                            rule, llm, guess))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# console output for --dry-run
# --------------------------------------------------------------------------
def format_dry_run(result: PipelineResult, verbose: bool = False) -> str:
    out: List[str] = []
    for res in result.results:
        if res.action == "skip":
            out.append("round %d: skip - %s" % (res.round_id, res.reason))
            continue
        counts = _method_counts(res.answers)
        out.append("round %d: %s - %s | rule=%d llm=%d guess=%d | file %s" % (
            res.round_id, res.action, res.reason, counts[0], counts[1], counts[2],
            "would change" if res.file_changed else "unchanged"))
        letters = "".join(res.answers[i].letter for i in ALL_INDEXES)
        out.append("  answers: %s" % letters)
        for i in ALL_INDEXES:
            a = res.answers[i]
            if verbose or a.rank < RULE_RANK:
                out.append("  q%-2d %s  %-28s %s" % (i, a.letter, a.method, a.detail))
    return "\n".join(out)


# --------------------------------------------------------------------------
# --check-practice
# --------------------------------------------------------------------------
@dataclass
class PracticeReport:
    total: int = 0
    correct: int = 0
    abstained: int = 0
    by_family: Dict[str, List[int]] = field(default_factory=dict)  # family -> [correct, total]
    wrong: List[Tuple[int, int, str, str, str]] = field(default_factory=list)

    @property
    def perfect(self) -> bool:
        return self.total > 0 and self.correct == self.total

    def format(self) -> str:
        lines = []
        for fam in sorted(self.by_family):
            c, t = self.by_family[fam]
            lines.append("  %-24s %4d/%-4d %6.2f%%" % (fam, c, t, 100.0 * c / t if t else 0.0))
        pct = 100.0 * self.correct / self.total if self.total else 0.0
        lines.append("accuracy: %d/%d (%.2f%%), abstained: %d" % (self.correct, self.total, pct, self.abstained))
        for rid, qi, want, got, method in self.wrong[:20]:
            lines.append("  MISS round %d q%d: key %s, got %s (%s)" % (rid, qi, want, got, method))
        if len(self.wrong) > 20:
            lines.append("  ... and %d more" % (len(self.wrong) - 20))
        return "\n".join(lines)


def check_practice(questions_path: str, keys_path: str, solver: Optional[Callable] = None) -> PracticeReport:
    solver = solver or default_solver()
    with open(questions_path, "rb") as fh:
        rounds = parse_questions_csv(fh.read().decode("utf-8", errors="replace"))
    keys: Dict[Tuple[int, int], str] = {}
    with open(keys_path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            keys[(int(row["round_id"]), int(row["q_index"]))] = row["answer"].strip().upper()
    report = PracticeReport()
    for rnd in rounds:
        for qi, question in sorted(rnd.questions.items()):
            want = keys.get((rnd.round_id, qi))
            if want is None:
                continue
            report.total += 1
            ans, note = solve_one(solver, question)
            if ans is None:
                report.abstained += 1
                fam = "(abstained)"
                got, method = "-", note
            else:
                fam = ans.method.split(":", 1)[1] if ":" in ans.method else ans.method
                got, method = ans.letter, ans.method
            stat = report.by_family.setdefault(fam, [0, 0])
            stat[1] += 1
            if got == want:
                stat[0] += 1
                report.correct += 1
            else:
                report.wrong.append((rnd.round_id, qi, want, got, method))
    return report
