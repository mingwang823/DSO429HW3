"""Optional LLM fallback for questions the rule solvers abstain on.

Uses GitHub Models (free for GitHub accounts, authenticates with the Actions
``GITHUB_TOKEN``).  The whole module is best-effort: every public function
degrades to "nothing answered" instead of raising, and model output is only
ever parsed as JSON letters - it is never executed.

Two limits protect the free tier and the run time:

* a per-process budget (``RunBudget``): at most ``MAX_REQUESTS`` HTTP requests and
  ``LLM_TIME_BUDGET`` seconds in total, shared by every call made by the process;
* ``MemoizedLLM``: a question (text + options) that was already asked in this
  process is never sent again, even when the git flow retries a rejected push.

``selftest()`` backs ``python3 run.py --llm-selftest``.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .source import ssl_context

log = logging.getLogger("quizbot.llm")

CATALOG_URL = "https://models.github.ai/catalog/models"
INFERENCE_URL = "https://models.github.ai/inference/chat/completions"
DEFAULT_MODELS = "openai/gpt-5,openai/o3,openai/o4-mini,openai/gpt-4.1,deepseek/deepseek-r1,openai/gpt-4o"
MAX_REQUESTS = 6          # hard cap on HTTP requests per run (catalog + inference)
CHUNK_SIZE = 25           # questions per request
REQUEST_TIMEOUT = 60      # seconds, per inference request
CATALOG_TIMEOUT = 30
LLM_TIME_BUDGET = 120.0   # seconds of LLM calls per run (process), all requests together
MIN_REQUEST_SECONDS = 5.0 # do not start a request with less budget than this left
LETTERS = ("A", "B", "C", "D")

SYSTEM_PROMPT = (
    "You are a careful solver of business-math multiple-choice questions "
    "(interest, margins, discounts, dates, ratios, averages, currency, probability). "
    "Exactly one option of each question is correct. The wrong options are computed "
    "with plausible wrong formulas, so work out the exact value with the formula the "
    "wording implies and pick the option that equals it. "
    "Reply with ONLY one JSON object and no other text, in this exact shape: "
    '{"answers":[{"id":"<question id>","work":"<one-line calculation>","answer":"<A, B, C or D>"}]}. '
    "Include every question id exactly once."
)


class LLMRequestError(Exception):
    """An HTTP/network/JSON failure; status is 0 for network errors and timeouts."""

    def __init__(self, status: int, message: str = ""):
        super().__init__("HTTP %s: %s" % (status, message))
        self.status = status
        self.message = message


@dataclass(frozen=True)
class LLMItem:
    id: str
    text: str
    options: Dict[str, str]


@dataclass(frozen=True)
class LLMAnswer:
    letter: str
    method: str   # "llm:<model>" or "llm-disagree:<model>"
    detail: str = ""
    models: Tuple[str, ...] = ()   # every model that voted on this question, first model first


class LLMResults(dict):
    """``{item id: LLMAnswer}`` plus ``unattempted``: ids that were never sent to a model
    (no token, request/time budget used up, every model already failed).  The pipeline
    only starts its 30-minute retry throttle for questions that were really attempted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unattempted: Set[str] = set()


class RunBudget:
    """Request cap + time budget + model-list cache shared by all LLM calls of one process."""

    def __init__(self, cap: int = MAX_REQUESTS, seconds: float = LLM_TIME_BUDGET,
                 clock: Callable[[], float] = time.monotonic):
        self.cap = cap
        self.seconds = seconds
        self.clock = clock
        self.used = 0                       # HTTP requests issued (catalog included)
        self.spent = 0.0                    # seconds spent inside LLM HTTP requests
        self.models: Optional[List[str]] = None   # discovered once per process
        self.requests: List[Tuple[str, float, str]] = []   # (label, seconds, outcome) for the self-test

    def time_left(self) -> float:
        return max(0.0, self.seconds - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.cap or self.time_left() < MIN_REQUEST_SECONDS


# --------------------------------------------------------------------------
# HTTP + auth
# --------------------------------------------------------------------------
def _http_json(method: str, url: str, headers: Mapping[str, str], body: Optional[dict] = None,
               timeout: int = REQUEST_TIMEOUT):
    """One JSON request.  Every failure - HTTP status, network, TLS, timeout, a truncated
    body (``http.client.IncompleteRead``), a reply that is not JSON - becomes LLMRequestError."""
    try:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        hdrs = {"User-Agent": "dso429-quizbot"}
        hdrs.update(headers)
        if data is not None:
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            snippet = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            snippet = ""
        raise LLMRequestError(exc.code, snippet)
    except (urllib.error.URLError, OSError) as exc:  # includes socket timeouts and TLS errors
        raise LLMRequestError(0, str(exc))
    except http.client.HTTPException as exc:         # IncompleteRead, RemoteDisconnected, BadStatusLine, ...
        raise LLMRequestError(0, "%s: %s" % (type(exc).__name__, exc))
    except ValueError as exc:                        # bad JSON / bad UTF-8 / unserialisable body
        raise LLMRequestError(-1, "response is not JSON: %s" % exc)


def get_token(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """GITHUB_TOKEN / GH_TOKEN, else ``gh auth token`` if the gh CLI exists."""
    env = os.environ if env is None else env
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = (env.get(key) or "").strip()
        if value:
            return value
    gh = shutil.which("gh")
    if gh:
        try:
            proc = subprocess.run([gh, "auth", "token"], capture_output=True, text=True, timeout=10)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def _api_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": "Bearer %s" % token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# --------------------------------------------------------------------------
# model selection
# --------------------------------------------------------------------------
def preferred_models(env: Optional[Mapping[str, str]] = None) -> List[str]:
    env = os.environ if env is None else env
    raw = (env.get("QUIZ_LLM_MODELS") or "").strip() or DEFAULT_MODELS
    seen: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part and part not in seen:
            seen.append(part)
    return seen


def _catalog_ids(catalog) -> set:
    if isinstance(catalog, dict):
        for key in ("models", "data", "items"):
            if isinstance(catalog.get(key), list):
                catalog = catalog[key]
                break
    ids = set()
    if isinstance(catalog, list):
        for entry in catalog:
            if isinstance(entry, str):
                ids.add(entry.lower())
            elif isinstance(entry, dict):
                for key in ("id", "name"):
                    if isinstance(entry.get(key), str):
                        ids.add(entry[key].lower())
    return ids


def discover_models(token: str, env: Optional[Mapping[str, str]] = None,
                    budget: Optional[RunBudget] = None) -> List[str]:
    """Preference list filtered to the live catalog; as-is when the catalog is unusable."""
    prefs = preferred_models(env)
    timeout = CATALOG_TIMEOUT
    clock = budget.clock if budget is not None else time.monotonic
    if budget is not None:
        timeout = max(1, min(CATALOG_TIMEOUT, int(budget.time_left())))
    started = clock()
    outcome = "ok"
    try:
        catalog = _http_json("GET", CATALOG_URL, _api_headers(token), None, timeout)
    except Exception as exc:  # LLMRequestError, or anything a test double raises
        outcome = "failed: %s" % exc
        log.warning("model catalog unavailable (%s); trying preferred models as-is", exc)
        return prefs
    finally:
        if budget is not None:
            elapsed = max(0.0, clock() - started)
            budget.spent += elapsed
            budget.requests.append(("model catalog", elapsed, outcome))
    ids = _catalog_ids(catalog)
    filtered = [m for m in prefs if m.lower() in ids]
    if not filtered:
        log.warning("catalog lists none of the preferred models; trying them as-is")
        return prefs
    return filtered


def is_reasoning_model(model: str) -> bool:
    """o-series, gpt-5 and DeepSeek-R models reject/ignore a custom temperature."""
    base = model.split("/")[-1].lower()
    return re.match(r"(o\d|gpt-5|deepseek-r)", base) is not None


# --------------------------------------------------------------------------
# prompt + parsing
# --------------------------------------------------------------------------
def format_user_prompt(items: Sequence[LLMItem]) -> str:
    parts = ["Answer each question. Question ids are in brackets.\n"]
    for item in items:
        parts.append("[%s] %s" % (item.id, item.text))
        for letter in LETTERS:
            parts.append("%s) %s" % (letter, item.options.get(letter, "")))
        parts.append("")
    return "\n".join(parts)


def build_request_body(model: str, items: Sequence[LLMItem]) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(items)},
        ],
    }
    if not is_reasoning_model(model):
        body["temperature"] = 0
    return body


_LETTER_RE = re.compile(r"^(?:(?:OPTION|ANSWER|CHOICE)\s*[:\-]?\s*)?\(?([ABCD])\)?\.?$")


def normalize_letter(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    m = _LETTER_RE.match(value.strip().upper())
    return m.group(1) if m else None


def parse_answers(content, valid_ids) -> Dict[str, Tuple[str, str]]:
    """Extract ``{id: (letter, work)}`` from a model reply.

    Finds the first JSON object (anywhere in the reply: bare, fenced, or
    surrounded by prose) that has an ``answers`` list.  Entries with an unknown
    id or a letter outside A-D are dropped; the first entry per id wins.
    """
    if not isinstance(content, str) or not content:
        return {}
    valid = set(valid_ids)
    decoder = json.JSONDecoder()
    pos = content.find("{")
    while pos != -1:
        try:
            obj, _end = decoder.raw_decode(content, pos)
        except ValueError:
            pos = content.find("{", pos + 1)
            continue
        if isinstance(obj, dict) and isinstance(obj.get("answers"), list):
            out: Dict[str, Tuple[str, str]] = {}
            for entry in obj["answers"]:
                if not isinstance(entry, dict):
                    continue
                qid = entry.get("id")
                letter = normalize_letter(entry.get("answer"))
                if not isinstance(qid, str) or qid not in valid or letter is None or qid in out:
                    continue
                work = entry.get("work")
                out[qid] = (letter, work if isinstance(work, str) else "")
            return out
        pos = content.find("{", pos + 1)
    return {}


def _reply_text(resp) -> str:
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def _ask(model: str, chunk: Sequence[LLMItem], token: str, budget: RunBudget,
         dead: set) -> Tuple[Optional[Dict[str, Tuple[str, str]]], bool]:
    """One inference request.  Returns (parsed answers or None, whether a request was issued).

    Never raises: any failure of this call only loses this call.
    """
    remaining = budget.time_left()
    if remaining < MIN_REQUEST_SECONDS:
        log.info("LLM time budget (%.0f s) used up; not asking %s", budget.seconds, model)
        return None, False
    budget.used += 1
    ids = [i.id for i in chunk]
    timeout = max(1, min(REQUEST_TIMEOUT, int(remaining)))
    started = budget.clock()
    outcome = "ok"
    try:
        resp = _http_json("POST", INFERENCE_URL, _api_headers(token), build_request_body(model, chunk), timeout)
        parsed = parse_answers(_reply_text(resp), ids)
        if not parsed:
            outcome = "no usable answers"
            log.warning("model %s returned no usable answers", model)
            return None, True
        log.info("model %s answered %d/%d questions", model, len(parsed), len(ids))
        return parsed, True
    except LLMRequestError as exc:
        outcome = "failed: %s" % exc
        log.warning("model %s failed (%s); falling through to the next model", model, exc)
        dead.add(model)
        return None, True
    except Exception as exc:  # defence in depth: a bad call must not discard earlier answers
        outcome = "failed: %s: %s" % (type(exc).__name__, exc)
        log.warning("model %s raised %s: %s; falling through to the next model", model, type(exc).__name__, exc)
        dead.add(model)
        return None, True
    finally:
        elapsed = max(0.0, budget.clock() - started)
        budget.spent += elapsed
        budget.requests.append((model, elapsed, outcome))


def _assemble(chunks, per_chunk) -> Dict[str, LLMAnswer]:
    out: Dict[str, LLMAnswer] = {}
    for ci, chunk in enumerate(chunks):
        opinions = per_chunk[ci]
        for item in chunk:
            votes = [(m, res[item.id]) for m, res in opinions if item.id in res]
            if not votes:
                continue
            first_model, (first_letter, work) = votes[0]
            detail = "%s->%s" % (first_model, first_letter)
            models = [first_model]
            if len(votes) >= 2:
                second_model, (second_letter, _) = votes[1]
                detail += "; %s->%s" % (second_model, second_letter)
                models.append(second_model)
                kind = "llm" if second_letter == first_letter else "llm-disagree"
            else:
                kind = "llm"
            if work:
                detail += " (%s)" % " ".join(work.split())[:120]
            out[item.id] = LLMAnswer(first_letter, "%s:%s" % (kind, first_model), detail, tuple(models))
    return out


def _answer(items: Sequence[LLMItem], token: Optional[str], models: Optional[Sequence[str]],
            budget: RunBudget, chunk_size: int) -> LLMResults:
    results = LLMResults()
    if not items:
        return results
    everything = [i.id for i in items]
    token = token or get_token()
    if not token:
        log.info("LLM fallback unavailable: no GitHub token")
        results.unattempted.update(everything)
        return results
    if models:
        model_list = list(models)
    elif budget.models is not None:
        model_list = list(budget.models)
    else:
        if budget.exhausted:
            log.info("LLM fallback skipped: request/time budget already used up")
            results.unattempted.update(everything)
            return results
        budget.used += 1             # the catalog request counts against the cap
        model_list = discover_models(token, budget=budget)
        budget.models = list(model_list)
    if not model_list:
        log.info("LLM fallback unavailable: no models")
        results.unattempted.update(everything)
        return results
    chunk_size = max(1, chunk_size)
    chunks = [list(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    dead: set = set()
    per_chunk: List[List[Tuple[str, Dict[str, Tuple[str, str]]]]] = [[] for _ in chunks]
    attempted: Set[int] = set()      # chunks for which at least one request was really issued

    try:
        # Pass 1: every chunk gets a first opinion (falling through dead/failing models).
        for ci, chunk in enumerate(chunks):
            for model in model_list:
                if budget.exhausted:
                    break
                if model in dead:
                    continue
                res, issued = _ask(model, chunk, token, budget, dead)
                if issued:
                    attempted.add(ci)
                if res:
                    per_chunk[ci].append((model, res))
                    break
        # Pass 2: a second, different model for agreement, while the budget lasts.
        for ci, chunk in enumerate(chunks):
            if not per_chunk[ci]:
                continue
            primary = per_chunk[ci][0][0]
            for model in model_list:
                if budget.exhausted:
                    break
                if model == primary or model in dead:
                    continue
                res, _issued = _ask(model, chunk, token, budget, dead)
                if res:
                    per_chunk[ci].append((model, res))
                    break
    except Exception as exc:  # whatever happens, keep what earlier calls produced
        log.warning("LLM fallback interrupted by %s: %s; keeping the answers collected so far", type(exc).__name__, exc)
    if budget.used >= budget.cap:
        log.info("LLM request cap (%d) reached", budget.cap)
    elif budget.time_left() < MIN_REQUEST_SECONDS:
        log.info("LLM time budget (%.0f s) used up", budget.seconds)

    try:
        results.update(_assemble(chunks, per_chunk))
    except Exception as exc:
        log.warning("could not assemble LLM answers: %s: %s", type(exc).__name__, exc)
    for ci, chunk in enumerate(chunks):
        if ci not in attempted:
            results.unattempted.update(i.id for i in chunk)
    return results


def answer_questions(items: Sequence[LLMItem], *, token: Optional[str] = None,
                     models: Optional[Sequence[str]] = None, max_requests: int = MAX_REQUESTS,
                     chunk_size: int = CHUNK_SIZE, budget: Optional[RunBudget] = None) -> LLMResults:
    """Answer questions with GitHub Models.  Never raises; returns {} when unavailable.

    ``budget`` is shared across calls of one process (request cap, time budget, model list);
    without it each call gets a fresh one.
    """
    budget = budget or RunBudget(max_requests)
    try:
        return _answer(items, token, models, budget, chunk_size)
    except Exception as exc:  # the fallback must never break the pipeline
        log.warning("LLM fallback failed: %s: %s", type(exc).__name__, exc)
        out = LLMResults()
        out.unattempted.update(getattr(i, "id", "") for i in items)
        return out


# --------------------------------------------------------------------------
# memoisation + the callable the pipeline uses
# --------------------------------------------------------------------------
def _item_key(item: LLMItem) -> tuple:
    return (item.text, tuple(item.options.get(letter, "") for letter in LETTERS))


class MemoizedLLM:
    """Wrap an ``llm_fn(items) -> {id: LLMAnswer}`` so a question is asked at most once per process.

    Keyed by question text + options (not by round/question id), so the same question in two
    rounds costs one answer, and a git push retry that re-runs the pipeline re-uses the earlier
    answer - including the fact that the model had none.  Items that were never really sent
    (``LLMResults.unattempted``) are not remembered, so they can still be asked later.
    """

    def __init__(self, fn: Callable[[Sequence[LLMItem]], Mapping[str, LLMAnswer]]):
        self._fn = fn
        self._memo: Dict[tuple, Optional[LLMAnswer]] = {}
        self.calls = 0                      # how many times the wrapped fn was really invoked

    def __call__(self, items: Sequence[LLMItem]) -> LLMResults:
        items = list(items)
        fresh: Dict[tuple, LLMItem] = {}
        for item in items:
            key = _item_key(item)
            if key not in self._memo and key not in fresh:
                fresh[key] = item
        if fresh:
            self.calls += 1
            got = self._fn(list(fresh.values()))
            if got is None:              # NB: not `or {}` - an empty LLMResults still carries .unattempted
                got = {}
            unattempted = set(getattr(got, "unattempted", ()) or ())
            for key, item in fresh.items():
                if item.id in unattempted:
                    continue
                answer = got.get(item.id)
                self._memo[key] = answer if isinstance(answer, LLMAnswer) else None
        out = LLMResults()
        for item in items:
            key = _item_key(item)
            if key not in self._memo:
                out.unattempted.add(item.id)
            elif self._memo[key] is not None:
                out[item.id] = self._memo[key]
        return out


def make_llm_fn(*, token: Optional[str] = None, models: Optional[Sequence[str]] = None,
                seconds: float = LLM_TIME_BUDGET, max_requests: int = MAX_REQUESTS,
                chunk_size: int = CHUNK_SIZE, clock: Callable[[], float] = time.monotonic) -> MemoizedLLM:
    """The real fallback for one process: GitHub Models under one shared budget, memoised."""
    budget = RunBudget(max_requests, seconds, clock)

    def ask(items: Sequence[LLMItem]) -> LLMResults:
        return answer_questions(items, token=token, models=models, chunk_size=chunk_size, budget=budget)

    client = MemoizedLLM(ask)
    client.budget = budget  # type: ignore[attr-defined]
    return client


# --------------------------------------------------------------------------
# self-test:  python3 run.py --llm-selftest
# --------------------------------------------------------------------------
SELFTEST_CASES: Tuple[Tuple[str, Dict[str, str], str], ...] = (
    ("What is 17 plus 25?",
     {"A": "40", "B": "41", "C": "42", "D": "43"}, "C"),
    ("What is 15% of 240?",
     {"A": "32", "B": "36", "C": "34", "D": "38"}, "B"),
    # a question type the rule solver does not know
    ("A loan of $10,000 at 6% simple interest for 3 years accrues how much interest?",
     {"A": "$600.00", "B": "$1,910.16", "C": "$180.00", "D": "$1,800.00"}, "D"),
)
SELFTEST_PASS_MARK = 2    # correct answers needed out of len(SELFTEST_CASES)
EXIT_OK, EXIT_FAILED, EXIT_UNAVAILABLE = 0, 1, 3


def selftest(llm_fn: Optional[Callable] = None, *, solver: Optional[Callable] = None,
             out: Callable[[str], None] = print, clock: Callable[[], float] = time.monotonic,
             token: Optional[str] = None) -> int:
    """Send 3 synthetic questions through the real fallback path and report.

    Exit code: 0 when at least 2 of 3 answers are correct, 3 when the LLM is unavailable
    (no token, or no usable answer at all), 1 otherwise.  Writes nothing to disk.
    ``solver`` (optional) is only used to show whether the rule solver would have answered.
    """
    out("LLM self-test: %d synthetic questions through the GitHub Models fallback" % len(SELFTEST_CASES))
    if llm_fn is None:
        token = token or get_token()
        if not token:
            out("LLM unavailable: no GITHUB_TOKEN / GH_TOKEN in the environment and no `gh auth token`")
            return EXIT_UNAVAILABLE
        llm_fn = make_llm_fn(token=token)
        out("models (preference order): %s" % ", ".join(preferred_models()))
    items = [LLMItem("selftest%d" % n, text, dict(opts)) for n, (text, opts, _want) in enumerate(SELFTEST_CASES, 1)]
    started = clock()
    try:
        answers = llm_fn(items) or {}
    except Exception as exc:
        out("LLM unavailable: %s: %s" % (type(exc).__name__, exc))
        return EXIT_UNAVAILABLE
    latency = clock() - started

    budget = getattr(llm_fn, "budget", None)
    for label, seconds, outcome in (budget.requests if budget is not None else []):
        out("  request to %-28s %6.1f s  %s" % (label, seconds, outcome))
    if not answers:
        out("LLM unavailable: no usable answer came back (total %.1f s)" % latency)
        return EXIT_UNAVAILABLE

    correct = 0
    models_seen: List[str] = []
    for item, (_text, _opts, want) in zip(items, SELFTEST_CASES):
        got = answers.get(item.id)
        rule = ""
        if solver is not None:
            try:
                sol = solver(item.text, dict(item.options))
                letter = getattr(sol, "letter", None)
                rule = "rule solver: %s" % (("answers " + str(letter)) if letter else "abstains (%s)" % getattr(sol, "method", "?"))
            except Exception as exc:
                rule = "rule solver raised %s" % type(exc).__name__
        if got is None:
            out("  [MISS] %s: no answer (expected %s)  %s" % (item.id, want, rule))
            continue
        ok = got.letter == want
        correct += 1 if ok else 0
        for m in (got.models or (got.method.split(":", 1)[-1],)):
            if m not in models_seen:
                models_seen.append(m)
        out("  [%s] %s: expected %s, got %s  via %s  (%s)  %s" % (
            "PASS" if ok else "FAIL", item.id, want, got.letter, got.method, got.detail, rule))
        out("         %s" % item.text)
    out("models that answered: %s" % (", ".join(models_seen) or "-"))
    out("latency: %.1f s for the whole batch" % latency)
    out("correct: %d/%d (need %d)" % (correct, len(SELFTEST_CASES), SELFTEST_PASS_MARK))
    return EXIT_OK if correct >= SELFTEST_PASS_MARK else EXIT_FAILED
