"""Optional LLM fallback for questions the rule solvers abstain on.

Uses GitHub Models (free for GitHub accounts, authenticates with the Actions
``GITHUB_TOKEN``).  The whole module is best-effort: every public function
degrades to "nothing answered" instead of raising, and model output is only
ever parsed as JSON letters - it is never executed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .source import ssl_context

log = logging.getLogger("quizbot.llm")

CATALOG_URL = "https://models.github.ai/catalog/models"
INFERENCE_URL = "https://models.github.ai/inference/chat/completions"
DEFAULT_MODELS = "openai/gpt-5,openai/o3,openai/o4-mini,openai/gpt-4.1,deepseek/deepseek-r1,openai/gpt-4o"
MAX_REQUESTS = 6          # hard cap on HTTP requests per run (catalog + inference)
CHUNK_SIZE = 25           # questions per request
REQUEST_TIMEOUT = 60      # seconds, per inference request
CATALOG_TIMEOUT = 30
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


# --------------------------------------------------------------------------
# HTTP + auth
# --------------------------------------------------------------------------
def _http_json(method: str, url: str, headers: Mapping[str, str], body: Optional[dict] = None,
               timeout: int = REQUEST_TIMEOUT):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = {"User-Agent": "dso429-quizbot"}
    hdrs.update(headers)
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            snippet = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            snippet = ""
        raise LLMRequestError(exc.code, snippet)
    except (urllib.error.URLError, OSError) as exc:  # includes socket timeouts
        raise LLMRequestError(0, str(exc))
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as exc:
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


def discover_models(token: str, env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Preference list filtered to the live catalog; as-is when the catalog is unusable."""
    prefs = preferred_models(env)
    try:
        catalog = _http_json("GET", CATALOG_URL, _api_headers(token), None, CATALOG_TIMEOUT)
    except LLMRequestError as exc:
        log.warning("model catalog unavailable (%s); trying preferred models as-is", exc)
        return prefs
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
class _Budget:
    def __init__(self, cap: int):
        self.cap = cap
        self.used = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.cap


def _ask(model: str, chunk: Sequence[LLMItem], token: str, budget: _Budget, dead: set):
    """One inference request.  Returns parsed answers, or None on any failure."""
    budget.used += 1
    ids = [i.id for i in chunk]
    try:
        resp = _http_json("POST", INFERENCE_URL, _api_headers(token), build_request_body(model, chunk), REQUEST_TIMEOUT)
    except LLMRequestError as exc:
        log.warning("model %s failed (%s); falling through to the next model", model, exc)
        dead.add(model)
        return None
    parsed = parse_answers(_reply_text(resp), ids)
    if not parsed:
        log.warning("model %s returned no usable answers", model)
        return None
    log.info("model %s answered %d/%d questions", model, len(parsed), len(ids))
    return parsed


def _answer(items: Sequence[LLMItem], token: Optional[str], models: Optional[Sequence[str]],
            max_requests: int, chunk_size: int) -> Dict[str, LLMAnswer]:
    if not items:
        return {}
    token = token or get_token()
    if not token:
        log.info("LLM fallback unavailable: no GitHub token")
        return {}
    budget = _Budget(max_requests)   # the cap covers every HTTP request of the run, catalog included
    if models:
        model_list = list(models)
    else:
        budget.used += 1             # the catalog request
        model_list = discover_models(token)
    if not model_list:
        log.info("LLM fallback unavailable: no models")
        return {}
    chunk_size = max(1, chunk_size)
    chunks = [list(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    dead: set = set()
    per_chunk: List[List[Tuple[str, Dict[str, Tuple[str, str]]]]] = [[] for _ in chunks]

    # Pass 1: every chunk gets a first opinion (falling through dead/failing models).
    for ci, chunk in enumerate(chunks):
        for model in model_list:
            if budget.exhausted:
                break
            if model in dead:
                continue
            res = _ask(model, chunk, token, budget, dead)
            if res:
                per_chunk[ci].append((model, res))
                break
    # Pass 2: a second, different model for agreement, while the request budget lasts.
    for ci, chunk in enumerate(chunks):
        if not per_chunk[ci]:
            continue
        primary = per_chunk[ci][0][0]
        for model in model_list:
            if budget.exhausted:
                break
            if model == primary or model in dead:
                continue
            res = _ask(model, chunk, token, budget, dead)
            if res:
                per_chunk[ci].append((model, res))
                break
    if budget.exhausted:
        log.info("LLM request cap (%d) reached", max_requests)

    out: Dict[str, LLMAnswer] = {}
    for ci, chunk in enumerate(chunks):
        opinions = per_chunk[ci]
        for item in chunk:
            votes = [(m, res[item.id]) for m, res in opinions if item.id in res]
            if not votes:
                continue
            first_model, (first_letter, work) = votes[0]
            detail = "%s->%s" % (first_model, first_letter)
            if len(votes) >= 2:
                second_model, (second_letter, _) = votes[1]
                detail += "; %s->%s" % (second_model, second_letter)
                kind = "llm" if second_letter == first_letter else "llm-disagree"
            else:
                kind = "llm"
            if work:
                detail += " (%s)" % " ".join(work.split())[:120]
            out[item.id] = LLMAnswer(first_letter, "%s:%s" % (kind, first_model), detail)
    return out


def answer_questions(items: Sequence[LLMItem], *, token: Optional[str] = None,
                     models: Optional[Sequence[str]] = None, max_requests: int = MAX_REQUESTS,
                     chunk_size: int = CHUNK_SIZE) -> Dict[str, LLMAnswer]:
    """Answer questions with GitHub Models.  Never raises; returns {} when unavailable."""
    try:
        return _answer(items, token, models, max_requests, chunk_size)
    except Exception as exc:  # the fallback must never break the pipeline
        log.warning("LLM fallback failed: %s: %s", type(exc).__name__, exc)
        return {}
