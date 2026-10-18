"""Fetch and parse the course's hourly-quiz ``questions.csv``.

Fetch order (freshest first, GitHub's raw CDN can lag several minutes):

1. ``git clone --depth 1 --branch main`` of the quiz repo into a temp dir
2. GitHub REST contents API (``Accept: application/vnd.github.raw``)
3. ``raw.githubusercontent.com`` with a cache-busting query string

Every source is header-validated (by column *name*: extra or reordered columns are
fine, but all 8 required names must be present); a source that returns something
other than the expected CSV (an HTML error page, a truncated file, ...) falls
through to the next one.  Parsing never crashes the run: malformed rows are
skipped with a warning, and an unterminated quote costs at most its own row.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("quizbot.source")

DEFAULT_REPO = "FaisalXL/dso429-hourly-quiz"
CSV_NAME = "questions.csv"
EXPECTED_HEADER = [
    "round_id",
    "q_index",
    "released_at",
    "question_text",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
]
NUM_QUESTIONS = 25
MAX_ROUND_ID = 9999          # answer files are named round-NNNN.json
OPTION_LETTERS = ("A", "B", "C", "D")
HTTP_TIMEOUT = 30
CLONE_TIMEOUT = 90
USER_AGENT = "dso429-quizbot"


class SourceError(Exception):
    """The questions could not be fetched or the file is not a valid questions.csv."""


@dataclass(frozen=True)
class Question:
    text: str
    options: Dict[str, str]


@dataclass
class Round:
    round_id: int
    released_at: datetime  # timezone-aware, UTC
    questions: Dict[int, Question] = field(default_factory=dict)

    def content_hash(self) -> str:
        """sha256 of the question content (not of the release time)."""
        payload = [
            [idx, q.text, [q.options.get(l, "") for l in OPTION_LETTERS]]
            for idx, q in sorted(self.questions.items())
        ]
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------
def parse_timestamp(value: str) -> datetime:
    """Parse ISO-8601 (``...Z`` or offset; naive means UTC) into aware UTC."""
    s = value.strip()
    if not s:
        raise ValueError("empty timestamp")
    if s[-1] in "zZ":
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_timestamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------------
# A physical line that begins a record: round_id,q_index,<ISO date>T...  Used to
# recover from an unterminated quote, which would otherwise swallow every row
# after it.  (Only meaningful while round_id, q_index, released_at come first,
# which is how the instructors' file is laid out; with another column order the
# recovery simply never triggers.)
_RECORD_START = re.compile(r"^\d+,\d+,\d{4}-\d{2}-\d{2}T", re.MULTILINE)
_EMBEDDED_START = re.compile(r"\n\d+,\d+,\d{4}-\d{2}-\d{2}T")

Record = Tuple[int, List[str]]


def _clean(text: str) -> str:
    """Drop a BOM and any leading blank space so the header is the first record."""
    return text.lstrip("\ufeff \t\r\n")


def _header_index(cells: List[str]) -> Dict[str, int]:
    """{column name: position}; names are compared stripped and case-insensitively.

    Extra and reordered columns are fine, but every required name must exist -
    which is also what rejects an HTML error page or a JSON document.
    """
    index: Dict[str, int] = {}
    for pos, cell in enumerate(cells):
        name = cell.replace("\ufeff", "").strip().lower()
        index.setdefault(name, pos)
    missing = [name for name in EXPECTED_HEADER if name not in index]
    if missing:
        shown = [c.strip()[:40] for c in cells[:10]]
        raise SourceError("unexpected questions.csv header: missing %s (got %r)" % (", ".join(missing), shown))
    return index


def check_header(text: str) -> Dict[str, int]:
    """Raise SourceError unless the first CSV record names all 8 required columns."""
    text = _clean(text)
    try:
        header = next(csv.reader(io.StringIO(text, newline="")))
    except StopIteration:
        raise SourceError("questions.csv is empty")
    except csv.Error as exc:
        raise SourceError("questions.csv header unreadable: %s" % exc)
    return _header_index(header)


def _read_records(text: str, line_offset: int = 0) -> List[Record]:
    """csv-parse ``text`` into [(line number, cells)], skipping blank rows and csv errors."""
    reader = csv.reader(io.StringIO(text, newline=""))
    out: List[Record] = []
    while True:
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:
            log.warning("CSV error near line %d: %s (row skipped)", line_offset + reader.line_num, exc)
            continue
        if not row or all(not c.strip() for c in row):
            continue
        out.append((line_offset + reader.line_num, row))
    return out


def _records_by_start(text: str) -> List[Record]:
    """Parse each record on its own, split at every line that looks like a record start.

    An unterminated quote can then cost at most the row it is in.
    """
    starts = [m.start() for m in _RECORD_START.finditer(text)]
    out: List[Record] = []
    for i, begin in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(text)
        out.extend(_read_records(text[begin:stop], line_offset=text.count("\n", 0, begin)))
    return out


def parse_questions_csv(text: str) -> List[Round]:
    """Parse CSV text into rounds sorted by round id.  Bad rows are skipped."""
    text = _clean(text)
    index = check_header(text)
    records = _read_records(text)
    if not records:                      # cannot happen after check_header; stay safe anyway
        raise SourceError("questions.csv is empty")
    header_cells = records[0][1]
    data = records[1:]

    # A stray quote makes the csv module read to the end of the file as one field.
    # Detect that (fewer records than lines that look like record starts, or a
    # start line buried inside a field) and re-parse record by record.
    starts = len(_RECORD_START.findall(text))
    swallowed = any(_EMBEDDED_START.search(cell) for _, row in data for cell in row)
    if len(data) < starts or swallowed:
        recovered = _records_by_start(text)
        if len(recovered) > len(data):
            log.warning("CSV has %d record-looking lines but parsed %d rows (unterminated quote?); "
                        "re-parsed record by record: %d rows", starts, len(data), len(recovered))
            data = recovered

    width = len(header_cells)
    rounds: Dict[int, Round] = {}
    release_times: Dict[int, set] = {}
    for lineno, row in data:
        if len(row) != width:
            log.warning("line %d: expected %d columns, got %d (row skipped)", lineno, width, len(row))
            continue

        def cell(name: str) -> str:
            return row[index[name]].strip()

        try:
            round_id = int(cell("round_id"))
            q_index = int(cell("q_index"))
            released = parse_timestamp(cell("released_at"))
        except (ValueError, OverflowError) as exc:
            log.warning("line %d: bad round_id/q_index/released_at (%s); row skipped", lineno, exc)
            continue
        if not 1 <= round_id <= MAX_ROUND_ID:
            log.warning("line %d: round_id %d outside 1..%d; row skipped", lineno, round_id, MAX_ROUND_ID)
            continue
        if not 1 <= q_index <= NUM_QUESTIONS:
            log.warning("line %d: round %d q_index %d outside 1..%d; row skipped", lineno, round_id, q_index, NUM_QUESTIONS)
            continue
        text_q = cell("question_text")
        options = {letter: cell("option_%s" % letter.lower()) for letter in OPTION_LETTERS}
        if not text_q:
            log.warning("line %d: round %d q%d has empty question text; row skipped", lineno, round_id, q_index)
            continue
        if not any(options.values()):
            log.warning("line %d: round %d q%d has no options; row skipped", lineno, round_id, q_index)
            continue
        rnd = rounds.get(round_id)
        if rnd is None:
            rnd = rounds[round_id] = Round(round_id=round_id, released_at=released)
            release_times[round_id] = set()
        release_times[round_id].add(released)
        if q_index in rnd.questions:
            log.warning("line %d: duplicate round %d q%d; keeping the later row", lineno, round_id, q_index)
        rnd.questions[q_index] = Question(text=text_q, options=options)
    for rid, times in release_times.items():
        if len(times) > 1:
            earliest = min(times)
            log.warning("round %d has %d different released_at values; using the earliest (%s)", rid, len(times), format_timestamp(earliest))
            rounds[rid].released_at = earliest
    return [rounds[k] for k in sorted(rounds)]


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------
_CA_BUNDLES = (
    "/etc/ssl/cert.pem",                       # macOS
    "/etc/ssl/certs/ca-certificates.crt",      # Debian / Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",        # RHEL / Fedora
)
_ssl_ctx: Optional[ssl.SSLContext] = None


def ssl_context() -> ssl.SSLContext:
    """Verifying TLS context that also works with a python.org build that has no CA bundle.

    Verification is never switched off: if the default context has no trusted roots
    (common on macOS python.org installs), certifi or the system bundle is loaded.
    """
    global _ssl_ctx
    if _ssl_ctx is not None:
        return _ssl_ctx
    ctx = ssl.create_default_context()
    paths = ssl.get_default_verify_paths()
    has_roots = ctx.cert_store_stats().get("x509_ca", 0) > 0 or (paths.capath and os.path.isdir(paths.capath))
    if not has_roots:
        candidates = []
        try:
            import certifi  # optional, not required

            candidates.append(certifi.where())
        except Exception:
            pass
        candidates.extend(_CA_BUNDLES)
        for path in candidates:
            if path and os.path.exists(path):
                try:
                    ctx.load_verify_locations(cafile=path)
                    break
                except (OSError, ssl.SSLError):
                    continue
    _ssl_ctx = ctx
    return ctx


def _http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = HTTP_TIMEOUT) -> bytes:
    hdrs = {"User-Agent": USER_AGENT}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
        return resp.read()


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def fetch_via_clone(repo: str) -> str:
    url = "https://github.com/%s.git" % repo
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    with tempfile.TemporaryDirectory(prefix="quizsrc-") as tmp:
        dest = os.path.join(tmp, "repo")
        cmd = ["git", "clone", "--depth", "1", "--branch", "main", "--quiet", url, dest]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CLONE_TIMEOUT, env=env)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SourceError("git clone failed: %s" % exc)
        if proc.returncode != 0:
            raise SourceError("git clone exit %s: %s" % (proc.returncode, (proc.stderr or "").strip()[:200]))
        path = os.path.join(dest, CSV_NAME)
        try:
            with open(path, "rb") as fh:
                return _decode(fh.read())
        except OSError as exc:
            raise SourceError("clone has no %s: %s" % (CSV_NAME, exc))


def fetch_via_api(repo: str) -> str:
    url = "https://api.github.com/repos/%s/contents/%s" % (repo, CSV_NAME)
    headers = {
        "Accept": "application/vnd.github.raw",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = "Bearer %s" % token
    try:
        return _decode(_http_get(url, headers))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise SourceError("GitHub API fetch failed: %s" % exc)


def fetch_via_raw(repo: str) -> str:
    url = "https://raw.githubusercontent.com/%s/main/%s?cb=%d" % (repo, CSV_NAME, int(time.time()))
    try:
        return _decode(_http_get(url, {"Cache-Control": "no-cache"}))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise SourceError("raw.githubusercontent.com fetch failed: %s" % exc)


def fetch_questions_text(repo: Optional[str] = None) -> str:
    """Return validated CSV text from the first source that works."""
    repo = repo or os.environ.get("QUIZ_SOURCE_REPO", "").strip() or DEFAULT_REPO
    sources = [
        ("git clone", fetch_via_clone),
        ("GitHub API", fetch_via_api),
        ("raw CDN", fetch_via_raw),
    ]
    errors: List[str] = []
    for name, fn in sources:
        try:
            text = fn(repo)
            check_header(text)
        except Exception as exc:  # any failure of one source -> try the next
            log.warning("source %s failed: %s", name, exc)
            errors.append("%s: %s" % (name, exc))
            continue
        log.info("fetched %s via %s (%d bytes)", CSV_NAME, name, len(text))
        return text
    raise SourceError("all sources failed: " + " | ".join(errors))


def load_rounds(questions_file: Optional[str] = None, repo: Optional[str] = None) -> List[Round]:
    """Rounds from a local file (tests/overrides) or from the network."""
    if questions_file:
        try:
            with open(questions_file, "rb") as fh:
                text = _decode(fh.read())
        except OSError as exc:
            raise SourceError("cannot read %s: %s" % (questions_file, exc))
    else:
        text = fetch_questions_text(repo)
    return parse_questions_csv(text)
