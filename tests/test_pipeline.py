"""Offline tests for the pipeline half of the bot (source, llm, pipeline, gitops, run.py).

No network, no real solver: a fake solver and fake HTTP/git layers are injected.
Run:  python3 -m unittest tests.test_pipeline -v
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import quizbot  # noqa: E402
from quizbot import gitops, llm, pipeline, source  # noqa: E402
from quizbot.source import Question, Round  # noqa: E402

import run as run_module  # noqa: E402

UTC = timezone.utc
REL = datetime(2026, 10, 17, 21, 0, 0, tzinfo=UTC)  # release time of round 49
Sol = namedtuple("Sol", "letter method detail")
HEADER = "round_id,q_index,released_at,question_text,option_a,option_b,option_c,option_d"


_NET_GUARD = None


def _no_network(*args, **kwargs):
    raise AssertionError("test tried to use the network")


def setUpModule():
    global _NET_GUARD
    logging.disable(logging.CRITICAL)
    # Safety net: any test that forgets to mock HTTP fails loudly instead of going online.
    _NET_GUARD = mock.patch("urllib.request.urlopen", _no_network)
    _NET_GUARD.start()


def tearDownModule():
    if _NET_GUARD is not None:
        _NET_GUARD.stop()
    logging.disable(logging.NOTSET)


class FakeSolver:
    """'RULE B ...' is always solvable; 'HARD B ...' only after learn_hard is set."""

    def __init__(self):
        self.learn_hard = False
        self.calls = 0

    def __call__(self, text, options):
        self.calls += 1
        m = re.match(r"(RULE|HARD) ([ABCD])\b", text)
        if m and (m.group(1) == "RULE" or self.learn_hard):
            return Sol(m.group(2), "rule:fake", "solved %s" % m.group(2))
        return Sol(None, "abstain:unmatched", "")


def make_round(round_id, released=REL, texts=None, n=25, letter="B", kind="RULE"):
    if texts is None:
        texts = ["%s %s question %d" % (kind, letter, i) for i in range(1, n + 1)]
    questions = {}
    for i, text in enumerate(texts, start=1):
        questions[i] = Question(text, {"A": "1", "B": "2", "C": "3", "D": "4"})
    return Round(round_id, released, questions)


def csv_line(round_id, q_index, released, text, opts=("1", "2", "3", "4")):
    def q(s):
        return '"%s"' % s.replace('"', '""')

    return ",".join([str(round_id), str(q_index), released, q(text)] + [q(o) for o in opts])


def make_csv(lines, header=HEADER, newline="\n", bom=False):
    text = newline.join([header] + list(lines)) + newline
    return ("﻿" if bom else "") + text


class TempRootTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="quizbot-test-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.solver = FakeSolver()

    def run_pipe(self, rounds, now, **kw):
        kw.setdefault("solver", self.solver)
        return pipeline.run_pipeline(rounds, root=self.root, now=now, **kw)

    def read_answers(self, rid):
        with open(pipeline.answer_path(self.root, rid), encoding="utf-8") as fh:
            return json.load(fh)

    def read_state(self):
        with open(pipeline.state_path(self.root), encoding="utf-8") as fh:
            return json.load(fh)

    def snapshot(self):
        snap = {}
        for base, _dirs, files in os.walk(self.root):
            for name in files:
                p = os.path.join(base, name)
                st = os.stat(p)
                with open(p, "rb") as fh:
                    snap[os.path.relpath(p, self.root)] = (st.st_mtime_ns, fh.read())
        return snap


# ==========================================================================
# source: CSV parsing + fetch fallbacks
# ==========================================================================
class CsvParsingTest(unittest.TestCase):
    def test_basic_round_grouping_and_sorting(self):
        text = make_csv([
            csv_line(50, 1, "2026-10-17T22:00:00Z", "later"),
            csv_line(49, 2, "2026-10-17T21:00:00Z", "second"),
            csv_line(49, 1, "2026-10-17T21:00:00Z", "first"),
        ])
        rounds = source.parse_questions_csv(text)
        self.assertEqual([r.round_id for r in rounds], [49, 50])
        self.assertEqual(sorted(rounds[0].questions), [1, 2])
        self.assertEqual(rounds[0].questions[1].text, "first")
        self.assertEqual(rounds[0].released_at, datetime(2026, 10, 17, 21, 0, tzinfo=UTC))
        self.assertEqual(rounds[0].questions[1].options, {"A": "1", "B": "2", "C": "3", "D": "4"})

    def test_quoted_commas_and_quotes(self):
        line = csv_line(49, 1, "2026-10-17T21:00:00Z", 'Costs are $12,000.00, and "x" is 5%, right?',
                        ("$1,533", "$2,000.50", '"quoted"', "n/a, none"))
        rounds = source.parse_questions_csv(make_csv([line]))
        q = rounds[0].questions[1]
        self.assertEqual(q.text, 'Costs are $12,000.00, and "x" is 5%, right?')
        self.assertEqual(q.options["A"], "$1,533")
        self.assertEqual(q.options["C"], '"quoted"')
        self.assertEqual(q.options["D"], "n/a, none")

    def test_bom_and_crlf(self):
        text = make_csv([csv_line(49, 1, "2026-10-17T21:00:00Z", "a"), csv_line(49, 2, "2026-10-17T21:00:00Z", "b")],
                        newline="\r\n", bom=True)
        rounds = source.parse_questions_csv(text)
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0].questions[2].text, "b")
        self.assertEqual(rounds[0].questions[2].options["D"], "4")

    def test_unquoted_fields_and_blank_lines(self):
        text = HEADER + "\n49,1,2026-10-17T21:00:00Z,Plain question,1,2,3,4\n\n\n49,2,2026-10-17T21:00:00Z,Another,5,6,7,8\n"
        rounds = source.parse_questions_csv(text)
        self.assertEqual(sorted(rounds[0].questions), [1, 2])

    def test_embedded_newline_in_quoted_field(self):
        text = HEADER + '\n49,1,2026-10-17T21:00:00Z,"line one\nline two",1,2,3,4\n'
        rounds = source.parse_questions_csv(text)
        self.assertEqual(rounds[0].questions[1].text, "line one\nline two")

    def test_malformed_rows_are_skipped_not_fatal(self):
        good = csv_line(49, 1, "2026-10-17T21:00:00Z", "good")
        lines = [
            good,
            "49,2,2026-10-17T21:00:00Z,too,few,columns",                                  # wrong column count
            "x,3,2026-10-17T21:00:00Z,bad round,1,2,3,4",                                  # non-int round_id
            "49,y,2026-10-17T21:00:00Z,bad index,1,2,3,4",                                 # non-int q_index
            "49,4,not-a-date,bad date,1,2,3,4",                                            # bad timestamp
            "49,0,2026-10-17T21:00:00Z,index zero,1,2,3,4",                                # q_index < 1
            "49,26,2026-10-17T21:00:00Z,index 26,1,2,3,4",                                 # q_index > 25
            "49,5,2026-10-17T21:00:00Z,,1,2,3,4",                                          # empty question
            "49,6,2026-10-17T21:00:00Z,no options,,,,",                                    # no options
            "0,7,2026-10-17T21:00:00Z,round zero,1,2,3,4",                                 # round_id < 1
            csv_line(49, 8, "2026-10-17T21:00:00Z", "also good"),
        ]
        rounds = source.parse_questions_csv(make_csv(lines))
        self.assertEqual([r.round_id for r in rounds], [49])
        self.assertEqual(sorted(rounds[0].questions), [1, 8])

    def test_duplicate_row_keeps_later_and_mixed_release_uses_earliest(self):
        lines = [
            csv_line(49, 1, "2026-10-17T21:00:00Z", "old"),
            csv_line(49, 1, "2026-10-17T20:59:00Z", "new"),
        ]
        rounds = source.parse_questions_csv(make_csv(lines))
        self.assertEqual(rounds[0].questions[1].text, "new")
        self.assertEqual(rounds[0].released_at, datetime(2026, 10, 17, 20, 59, tzinfo=UTC))

    def test_header_validation(self):
        for bad in ("", "<html>404</html>", "round_id,q_index\n1,2\n",
                    HEADER + ",extra\n", HEADER.replace("q_index", "question_index") + "\n"):
            with self.assertRaises(source.SourceError):
                source.parse_questions_csv(bad)

    def test_timestamp_forms(self):
        self.assertEqual(source.parse_timestamp("2026-10-15T21:00:00Z"), datetime(2026, 10, 15, 21, tzinfo=UTC))
        self.assertEqual(source.parse_timestamp("2026-10-15T14:00:00-07:00"), datetime(2026, 10, 15, 21, tzinfo=UTC))
        self.assertEqual(source.parse_timestamp("2026-10-15T21:00:00"), datetime(2026, 10, 15, 21, tzinfo=UTC))
        self.assertEqual(source.format_timestamp(datetime(2026, 10, 15, 21, tzinfo=UTC)), "2026-10-15T21:00:00Z")
        with self.assertRaises(ValueError):
            source.parse_timestamp("yesterday")

    def test_content_hash_depends_on_content_only(self):
        a = make_round(49)
        b = make_round(49, released=REL + timedelta(hours=5))
        c = make_round(49, texts=["RULE B question %d" % i for i in range(1, 25)] + ["RULE B changed"])
        self.assertEqual(a.content_hash(), b.content_hash())
        self.assertNotEqual(a.content_hash(), c.content_hash())
        self.assertEqual(len(a.content_hash()), 64)

    def test_load_rounds_from_file_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q.csv")
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(make_csv([csv_line(49, 1, "2026-10-17T21:00:00Z", "q")], newline="\r\n", bom=True))
            self.assertEqual(source.load_rounds(path)[0].round_id, 49)
            with self.assertRaises(source.SourceError):
                source.load_rounds(os.path.join(tmp, "missing.csv"))


class FetchFallbackTest(unittest.TestCase):
    GOOD = make_csv([csv_line(49, 1, "2026-10-17T21:00:00Z", "q")])

    def test_clone_command_and_result(self):
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            dest = cmd[-1]
            os.makedirs(dest)
            with open(os.path.join(dest, "questions.csv"), "w", encoding="utf-8") as fh:
                fh.write(self.GOOD)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch("quizbot.source.subprocess.run", side_effect=fake_run):
            text = source.fetch_via_clone("Owner/Repo")
        self.assertEqual(text, self.GOOD)
        cmd = seen["cmd"]
        self.assertEqual(cmd[:2], ["git", "clone"])
        for token in ("--depth", "1", "--branch", "main", "https://github.com/Owner/Repo.git"):
            self.assertIn(token, cmd)

    def test_clone_failure_raises_source_error(self):
        with mock.patch("quizbot.source.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 128, "", "fatal: not found")):
            with self.assertRaises(source.SourceError):
                source.fetch_via_clone("Owner/Repo")

    def test_api_uses_raw_accept_and_token(self):
        calls = []

        def fake_get(url, headers=None, timeout=0):
            calls.append((url, headers))
            return self.GOOD.encode("utf-8")

        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "tok123"}), mock.patch("quizbot.source._http_get", fake_get):
            self.assertEqual(source.fetch_via_api("Owner/Repo"), self.GOOD)
        url, headers = calls[0]
        self.assertEqual(url, "https://api.github.com/repos/Owner/Repo/contents/questions.csv")
        self.assertEqual(headers["Accept"], "application/vnd.github.raw")
        self.assertEqual(headers["Authorization"], "Bearer tok123")

    def test_api_without_token_has_no_auth_header(self):
        calls = []
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("quizbot.source._http_get", lambda u, h=None, timeout=0: calls.append(h) or b"x"):
            source.fetch_via_api("Owner/Repo")
        self.assertNotIn("Authorization", calls[0])

    def test_raw_url_has_cache_buster(self):
        calls = []
        with mock.patch("quizbot.source._http_get", lambda u, h=None, timeout=0: calls.append(u) or b"x"):
            source.fetch_via_raw("Owner/Repo")
        self.assertRegex(calls[0], r"^https://raw\.githubusercontent\.com/Owner/Repo/main/questions\.csv\?cb=\d+$")

    def test_falls_through_clone_to_api_to_raw(self):
        order = []

        def clone(repo):
            order.append("clone")
            raise source.SourceError("no git")

        def api(repo):
            order.append("api")
            return "<html>rate limited</html>"      # fails header validation

        def raw(repo):
            order.append("raw")
            return self.GOOD

        with mock.patch.multiple("quizbot.source", fetch_via_clone=clone, fetch_via_api=api, fetch_via_raw=raw):
            self.assertEqual(source.fetch_questions_text("Owner/Repo"), self.GOOD)
        self.assertEqual(order, ["clone", "api", "raw"])

    def test_stops_at_first_good_source_and_uses_env_repo(self):
        seen = []

        def clone(repo):
            seen.append(repo)
            return self.GOOD

        def boom(repo):
            raise AssertionError("must not be called")

        with mock.patch.dict(os.environ, {"QUIZ_SOURCE_REPO": "Me/Mine"}), \
                mock.patch.multiple("quizbot.source", fetch_via_clone=clone, fetch_via_api=boom, fetch_via_raw=boom):
            source.fetch_questions_text()
        self.assertEqual(seen, ["Me/Mine"])

    def test_ssl_context_loads_system_bundle_only_when_default_has_no_roots(self):
        class FakeCtx:
            def __init__(self, roots):
                self.roots = roots
                self.loaded = []

            def cert_store_stats(self):
                return {"x509_ca": self.roots}

            def load_verify_locations(self, cafile=None, **kw):
                self.loaded.append(cafile)

        Paths = namedtuple("Paths", "cafile capath")
        for roots, expect in ((0, ["/etc/ssl/cert.pem"]), (5, [])):
            ctx = FakeCtx(roots)
            with mock.patch.object(source, "_ssl_ctx", None), \
                    mock.patch("quizbot.source.ssl.create_default_context", return_value=ctx), \
                    mock.patch("quizbot.source.ssl.get_default_verify_paths", return_value=Paths(None, None)), \
                    mock.patch("quizbot.source.os.path.exists", lambda p: p == "/etc/ssl/cert.pem"):
                self.assertIs(source.ssl_context(), ctx)
            self.assertEqual(ctx.loaded, expect)

    def test_all_sources_fail(self):
        def bad(repo):
            raise source.SourceError("down")

        with mock.patch.multiple("quizbot.source", fetch_via_clone=bad, fetch_via_api=bad, fetch_via_raw=bad):
            with self.assertRaises(source.SourceError) as ctx:
                source.fetch_questions_text("Owner/Repo")
        self.assertIn("all sources failed", str(ctx.exception))


# ==========================================================================
# pipeline
# ==========================================================================
class EditableWindowTest(unittest.TestCase):
    def test_boundaries(self):
        self.assertTrue(pipeline.is_editable(REL + timedelta(hours=2, minutes=49), REL))
        self.assertTrue(pipeline.is_editable(REL + timedelta(hours=2, minutes=49, seconds=59), REL))
        self.assertFalse(pipeline.is_editable(REL + timedelta(hours=2, minutes=50), REL))
        self.assertFalse(pipeline.is_editable(REL + timedelta(hours=2, minutes=51), REL))
        self.assertTrue(pipeline.is_editable(REL - timedelta(minutes=1), REL))

    def test_method_rank(self):
        self.assertEqual(pipeline.method_rank("rule:compound_interest"), 3)
        self.assertEqual(pipeline.method_rank("llm:openai/gpt-4.1"), 2)
        self.assertEqual(pipeline.method_rank("llm-disagree:openai/o3"), 2)
        self.assertEqual(pipeline.method_rank("guess"), 1)
        self.assertEqual(pipeline.method_rank("guess:missing"), 1)
        self.assertEqual(pipeline.method_rank(""), 0)
        self.assertEqual(pipeline.method_rank(None), 0)
        self.assertEqual(pipeline.method_rank("weird"), 0)


class AnswerFileTest(TempRootTest):
    def test_new_round_json_shape_is_exact(self):
        letters = "ABCD" * 6 + "A"
        texts = ["RULE %s q%d" % (letters[i], i + 1) for i in range(25)]
        result = self.run_pipe([make_round(49, texts=texts)], REL + timedelta(minutes=7))
        self.assertEqual(result.new_rounds, [49])
        path = os.path.join(self.root, "answers", "round-0049.json")
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        expected = {"round_id": 49, "answers": {"q%d" % (i + 1): letters[i] for i in range(25)}}
        self.assertEqual(raw, json.dumps(expected, indent=2) + "\n")
        obj = json.loads(raw)
        self.assertEqual(list(obj), ["round_id", "answers"])
        self.assertEqual(list(obj["answers"]), ["q%d" % i for i in range(1, 26)])  # q2 before q10
        self.assertIs(type(obj["round_id"]), int)
        self.assertEqual(sorted(os.listdir(os.path.join(self.root, "answers"))), ["round-0049.json"])

    def test_zero_padded_names_and_answers_dir_created_on_write(self):
        self.assertFalse(os.path.exists(os.path.join(self.root, "answers")))
        self.run_pipe([make_round(7, REL), make_round(168, REL)], REL)
        self.assertEqual(sorted(os.listdir(os.path.join(self.root, "answers"))),
                         ["round-0007.json", "round-0168.json"])

    def test_validate_answers_obj_rejects_bad_shapes(self):
        good = pipeline.build_answers_obj(49, {i: "A" for i in range(1, 26)})
        pipeline.validate_answers_obj(good, 49)

        def mutate(fn):
            obj = json.loads(json.dumps(good))
            fn(obj)
            return obj

        bad = [
            mutate(lambda o: o.pop("round_id")),
            mutate(lambda o: o.update(extra=1)),
            mutate(lambda o: o.update(round_id="49")),
            mutate(lambda o: o.update(round_id=True)),
            mutate(lambda o: o.update(round_id=50)),
            mutate(lambda o: o["answers"].pop("q25")),
            mutate(lambda o: o["answers"].update(q26="A")),
            mutate(lambda o: o["answers"].update(q3="E")),
            mutate(lambda o: o["answers"].update(q3="a")),
            mutate(lambda o: o["answers"].update(q3=None)),
            mutate(lambda o: o.update(answers=list(o["answers"].values()))),
            {"answers": good["answers"], "round_id": 49},       # wrong key order
            [],
        ]
        for obj in bad:
            with self.assertRaises(ValueError, msg=repr(obj)[:80]):
                pipeline.validate_answers_obj(obj, 49)

    def test_atomic_write_leaves_no_temp_file_on_failure(self):
        target = os.path.join(self.root, "answers", "round-0049.json")
        with mock.patch("quizbot.pipeline.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                pipeline.write_json_atomic(target, {"a": 1})
        self.assertEqual(os.listdir(os.path.join(self.root, "answers")), [])
        pipeline.write_json_atomic(target, {"b": 2, "a": 1}, sort_keys=True)
        with open(target, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), '{\n  "a": 1,\n  "b": 2\n}\n')
        self.assertEqual(os.listdir(os.path.join(self.root, "answers")), ["round-0049.json"])

    def test_stray_temp_files_are_removed(self):
        os.makedirs(os.path.join(self.root, "answers"))
        stray = os.path.join(self.root, "answers", ".tmp-abc123.json")
        open(stray, "w").close()
        self.run_pipe([make_round(49)], REL)
        self.assertFalse(os.path.exists(stray))
        self.assertEqual(os.listdir(os.path.join(self.root, "answers")), ["round-0049.json"])


class StateAndStatusTest(TempRootTest):
    def test_state_shape(self):
        texts = ["RULE C q%d" % i for i in range(1, 24)] + ["HARD D q24", "HARD D q25"]
        now = REL + timedelta(minutes=12, seconds=30)
        self.run_pipe([make_round(49, texts=texts)], now)
        state = self.read_state()
        self.assertEqual(list(state), ["49"])
        entry = state["49"]
        self.assertEqual(sorted(entry), ["answered_at", "details", "issue_opened", "methods", "questions_sha256",
                                         "released_at", "solver_version", "updated_at"])
        self.assertEqual(entry["released_at"], "2026-10-17T21:00:00Z")
        self.assertEqual(entry["answered_at"], "2026-10-17T21:12:30Z")
        self.assertEqual(entry["updated_at"], "2026-10-17T21:12:30Z")
        self.assertEqual(entry["solver_version"], quizbot.SOLVER_VERSION)
        self.assertFalse(entry["issue_opened"])
        self.assertEqual(entry["questions_sha256"], make_round(49, texts=texts).content_hash())
        self.assertEqual(sorted(entry["methods"]), sorted("q%d" % i for i in range(1, 26)))
        self.assertEqual(entry["methods"]["q1"], "rule:fake")
        self.assertEqual(entry["methods"]["q24"], "guess")
        self.assertEqual(sorted(entry["details"]), sorted(entry["methods"]))
        with open(pipeline.state_path(self.root), encoding="utf-8") as fh:
            raw = fh.read()
        self.assertEqual(raw, json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n")

    def test_status_md_has_utc_pacific_and_table(self):
        now = datetime(2026, 10, 17, 21, 7, 12, tzinfo=UTC)
        self.run_pipe([make_round(49)], now)
        with open(os.path.join(self.root, "STATUS.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("2026-10-17 21:07:12 UTC", text)
        self.assertIn("2026-10-17 14:07:12 PDT", text)
        self.assertIn("Answered rounds: 1", text)
        self.assertRegex(text, r"\| 49 \| 2026-10-17 21:00:00 \| 2026-10-17 21:07:12 \| 7 \| 25 \| 0 \| 0 \|")

    def test_status_lists_only_latest_15_rounds(self):
        rounds = [make_round(rid, released=REL + timedelta(hours=rid - 49)) for rid in range(49, 69)]
        self.run_pipe(rounds, REL + timedelta(days=1))
        with open(os.path.join(self.root, "STATUS.md"), encoding="utf-8") as fh:
            text = fh.read()
        rows = [l for l in text.splitlines() if re.match(r"\| \d+ \|", l)]
        self.assertEqual(len(rows), 15)
        self.assertTrue(rows[0].startswith("| 68 |"))
        self.assertTrue(rows[-1].startswith("| 54 |"))
        self.assertIn("Answered rounds: 20", text)

    def test_pacific_conversion_matches_fallback(self):
        instants = [
            datetime(2026, 10, 17, 21, 0, tzinfo=UTC),
            datetime(2026, 3, 8, 9, 59, tzinfo=UTC),    # just before spring forward
            datetime(2026, 3, 8, 10, 0, tzinfo=UTC),    # PDT starts
            datetime(2026, 11, 1, 8, 59, tzinfo=UTC),   # last PDT minute
            datetime(2026, 11, 1, 9, 0, tzinfo=UTC),    # PST starts
            datetime(2026, 12, 1, 0, 0, tzinfo=UTC),
        ]
        real = [pipeline.to_pacific(dt) for dt in instants]
        with mock.patch("zoneinfo.ZoneInfo", side_effect=KeyError("no tzdata")):
            fallback = [pipeline.to_pacific(dt) for dt in instants]
        for dt, a, b in zip(instants, real, fallback):
            self.assertEqual((a[0].replace(tzinfo=None), a[1]), (b[0].replace(tzinfo=None), b[1]), str(dt))
        self.assertEqual(fallback[0][1], "PDT")
        self.assertEqual(fallback[0][0].hour, 14)
        self.assertEqual(fallback[4][1], "PST")
        self.assertEqual(fallback[4][0].hour, 1)


class DecisionRulesTest(TempRootTest):
    def test_multiple_rounds_at_once(self):
        rounds = [make_round(49, REL), make_round(50, REL + timedelta(hours=1)), make_round(51, REL + timedelta(hours=2))]
        result = self.run_pipe(rounds, REL + timedelta(hours=2, minutes=5))
        self.assertEqual(result.new_rounds, [49, 50, 51])
        self.assertEqual(result.updated_rounds, [])
        self.assertEqual(sorted(self.read_state()), ["49", "50", "51"])
        for rid in (49, 50, 51):
            self.assertEqual(set(self.read_answers(rid)["answers"].values()), {"B"})

    def test_late_round_is_still_answered(self):
        result = self.run_pipe([make_round(49)], REL + timedelta(hours=6))
        self.assertEqual(result.new_rounds, [49])
        self.assertTrue(os.path.exists(pipeline.answer_path(self.root, 49)))

    def test_idempotent_second_run_writes_nothing(self):
        rounds = [make_round(49), make_round(50, kind="HARD")]   # round 50 has guesses that stay unresolved
        now = REL + timedelta(minutes=10)
        first = self.run_pipe(rounds, now)
        self.assertTrue(first.changed)
        before = self.snapshot()
        second = self.run_pipe(rounds, now + timedelta(minutes=5))
        self.assertFalse(second.changed)
        self.assertEqual(second.written, [])
        self.assertEqual(self.snapshot(), before)           # identical bytes AND mtimes: nothing was rewritten
        third = self.run_pipe(rounds, now + timedelta(hours=5))
        self.assertFalse(third.changed)
        self.assertEqual(self.snapshot(), before)

    def test_method_upgrade_guess_to_rule_inside_window(self):
        rnd = make_round(49, kind="HARD")
        self.run_pipe([rnd], REL + timedelta(minutes=5))
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"A"})
        self.assertEqual(set(self.read_state()["49"]["methods"].values()), {"guess"})
        answered_at = self.read_state()["49"]["answered_at"]

        self.solver.learn_hard = True
        result = self.run_pipe([rnd], REL + timedelta(hours=2, minutes=49))
        self.assertEqual(result.updated_rounds, [49])
        self.assertEqual(result.new_rounds, [])
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"B"})
        entry = self.read_state()["49"]
        self.assertEqual(set(entry["methods"].values()), {"rule:fake"})
        self.assertEqual(entry["answered_at"], answered_at)           # first answer time is kept
        self.assertEqual(entry["updated_at"], "2026-10-17T23:49:00Z")

    def test_no_upgrade_after_the_window(self):
        rnd = make_round(49, kind="HARD")
        self.run_pipe([rnd], REL + timedelta(minutes=5))
        self.solver.learn_hard = True
        before = self.snapshot()
        result = self.run_pipe([rnd], REL + timedelta(hours=2, minutes=51))
        self.assertFalse(result.changed)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"A"})
        self.assertEqual(result.results[0].action, "skip")

    def test_exactly_2h50_is_already_locked(self):
        rnd = make_round(49, kind="HARD")
        self.run_pipe([rnd], REL + timedelta(minutes=5))
        self.solver.learn_hard = True
        result = self.run_pipe([rnd], REL + timedelta(hours=2, minutes=50))
        self.assertFalse(result.changed)

    def test_upgrade_only_touches_questions_below_rule(self):
        texts = ["RULE C q%d" % i for i in range(1, 21)] + ["HARD D q%d" % i for i in range(21, 26)]
        self.run_pipe([make_round(49, texts=texts)], REL + timedelta(minutes=1))
        self.solver.learn_hard = True
        calls_before = self.solver.calls
        self.run_pipe([make_round(49, texts=texts)], REL + timedelta(minutes=20))
        self.assertEqual(self.solver.calls - calls_before, 5)       # only the 5 non-rule questions re-solved
        answers = self.read_answers(49)["answers"]
        self.assertEqual([answers["q%d" % i] for i in (1, 20)], ["C", "C"])
        self.assertEqual([answers["q%d" % i] for i in (21, 25)], ["D", "D"])

    def test_content_hash_change_resolves_everything_inside_window(self):
        rnd = make_round(49, letter="B")
        self.run_pipe([rnd], REL + timedelta(minutes=5))
        changed = make_round(49, texts=["RULE D q%d" % i for i in range(1, 26)])
        result = self.run_pipe([changed], REL + timedelta(minutes=30))
        self.assertEqual(result.updated_rounds, [49])
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"D"})
        self.assertEqual(self.read_state()["49"]["questions_sha256"], changed.content_hash())

    def test_content_hash_change_after_window_is_ignored(self):
        rnd = make_round(49, letter="B")
        self.run_pipe([rnd], REL + timedelta(minutes=5))
        changed = make_round(49, texts=["RULE D q%d" % i for i in range(1, 26)])
        before = self.snapshot()
        result = self.run_pipe([changed], REL + timedelta(hours=3, minutes=1))
        self.assertFalse(result.changed)
        self.assertEqual(self.snapshot(), before)

    def test_rule_answers_are_never_downgraded(self):
        rnd = make_round(49, letter="C")
        self.run_pipe([rnd], REL + timedelta(minutes=5))
        broken = mock.Mock(return_value=Sol(None, "abstain:broken", ""))
        before = self.snapshot()
        result = self.run_pipe([rnd], REL + timedelta(minutes=10), solver=broken)
        self.assertFalse(result.changed)
        self.assertEqual(self.snapshot(), before)

    def test_no_state_entry_leaves_file_untouched(self):
        os.makedirs(os.path.join(self.root, "answers"))
        manual = json.dumps({"round_id": 49, "answers": {"q%d" % i: "D" for i in range(1, 26)}}, indent=2) + "\n"
        path = pipeline.answer_path(self.root, 49)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(manual)
        result = self.run_pipe([make_round(49, letter="B")], REL + timedelta(minutes=5))
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), manual)
        self.assertFalse(result.changed)
        self.assertEqual(result.results[0].action, "skip")
        self.assertFalse(os.path.exists(pipeline.state_path(self.root)))

    def test_invalid_file_without_state_is_untouched_but_with_state_is_regenerated(self):
        os.makedirs(os.path.join(self.root, "answers"))
        path = pipeline.answer_path(self.root, 49)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.run_pipe([make_round(49)], REL + timedelta(minutes=5))
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "{not json")
        # now pretend the bot owns it
        os.remove(path)
        self.run_pipe([make_round(49)], REL + timedelta(minutes=6))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{corrupt")
        result = self.run_pipe([make_round(49)], REL + timedelta(hours=5))
        self.assertEqual(result.new_rounds, [49])
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"B"})

    def test_deleted_answer_file_is_rewritten_even_after_window(self):
        self.run_pipe([make_round(49)], REL + timedelta(minutes=5))
        os.remove(pipeline.answer_path(self.root, 49))
        result = self.run_pipe([make_round(49)], REL + timedelta(hours=4))
        self.assertEqual(result.new_rounds, [49])
        self.assertTrue(os.path.exists(pipeline.answer_path(self.root, 49)))

    def test_fewer_than_25_questions_pads_with_guess_and_is_revisited(self):
        short = make_round(49, n=20)
        self.run_pipe([short], REL + timedelta(minutes=2))
        answers = self.read_answers(49)["answers"]
        self.assertEqual(len(answers), 25)
        self.assertEqual(answers["q20"], "B")
        self.assertEqual([answers["q%d" % i] for i in range(21, 26)], ["A"] * 5)
        methods = self.read_state()["49"]["methods"]
        self.assertEqual(methods["q20"], "rule:fake")
        self.assertEqual({methods["q%d" % i] for i in range(21, 26)}, {"guess:missing"})

        # the remaining 5 questions show up later (content hash changes) -> everything re-solved
        full = make_round(49, n=25)
        result = self.run_pipe([full], REL + timedelta(minutes=9))
        self.assertEqual(result.updated_rounds, [49])
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"B"})
        self.assertEqual(set(self.read_state()["49"]["methods"].values()), {"rule:fake"})

    def test_short_round_is_stable_on_rerun(self):
        short = make_round(49, n=20)
        self.run_pipe([short], REL + timedelta(minutes=2))
        before = self.snapshot()
        result = self.run_pipe([short], REL + timedelta(minutes=7))
        self.assertFalse(result.changed)
        self.assertEqual(self.snapshot(), before)

    def test_dry_run_writes_nothing(self):
        result = self.run_pipe([make_round(49)], REL + timedelta(minutes=5), dry_run=True)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.written, [])
        self.assertEqual(os.listdir(self.root), [])
        self.assertEqual(result.results[0].action, "new")
        self.assertEqual(result.results[0].answers[1].letter, "B")
        self.assertIn("round 49: new", pipeline.format_dry_run(result))

    def test_min_round_filter(self):
        result = self.run_pipe([make_round(1), make_round(49)], REL, min_round=49)
        self.assertEqual([r.round_id for r in result.results], [49])
        self.assertEqual(os.listdir(os.path.join(self.root, "answers")), ["round-0049.json"])

    def test_solver_exception_becomes_guess(self):
        def explode(text, options):
            raise RuntimeError("boom")

        self.run_pipe([make_round(49)], REL, solver=explode)
        self.assertEqual(set(self.read_state()["49"]["methods"].values()), {"guess"})
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"A"})

    def test_solver_returning_invalid_letter_is_treated_as_abstain(self):
        weird = lambda text, options: Sol("Z", "rule:bad", "")
        self.run_pipe([make_round(49)], REL, solver=weird)
        self.assertEqual(set(self.read_state()["49"]["methods"].values()), {"guess"})

    def test_issue_opened_flag_survives_updates(self):
        rnd = make_round(49, kind="HARD")
        self.run_pipe([rnd], REL + timedelta(minutes=5))
        state = self.read_state()
        state["49"]["issue_opened"] = True
        pipeline.save_state(self.root, state)
        self.solver.learn_hard = True
        self.run_pipe([rnd], REL + timedelta(minutes=30))
        self.assertTrue(self.read_state()["49"]["issue_opened"])


class LLMIntegrationTest(TempRootTest):
    def fake_llm(self, letter="C", model="m1"):
        calls = []

        def fn(items):
            calls.append(list(items))
            return {i.id: llm.LLMAnswer(letter, "llm:%s" % model, "asked") for i in items}

        fn.calls = calls
        return fn

    def test_abstained_questions_are_batched_into_one_llm_call(self):
        rounds = [make_round(49, kind="HARD"), make_round(50, kind="HARD", released=REL + timedelta(hours=1))]
        fn = self.fake_llm("C")
        self.run_pipe(rounds, REL + timedelta(hours=1, minutes=5), llm_fn=fn)
        self.assertEqual(len(fn.calls), 1)
        self.assertEqual(len(fn.calls[0]), 50)
        self.assertEqual(fn.calls[0][0].id, "r49q1")
        self.assertEqual(fn.calls[0][-1].id, "r50q25")
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"C"})
        self.assertEqual(set(self.read_state()["50"]["methods"].values()), {"llm:m1"})

    def test_rule_solved_questions_are_not_sent_to_the_llm(self):
        texts = ["RULE B q%d" % i for i in range(1, 24)] + ["HARD B q24", "HARD B q25"]
        fn = self.fake_llm("D")
        self.run_pipe([make_round(49, texts=texts)], REL, llm_fn=fn)
        self.assertEqual([i.id for i in fn.calls[0]], ["r49q24", "r49q25"])
        answers = self.read_answers(49)["answers"]
        self.assertEqual((answers["q23"], answers["q24"], answers["q25"]), ("B", "D", "D"))

    def test_llm_answer_is_upgraded_by_rule_and_not_reasked(self):
        rnd = make_round(49, kind="HARD", letter="B")
        fn = self.fake_llm("C")
        self.run_pipe([rnd], REL + timedelta(minutes=2), llm_fn=fn)
        self.assertEqual(len(fn.calls), 1)
        self.run_pipe([rnd], REL + timedelta(minutes=8), llm_fn=fn)       # still abstains; rank 2 -> no new LLM call
        self.assertEqual(len(fn.calls), 1)
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"C"})
        self.solver.learn_hard = True
        self.run_pipe([rnd], REL + timedelta(minutes=14), llm_fn=fn)
        self.assertEqual(len(fn.calls), 1)
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"B"})
        self.assertEqual(set(self.read_state()["49"]["methods"].values()), {"rule:fake"})

    def test_guess_is_retried_with_llm_later(self):
        rnd = make_round(49, kind="HARD")
        self.run_pipe([rnd], REL + timedelta(minutes=2), llm_fn=lambda items: {})   # LLM unavailable
        self.assertEqual(set(self.read_state()["49"]["methods"].values()), {"guess"})
        fn = self.fake_llm("D")
        result = self.run_pipe([rnd], REL + timedelta(minutes=8), llm_fn=fn)
        self.assertEqual(result.updated_rounds, [49])
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"D"})

    def test_llm_failure_never_breaks_the_run(self):
        def boom(items):
            raise RuntimeError("llm down")

        self.run_pipe([make_round(49, kind="HARD")], REL, llm_fn=boom)
        self.assertEqual(set(self.read_state()["49"]["methods"].values()), {"guess"})

    def test_llm_not_called_when_nothing_abstains_or_disabled(self):
        fn = self.fake_llm()
        self.run_pipe([make_round(49)], REL, llm_fn=fn)
        self.assertEqual(fn.calls, [])
        self.run_pipe([make_round(50, kind="HARD", released=REL)], REL, llm_fn=None)
        self.assertEqual(set(self.read_state()["50"]["methods"].values()), {"guess"})

    def test_llm_answer_with_bad_letter_is_ignored(self):
        bad = lambda items: {i.id: llm.LLMAnswer("Z", "llm:m", "") for i in items}
        self.run_pipe([make_round(49, kind="HARD")], REL, llm_fn=bad)
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"A"})


# ==========================================================================
# llm
# ==========================================================================
def reply(answers):
    return json.dumps({"answers": [{"id": k, "work": "w", "answer": v} for k, v in answers.items()]})


class FakeModels:
    """Stand-in for llm._http_json: records POST bodies, serves per-model replies."""

    def __init__(self, replies, catalog="unset"):
        self.replies = replies
        self.catalog = catalog
        self.requests = []

    def __call__(self, method, url, headers, body=None, timeout=60):
        if url == llm.CATALOG_URL:
            if self.catalog == "unset":
                raise llm.LLMRequestError(500, "no catalog")
            return self.catalog
        assert url == llm.INFERENCE_URL, url
        assert headers["Authorization"] == "Bearer tok"
        self.requests.append(body)
        out = self.replies[body["model"]]
        if isinstance(out, Exception):
            raise out
        if callable(out):
            out = out(body)
        return {"choices": [{"message": {"content": out}}]}

    def models_used(self):
        return [r["model"] for r in self.requests]


def items(n, prefix="r49q"):
    return [llm.LLMItem("%s%d" % (prefix, i), "Question %d?" % i, {"A": "1", "B": "2", "C": "3", "D": "4"})
            for i in range(1, n + 1)]


class LLMParsingTest(unittest.TestCase):
    IDS = ["r49q1", "r49q2", "r49q3"]

    def test_bare_json(self):
        got = llm.parse_answers(reply({"r49q1": "A", "r49q2": "D"}), self.IDS)
        self.assertEqual(got, {"r49q1": ("A", "w"), "r49q2": ("D", "w")})

    def test_fenced_json(self):
        content = "```json\n" + reply({"r49q1": "B", "r49q3": "C"}) + "\n```"
        self.assertEqual({k: v[0] for k, v in llm.parse_answers(content, self.IDS).items()},
                         {"r49q1": "B", "r49q3": "C"})

    def test_extra_prose_and_stray_braces(self):
        content = ("Sure! Here is the {result} you asked for:\n\n"
                   + reply({"r49q2": "C"}) + "\n\nLet me know if you need {anything} else.")
        self.assertEqual({k: v[0] for k, v in llm.parse_answers(content, self.IDS).items()}, {"r49q2": "C"})

    def test_braces_inside_work_string(self):
        content = json.dumps({"answers": [{"id": "r49q1", "work": "set {x} = 5 -> \"y\"", "answer": "D"}]})
        self.assertEqual(llm.parse_answers(content, self.IDS), {"r49q1": ("D", 'set {x} = 5 -> "y"')})

    def test_bad_letters_unknown_ids_and_duplicates_are_dropped(self):
        content = json.dumps({"answers": [
            {"id": "r49q1", "answer": "E"},
            {"id": "r49q2", "answer": "AB"},
            {"id": "r49q3", "answer": None},
            {"id": "r49q9", "answer": "A"},                 # unknown id
            {"id": 5, "answer": "A"},
            "junk",
            {"id": "r49q1", "answer": "c"},                 # lower case is fine
            {"id": "r49q1", "answer": "D"},                 # duplicate: first valid wins
            {"id": "r49q2", "answer": "(b)"},
        ]})
        got = llm.parse_answers(content, self.IDS)
        self.assertEqual({k: v[0] for k, v in got.items()}, {"r49q1": "C", "r49q2": "B"})

    def test_letter_normalisation(self):
        for raw, want in (("A", "A"), (" d ", "D"), ("b.", "B"), ("(C)", "C"), ("Option B", "B"), ("answer: a", "A"),
                          ("E", None), ("", None), ("AB", None), ("A, B", None), ("none", None), (3, None), (None, None)):
            self.assertEqual(llm.normalize_letter(raw), want, repr(raw))

    def test_garbage(self):
        for content in ("", "no json here", "{", "{}", '{"answers": "nope"}', '{"foo": []}', None, 12, "[1,2]"):
            self.assertEqual(llm.parse_answers(content, self.IDS), {})

    def test_reply_text_variants(self):
        self.assertEqual(llm._reply_text({"choices": [{"message": {"content": "hi"}}]}), "hi")
        self.assertEqual(llm._reply_text({"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}), "ab")
        self.assertEqual(llm._reply_text({}), "")
        self.assertEqual(llm._reply_text({"choices": []}), "")


class LLMRunTest(unittest.TestCase):
    def ask(self, fake, its, **kw):
        kw.setdefault("token", "tok")
        with mock.patch("quizbot.llm._http_json", fake):
            return llm.answer_questions(its, **kw)

    def test_two_models_agree(self):
        its = items(3)
        fake = FakeModels({"m1": lambda b: reply({i.id: "B" for i in its}),
                           "m2": lambda b: reply({i.id: "B" for i in its})})
        got = self.ask(fake, its, models=["m1", "m2", "m3"])
        self.assertEqual(fake.models_used(), ["m1", "m2"])        # never more than 2 models per chunk
        self.assertEqual({k: (v.letter, v.method) for k, v in got.items()},
                         {i.id: ("B", "llm:m1") for i in its})

    def test_two_models_disagree_takes_first_and_flags_it(self):
        its = items(2)
        fake = FakeModels({"m1": lambda b: reply({"r49q1": "A", "r49q2": "B"}),
                           "m2": lambda b: reply({"r49q1": "A", "r49q2": "C"})})
        got = self.ask(fake, its, models=["m1", "m2"])
        self.assertEqual((got["r49q1"].letter, got["r49q1"].method), ("A", "llm:m1"))
        self.assertEqual((got["r49q2"].letter, got["r49q2"].method), ("B", "llm-disagree:m1"))
        self.assertIn("m2->C", got["r49q2"].detail)

    def test_http_429_falls_through_to_next_model(self):
        its = items(2)
        fake = FakeModels({"m1": llm.LLMRequestError(429, "rate limited"),
                           "m2": lambda b: reply({i.id: "D" for i in its}),
                           "m3": lambda b: reply({i.id: "D" for i in its})})
        got = self.ask(fake, its, models=["m1", "m2", "m3"])
        self.assertEqual(fake.models_used(), ["m1", "m2", "m3"])
        self.assertEqual({v.method for v in got.values()}, {"llm:m2"})
        self.assertEqual({v.letter for v in got.values()}, {"D"})

    def test_timeout_and_unusable_reply_fall_through(self):
        its = items(1)
        fake = FakeModels({"m1": llm.LLMRequestError(0, "timed out"),
                           "m2": "I cannot answer that.",
                           "m3": lambda b: reply({"r49q1": "C"})})
        got = self.ask(fake, its, models=["m1", "m2", "m3"])
        self.assertEqual(got["r49q1"].method, "llm:m3")

    def test_single_working_model(self):
        its = items(1)
        fake = FakeModels({"m1": lambda b: reply({"r49q1": "A"}), "m2": llm.LLMRequestError(429)})
        got = self.ask(fake, its, models=["m1", "m2"])
        self.assertEqual(got["r49q1"].method, "llm:m1")

    def test_all_models_fail_returns_empty(self):
        fake = FakeModels({"m1": llm.LLMRequestError(429), "m2": llm.LLMRequestError(500)})
        self.assertEqual(self.ask(fake, items(3), models=["m1", "m2"]), {})

    def test_dead_model_is_not_retried_for_later_chunks(self):
        its = items(30)
        fake = FakeModels({"m1": llm.LLMRequestError(429), "m2": lambda b: reply({"r49q%d" % i: "A" for i in range(1, 31)})})
        self.ask(fake, its, models=["m1", "m2"], chunk_size=10)
        self.assertEqual(fake.models_used().count("m1"), 1)

    def test_request_cap(self):
        its = items(175)                                          # 7 chunks of 25
        fake = FakeModels({"m1": lambda b: reply({"r49q%d" % i: "A" for i in range(1, 176)}),
                           "m2": lambda b: reply({"r49q%d" % i: "A" for i in range(1, 176)})})
        got = self.ask(fake, its, models=["m1", "m2"])
        self.assertEqual(len(fake.requests), llm.MAX_REQUESTS)
        self.assertEqual(llm.MAX_REQUESTS, 6)
        self.assertEqual(len(got), 150)                           # 6 chunks got a first opinion, chunk 7 did not

    def test_cap_is_shared_between_first_and_second_opinions(self):
        its = items(50)                                           # 2 chunks -> 4 requests with 2 models
        fake = FakeModels({"m1": lambda b: reply({"r49q%d" % i: "A" for i in range(1, 51)}),
                           "m2": lambda b: reply({"r49q%d" % i: "A" for i in range(1, 51)})})
        self.ask(fake, its, models=["m1", "m2"])
        self.assertEqual(len(fake.requests), 4)
        fake = FakeModels({"m1": lambda b: reply({"r49q%d" % i: "A" for i in range(1, 51)}),
                           "m2": lambda b: reply({"r49q%d" % i: "A" for i in range(1, 51)})})
        self.ask(fake, its, models=["m1", "m2"], max_requests=3)
        self.assertEqual(len(fake.requests), 3)

    def test_cap_includes_the_catalog_request(self):
        its = items(175)
        everything = {"r49q%d" % i: "A" for i in range(1, 176)}
        catalog = [{"id": "m1"}, {"id": "m2"}]
        fake = FakeModels({"m1": lambda b: reply(everything), "m2": lambda b: reply(everything)}, catalog=catalog)
        http_calls = []

        def counting(method, url, headers, body=None, timeout=60):
            http_calls.append(url)
            return fake(method, url, headers, body, timeout)

        env = {"QUIZ_LLM_MODELS": "m1,m2"}
        with mock.patch("quizbot.llm._http_json", counting), mock.patch.dict(os.environ, env):
            got = llm.answer_questions(its, token="tok")
        self.assertEqual(len(http_calls), llm.MAX_REQUESTS)
        self.assertEqual(http_calls[0], llm.CATALOG_URL)
        self.assertEqual(len(fake.requests), llm.MAX_REQUESTS - 1)
        self.assertEqual(len(got), 125)

    def test_temperature_only_for_non_reasoning_models(self):
        its = items(1)
        models = ["openai/gpt-5", "openai/o3", "openai/o4-mini", "deepseek/deepseek-r1", "openai/gpt-4.1", "openai/gpt-4o"]
        fake = FakeModels({m: llm.LLMRequestError(500) for m in models})
        self.ask(fake, its, models=models, max_requests=10)
        sent = {r["model"]: r for r in fake.requests}
        for m in models[:4]:
            self.assertNotIn("temperature", sent[m], m)
        for m in models[4:]:
            self.assertEqual(sent[m]["temperature"], 0, m)
        body = sent["openai/gpt-4.1"]
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertIn("[r49q1] Question 1?", body["messages"][1]["content"])
        self.assertIn("D) 4", body["messages"][1]["content"])
        self.assertEqual(set(body), {"model", "messages", "temperature"})

    def test_never_raises(self):
        def boom(*a, **k):
            raise RuntimeError("kaboom")

        with mock.patch("quizbot.llm._http_json", boom):
            self.assertEqual(llm.answer_questions(items(2), token="tok", models=["m1"]), {})
        with mock.patch("quizbot.llm.get_token", side_effect=ValueError("x")):
            self.assertEqual(llm.answer_questions(items(2)), {})

    def test_no_token_means_no_requests(self):
        fake = FakeModels({})
        with mock.patch("quizbot.llm.get_token", return_value=None), mock.patch("quizbot.llm._http_json", fake):
            self.assertEqual(llm.answer_questions(items(2)), {})
        self.assertEqual(fake.requests, [])

    def test_empty_items(self):
        self.assertEqual(llm.answer_questions([]), {})

    def test_discovery_uses_catalog_and_preference_order(self):
        catalog = [{"id": "openai/gpt-4o"}, {"id": "openai/o3"}, {"id": "meta/llama"}, {"id": "openai/gpt-4.1"}]
        fake = FakeModels({}, catalog=catalog)
        with mock.patch("quizbot.llm._http_json", fake):
            self.assertEqual(llm.discover_models("tok", env={}), ["openai/o3", "openai/gpt-4.1", "openai/gpt-4o"])
            self.assertEqual(llm.discover_models("tok", env={"QUIZ_LLM_MODELS": "openai/gpt-4o, openai/o3,nope/x"}),
                             ["openai/gpt-4o", "openai/o3"])

    def test_discovery_falls_back_when_catalog_is_down_or_useless(self):
        defaults = llm.preferred_models({})
        self.assertEqual(defaults, llm.DEFAULT_MODELS.split(","))
        with mock.patch("quizbot.llm._http_json", FakeModels({})):        # catalog -> 500
            self.assertEqual(llm.discover_models("tok", env={}), defaults)
        with mock.patch("quizbot.llm._http_json", FakeModels({}, catalog=[{"id": "other/model"}])):
            self.assertEqual(llm.discover_models("tok", env={}), defaults)
        with mock.patch("quizbot.llm._http_json", FakeModels({}, catalog={"models": [{"id": "openai/o3"}]})):
            self.assertEqual(llm.discover_models("tok", env={}), ["openai/o3"])

    def test_get_token_sources(self):
        self.assertEqual(llm.get_token({"GITHUB_TOKEN": " abc ", "GH_TOKEN": "zzz"}), "abc")
        self.assertEqual(llm.get_token({"GH_TOKEN": "zzz"}), "zzz")
        with mock.patch("quizbot.llm.shutil.which", return_value=None):
            self.assertIsNone(llm.get_token({}))
        with mock.patch("quizbot.llm.shutil.which", return_value="/usr/bin/gh"), \
                mock.patch("quizbot.llm.subprocess.run",
                           return_value=subprocess.CompletedProcess([], 0, "ghp_fromcli\n", "")) as run:
            self.assertEqual(llm.get_token({}), "ghp_fromcli")
            self.assertEqual(run.call_args[0][0], ["/usr/bin/gh", "auth", "token"])
        with mock.patch("quizbot.llm.shutil.which", return_value="/usr/bin/gh"), \
                mock.patch("quizbot.llm.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "not logged in")):
            self.assertIsNone(llm.get_token({}))

    def test_http_json_maps_errors(self):
        import urllib.error

        err = urllib.error.HTTPError("http://x", 429, "Too Many", {}, io.BytesIO(b"slow down"))
        with mock.patch("quizbot.llm.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(llm.LLMRequestError) as ctx:
                llm._http_json("POST", "http://x", {}, {"a": 1})
        self.assertEqual(ctx.exception.status, 429)
        with mock.patch("quizbot.llm.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(llm.LLMRequestError) as ctx:
                llm._http_json("GET", "http://x", {})
        self.assertEqual(ctx.exception.status, 0)


# ==========================================================================
# gitops
# ==========================================================================
class FakeRunner:
    """Scripted stand-in for subprocess.run used by gitops.Git."""

    def __init__(self, handler=None):
        self.calls = []
        self.handler = handler or (lambda args: (0, "", ""))

    def __call__(self, cmd, **kwargs):
        args = cmd[1:]
        self.calls.append(args)
        rc, out, err = self.handler(args)
        return subprocess.CompletedProcess(cmd, rc, out, err)

    def commands(self):
        return [" ".join(a) for a in self.calls]


def fake_result(new=(), updated=(), rounds=None):
    res = pipeline.PipelineResult(results=[], state={}, rounds=rounds or {})
    res.results = [pipeline.RoundResult(r, "new", is_new=True, file_changed=True) for r in new] + \
                  [pipeline.RoundResult(r, "upgrade", file_changed=True) for r in updated]
    return res


class CommitMessageTest(unittest.TestCase):
    def test_messages(self):
        self.assertEqual(gitops.commit_message([49], []), "Answer round 49")
        self.assertEqual(gitops.commit_message([49, 50], []), "Answer rounds 49, 50")
        self.assertEqual(gitops.commit_message([], [49]), "Update round 49")
        self.assertEqual(gitops.commit_message([], [49, 50]), "Update rounds 49, 50")
        self.assertEqual(gitops.commit_message([51], [49]), "Answer round 51; Update round 49")
        self.assertEqual(gitops.commit_message(list(range(1, 49)), []), "Answer rounds 1-48")
        self.assertEqual(gitops.commit_message([50, 49, 49], []), "Answer rounds 49, 50")
        self.assertEqual(gitops.commit_message([], []), "Update quiz state")

    def test_round_list_compression(self):
        self.assertEqual(gitops.format_round_list([49, 50, 51]), "49-51")
        self.assertEqual(gitops.format_round_list([49, 51, 52, 53, 60]), "49, 51-53, 60")
        self.assertEqual(gitops.format_round_list([3]), "3")

    def test_result_properties_drive_message(self):
        res = fake_result(new=[50], updated=[49])
        self.assertEqual(gitops.commit_message(res.new_rounds, res.updated_rounds), "Answer round 50; Update round 49")


class GitFlowTest(unittest.TestCase):
    def flow(self, handler=None, env=None, pipeline_results=None, sleeps=None):
        runner = FakeRunner(handler)
        sleeps = sleeps if sleeps is not None else []
        results = list(pipeline_results or [fake_result(new=[49])])
        calls = []

        def pipeline_fn():
            calls.append(1)
            return results.pop(0) if len(results) > 1 else results[0]

        rc = gitops.run_git_flow(pipeline_fn, root="/repo", env=env if env is not None else {}, runner=runner,
                                 sleep=sleeps.append)
        return rc, runner, sleeps, calls

    @staticmethod
    def dirty(args):
        if args[:2] == ["status", "--porcelain"]:
            return 0, " M answers/round-0049.json\n", ""
        return 0, "", ""

    def test_reset_only_when_allowed(self):
        _, runner, _, _ = self.flow(env={})
        cmds = runner.commands()
        self.assertFalse(any(c.startswith("reset") for c in cmds), cmds)
        self.assertIn("pull --rebase --autostash origin main", cmds)
        _, runner, _, _ = self.flow(env={"QUIZBOT_ALLOW_RESET": "1"})
        cmds = runner.commands()
        self.assertEqual(cmds[:2], ["fetch origin main", "reset --hard origin/main"])
        self.assertFalse(any(c.startswith("pull") for c in cmds))
        _, runner, _, _ = self.flow(env={"QUIZBOT_ALLOW_RESET": "0"})
        self.assertFalse(any(c.startswith("reset") for c in runner.commands()))

    def test_nothing_to_commit_exits_zero_without_push(self):
        rc, runner, sleeps, calls = self.flow(env={"QUIZBOT_ALLOW_RESET": "1"})
        self.assertEqual(rc, 0)
        cmds = runner.commands()
        self.assertIn("status --porcelain -- answers state STATUS.md", cmds)
        self.assertFalse(any(c.startswith(("add", "commit", "push")) for c in cmds))
        self.assertEqual(sleeps, [])
        self.assertEqual(len(calls), 1)

    def test_commit_and_push(self):
        rc, runner, sleeps, _ = self.flow(self.dirty, env={"QUIZBOT_ALLOW_RESET": "1", "GITHUB_ACTIONS": "true"})
        self.assertEqual(rc, 0)
        cmds = runner.commands()
        self.assertIn("add -A -- answers state STATUS.md", cmds)
        commit = [c for c in runner.calls if "commit" in c][0]
        self.assertEqual(commit[-2:], ["-m", "Answer round 49"])
        self.assertIn("user.name=github-actions[bot]", commit)
        self.assertIn("user.email=41898282+github-actions[bot]@users.noreply.github.com", commit)
        self.assertEqual(cmds[-1], "push origin HEAD:main")
        self.assertEqual(sleeps, [])

    def test_local_identity_is_used_when_configured(self):
        def handler(args):
            if args == ["config", "user.name"]:
                return 0, "Ming\n", ""
            if args == ["config", "user.email"]:
                return 0, "ming@example.com\n", ""
            return self.dirty(args)

        rc, runner, _, _ = self.flow(handler, env={"QUIZBOT_ALLOW_RESET": "1"})
        commit = [c for c in runner.calls if "commit" in c][0]
        self.assertEqual(rc, 0)
        self.assertNotIn("-c", commit)
        # no configured identity -> falls back to the bot identity
        rc, runner, _, _ = self.flow(self.dirty, env={"QUIZBOT_ALLOW_RESET": "1"})
        commit = [c for c in runner.calls if "commit" in c][0]
        self.assertIn("user.name=github-actions[bot]", commit)

    def test_push_rejected_then_retry_reruns_pipeline(self):
        pushes = []

        def handler(args):
            if args[0] == "push":
                pushes.append(1)
                return (1, "", "! [rejected] non-fast-forward") if len(pushes) == 1 else (0, "", "")
            return self.dirty(args)

        rc, runner, sleeps, calls = self.flow(handler, env={"QUIZBOT_ALLOW_RESET": "1"})
        self.assertEqual(rc, 0)
        self.assertEqual(sleeps, [5])
        self.assertEqual(len(calls), 2)                      # pipeline re-run after resync
        self.assertEqual(runner.commands().count("reset --hard origin/main"), 2)

    def test_gives_up_after_five_attempts_with_backoff(self):
        def handler(args):
            if args[0] == "push":
                return 1, "", "rejected"
            return self.dirty(args)

        rc, runner, sleeps, calls = self.flow(handler, env={"QUIZBOT_ALLOW_RESET": "1"})
        self.assertEqual(rc, 1)
        self.assertEqual(sleeps, [5, 10, 20, 30])
        self.assertEqual(len(calls), 5)
        self.assertEqual(runner.commands().count("push origin HEAD:main"), 5)

    def test_sync_failure_consumes_attempts(self):
        def handler(args):
            return (1, "", "network down") if args[0] == "fetch" else (0, "", "")

        rc, runner, sleeps, calls = self.flow(handler, env={"QUIZBOT_ALLOW_RESET": "1"})
        self.assertEqual(rc, 1)
        self.assertEqual(sleeps, [5, 10, 20, 30])
        self.assertEqual(calls, [])

    def test_source_error_exits_nonzero_without_retry(self):
        def pipeline_fn():
            raise source.SourceError("all sources failed")

        sleeps = []
        rc = gitops.run_git_flow(pipeline_fn, root="/repo", env={"QUIZBOT_ALLOW_RESET": "1"},
                                 runner=FakeRunner(), sleep=sleeps.append)
        self.assertEqual(rc, 1)
        self.assertEqual(sleeps, [])

    def test_unpushed_local_commit_is_pushed_even_if_tree_is_clean(self):
        def handler(args):
            if args[0] == "rev-list":
                return 0, "1\n", ""
            return 0, "", ""

        rc, runner, _, _ = self.flow(handler, env={})
        self.assertEqual(rc, 0)
        self.assertEqual(runner.commands()[-1], "push origin HEAD:main")
        self.assertFalse(any(c.startswith("commit") for c in runner.commands()))

    def test_rebase_failure_aborts_rebase(self):
        def handler(args):
            return (1, "", "conflict") if args[0] == "pull" else (0, "", "")

        rc, runner, _, calls = self.flow(handler, env={})
        self.assertEqual(rc, 1)
        self.assertIn("rebase --abort", runner.commands())
        self.assertEqual(calls, [])


class IssueNotificationTest(TempRootTest):
    ENV = {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "tok", "GITHUB_REPOSITORY": "me/DSO429HW3"}

    def setUp(self):
        super().setUp()
        texts = ["RULE B q%d" % i for i in range(1, 24)] + ["HARD C q24", "HARD C q25"]
        self.rounds = {
            49: make_round(49, texts=texts),
            50: make_round(50, released=REL + timedelta(hours=1)),                 # all rules -> no issue
            51: make_round(51, released=REL + timedelta(hours=2), n=24),           # only guess:missing -> no issue
        }
        pipeline.run_pipeline(list(self.rounds.values()), root=self.root, now=REL + timedelta(hours=2, minutes=3),
                              solver=self.solver, llm_fn=None)

    def make_request(self, existing=(), fail_post=False):
        calls = []

        def request(method, url, token, body=None, timeout=20):
            calls.append((method, url, body))
            if method == "GET":
                return [{"title": t} for t in existing]
            if fail_post:
                raise OSError("422")
            return {"number": 1}

        request.calls = calls
        return request

    def test_one_issue_per_round_and_never_duplicated(self):
        req = self.make_request()
        opened = gitops.notify_issues(self.root, self.rounds, self.ENV, req)
        self.assertEqual(opened, [49])
        posts = [c for c in req.calls if c[0] == "POST"]
        self.assertEqual(len(posts), 1)
        method, url, body = posts[0]
        self.assertEqual(url, "https://api.github.com/repos/me/DSO429HW3/issues")
        self.assertEqual(body["title"], "Round 49: 2 question(s) not solved by rules")
        self.assertIn("HARD C q24", body["body"])
        self.assertIn("A) 1", body["body"])
        self.assertIn("Chosen: **A**", body["body"])
        self.assertIn("`guess`", body["body"])
        self.assertNotIn("q23", body["body"])
        self.assertTrue(self.read_state()["49"]["issue_opened"])
        self.assertFalse(self.read_state()["50"]["issue_opened"])
        self.assertFalse(self.read_state()["51"]["issue_opened"])

        req2 = self.make_request()
        self.assertEqual(gitops.notify_issues(self.root, self.rounds, self.ENV, req2), [])
        self.assertEqual(req2.calls, [])                       # state says opened: not even a GET

    def test_existing_issue_with_same_round_prefix_is_not_duplicated(self):
        req = self.make_request(existing=["Round 49: 3 question(s) not solved by rules", "Unrelated"])
        self.assertEqual(gitops.notify_issues(self.root, self.rounds, self.ENV, req), [])
        self.assertEqual([c for c in req.calls if c[0] == "POST"], [])
        self.assertTrue(self.read_state()["49"]["issue_opened"])

    def test_only_in_actions_with_token_and_repo(self):
        for env in ({}, {**self.ENV, "GITHUB_ACTIONS": "false"}, {**self.ENV, "GITHUB_TOKEN": ""},
                    {k: v for k, v in self.ENV.items() if k != "GITHUB_REPOSITORY"}):
            req = self.make_request()
            self.assertEqual(gitops.notify_issues(self.root, self.rounds, env, req), [])
            self.assertEqual(req.calls, [], env)
        self.assertFalse(self.read_state()["49"]["issue_opened"])

    def test_failure_is_logged_not_fatal_and_retried_next_time(self):
        req = self.make_request(fail_post=True)
        self.assertEqual(gitops.notify_issues(self.root, self.rounds, self.ENV, req), [])
        self.assertFalse(self.read_state()["49"]["issue_opened"])
        req2 = self.make_request()
        self.assertEqual(gitops.notify_issues(self.root, self.rounds, self.ENV, req2), [49])

    def test_listing_failure_still_opens_issue(self):
        calls = []

        def request(method, url, token, body=None, timeout=20):
            calls.append(method)
            if method == "GET":
                raise OSError("boom")
            return {}

        self.assertEqual(gitops.notify_issues(self.root, self.rounds, self.ENV, request), [49])
        self.assertEqual(calls, ["GET", "POST"])

    def test_never_raises(self):
        def request(*a, **k):
            raise RuntimeError("x")

        self.assertEqual(gitops.notify_issues(self.root, self.rounds, self.ENV, request), [])
        self.assertEqual(gitops.notify_issues("/nonexistent/dir", self.rounds, self.ENV, self.make_request()), [])


@unittest.skipUnless(shutil.which("git"), "git not installed")
class GitEndToEndTest(TempRootTest):
    """Real git against a local bare repo: commit message, files pushed, idempotent second run."""

    def git(self, cwd, *args):
        return subprocess.run(["git"] + list(args), cwd=cwd, check=True, capture_output=True, text=True).stdout

    def test_flow_pushes_answers_and_second_run_is_a_noop(self):
        env = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
        with mock.patch.dict(os.environ, env):
            bare = os.path.join(self.root, "remote.git")
            seed = os.path.join(self.root, "seed")
            work = os.path.join(self.root, "work")
            self.git(self.root, "init", "--bare", bare)
            self.git(bare, "symbolic-ref", "HEAD", "refs/heads/main")
            self.git(self.root, "clone", bare, seed)
            self.git(seed, "checkout", "-b", "main")
            with open(os.path.join(seed, ".gitignore"), "w") as fh:
                fh.write("__pycache__/\n")
            self.git(seed, "add", ".gitignore")
            self.git(seed, "commit", "-m", "init")
            self.git(seed, "push", "origin", "main")
            self.git(self.root, "clone", bare, work)

            rounds = [make_round(49), make_round(50, released=REL + timedelta(hours=1))]
            now = REL + timedelta(hours=1, minutes=6)

            def pipeline_fn():
                return pipeline.run_pipeline(rounds, root=work, now=now, solver=self.solver)

            flow_env = {"QUIZBOT_ALLOW_RESET": "1"}
            self.assertEqual(gitops.run_git_flow(pipeline_fn, root=work, env=flow_env, sleep=lambda s: None), 0)
            log = self.git(bare, "log", "--format=%s", "main").splitlines()
            self.assertEqual(log[0], "Answer rounds 49, 50")
            files = self.git(bare, "ls-tree", "-r", "--name-only", "main").splitlines()
            self.assertEqual(sorted(files), [".gitignore", "STATUS.md", "answers/round-0049.json",
                                             "answers/round-0050.json", "state/rounds.json"])
            self.assertEqual(self.git(work, "status", "--porcelain").strip(), "")

            commits_before = self.git(bare, "rev-list", "--count", "main").strip()
            self.assertEqual(gitops.run_git_flow(pipeline_fn, root=work, env=flow_env, sleep=lambda s: None), 0)
            self.assertEqual(self.git(bare, "rev-list", "--count", "main").strip(), commits_before)


# ==========================================================================
# run.py
# ==========================================================================
class CliTest(TempRootTest):
    def write_csv(self, rounds_lines):
        path = os.path.join(self.root, "q.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(make_csv(rounds_lines))
        return path

    def main(self, argv, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = run_module.main(argv, root=self.root, solver=self.solver, **kw)
        return rc, out.getvalue()

    def lines(self, rid=49, released="2026-10-17T21:00:00Z"):
        return [csv_line(rid, i, released, "RULE C q%d" % i) for i in range(1, 26)]

    def test_plain_run_writes_files(self):
        path = self.write_csv(self.lines())
        rc, _ = self.main(["--questions-file", path, "--now", "2026-10-17T21:05:00Z", "--no-llm"])
        self.assertEqual(rc, 0)
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"C"})
        self.assertEqual(self.read_state()["49"]["answered_at"], "2026-10-17T21:05:00Z")

    def test_dry_run_prints_and_writes_nothing(self):
        path = self.write_csv(self.lines())
        rc, out = self.main(["--dry-run", "--no-llm", "--questions-file", path, "--now", "2026-10-17T21:05:00Z"])
        self.assertEqual(rc, 0)
        self.assertIn("round 49: new", out)
        self.assertIn("answers: " + "C" * 25, out)
        self.assertEqual([n for n in os.listdir(self.root) if n != "q.csv"], [])

    def test_no_llm_flag_prevents_llm_calls(self):
        lines = [csv_line(49, i, "2026-10-17T21:00:00Z", "HARD C q%d" % i) for i in range(1, 26)]
        path = self.write_csv(lines)
        spy = mock.Mock(return_value={})
        self.main(["--questions-file", path, "--now", "2026-10-17T21:05:00Z", "--no-llm"], llm_fn=spy)
        spy.assert_not_called()
        self.main(["--questions-file", path, "--now", "2026-10-17T21:06:00Z"], llm_fn=spy)
        spy.assert_called_once()

    def test_git_flag_runs_pipeline_then_commits_and_pushes(self):
        path = self.write_csv(self.lines())

        def handler(args):
            if args[:2] == ["status", "--porcelain"]:
                return 0, "?? answers/round-0049.json\n", ""
            return 0, "", ""

        runner = FakeRunner(handler)
        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_ACTIONS", "GITHUB_TOKEN", "GH_TOKEN")}
        env["QUIZBOT_ALLOW_RESET"] = "1"
        with mock.patch.dict(os.environ, env, clear=True):
            rc, _ = self.main(["--git", "--no-llm", "--questions-file", path, "--now", "2026-10-17T21:05:00Z"],
                              runner=runner)
        self.assertEqual(rc, 0)
        cmds = runner.commands()
        self.assertEqual(cmds[:2], ["fetch origin main", "reset --hard origin/main"])
        self.assertTrue(any("commit" in c and "Answer round 49" in c for c in cmds))
        self.assertEqual(cmds[-1], "push origin HEAD:main")
        self.assertTrue(os.path.exists(pipeline.answer_path(self.root, 49)))

    def test_bad_arguments(self):
        self.assertEqual(self.main(["--git", "--dry-run"])[0], 2)
        self.assertEqual(self.main(["--now", "garbage"])[0], 2)
        self.assertEqual(self.main(["--questions-file", os.path.join(self.root, "nope.csv"), "--no-llm"])[0], 1)

    def test_check_practice_scoring_and_exit_code(self):
        fixtures = os.path.join(self.root, "tests", "fixtures")
        os.makedirs(fixtures)
        for name in ("practice_questions.csv", "practice_keys.csv"):
            shutil.copy(os.path.join(ROOT, "tests", "fixtures", name), os.path.join(fixtures, name))
        with open(os.path.join(fixtures, "practice_questions.csv"), encoding="utf-8") as fh:
            rounds = source.parse_questions_csv(fh.read())
        import csv as csv_mod
        with open(os.path.join(fixtures, "practice_keys.csv"), encoding="utf-8", newline="") as fh:
            keys = {(int(r["round_id"]), int(r["q_index"])): r["answer"] for r in csv_mod.DictReader(fh)}
        by_text = {}
        for rnd in rounds:
            for qi, q in rnd.questions.items():
                by_text[q.text] = keys[(rnd.round_id, qi)]
        oracle = lambda text, options: Sol(by_text[text], "rule:oracle", "")
        rc, out = self._check(oracle)
        self.assertEqual(rc, 0)
        self.assertEqual(len(by_text), 1200)
        self.assertIn("accuracy: 1200/1200", out)
        self.assertIn("oracle", out)
        abstainer = lambda text, options: Sol(None, "abstain:never", "")
        rc, out = self._check(abstainer)
        self.assertEqual(rc, 1)
        self.assertIn("abstained", out)

    def _check(self, solver):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = run_module.main(["--check-practice"], root=self.root, solver=solver)
        return rc, out.getvalue()


if __name__ == "__main__":
    unittest.main()
