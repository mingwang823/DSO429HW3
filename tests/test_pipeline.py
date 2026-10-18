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
        # an extra column is fine now (see test_header_is_matched_by_name_...); a missing name is not
        for bad in ("", "<html>404</html>", "round_id,q_index\n1,2\n",
                    HEADER.replace("q_index", "question_index") + "\n"):
            with self.assertRaises(source.SourceError):
                source.parse_questions_csv(bad)

    # ---- finding 1: header matched by NAME --------------------------------------
    def test_header_is_matched_by_name_extra_and_reordered_columns(self):
        row = ["49", "1", "2026-10-17T21:00:00Z", "Q one", "o1", "o2", "o3", "o4"]
        # the reviewer's repro: one extra trailing column
        rounds = source.parse_questions_csv(HEADER + ",extra\n49,1,2026-10-17T21:00:00Z,Q,A,B,C,D,x\n")
        self.assertEqual(rounds[0].questions[1].options, {"A": "A", "B": "B", "C": "C", "D": "D"})
        # extra column in the middle, columns reordered, a trailing comment column
        names = ["note", "option_d", "question_text", "released_at", "option_b", "q_index", "option_a",
                 "round_id", "option_c", "id"]
        values = {"note": "n", "option_d": "o4", "question_text": "Q one", "released_at": "2026-10-17T21:00:00Z",
                  "option_b": "o2", "q_index": "1", "option_a": "o1", "round_id": "49", "option_c": "o3", "id": "99"}
        text = ",".join(names) + "\n" + ",".join(values[n] for n in names) + "\n"
        rounds = source.parse_questions_csv(text)
        self.assertEqual([r.round_id for r in rounds], [49])
        q = rounds[0].questions[1]
        self.assertEqual((q.text, q.options), ("Q one", {"A": "o1", "B": "o2", "C": "o3", "D": "o4"}))
        self.assertEqual(rounds[0].released_at, datetime(2026, 10, 17, 21, 0, tzinfo=UTC))
        # a plain row still works with the original layout
        self.assertEqual(source.parse_questions_csv(make_csv([",".join(row)]))[0].questions[1].text, "Q one")

    def test_header_names_tolerate_case_whitespace_quotes_bom_and_leading_blank_lines(self):
        header = '\ufeff\n  \n" Round_ID ",Q_Index , released_at,Question_Text,OPTION_A,option_b,option_c ,"option_d"'
        text = header + "\n49,1,2026-10-17T21:00:00Z,Q,1,2,3,4\n"
        rounds = source.parse_questions_csv(text)
        self.assertEqual(rounds[0].questions[1].options["D"], "4")
        source.check_header(text)       # the fetch layer validates with check_header

    def test_header_still_rejects_html_json_and_garbage(self):
        bad = [
            "<!DOCTYPE html><html><body>rate limited, a, b</body></html>",
            '{"message": "Not Found", "documentation_url": "https://docs.github.com"}',
            '[{"round_id": 1}]',
            "just some text\nwith, commas, and lines\n",
            "round_id,q_index,released_at,question_text,option_a,option_b,option_c\n1,1,x,y,1,2,3\n",   # no option_d
            HEADER.replace("option_d", "option_e") + "\n",
            "\ufeff",
            "   \n\n",
            '"unterminated,header\nnext line',
        ]
        for text in bad:
            with self.assertRaises(source.SourceError, msg=text[:40]):
                source.parse_questions_csv(text)
            with self.assertRaises(source.SourceError, msg=text[:40]):
                source.check_header(text)

    def test_rows_must_match_the_header_width(self):
        # with 9 header columns, a row of 8 cells is misaligned and is skipped (not guessed at)
        text = HEADER + ",extra\n49,1,2026-10-17T21:00:00Z,Q,1,2,3,4\n49,2,2026-10-17T21:00:00Z,Q,1,2,3,4,x\n"
        rounds = source.parse_questions_csv(text)
        self.assertEqual(sorted(rounds[0].questions), [2])

    # ---- finding 5: round_id range ---------------------------------------------------
    def test_round_id_must_be_1_to_9999(self):
        lines = [
            csv_line(1, 1, "2026-10-17T21:00:00Z", "low edge"),
            csv_line(9999, 1, "2026-10-17T21:00:00Z", "high edge"),
            csv_line(10000, 1, "2026-10-17T21:00:00Z", "too big"),
            csv_line(100000000000000000000, 1, "2026-10-17T21:00:00Z", "absurd"),
            csv_line(0, 1, "2026-10-17T21:00:00Z", "zero"),
            csv_line(-5, 1, "2026-10-17T21:00:00Z", "negative"),
            csv_line("9" * 5000, 1, "2026-10-17T21:00:00Z", "digit bomb"),
            csv_line(49, 1, "2026-10-17T21:00:00Z", "normal"),
        ]
        with self.assertLogs("quizbot.source", level="WARNING") as logs:
            logging.disable(logging.NOTSET)
            try:
                rounds = source.parse_questions_csv(make_csv(lines))
            finally:
                logging.disable(logging.CRITICAL)
        self.assertEqual([r.round_id for r in rounds], [1, 49, 9999])
        self.assertTrue(any("outside 1..9999" in m for m in logs.output), logs.output)

    def test_bogus_round_ids_never_create_answer_files(self):
        root = tempfile.mkdtemp(prefix="quizbot-test-")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        lines = [csv_line(rid, i, "2026-10-17T21:00:00Z", "RULE C q%d" % i)
                 for rid in (4900, 100000000000000000000, 10000) for i in range(1, 26)]
        rounds = source.parse_questions_csv(make_csv(lines))
        pipeline.run_pipeline(rounds, root=root, now=REL + timedelta(hours=1), solver=FakeSolver())
        self.assertEqual(os.listdir(os.path.join(root, "answers")), ["round-4900.json"])

    # ---- finding 6: an unterminated quote costs at most its own row --------------------
    def test_unterminated_quote_costs_only_its_row(self):
        # the reviewer's repro: one bad row, then a good round-50 row
        text = HEADER + '\n49,1,2026-10-17T21:00:00Z,"oops,A,B,C,D\n' + \
            csv_line(50, 1, "2026-10-17T22:00:00Z", "still here") + "\n"
        rounds = source.parse_questions_csv(text)
        self.assertEqual([r.round_id for r in rounds], [50])
        self.assertEqual(rounds[0].questions[1].text, "still here")

    def test_unterminated_quote_in_the_middle_keeps_every_other_row(self):
        lines = [csv_line(49, i, "2026-10-17T21:00:00Z", "q%d" % i) for i in range(1, 11)]
        lines[4] = '49,5,2026-10-17T21:00:00Z,"never closed,1,2,3,4'
        lines += [csv_line(50, i, "2026-10-17T22:00:00Z", "r50 q%d" % i) for i in range(1, 26)]
        for newline in ("\n", "\r\n"):
            rounds = source.parse_questions_csv(make_csv(lines, newline=newline))
            self.assertEqual([r.round_id for r in rounds], [49, 50], repr(newline))
            self.assertEqual(sorted(rounds[0].questions), [1, 2, 3, 4, 6, 7, 8, 9, 10], repr(newline))
            self.assertEqual(len(rounds[1].questions), 25)
            self.assertEqual(rounds[1].questions[25].text, "r50 q25")

    def test_unterminated_quote_in_first_and_last_row(self):
        good = [csv_line(49, i, "2026-10-17T21:00:00Z", "q%d" % i) for i in (2, 3)]
        lines = ['49,1,2026-10-17T21:00:00Z,"bad first,1,2,3,4'] + good + ['49,4,2026-10-17T21:00:00Z,"bad last,1,2,3,4']
        rounds = source.parse_questions_csv(make_csv(lines))
        self.assertEqual(sorted(rounds[0].questions), [2, 3])

    def test_recovery_leaves_legit_multiline_fields_alone(self):
        lines = ['49,1,2026-10-17T21:00:00Z,"line one\nline two",1,2,3,4',
                 csv_line(49, 2, "2026-10-17T21:00:00Z", "plain"),
                 '49,3,2026-10-17T21:00:00Z,"a, b\r\nc",1,2,3,4']
        rounds = source.parse_questions_csv(make_csv(lines))
        self.assertEqual(rounds[0].questions[1].text, "line one\nline two")
        self.assertEqual(rounds[0].questions[3].text, "a, b\r\nc")
        self.assertEqual(sorted(rounds[0].questions), [1, 2, 3])

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
        result = self.run_pipe([rnd], REL + timedelta(minutes=33), llm_fn=fn)       # 31 min later: asked again
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


class NotReleasedYetTest(TempRootTest):
    """Finding 4: a round whose released_at is more than 60 s ahead is not answered."""

    def test_future_round_is_skipped_with_a_reason_and_writes_nothing(self):
        future = make_round(49, released=REL + timedelta(hours=38))
        result = self.run_pipe([future], REL)
        self.assertEqual([(r.round_id, r.action, r.reason) for r in result.results],
                         [(49, "skip", "not released yet")])
        self.assertFalse(result.changed)
        self.assertEqual(os.listdir(self.root), [])               # no answers/, no state, no STATUS.md

    def test_the_60_second_boundary(self):
        rnd = make_round(49)
        just_over = self.run_pipe([rnd], REL - timedelta(seconds=61))
        self.assertEqual(just_over.results[0].action, "skip")
        self.assertEqual(os.listdir(self.root), [])
        within_slack = self.run_pipe([rnd], REL - timedelta(seconds=60))      # exactly 60 s: clock skew allowed
        self.assertEqual(within_slack.new_rounds, [49])

    def test_round_is_answered_once_it_is_released(self):
        rnd = make_round(49)
        self.run_pipe([rnd], REL - timedelta(minutes=5))
        self.assertFalse(os.path.exists(pipeline.answer_path(self.root, 49)))
        result = self.run_pipe([rnd], REL + timedelta(seconds=30))
        self.assertEqual(result.new_rounds, [49])
        self.assertEqual(self.read_state()["49"]["answered_at"], "2026-10-17T21:00:30Z")   # never negative

    def test_released_rounds_are_answered_while_a_later_one_waits(self):
        rounds = [make_round(49), make_round(50, released=REL + timedelta(hours=1)),
                  make_round(51, released=REL + timedelta(hours=2))]
        result = self.run_pipe(rounds, REL + timedelta(hours=1, minutes=2))
        self.assertEqual(result.new_rounds, [49, 50])
        self.assertEqual([r.action for r in result.results], ["new", "new", "skip"])
        self.assertEqual(sorted(os.listdir(os.path.join(self.root, "answers"))), ["round-0049.json", "round-0050.json"])
        self.assertEqual(sorted(self.read_state()), ["49", "50"])

    def test_future_round_with_an_existing_file_is_left_alone(self):
        rnd = make_round(49)
        self.run_pipe([rnd], REL + timedelta(minutes=1))
        before = self.snapshot()
        moved = make_round(49, released=REL + timedelta(days=1))          # the instructors re-scheduled it
        result = self.run_pipe([moved], REL + timedelta(minutes=2))
        self.assertEqual(result.results[0].reason, "not released yet")
        self.assertEqual(self.snapshot(), before)

    def test_dry_run_reports_it(self):
        result = self.run_pipe([make_round(49, released=REL + timedelta(hours=3))], REL, dry_run=True)
        self.assertIn("round 49: skip - not released yet", pipeline.format_dry_run(result))


class LLMThrottleTest(TempRootTest):
    """Finding 2: guess-ranked questions go back to the LLM at most once per 30 minutes."""

    def counting_llm(self, answer=None):
        calls = []

        def fn(items):
            calls.append([i.id for i in items])
            if answer is None:
                return {}
            return {i.id: llm.LLMAnswer(answer, "llm:m1", "asked") for i in items}

        fn.calls = calls
        return fn

    def test_not_resent_within_30_minutes_but_resent_after(self):
        rnd = make_round(49, kind="HARD")
        fn = self.counting_llm()                                   # the LLM never answers
        t0 = REL + timedelta(minutes=1)
        self.run_pipe([rnd], t0, llm_fn=fn)
        self.assertEqual(len(fn.calls), 1)
        self.assertEqual(self.read_state()["49"]["llm_attempted_at"]["q1"], "2026-10-17T21:01:00Z")
        for delta in (timedelta(minutes=1), timedelta(minutes=15), timedelta(minutes=29, seconds=59)):
            result = self.run_pipe([rnd], t0 + delta, llm_fn=fn)
            self.assertEqual(len(fn.calls), 1, delta)
            self.assertFalse(result.changed, delta)                # not even a state rewrite
        self.run_pipe([rnd], t0 + timedelta(minutes=30), llm_fn=fn)
        self.assertEqual(len(fn.calls), 2)
        self.assertEqual(self.read_state()["49"]["llm_attempted_at"]["q1"], "2026-10-17T21:31:00Z")
        self.run_pipe([rnd], t0 + timedelta(minutes=45), llm_fn=fn)
        self.assertEqual(len(fn.calls), 2)

    def test_throttle_is_per_question(self):
        texts = ["HARD B q%d" % i for i in range(1, 26)]
        rnd = make_round(49, texts=texts)
        fn = self.counting_llm()
        self.run_pipe([rnd], REL + timedelta(minutes=1), llm_fn=fn)
        state = self.read_state()
        state["49"]["llm_attempted_at"].pop("q7")                  # pretend q7 was never attempted
        state["49"]["llm_attempted_at"]["q8"] = "2026-10-17T20:00:00Z"   # q8 long ago
        pipeline.save_state(self.root, state)
        self.run_pipe([rnd], REL + timedelta(minutes=10), llm_fn=fn)
        self.assertEqual(fn.calls[1], ["r49q7", "r49q8"])

    def test_answered_questions_drop_out_of_the_attempt_list(self):
        rnd = make_round(49, kind="HARD")
        self.run_pipe([rnd], REL + timedelta(minutes=1), llm_fn=self.counting_llm())
        self.assertEqual(len(self.read_state()["49"]["llm_attempted_at"]), 25)
        self.run_pipe([rnd], REL + timedelta(minutes=40), llm_fn=self.counting_llm("C"))
        self.assertNotIn("llm_attempted_at", self.read_state()["49"])
        self.assertEqual(set(self.read_answers(49)["answers"].values()), {"C"})

    def test_a_successful_llm_answer_is_not_asked_again(self):
        rnd = make_round(49, kind="HARD")
        fn = self.counting_llm("D")
        self.run_pipe([rnd], REL + timedelta(minutes=1), llm_fn=fn)
        self.run_pipe([rnd], REL + timedelta(hours=1), llm_fn=fn)
        self.assertEqual(len(fn.calls), 1)

    def test_changed_content_restarts_the_clock(self):
        fn = self.counting_llm()
        self.run_pipe([make_round(49, kind="HARD", letter="B")], REL + timedelta(minutes=1), llm_fn=fn)
        changed = make_round(49, kind="HARD", letter="C")
        self.run_pipe([changed], REL + timedelta(minutes=5), llm_fn=fn)
        self.assertEqual(len(fn.calls), 2)                         # new questions are asked at once
        self.assertEqual(self.read_state()["49"]["llm_attempted_at"]["q1"], "2026-10-17T21:05:00Z")

    def test_questions_that_were_never_sent_do_not_start_the_clock(self):
        rnd = make_round(49, kind="HARD")
        calls = []

        def unavailable(items):
            calls.append(len(items))
            res = llm.LLMResults()
            res.unattempted.update(i.id for i in items)
            return res

        self.run_pipe([rnd], REL + timedelta(minutes=1), llm_fn=unavailable)
        self.assertNotIn("llm_attempted_at", self.read_state()["49"])
        before = self.snapshot()
        result = self.run_pipe([rnd], REL + timedelta(minutes=2), llm_fn=unavailable)
        self.assertEqual(calls, [25, 25])                          # still eligible next time ...
        self.assertFalse(result.changed)                           # ... and nothing was written for it
        self.assertEqual(self.snapshot(), before)

    def test_an_llm_that_raises_counts_as_an_attempt(self):
        rnd = make_round(49, kind="HARD")
        calls = []

        def boom(items):
            calls.append(1)
            raise RuntimeError("down")

        self.run_pipe([rnd], REL + timedelta(minutes=1), llm_fn=boom)
        self.run_pipe([rnd], REL + timedelta(minutes=2), llm_fn=boom)
        self.assertEqual(len(calls), 1)

    def test_attempt_bookkeeping_keeps_the_original_timestamps(self):
        rnd = make_round(49, kind="HARD")
        self.run_pipe([rnd], REL + timedelta(minutes=1), llm_fn=self.counting_llm())
        entry = self.read_state()["49"]
        self.run_pipe([rnd], REL + timedelta(minutes=35), llm_fn=self.counting_llm())
        later = self.read_state()["49"]
        self.assertEqual(later["updated_at"], entry["updated_at"])          # only the attempt times moved
        self.assertEqual(later["methods"], entry["methods"])
        self.assertEqual(later["llm_attempted_at"]["q1"], "2026-10-17T21:35:00Z")

    def test_malformed_attempt_times_are_ignored(self):
        rnd = make_round(49, kind="HARD")
        self.run_pipe([rnd], REL + timedelta(minutes=1), llm_fn=None)
        state = self.read_state()
        state["49"]["llm_attempted_at"] = {"q1": "garbage", "q2": None, "zz": "2026-10-17T21:00:00Z", "q3": 5}
        pipeline.save_state(self.root, state)
        fn = self.counting_llm()
        self.run_pipe([rnd], REL + timedelta(minutes=2), llm_fn=fn)
        self.assertEqual(len(fn.calls[0]), 25)

    def test_future_dated_attempt_does_not_block_forever(self):
        rnd = make_round(49, kind="HARD")
        self.run_pipe([rnd], REL + timedelta(minutes=1), llm_fn=None)
        state = self.read_state()
        state["49"]["llm_attempted_at"] = {"q1": "2099-01-01T00:00:00Z"}
        pipeline.save_state(self.root, state)
        fn = self.counting_llm()
        self.run_pipe([rnd], REL + timedelta(minutes=2), llm_fn=fn)
        self.assertIn("r49q1", fn.calls[0])


class TwoPassTest(TempRootTest):
    """Finding 2: pass 1 (llm_fn=None) writes rule answers + guess placeholders; pass 2 upgrades them."""

    def mixed_round(self):
        texts = ["RULE C q%d" % i for i in range(1, 21)] + ["HARD D q%d" % i for i in range(21, 26)]
        return make_round(49, texts=texts)

    def test_pass_one_never_calls_the_llm_and_pass_two_upgrades(self):
        rnd = self.mixed_round()
        first = self.run_pipe([rnd], REL + timedelta(minutes=2), llm_fn=None)
        self.assertEqual(first.new_rounds, [49])
        answers = self.read_answers(49)["answers"]
        self.assertEqual([answers["q20"], answers["q21"]], ["C", "A"])
        self.assertEqual(self.read_state()["49"]["methods"]["q21"], "guess")
        answered_at = self.read_state()["49"]["answered_at"]

        fn = mock.Mock(side_effect=lambda items: {i.id: llm.LLMAnswer("B", "llm:m1", "asked") for i in items})
        second = self.run_pipe([rnd], REL + timedelta(minutes=3), llm_fn=fn)
        self.assertEqual([i.id for i in fn.call_args[0][0]], ["r49q%d" % i for i in range(21, 26)])
        self.assertEqual(second.new_rounds, [])
        self.assertEqual(second.updated_rounds, [49])
        answers = self.read_answers(49)["answers"]
        self.assertEqual([answers["q20"], answers["q21"], answers["q25"]], ["C", "B", "B"])
        self.assertEqual(self.read_state()["49"]["answered_at"], answered_at)      # the rule answers' time stands

    def test_updated_rounds_counts_an_improvement_that_keeps_the_letter(self):
        rnd = self.mixed_round()
        self.run_pipe([rnd], REL + timedelta(minutes=2), llm_fn=None)
        same_as_placeholder = lambda items: {i.id: llm.LLMAnswer("A", "llm:m1", "asked") for i in items}
        result = self.run_pipe([rnd], REL + timedelta(minutes=3), llm_fn=same_as_placeholder)
        self.assertFalse(result.results[0].file_changed)
        self.assertEqual(result.updated_rounds, [49])              # still worth a "Update round 49" commit
        self.assertEqual(self.read_state()["49"]["methods"]["q21"], "llm:m1")

    def test_late_round_is_upgraded_in_pass_two_only_if_pass_one_just_answered_it(self):
        rnd = self.mixed_round()
        late = REL + timedelta(hours=5)
        first = self.run_pipe([rnd], late, llm_fn=None)
        self.assertEqual(first.new_rounds, [49])
        fn = mock.Mock(side_effect=lambda items: {i.id: llm.LLMAnswer("B", "llm:m1", "asked") for i in items})
        # without the hint the round is past its edit window and is (correctly) left alone
        untouched = self.run_pipe([rnd], late, llm_fn=fn)
        fn.assert_not_called()
        self.assertEqual(untouched.results[0].action, "skip")
        # run.py passes the rounds pass 1 created, so a late round still gets its LLM answers
        upgraded = self.run_pipe([rnd], late, llm_fn=fn, also_editable=[49])
        self.assertEqual(upgraded.updated_rounds, [49])
        self.assertEqual(self.read_answers(49)["answers"]["q25"], "B")

    def test_also_editable_does_not_touch_other_locked_rounds(self):
        locked = make_round(48, kind="HARD", released=REL - timedelta(days=1))
        self.run_pipe([locked], REL - timedelta(days=1) + timedelta(minutes=1), llm_fn=None)
        fn = mock.Mock(return_value={})
        result = self.run_pipe([locked, self.mixed_round()], REL + timedelta(minutes=2), llm_fn=fn, also_editable=[49])
        self.assertEqual([r.action for r in result.results], ["skip", "new"])
        self.assertTrue(all(i.id.startswith("r49") for i in fn.call_args[0][0]))


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


class LLMResilienceTest(unittest.TestCase):
    """Finding 3: one failing call never discards what earlier calls returned."""

    def ask(self, fake, its, **kw):
        kw.setdefault("token", "tok")
        with mock.patch("quizbot.llm._http_json", fake):
            return llm.answer_questions(its, **kw)

    @staticmethod
    def urlopen_sequence(*steps):
        """A urlopen replacement: each step is a JSON-able reply body or an exception to raise."""
        queue = list(steps)

        class Resp:
            def __init__(self, step):
                self.step = step

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                if isinstance(self.step, BaseException):
                    raise self.step
                return self.step if isinstance(self.step, bytes) else json.dumps(self.step).encode("utf-8")

        def urlopen(req, timeout=None, context=None):
            step = queue.pop(0)
            if isinstance(step, urllib_error().URLError):
                raise step
            return Resp(step)

        return urlopen

    def test_http_json_maps_truncated_and_broken_bodies(self):
        import http.client

        cases = [
            http.client.IncompleteRead(b"partial", 100),
            http.client.RemoteDisconnected("closed"),
            http.client.BadStatusLine("garbage"),
            b"<html>not json</html>",
            b"\xff\xfe not utf-8",
            b"",
            ConnectionResetError("reset"),
        ]
        for step in cases:
            with mock.patch("quizbot.llm.urllib.request.urlopen", self.urlopen_sequence(step)):
                with self.assertRaises(llm.LLMRequestError, msg=repr(step)):
                    llm._http_json("POST", "http://x", {}, {"a": 1})

    def test_http_json_maps_connect_failures(self):
        err = urllib_error().URLError("dns failure")
        with mock.patch("quizbot.llm.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(llm.LLMRequestError) as ctx:
                llm._http_json("GET", "http://x", {})
        self.assertEqual(ctx.exception.status, 0)

    def test_incomplete_read_on_the_second_call_keeps_the_first_models_answers(self):
        import http.client

        its = items(3)
        good = {"choices": [{"message": {"content": reply({i.id: "B" for i in its})}}]}
        urlopen = self.urlopen_sequence(good, http.client.IncompleteRead(b"x", 10))
        with mock.patch("quizbot.llm.urllib.request.urlopen", urlopen):
            got = llm.answer_questions(its, token="tok", models=["openai/o3", "openai/gpt-4o"])
        self.assertEqual({k: (v.letter, v.method) for k, v in got.items()}, {i.id: ("B", "llm:openai/o3") for i in its})

    def test_failure_on_chunk_two_keeps_chunk_one(self):
        import http.client

        its = items(4)
        first = {"choices": [{"message": {"content": reply({"r49q1": "C", "r49q2": "D"})}}]}
        urlopen = self.urlopen_sequence(first, http.client.IncompleteRead(b"x", 10))
        with mock.patch("quizbot.llm.urllib.request.urlopen", urlopen):
            got = llm.answer_questions(its, token="tok", models=["m1"], chunk_size=2, max_requests=2)
        self.assertEqual({k: v.letter for k, v in got.items()}, {"r49q1": "C", "r49q2": "D"})
        self.assertEqual(got.unattempted, set())                  # chunk 2 was really tried (and failed)

    def test_any_exception_type_from_one_call_only_loses_that_call(self):
        its = items(2)
        fake = FakeModels({"m1": lambda b: reply({"r49q1": "A", "r49q2": "B"}), "m2": RuntimeError("surprise")})
        got = self.ask(fake, its, models=["m1", "m2"])
        self.assertEqual({k: v.letter for k, v in got.items()}, {"r49q1": "A", "r49q2": "B"})
        self.assertEqual({v.method for v in got.values()}, {"llm:m1"})

    def test_exception_while_parsing_the_second_reply_keeps_the_first(self):
        its = items(2)
        fake = FakeModels({"m1": lambda b: reply({"r49q1": "A", "r49q2": "B"}), "m2": lambda b: reply({})})
        with mock.patch("quizbot.llm.parse_answers", side_effect=[{"r49q1": ("A", ""), "r49q2": ("B", "")},
                                                                   RuntimeError("parser bug")]):
            got = self.ask(fake, its, models=["m1", "m2"])
        self.assertEqual({k: v.letter for k, v in got.items()}, {"r49q1": "A", "r49q2": "B"})

    def test_failure_after_pass_one_keeps_pass_one_answers(self):
        its = items(2)
        fake = FakeModels({"m1": lambda b: reply({"r49q1": "A", "r49q2": "B"}), "m2": lambda b: reply({})})
        calls = []

        def flaky_ask(model, chunk, token, budget, dead):
            calls.append(model)
            if len(calls) == 2:
                raise MemoryError("boom")                       # escapes even _ask's own handlers
            return llm_ask(model, chunk, token, budget, dead)

        llm_ask = llm._ask
        with mock.patch("quizbot.llm._http_json", fake), mock.patch("quizbot.llm._ask", flaky_ask):
            got = llm.answer_questions(its, token="tok", models=["m1", "m2"])
        self.assertEqual({k: v.letter for k, v in got.items()}, {"r49q1": "A", "r49q2": "B"})

    def test_ask_never_raises(self):
        budget = llm.RunBudget()
        fake = FakeModels({"m1": RuntimeError("x")})
        with mock.patch("quizbot.llm._http_json", fake):
            self.assertEqual(llm._ask("m1", items(1), "tok", budget, set()), (None, True))


def urllib_error():
    import urllib.error

    return urllib.error


class LLMBudgetTest(unittest.TestCase):
    """Finding 2: overall LLM budget per process (120 s, 6 requests, one catalog fetch)."""

    class Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

    def slow_fake(self, clock, seconds_per_request, replies=None):
        everything = {"r49q%d" % i: "A" for i in range(1, 101)}
        fake = FakeModels(replies or {"m1": lambda b: reply(everything), "m2": lambda b: reply(everything),
                                      "m3": lambda b: reply(everything)})
        inner = fake.__call__

        def timed(method, url, headers, body=None, timeout=60):
            fake.timeouts = getattr(fake, "timeouts", []) + [timeout]
            clock.t += seconds_per_request
            return inner(method, url, headers, body, timeout)

        return fake, timed

    def test_default_budget_is_120_seconds(self):
        self.assertEqual(llm.LLM_TIME_BUDGET, 120)
        self.assertEqual(llm.RunBudget().seconds, 120)

    def test_requests_stop_when_the_time_budget_is_used_up(self):
        clock = self.Clock()
        fake, timed = self.slow_fake(clock, 50)
        budget = llm.RunBudget(cap=6, seconds=120, clock=clock)
        with mock.patch("quizbot.llm._http_json", timed):
            got = llm.answer_questions(items(100), token="tok", models=["m1", "m2", "m3"], chunk_size=25, budget=budget)
        # 50 s + 50 s, then 20 s left: a third request still starts (with a 20 s timeout); a fourth does not
        self.assertEqual(len(fake.requests), 3)
        self.assertEqual(fake.timeouts, [60, 60, 20])
        self.assertEqual(budget.spent, 150)
        self.assertEqual(len(got), 75)                       # chunks 1-3 answered, chunk 4 never asked
        self.assertEqual(got.unattempted, {"r49q%d" % i for i in range(76, 101)})

    def test_budget_is_shared_between_calls_of_one_process(self):
        clock = self.Clock()
        fake, timed = self.slow_fake(clock, 70)
        budget = llm.RunBudget(cap=6, seconds=120, clock=clock)
        with mock.patch("quizbot.llm._http_json", timed):
            first = llm.answer_questions(items(2), token="tok", models=["m1"], budget=budget)
            second = llm.answer_questions(items(2, prefix="r50q"), token="tok", models=["m1"], budget=budget)
            third = llm.answer_questions(items(2, prefix="r51q"), token="tok", models=["m1"], budget=budget)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 0)                     # 70 s used, 50 s left, request starts but gets nothing ...
        self.assertEqual(len(third), 0)
        self.assertEqual(len(fake.requests), 2)              # ... and then the budget is gone
        self.assertEqual(third.unattempted, {"r51q1", "r51q2"})

    def test_request_cap_is_shared_between_calls(self):
        fake = FakeModels({"m1": lambda b: reply({"r49q%d" % i: "A" for i in range(1, 11)})})
        budget = llm.RunBudget(cap=3)
        with mock.patch("quizbot.llm._http_json", fake):
            for _ in range(5):
                llm.answer_questions(items(2), token="tok", models=["m1"], budget=budget)
        self.assertEqual(len(fake.requests), 3)

    def test_catalog_is_fetched_once_per_process(self):
        everything = {"r49q%d" % i: "A" for i in range(1, 11)}
        fake = FakeModels({"openai/o3": lambda b: reply(everything)}, catalog=[{"id": "openai/o3"}])
        urls = []

        def counting(method, url, headers, body=None, timeout=60):
            urls.append(url)
            return fake(method, url, headers, body, timeout)

        budget = llm.RunBudget()
        with mock.patch("quizbot.llm._http_json", counting), mock.patch.dict(os.environ, {"QUIZ_LLM_MODELS": "openai/o3"}):
            llm.answer_questions(items(2), token="tok", budget=budget)
            llm.answer_questions(items(2, prefix="r50q"), token="tok", budget=budget)
        self.assertEqual(urls.count(llm.CATALOG_URL), 1)
        self.assertEqual(budget.used, 3)

    def test_slow_catalog_counts_against_the_time_budget(self):
        clock = self.Clock()
        budget = llm.RunBudget(seconds=120, clock=clock)

        def slow_catalog(method, url, headers, body=None, timeout=60):
            clock.t += 25
            raise llm.LLMRequestError(500, "down")

        with mock.patch("quizbot.llm._http_json", slow_catalog):
            llm.discover_models("tok", env={}, budget=budget)
        self.assertEqual(budget.spent, 25)

    def test_no_token_or_no_budget_means_nothing_was_attempted(self):
        its = items(3)
        with mock.patch("quizbot.llm.get_token", return_value=None):
            got = llm.answer_questions(its)
        self.assertEqual((dict(got), got.unattempted), ({}, {i.id for i in its}))
        spent = llm.RunBudget(cap=1)
        spent.used = 1
        with mock.patch("quizbot.llm._http_json", FakeModels({})):
            got = llm.answer_questions(its, token="tok", budget=spent)
        self.assertEqual(got.unattempted, {i.id for i in its})

    def test_questions_after_every_model_died_are_unattempted(self):
        its = items(4)
        fake = FakeModels({"m1": llm.LLMRequestError(429)})
        with mock.patch("quizbot.llm._http_json", fake):
            got = llm.answer_questions(its, token="tok", models=["m1"], chunk_size=2)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(got.unattempted, {"r49q3", "r49q4"})    # chunk 1 was tried and failed; chunk 2 never sent

    def test_inference_timeout_never_exceeds_the_remaining_budget(self):
        clock = self.Clock()
        clock.t = 0
        budget = llm.RunBudget(seconds=30, clock=clock)
        fake = FakeModels({"m1": lambda b: reply({"r49q1": "A"})})
        seen = []

        def spy(method, url, headers, body=None, timeout=60):
            seen.append(timeout)
            return fake(method, url, headers, body, timeout)

        with mock.patch("quizbot.llm._http_json", spy):
            llm.answer_questions(items(1), token="tok", models=["m1"], budget=budget)
        self.assertEqual(seen, [30])


class MemoizedLLMTest(unittest.TestCase):
    def fn(self, letter="B"):
        calls = []

        def inner(its):
            calls.append([i.id for i in its])
            return {i.id: llm.LLMAnswer(letter, "llm:m1", "d") for i in its}

        inner.calls = calls
        return inner

    def test_a_question_is_asked_once_per_process(self):
        inner = self.fn()
        memo = llm.MemoizedLLM(inner)
        first = memo(items(3))
        again = memo(items(3))
        self.assertEqual(inner.calls, [["r49q1", "r49q2", "r49q3"]])
        self.assertEqual(dict(first), dict(again))
        self.assertEqual(memo.calls, 1)

    def test_key_is_text_plus_options_not_the_id(self):
        inner = self.fn()
        memo = llm.MemoizedLLM(inner)
        memo(items(2))
        got = memo(items(2, prefix="r50q"))                       # same questions in another round
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(sorted(got), ["r50q1", "r50q2"])
        other_options = [llm.LLMItem("r51q1", "Question 1?", {"A": "9", "B": "2", "C": "3", "D": "4"})]
        memo(other_options)
        self.assertEqual(inner.calls[1], ["r51q1"])               # same text, different options: new question

    def test_duplicates_inside_one_call_are_sent_once(self):
        inner = self.fn()
        memo = llm.MemoizedLLM(inner)
        dup = [llm.LLMItem("a", "Same?", {"A": "1"}), llm.LLMItem("b", "Same?", {"A": "1"})]
        got = memo(dup)
        self.assertEqual(inner.calls, [["a"]])
        self.assertEqual(sorted(got), ["a", "b"])

    def test_only_new_questions_are_sent(self):
        inner = self.fn()
        memo = llm.MemoizedLLM(inner)
        memo(items(2))
        memo(items(4))
        self.assertEqual(inner.calls, [["r49q1", "r49q2"], ["r49q3", "r49q4"]])

    def test_no_answer_is_remembered_too(self):
        calls = []

        def nothing(its):
            calls.append(len(its))
            return {}

        memo = llm.MemoizedLLM(nothing)
        self.assertEqual(dict(memo(items(2))), {})
        self.assertEqual(dict(memo(items(2))), {})
        self.assertEqual(calls, [2])

    def test_unattempted_items_are_not_remembered(self):
        calls = []

        def nothing_sent(its):
            calls.append([i.id for i in its])
            res = llm.LLMResults()
            res.unattempted.update(i.id for i in its)
            return res

        memo = llm.MemoizedLLM(nothing_sent)
        first = memo(items(2))
        memo(items(2))
        self.assertEqual(len(calls), 2)                           # asked again: it was never really tried
        self.assertEqual(first.unattempted, {"r49q1", "r49q2"})

    def test_exceptions_propagate_and_are_not_remembered(self):
        state = {"n": 0}

        def flaky(its):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("down")
            return {i.id: llm.LLMAnswer("C", "llm:m1") for i in its}

        memo = llm.MemoizedLLM(flaky)
        with self.assertRaises(RuntimeError):
            memo(items(1))
        self.assertEqual(memo(items(1))["r49q1"].letter, "C")

    def test_make_llm_fn_shares_one_budget_and_memo(self):
        fake = FakeModels({"m1": lambda b: reply({"r49q%d" % i: "A" for i in range(1, 6)})})
        client = llm.make_llm_fn(token="tok", models=["m1"], max_requests=2)
        with mock.patch("quizbot.llm._http_json", fake):
            client(items(3))
            client(items(3))                                      # memo: no request
            client(items(5))                                      # 2 new questions: one more request
            client(items(5, prefix="r50q"))                       # new ids, same questions: memo
        self.assertEqual(len(fake.requests), 2)
        self.assertEqual(client.budget.used, 2)
        self.assertEqual(client.budget.cap, 2)


class SelfTestTest(unittest.TestCase):
    def run_selftest(self, llm_fn=None, **kw):
        lines = []
        rc = llm.selftest(llm_fn, out=lines.append, **kw)
        return rc, "\n".join(lines)

    @staticmethod
    def answers(letters, method="llm:m1", models=("m1",)):
        def fn(its):
            return {i.id: llm.LLMAnswer(letter, method, "m1->%s" % letter, models)
                    for i, letter in zip(its, letters) if letter}
        return fn

    def test_the_expected_answers_are_actually_right(self):
        from decimal import Decimal

        def value(text):
            return Decimal(text.replace("$", "").replace(",", ""))

        q1, q2, q3 = llm.SELFTEST_CASES
        self.assertEqual(value(q1[1][q1[2]]), 17 + 25)
        self.assertEqual(value(q2[1][q2[2]]), Decimal("0.15") * 240)
        self.assertEqual(value(q3[1][q3[2]]), Decimal(10000) * Decimal("0.06") * 3)
        for text, options, want in llm.SELFTEST_CASES:
            self.assertEqual(sorted(options), ["A", "B", "C", "D"])
            self.assertEqual(len({v for v in options.values()}), 4)
            self.assertIn(want, "ABCD")
        self.assertGreater(len({want for _t, _o, want in llm.SELFTEST_CASES}), 1)    # not all the same letter
        self.assertIn("simple interest", q3[0])
        self.assertIn("%", q2[0])

    def test_all_correct_exits_0_and_reports_models_latency_and_per_question_result(self):
        want = [c[2] for c in llm.SELFTEST_CASES]
        clock = iter([100.0, 103.5]).__next__
        rc, text = self.run_selftest(self.answers(want, "llm:openai/gpt-4.1", ("openai/gpt-4.1", "openai/o3")), clock=clock)
        self.assertEqual(rc, 0)
        self.assertEqual(text.count("[PASS]"), 3)
        self.assertNotIn("[FAIL]", text)
        self.assertIn("via llm:openai/gpt-4.1", text)
        self.assertIn("models that answered: openai/gpt-4.1, openai/o3", text)
        self.assertIn("latency: 3.5 s", text)
        self.assertIn("correct: 3/3", text)
        self.assertIn("simple interest", text)

    def test_two_of_three_correct_passes(self):
        want = [c[2] for c in llm.SELFTEST_CASES]
        wrong = [w for w in "ABCD" if w != want[2]][0]
        rc, text = self.run_selftest(self.answers([want[0], want[1], wrong]))
        self.assertEqual(rc, 0)
        self.assertEqual((text.count("[PASS]"), text.count("[FAIL]")), (2, 1))

    def test_one_of_three_correct_fails_with_exit_1(self):
        want = [c[2] for c in llm.SELFTEST_CASES]
        wrong = lambda w: [x for x in "ABCD" if x != w][0]
        rc, text = self.run_selftest(self.answers([want[0], wrong(want[1]), wrong(want[2])]))
        self.assertEqual(rc, 1)
        self.assertIn("correct: 1/3", text)

    def test_a_missing_answer_counts_as_wrong(self):
        want = [c[2] for c in llm.SELFTEST_CASES]
        rc, text = self.run_selftest(self.answers([want[0], None, None]))
        self.assertEqual(rc, 1)
        self.assertIn("[MISS]", text)

    def test_unavailable_llm_exits_3(self):
        rc, text = self.run_selftest(lambda its: {})               # the sandbox answers "OK": no usable reply
        self.assertEqual(rc, 3)
        self.assertIn("unavailable", text)
        rc, _ = self.run_selftest(lambda its: None)
        self.assertEqual(rc, 3)

        def boom(its):
            raise RuntimeError("network")

        rc, text = self.run_selftest(boom)
        self.assertEqual(rc, 3)
        self.assertIn("RuntimeError", text)

    def test_no_token_exits_3_without_any_request(self):
        fake = FakeModels({})
        with mock.patch("quizbot.llm.get_token", return_value=None), mock.patch("quizbot.llm._http_json", fake):
            rc, text = self.run_selftest()
        self.assertEqual(rc, 3)
        self.assertIn("no GITHUB_TOKEN", text)
        self.assertEqual(fake.requests, [])

    def test_real_path_with_fake_http_lists_requests_and_models(self):
        want = {"selftest%d" % n: c[2] for n, c in enumerate(llm.SELFTEST_CASES, 1)}
        models = llm.preferred_models({})
        fake = FakeModels({m: (lambda b: reply(want)) for m in models})
        with mock.patch("quizbot.llm._http_json", fake), mock.patch.dict(os.environ, {"QUIZ_LLM_MODELS": ""}):
            rc, text = self.run_selftest(token="tok")
        self.assertEqual(rc, 0)
        self.assertEqual(fake.models_used(), models[:2])           # first opinion + one second opinion
        self.assertIn("request to %s" % models[0], text)
        self.assertIn("request to %s" % models[1], text)
        self.assertIn("models that answered: %s, %s" % (models[0], models[1]), text)
        self.assertIn("via llm:%s" % models[0], text)
        sent = fake.requests[0]["messages"][1]["content"]
        for text_q, _o, _w in llm.SELFTEST_CASES:
            self.assertIn(text_q, sent)

    def test_real_path_when_the_endpoint_returns_ok_is_unavailable(self):
        # what the local sandbox does: every call "succeeds" with a body that is not JSON
        def ok_body(method, url, headers, body=None, timeout=60):
            raise llm.LLMRequestError(-1, "response is not JSON: OK")

        with mock.patch("quizbot.llm._http_json", ok_body):
            rc, text = self.run_selftest(token="tok")
        self.assertEqual(rc, 3)
        self.assertIn("unavailable", text)

    def test_solver_status_is_shown_but_does_not_affect_the_result(self):
        want = [c[2] for c in llm.SELFTEST_CASES]
        solver = lambda text, options: Sol(None, "abstain:unmatched", "")
        rc, text = self.run_selftest(self.answers(want), solver=solver)
        self.assertEqual(rc, 0)
        self.assertIn("rule solver: abstains (abstain:unmatched)", text)
        rc, text = self.run_selftest(self.answers(want), solver=lambda t, o: Sol("A", "rule:x", ""))
        self.assertIn("rule solver: answers A", text)

        def broken(text, options):
            raise ValueError("x")

        rc, text = self.run_selftest(self.answers(want), solver=broken)
        self.assertEqual(rc, 0)
        self.assertIn("rule solver raised ValueError", text)


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


class TwoPhaseFlowTest(unittest.TestCase):
    """Finding 2 in gitops: rule answers are pushed first, the LLM upgrade pass is a second commit."""

    def flow(self, handler=None, env=None, llm_pass=True, phase1=None, phase2=None, issues=None):
        events = []
        sleeps = []

        def record(args):
            events.append(" ".join(args))
            return handler(args) if handler else (0, "", "")

        runner = FakeRunner(record)

        def pipeline_fn():
            events.append("PASS1")
            return phase1() if phase1 else fake_result(new=[49])

        def llm_pass_fn(first):
            events.append("PASS2")
            llm_pass_fn.seen.append(first)
            return phase2() if phase2 else fake_result(updated=[49])

        llm_pass_fn.seen = []

        def notify(root, rounds, env, request=None):
            events.append("ISSUES")
            return []

        with mock.patch.object(gitops, "notify_issues", notify if issues is None else issues):
            rc = gitops.run_git_flow(pipeline_fn, root="/repo", env=env if env is not None else {"QUIZBOT_ALLOW_RESET": "1"},
                                     runner=runner, sleep=sleeps.append,
                                     llm_pass_fn=llm_pass_fn if llm_pass else None)
        return rc, events, sleeps, llm_pass_fn

    @staticmethod
    def dirty(args):
        if args[:2] == ["status", "--porcelain"]:
            return 0, " M answers/round-0049.json\n", ""
        return 0, "", ""

    @staticmethod
    def commits(events):
        return [e.split(" -m ")[-1] for e in events if " commit " in " " + e + " " and "-m" in e]

    def test_rules_are_pushed_before_the_llm_pass_starts(self):
        rc, events, sleeps, _ = self.flow(self.dirty)
        self.assertEqual(rc, 0)
        first_push = events.index("push origin HEAD:main")
        second_push = len(events) - 1 - events[::-1].index("push origin HEAD:main")
        self.assertLess(events.index("PASS1"), first_push)
        self.assertLess(first_push, events.index("PASS2"))
        self.assertLess(events.index("PASS2"), second_push)
        self.assertNotEqual(first_push, second_push)
        self.assertEqual(sleeps, [])

    def test_two_commits_answer_then_update(self):
        rc, events, _, _ = self.flow(self.dirty)
        self.assertEqual(rc, 0)
        self.assertEqual(self.commits(events), ["Answer round 49", "Update round 49"])

    def test_llm_pass_receives_the_first_result_and_does_not_resync_on_its_first_attempt(self):
        rc, events, _, llm_pass = self.flow(self.dirty)
        self.assertEqual(events.count("reset --hard origin/main"), 1)         # phase 1 only
        self.assertEqual(events.count("fetch origin main"), 1)
        self.assertEqual(len(llm_pass.seen), 1)
        self.assertEqual(llm_pass.seen[0].new_rounds, [49])

    def test_issues_wait_for_the_llm_pass(self):
        _, events, _, _ = self.flow(self.dirty)
        self.assertEqual(events.count("ISSUES"), 1)
        self.assertGreater(events.index("ISSUES"), events.index("PASS2"))
        _, events, _, _ = self.flow(self.dirty, llm_pass=False)               # no LLM pass: opened in phase 1
        self.assertEqual(events.count("ISSUES"), 1)
        self.assertLess(events.index("ISSUES"), events.index("push origin HEAD:main"))

    def test_phase_one_failure_skips_the_llm_pass(self):
        def handler(args):
            return (1, "", "rejected") if args[0] == "push" else self.dirty(args)

        rc, events, sleeps, llm_pass = self.flow(handler)
        self.assertEqual(rc, 1)
        self.assertNotIn("PASS2", events)
        self.assertEqual(sleeps, [5, 10, 20, 30])

    def test_source_error_in_phase_one_skips_the_llm_pass(self):
        def broken():
            raise source.SourceError("no questions")

        rc, events, _, _ = self.flow(self.dirty, phase1=broken)
        self.assertEqual(rc, 1)
        self.assertNotIn("PASS2", events)

    def test_rejected_llm_push_resyncs_and_reruns_only_the_llm_pass(self):
        pushes = []

        def handler(args):
            if args[0] == "push":
                pushes.append(1)
                return (1, "", "! [rejected]") if len(pushes) == 2 else (0, "", "")
            return self.dirty(args)

        rc, events, sleeps, llm_pass = self.flow(handler)
        self.assertEqual(rc, 0)
        self.assertEqual(events.count("PASS1"), 1)
        self.assertEqual(events.count("PASS2"), 2)
        self.assertEqual(events.count("reset --hard origin/main"), 2)         # phase 1, then the phase-2 retry
        self.assertEqual(sleeps, [5])
        self.assertEqual(self.commits(events), ["Answer round 49", "Update round 49", "Update round 49"])
        self.assertEqual(llm_pass.seen[0], llm_pass.seen[1])                  # same phase-1 result both times

    def test_llm_push_gives_up_after_five_attempts_but_phase_one_stays_pushed(self):
        pushes = []

        def handler(args):
            if args[0] == "push":
                pushes.append(1)
                return (0, "", "") if len(pushes) == 1 else (1, "", "rejected")
            return self.dirty(args)

        rc, events, sleeps, _ = self.flow(handler)
        self.assertEqual(rc, 1)
        self.assertEqual(len(pushes), 6)                                      # 1 for phase 1 + 5 for phase 2
        self.assertEqual(events.count("PASS1"), 1)
        self.assertEqual(sleeps, [5, 10, 20, 30])

    def test_a_crash_in_the_llm_pass_is_reported_but_phase_one_is_already_pushed(self):
        def crash():
            raise RuntimeError("bug in the upgrade pass")

        rc, events, _, _ = self.flow(self.dirty, phase2=crash)
        self.assertEqual(rc, 1)
        self.assertEqual(self.commits(events), ["Answer round 49"])
        self.assertEqual(events.count("push origin HEAD:main"), 1)

    def test_clean_tree_after_both_passes_pushes_nothing(self):
        rc, events, _, _ = self.flow(None)                                    # status always clean
        self.assertEqual(rc, 0)
        self.assertEqual(events.count("PASS2"), 1)
        self.assertFalse(any(e.startswith(("commit", "push", "add")) for e in events))

    def test_state_only_llm_bookkeeping_is_committed_as_a_state_update(self):
        result = pipeline.PipelineResult(results=[pipeline.RoundResult(49, "upgrade", state_changed=True)], state={}, rounds={})
        rc, events, _, _ = self.flow(self.dirty, phase1=lambda: fake_result(), phase2=lambda: result)
        self.assertEqual(self.commits(events)[-1], "Update quiz state")

    def test_improved_round_with_unchanged_letters_is_named_in_the_commit(self):
        result = pipeline.PipelineResult(results=[pipeline.RoundResult(
            49, "upgrade", state_changed=True, changed_questions=[24, 25])], state={}, rounds={})
        self.assertEqual(result.updated_rounds, [49])
        rc, events, _, _ = self.flow(self.dirty, phase2=lambda: result)
        self.assertEqual(self.commits(events)[-1], "Update round 49")

    def test_skipped_rounds_are_never_listed_as_updated(self):
        result = pipeline.PipelineResult(results=[pipeline.RoundResult(49, "skip", "not released yet")], state={}, rounds={})
        self.assertEqual((result.updated_rounds, result.new_rounds, result.changed), ([], [], False))


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


@unittest.skipUnless(shutil.which("git"), "git not installed")
class TwoPhaseGitTest(TempRootTest):
    """`run.py --git` against a real local bare repo: rule answers land on the remote BEFORE the LLM is asked."""

    GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
               "QUIZBOT_ALLOW_RESET": "1"}

    def git(self, cwd, *args):
        return subprocess.run(["git"] + list(args), cwd=cwd, check=True, capture_output=True, text=True).stdout

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, self.GIT_ENV)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("GITHUB_ACTIONS", "GITHUB_TOKEN", "GH_TOKEN"):
            os.environ.pop(key, None)
        self.bare = os.path.join(self.root, "remote.git")
        seed = os.path.join(self.root, "seed")
        self.work = os.path.join(self.root, "work")
        self.rival = os.path.join(self.root, "rival")
        self.git(self.root, "init", "--bare", self.bare)
        self.git(self.bare, "symbolic-ref", "HEAD", "refs/heads/main")
        self.git(self.root, "clone", self.bare, seed)
        self.git(seed, "checkout", "-b", "main")
        with open(os.path.join(seed, ".gitignore"), "w") as fh:
            fh.write("__pycache__/\n")
        self.git(seed, "add", ".gitignore")
        self.git(seed, "commit", "-m", "init")
        self.git(seed, "push", "origin", "main")
        self.git(self.root, "clone", self.bare, self.work)
        self.git(self.root, "clone", self.bare, self.rival)
        self.csv = os.path.join(self.root, "q.csv")
        texts = ["RULE C q%d" % i for i in range(1, 21)] + ["HARD D q%d" % i for i in range(21, 26)]
        with open(self.csv, "w", encoding="utf-8", newline="") as fh:
            fh.write(make_csv([csv_line(49, i, "2026-10-17T21:00:00Z", t) for i, t in enumerate(texts, 1)]))
        patcher = mock.patch.object(gitops, "BACKOFFS", (0, 0, 0, 0))
        patcher.start()
        self.addCleanup(patcher.stop)

    def remote_log(self):
        return self.git(self.bare, "log", "--format=%s", "main").splitlines()

    def remote_answers(self, rid=49):
        return json.loads(self.git(self.bare, "show", "main:answers/round-%04d.json" % rid))["answers"]

    def remote_state(self):
        return json.loads(self.git(self.bare, "show", "main:state/rounds.json"))

    def main(self, llm_fn, now="2026-10-17T21:05:00Z", extra=()):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            return run_module.main(["--git", "--questions-file", self.csv, "--now", now] + list(extra),
                                   root=self.work, solver=self.solver, llm_fn=llm_fn)

    def llm_says(self, letter, spy=None):
        def fn(its):
            if spy is not None:
                spy(its)
            return {i.id: llm.LLMAnswer(letter, "llm:m1", "asked") for i in its}
        return fn

    def test_rule_answers_are_on_the_remote_before_the_llm_is_asked(self):
        seen = {}

        def spy(its):
            seen["log"] = self.remote_log()
            seen["answers"] = self.remote_answers()
            seen["methods"] = self.remote_state()["49"]["methods"]
            seen["ids"] = [i.id for i in its]

        rc = self.main(self.llm_says("B", spy))
        self.assertEqual(rc, 0)
        # what the remote held at the moment the LLM was called
        self.assertEqual(seen["log"], ["Answer round 49", "init"])
        self.assertEqual([seen["answers"]["q%d" % i] for i in (1, 20)], ["C", "C"])
        self.assertEqual([seen["answers"]["q%d" % i] for i in range(21, 26)], ["A"] * 5)       # guess placeholders
        self.assertEqual({seen["methods"]["q%d" % i] for i in range(21, 26)}, {"guess"})
        self.assertEqual(seen["ids"], ["r49q%d" % i for i in range(21, 26)])
        # and afterwards
        self.assertEqual(self.remote_log(), ["Update round 49", "Answer round 49", "init"])
        self.assertEqual([self.remote_answers()["q%d" % i] for i in (20, 21, 25)], ["C", "B", "B"])
        self.assertEqual(self.remote_state()["49"]["methods"]["q25"], "llm:m1")
        self.assertEqual(self.git(self.work, "status", "--porcelain").strip(), "")

    def test_a_slow_or_failing_llm_cannot_prevent_the_rule_push(self):
        def boom(its):
            raise RuntimeError("models.github.ai is down")

        rc = self.main(boom)
        self.assertEqual(rc, 0)
        self.assertEqual(self.remote_log()[-2], "Answer round 49")
        self.assertEqual([self.remote_answers()["q%d" % i] for i in (1, 20, 21)], ["C", "C", "A"])

    def test_push_race_during_the_llm_pass_does_not_ask_the_llm_twice(self):
        calls = []

        def rival_pushes_while_we_think(its):
            calls.append([i.id for i in its])
            self.git(self.rival, "pull", "-q", "origin", "main")
            with open(os.path.join(self.rival, "NOTES.txt"), "w") as fh:
                fh.write("rival\n")
            self.git(self.rival, "add", "NOTES.txt")
            self.git(self.rival, "commit", "-m", "rival commit")
            self.git(self.rival, "push", "-q", "origin", "main")

        rc = self.main(self.llm_says("D", rival_pushes_while_we_think))
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)                                          # memoised across the retry
        self.assertEqual(self.remote_log(), ["Update round 49", "rival commit", "Answer round 49", "init"])
        self.assertEqual(self.remote_answers()["q25"], "D")

    def test_late_round_still_gets_its_llm_upgrade(self):
        rc = self.main(self.llm_says("C"), now="2026-10-18T02:00:00Z")           # 5 h after release
        self.assertEqual(rc, 0)
        self.assertEqual(self.remote_log(), ["Update round 49", "Answer round 49", "init"])
        self.assertEqual(self.remote_answers()["q23"], "C")
        # but the next process leaves the (now locked) round alone
        before = self.remote_log()
        self.assertEqual(self.main(self.llm_says("A"), now="2026-10-18T02:05:00Z"), 0)
        self.assertEqual(self.remote_log(), before)

    def test_all_rule_round_is_a_single_commit_and_never_calls_the_llm(self):
        with open(self.csv, "w", encoding="utf-8", newline="") as fh:
            fh.write(make_csv([csv_line(49, i, "2026-10-17T21:00:00Z", "RULE B q%d" % i) for i in range(1, 26)]))
        spy = mock.Mock(return_value={})
        self.assertEqual(self.main(spy), 0)
        spy.assert_not_called()
        self.assertEqual(self.remote_log(), ["Answer round 49", "init"])

    def test_second_process_is_a_noop_and_does_not_reask_a_working_llm(self):
        calls = []
        self.assertEqual(self.main(self.llm_says("B", calls.append)), 0)
        log_after_first = self.remote_log()
        self.assertEqual(self.main(self.llm_says("B", calls.append), now="2026-10-17T21:07:00Z"), 0)
        self.assertEqual(self.remote_log(), log_after_first)
        self.assertEqual(len(calls), 1)

    def test_throttled_llm_failure_is_recorded_once_and_not_hammered(self):
        calls = []

        def nothing(its):
            calls.append(1)
            return {}

        self.assertEqual(self.main(nothing), 0)
        self.assertEqual(self.remote_log(), ["Update quiz state", "Answer round 49", "init"])   # attempt times recorded
        self.assertIn("llm_attempted_at", self.remote_state()["49"])
        for minute in ("21:06", "21:20", "21:34"):                                  # all < 30 min after 21:05
            self.assertEqual(self.main(nothing, now="2026-10-17T%s:00Z" % minute), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.remote_log()), 3)
        self.assertEqual(self.main(nothing, now="2026-10-17T21:36:00Z"), 0)         # 31 min later: asked again
        self.assertEqual(len(calls), 2)

    def test_no_llm_flag_is_a_single_phase(self):
        spy = mock.Mock(return_value={})
        rc = self.main(spy, extra=["--no-llm"])
        self.assertEqual(rc, 0)
        spy.assert_not_called()
        self.assertEqual(self.remote_log(), ["Answer round 49", "init"])

    def test_future_round_is_not_pushed(self):
        rc = self.main(self.llm_says("B"), now="2026-10-17T20:00:00Z")           # an hour before release
        self.assertEqual(rc, 0)
        self.assertEqual(self.remote_log(), ["init"])
        self.assertEqual(self.git(self.work, "status", "--porcelain").strip(), "")


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

    # ---- --llm-selftest -----------------------------------------------------
    def test_llm_selftest_exit_codes_and_no_files_written(self):
        want = [c[2] for c in llm.SELFTEST_CASES]

        def llm_fn(letters):
            return lambda its: {i.id: llm.LLMAnswer(l, "llm:m1", "m1->%s" % l, ("m1",)) for i, l in zip(its, letters)}

        rc, out = self.main(["--llm-selftest"], llm_fn=llm_fn(want))
        self.assertEqual(rc, 0)
        self.assertIn("models that answered: m1", out)
        self.assertIn("latency:", out)
        self.assertEqual(out.count("[PASS]"), 3)
        wrong = ["A" if w != "A" else "B" for w in want]
        rc, out = self.main(["--llm-selftest"], llm_fn=llm_fn(wrong))
        self.assertEqual(rc, 1)
        rc, out = self.main(["--llm-selftest"], llm_fn=lambda its: {})
        self.assertEqual(rc, 3)
        self.assertEqual(os.listdir(self.root), [])               # no answers/, state/ or STATUS.md anywhere

    def test_llm_selftest_never_touches_the_pipeline_or_git(self):
        runner = FakeRunner()
        with mock.patch("quizbot.pipeline.run_pipeline", side_effect=AssertionError("pipeline ran")), \
                mock.patch("quizbot.source.load_rounds", side_effect=AssertionError("fetched questions")):
            rc, _ = self.main(["--llm-selftest"], llm_fn=lambda its: {}, runner=runner)
        self.assertEqual(rc, 3)
        self.assertEqual(runner.calls, [])

    def test_llm_selftest_without_a_token_is_unavailable(self):
        with mock.patch("quizbot.llm.get_token", return_value=None), \
                mock.patch("quizbot.llm._http_json", side_effect=AssertionError("must not call out")):
            rc, out = self.main(["--llm-selftest"])
        self.assertEqual(rc, 3)
        self.assertIn("unavailable", out)

    def test_llm_selftest_cannot_be_combined_with_other_modes(self):
        for extra in ("--no-llm", "--git", "--dry-run", "--check-practice"):
            self.assertEqual(self.main(["--llm-selftest", extra])[0], 2, extra)

    def test_llm_selftest_shows_the_rule_solvers_view(self):
        rc, out = self.main(["--llm-selftest"], llm_fn=lambda its: {})
        rc, out = self.main(["--llm-selftest"], llm_fn=lambda its: {i.id: llm.LLMAnswer("A", "llm:m", "") for i in its})
        self.assertIn("rule solver: abstains", out)               # FakeSolver knows none of the 3 synthetic questions

    def _check(self, solver):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = run_module.main(["--check-practice"], root=self.root, solver=solver)
        return rc, out.getvalue()


if __name__ == "__main__":
    unittest.main()
