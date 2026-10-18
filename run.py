#!/usr/bin/env python3
"""DSO 429 HW3 hourly-quiz bot - command line entry point.

    python3 run.py --git                 full CI flow: sync, answer new rounds (rules first, pushed
                                         at once), then the LLM upgrade pass and a second push
    python3 run.py                       answer new rounds into answers/ (no git)
    python3 run.py --dry-run             solve and print, write nothing
    python3 run.py --check-practice      score the solver against the practice keys
    python3 run.py --llm-selftest        3 synthetic questions through the LLM fallback; writes nothing
                                         (exit 0 = at least 2/3 correct, 3 = LLM unavailable, 1 = wrong)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quizbot import gitops, llm, pipeline, source  # noqa: E402

log = logging.getLogger("quizbot")


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers[:] = [handler]
    root_logger.setLevel(level)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Answer the DSO 429 hourly quiz rounds.")
    p.add_argument("--git", action="store_true", help="full CI flow: sync with origin, answer, commit, push")
    p.add_argument("--dry-run", action="store_true", help="solve and print; write nothing")
    p.add_argument("--questions-file", metavar="PATH", help="read questions from this CSV instead of GitHub")
    p.add_argument("--now", metavar="ISO", help="override the clock, e.g. 2026-10-17T21:30:00Z")
    p.add_argument("--check-practice", action="store_true",
                   help="score the rule solver on the practice fixtures; exit 1 if below 100%%")
    p.add_argument("--no-llm", action="store_true", help="never call the LLM fallback")
    p.add_argument("--llm-selftest", action="store_true",
                   help="send 3 synthetic questions through the LLM fallback and report models, latency and "
                        "correctness; writes nothing; exit 0 if >= 2/3 correct, 3 if the LLM is unavailable, else 1")
    p.add_argument("-v", "--verbose", action="store_true", help="print every question in --dry-run output")
    return p


def run_check_practice(root: str, solver: Optional[Callable] = None) -> int:
    fixtures = os.path.join(root, "tests", "fixtures")
    questions = os.path.join(fixtures, "practice_questions.csv")
    keys = os.path.join(fixtures, "practice_keys.csv")
    solver = solver or pipeline.default_solver()
    report = pipeline.check_practice(questions, keys, solver)
    print(report.format())
    if report.total == 0:
        print("no practice questions found", file=sys.stderr)
        return 2
    return 0 if report.perfect else 1


def main(argv: Optional[List[str]] = None, *, root: str = ROOT, solver: Optional[Callable] = None,
         llm_fn: Optional[Callable] = None, runner: Optional[Callable] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    if args.git and args.dry_run:
        print("--git and --dry-run cannot be combined", file=sys.stderr)
        return 2
    if args.llm_selftest:
        if args.no_llm or args.git or args.dry_run or args.check_practice:
            print("--llm-selftest cannot be combined with --no-llm, --git, --dry-run or --check-practice",
                  file=sys.stderr)
            return 2
        return llm.selftest(llm.MemoizedLLM(llm_fn) if llm_fn is not None else None,
                            solver=solver or pipeline.default_solver())

    fixed_now = None
    if args.now:
        try:
            fixed_now = source.parse_timestamp(args.now)
        except ValueError as exc:
            print("bad --now value %r: %s" % (args.now, exc), file=sys.stderr)
            return 2

    if args.check_practice:
        return run_check_practice(root, solver)

    if args.no_llm:
        effective_llm = None
    elif llm_fn is not None:
        effective_llm = llm.MemoizedLLM(llm_fn)     # one answer per question per process, even across retries
    else:
        effective_llm = llm.make_llm_fn()           # shared 6-request / 120 s budget + memo for the whole process
    try:
        min_round = int(os.environ.get("QUIZ_MIN_ROUND", "1") or 1)
    except ValueError:
        min_round = 1

    def pipeline_fn(use_llm: bool = True, previous: Optional[pipeline.PipelineResult] = None
                    ) -> pipeline.PipelineResult:
        """One pipeline pass.  ``previous`` (the rules-only pass of the git flow) supplies the
        already-fetched rounds and the rounds it answered, which stay editable even if late."""
        now = fixed_now or datetime.now(timezone.utc)
        if previous is None:
            rounds = source.load_rounds(args.questions_file)
            log.info("%d round(s) in the source; clock %s", len(rounds), source.format_timestamp(now))
            fresh: List[int] = []
        else:
            rounds = list(previous.rounds.values())
            fresh = [r.round_id for r in previous.results if r.action == "new"]
            log.info("LLM upgrade pass; clock %s", source.format_timestamp(now))
        return pipeline.run_pipeline(rounds, root=root, now=now, solver=solver,
                                     llm_fn=effective_llm if use_llm else None,
                                     dry_run=args.dry_run, min_round=min_round, also_editable=fresh)

    if args.git:
        # Rules first and pushed at once; the (possibly slow) LLM only upgrades afterwards.
        if effective_llm is None:
            return gitops.run_git_flow(lambda: pipeline_fn(), root=root, runner=runner)
        return gitops.run_git_flow(lambda: pipeline_fn(use_llm=False), root=root, runner=runner,
                                   llm_pass_fn=lambda first: pipeline_fn(use_llm=True, previous=first))

    try:
        result = pipeline_fn()
    except source.SourceError as exc:
        log.error("cannot get the questions: %s", exc)
        return 1
    if args.dry_run:
        print(pipeline.format_dry_run(result, args.verbose))
        log.info("dry run: nothing written")
    else:
        shown = result.written[:6]
        more = " (+%d more)" % (len(result.written) - len(shown)) if len(result.written) > len(shown) else ""
        log.info("wrote %d file(s): %s%s", len(result.written), ", ".join(shown) or "-", more)
    return 0


if __name__ == "__main__":
    sys.exit(main())
