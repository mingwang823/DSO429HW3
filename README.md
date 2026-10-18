# DSO 429 HW3 - Hourly Quiz Bot

Answers the course's hourly quiz automatically. Every round has 25 multiple-choice business-math
questions; the bot solves them, writes `answers/round-NNNN.json` and pushes it to `main`, so the
push reaches GitHub within minutes of the release (100% timing credit needs it within 3 hours).

## How it works

1. **Fetch** `questions.csv` from [`FaisalXL/dso429-hourly-quiz`](https://github.com/FaisalXL/dso429-hourly-quiz)
   (`git clone` first because it is the freshest, then the GitHub API, then the raw CDN; set `QUIZ_SOURCE_REPO`
   to use another repo). Columns are matched by *name*, so extra or reordered columns do not matter; an HTML or
   JSON error page is rejected because the 8 required names are missing. A bad row (unterminated quote, round id
   outside 1-9999, ...) costs only that row, and a round whose `released_at` is more than 60 s in the future is
   skipped as "not released yet".
2. **Solve** each question with exact rules (`quizbot/solvers.py`): the question must match a known wording
   completely, the value is computed with the right formula, and it is matched to an option. A rule that is
   not 100% sure *abstains* instead of guessing.
3. **Write and push the rule answers first.** Questions the rules could not solve get a placeholder `A`
   (method `guess`). This commit ("Answer round 49") never waits for the LLM.
4. **LLM upgrade pass** for the abstained questions, then a second commit ("Update round 49") if it helped.
   The fallback is GitHub Models (free, uses the Actions `GITHUB_TOKEN`; two models must agree for a clean `llm`
   label). Limits, so a slow or rate-limited model can never hurt: at most 6 HTTP requests and 120 s of LLM time per
   run, a question is asked at most once per run even if the push has to be retried, and a question that is still a
   `guess` is not sent to the LLM again for 30 minutes (the time of the last attempt is kept in
   `state/rounds.json` as `llm_attempted_at`). If the LLM is unavailable the answer simply stays `A`/`guess`.
5. **Record** `state/rounds.json` (how each answer was produced) and `STATUS.md`, committed together with the
   answers as `github-actions[bot]`. A rejected push is retried up to 5 times (resync, re-solve, push).

Every round found in the CSV that has no answer file yet is answered, so several rounds released at
once are all handled. The practice rounds 1-48 are in the same CSV and are answered too (harmless, they are
already submitted; set `QUIZ_MIN_ROUND=49` to skip them).

### Re-checking inside the grading window

The last version received within 3 hours of release is the one that is graded. For 2 h 50 min after a
round's release the bot may improve its own answer: questions that were `guess`/`llm` are retried with
the rules (a method is never downgraded), and everything is re-solved if the question text itself
changed. After that the file is never touched again. A file that has no entry in `state/rounds.json`
(for example one you edited by hand) is never touched at all. A round that the bot only reaches late
(already past the window) is still answered, and still gets its LLM upgrade pass in the same run.

## Where it runs

- **GitHub Actions, `.github/workflows/hourly-quiz.yml`: a long-running, self-re-queueing watcher.**
  GitHub's cron fires only every few hours on this repo, so the cron is just a backstop. Each watcher job
  runs about 5.5 hours: it loops `git fetch && git reset --hard origin/main` followed by `python3 -B run.py --git`,
  sleeping 45 s between passes during the first 20 minutes of every hour (when new rounds appear) and 150 s
  otherwise. At the start of every watcher it queues its own successor with `gh workflow run` (the one event
  `GITHUB_TOKEN` may trigger); the successor waits in the `hourly-quiz` concurrency group and starts the moment
  the current job ends, and a final `always()` step re-queues one if none is waiting. The chain stops by itself
  after `QUIZBOT_STOP_AFTER` (2026-10-23T09:00Z, after the last round can earn credit). Because every pass starts
  from a fresh `origin/main`, code you push while a watcher is running takes effect on its next pass. If any
  pass fails, the job is marked failed at the end and GitHub emails you.
- **Tests, `.github/workflows/tests.yml`**: run on every push to `main` that changes code (not for
  `answers/`, `state/` or `STATUS.md`) and on **Run workflow**, separately from the watcher so a test never
  waits behind a 5.5 h job. It runs `python3 -B -m unittest discover -s tests -v` and then
  `python3 -B run.py --check-practice`, which must report 1200/1200. A red run emails you.
- **Optional local backup** (macOS): `bash scripts/local_backup/install.sh` clones the repo into
  `~/Library/Application Support/dso429-quizbot/repo` and runs it every 10 minutes with launchd
  (log: `~/Library/Logs/dso429-quizbot.log`). It needs git credentials that work without a prompt
  (`gh auth login && gh auth setup-git`). Remove it with `bash scripts/local_backup/uninstall.sh`
  (`--purge` also deletes the clone and log). Both runners are safe together: a rejected push just
  re-syncs and retries.

## Checking on it

- **`STATUS.md`** - written by the bot only when something changed, so an old "Last update" does not by
  itself mean the bot is down (look at the Actions tab for that).
  - *Last update*: UTC and Pacific time of the last change. *Answered rounds*: number of answer files
    (practice rounds included). *Solver version*: `SOLVER_VERSION` of the code that produced the last change.
  - The table lists the latest 15 rounds. *Released* comes from the quiz CSV; *Answered* is when the bot first
    pushed answers for that round (the rule answers, before any LLM upgrade); *Minutes after release* is the
    difference - under 180 means full timing credit. *rule / llm / guess* count the 25 questions by how they
    were answered: `rule` is a deterministic solver (trusted), `llm` came from GitHub Models, and any `guess`
    is a placeholder `A` that neither the rules nor the LLM could answer - open that round's issue and check it.
- **Actions** tab - the watcher job (one long run at a time, the next one queued) and the test runs;
  red means a pass failed, a push failed or a test failed.
- **Issues** - when a round used any non-rule answer, the bot opens one issue ("Round 49: 2 question(s)
  not solved by rules") with the questions, options, and chosen letters so you can double-check before
  the window closes. Needs the `issues: write` permission already set in the workflow.
- `state/rounds.json` - per-question method and detail (`rule:compound_interest`, `llm:openai/gpt-4.1`,
  `llm-disagree:...`, `guess`, `guess:missing`), plus `llm_attempted_at` for questions still waiting for the LLM.

## Running it yourself

```bash
python3 run.py --dry-run                      # fetch, solve, print; writes nothing
python3 run.py --dry-run --no-llm -v          # also print every question's method, never call the LLM
python3 run.py                                # answer new rounds into answers/ (no git)
python3 run.py --git                          # the full CI flow: sync, answer, push, LLM pass, push
python3 run.py --check-practice               # score the solver on the practice rounds (must be 1200/1200)
python3 run.py --llm-selftest                 # does the LLM fallback work? (see below)
python3 run.py --dry-run --questions-file tests/fixtures/practice_questions.csv --now 2026-10-16T00:00:00Z
```

`--git` hard-resets the checkout to `origin/main` only when `QUIZBOT_ALLOW_RESET=1` (set by the workflow
and the local backup clone). Without it the bot just uses `git pull --rebase`, so running it in your own
working folder never throws your changes away. The LLM fallback finds a token in `GITHUB_TOKEN`,
`GH_TOKEN`, or `gh auth token`; without one it is simply off. `QUIZ_LLM_MODELS` overrides the model list.

### LLM self-test

```bash
GITHUB_TOKEN=$(gh auth token) python3 run.py --llm-selftest
```

Sends 3 synthetic questions (an addition, a percent question, and a simple-interest loan the rule solver does
not know) through the same fallback code the bot uses, then prints which model(s) answered, the latency of each
request and of the batch, and whether each answer is correct. It writes no files and does not touch git or the
quiz repo. Exit code `0` = at least 2 of 3 correct, `3` = the LLM is unavailable (no token, no usable reply),
`1` = it answered but got fewer than 2 right. In GitHub Actions it needs `GITHUB_TOKEN` in the step's `env` and
the `models: read` permission; the manual workflow `.github/workflows/llm-selftest.yml` (**Run workflow** button)
runs exactly this inside Actions. On a machine whose network blocks `models.github.ai` it reports "unavailable"
(exit 3), which is the expected result there.

## Adding a new solver

New wordings or families may appear in graded rounds. In `quizbot/solvers.py`:

1. Add a family function that `re.fullmatch`es the *whole* normalised question and returns a computed value
   (anything unexpected must make it abstain, never guess).
2. Add it to the `SOLVERS` list; `quizbot/matching.py` maps the computed value to an option
   (and abstains when no single option fits).
3. Bump `SOLVER_VERSION` in `quizbot/__init__.py`.
4. Add a test with the new example under `tests/`, run the tests below, and push. Within the 2 h 50 min window
   the bot upgrades its earlier `guess`/`llm` answers for that round by itself.

## Tests

```bash
python3 -B -m unittest discover -s tests -v   # everything (what tests.yml runs)
python3 -B -m unittest tests.test_pipeline -v # pipeline: CSV parsing, window rules, LLM, two-phase git flow
python3 -B run.py --check-practice            # solver accuracy per question family (must be 1200/1200)
```

Only the Python 3 standard library is used (and the code also runs on macOS's system Python 3.9);
there is nothing to install.
