# DSO 429 HW3 - Hourly Quiz Bot

Answers the course's hourly quiz automatically. Every round has 25 multiple-choice business-math
questions; the bot solves them, writes `answers/round-NNNN.json` and pushes it to `main`, so the
push reaches GitHub within minutes of the release (100% timing credit needs it within 3 hours).

## How it works

1. **Fetch** `questions.csv` from [`FaisalXL/dso429-hourly-quiz`](https://github.com/FaisalXL/dso429-hourly-quiz)
   (`git clone` first because it is the freshest, then the GitHub API, then the raw CDN; set `QUIZ_SOURCE_REPO`
   to use another repo).
2. **Solve** each question with exact rules (`quizbot/solvers.py`): the question must match a known wording
   completely, the value is computed with the right formula, and it is matched to an option. A rule that is
   not 100% sure *abstains* instead of guessing.
3. **Fallback** for abstained questions: GitHub Models (free, uses the Actions `GITHUB_TOKEN`; two models must
   agree for a clean `llm` label; at most 6 requests per run). If that is unavailable the answer is `A`
   and the method is recorded as `guess`.
4. **Write** `answers/round-0049.json` (`{"round_id": 49, "answers": {"q1": "C", ..., "q25": "A"}}`),
   `state/rounds.json` (how each answer was produced) and `STATUS.md`.
5. **Commit and push** (`github-actions[bot]`), retrying up to 5 times if the push is rejected.

Every round found in the CSV that has no answer file yet is answered, so several rounds released at
once are all handled. The practice rounds 1-48 are in the same CSV and are answered too
(set `QUIZ_MIN_ROUND=49` to skip them).

### Re-checking inside the grading window

The last version received within 3 hours of release is the one that is graded. For 2 h 50 min after a
round's release the bot may improve its own answer: questions that were `guess`/`llm` are retried with
the rules (a method is never downgraded), and everything is re-solved if the question text itself
changed. After that the file is never touched again. A file that has no entry in `state/rounds.json`
(for example one you edited by hand) is never touched at all.

## Where it runs

- **GitHub Actions** (`.github/workflows/hourly-quiz.yml`): every 5 minutes (`2-59/5 * * * *`),
  on the **Run workflow** button, and on pushes to `main` that change code. Then it runs the unit tests,
  so a broken test makes the run fail and GitHub emails you, but never blocks answering.
  GitHub's scheduler can start runs late; the local backup below covers that.
- **Optional local backup** (macOS): `bash scripts/local_backup/install.sh` clones the repo into
  `~/Library/Application Support/dso429-quizbot/repo` and runs it every 10 minutes with launchd
  (log: `~/Library/Logs/dso429-quizbot.log`). It needs git credentials that work without a prompt
  (`gh auth login && gh auth setup-git`). Remove it with `bash scripts/local_backup/uninstall.sh`
  (`--purge` also deletes the clone and log). Both runners are safe together: a rejected push just
  re-syncs and retries.

## Checking on it

- `STATUS.md` - last update (UTC and Pacific), answered rounds, the latest 15 rounds with minutes after
  release and how many answers came from `rule` / `llm` / `guess`.
- **Actions** tab - one run per 5 minutes; red means a test or the push failed.
- **Issues** - when a round used any non-rule answer, the bot opens one issue ("Round 49: 2 question(s)
  not solved by rules") with the questions, options, and chosen letters so you can double-check before
  the window closes. Needs the `issues: write` permission already set in the workflow.
- `state/rounds.json` - per-question method and detail (`rule:compound_interest`, `llm:openai/gpt-4.1`,
  `llm-disagree:...`, `guess`, `guess:missing`).

## Running it yourself

```bash
python3 run.py --dry-run                      # fetch, solve, print; writes nothing
python3 run.py --dry-run --no-llm -v          # also print every question's method, never call the LLM
python3 run.py                                # answer new rounds into answers/ (no git)
python3 run.py --git                          # the full CI flow: sync, answer, commit, push
python3 run.py --check-practice               # score the solver on the practice rounds (must be 1200/1200)
python3 run.py --dry-run --questions-file tests/fixtures/practice_questions.csv --now 2026-10-16T00:00:00Z
```

`--git` hard-resets the checkout to `origin/main` only when `QUIZBOT_ALLOW_RESET=1` (set by the workflow
and the local backup clone). Without it the bot just uses `git pull --rebase`, so running it in your own
working folder never throws your changes away. The LLM fallback finds a token in `GITHUB_TOKEN`,
`GH_TOKEN`, or `gh auth token`; without one it is simply off. `QUIZ_LLM_MODELS` overrides the model list.

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
python3 -m unittest discover -s tests -v      # everything (what CI runs)
python3 -m unittest tests.test_pipeline -v    # pipeline: CSV parsing, window rules, LLM, git flow
python3 run.py --check-practice               # solver accuracy per question family
```

Only the Python 3 standard library is used; there is nothing to install.
