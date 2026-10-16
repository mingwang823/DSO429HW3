"""Score the deterministic solver on the 1,200 practice questions (rounds 1-48).

Requirement: 1200/1200 correct and zero abstains.  Per-family accuracy is printed.
"""
from __future__ import annotations

import csv
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from quizbot.solvers import FAMILIES, solve_question

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_practice() -> list[dict]:
    """Return [{round_id, q_index, text, options, key}] for every practice question."""
    with open(FIXTURES / "practice_keys.csv", newline="", encoding="utf-8-sig") as fh:
        keys = {
            (int(r["round_id"]), int(r["q_index"])): r["answer"].strip().upper()
            for r in csv.DictReader(fh)
        }
    rows = []
    with open(FIXTURES / "practice_questions.csv", newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rid, qi = int(r["round_id"]), int(r["q_index"])
            rows.append(
                {
                    "round_id": rid,
                    "q_index": qi,
                    "text": r["question_text"],
                    "options": {c.upper(): r[f"option_{c}"] for c in "abcd"},
                    "key": keys[(rid, qi)],
                }
            )
    return rows


class PracticeSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_practice()
        cls.results = [(row, solve_question(row["text"], row["options"])) for row in cls.rows]

    def test_fixture_shape(self) -> None:
        self.assertEqual(len(self.rows), 1200)
        self.assertEqual(len({(r["round_id"], r["q_index"]) for r in self.rows}), 1200)
        self.assertEqual({r["round_id"] for r in self.rows}, set(range(1, 49)))
        self.assertTrue(all(r["key"] in "ABCD" and len(r["key"]) == 1 for r in self.rows))

    def test_all_1200_correct_no_abstains(self) -> None:
        per_family: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # correct, total, abstain
        wrong, abstained = [], []
        methods = Counter()
        for row, sol in self.results:
            fam = sol.method.split(":", 1)[1].split("+", 1)[0] if sol.method.startswith("rule:") else "(abstain)"
            methods[sol.method] += 1
            stats = per_family[fam]
            stats[1] += 1
            if sol.letter is None:
                stats[2] += 1
                abstained.append((row, sol))
            elif sol.letter == row["key"]:
                stats[0] += 1
            else:
                wrong.append((row, sol))

        correct = sum(s[0] for s in per_family.values())
        print("\n--- practice accuracy per family ---")
        for fam in list(FAMILIES) + ["(abstain)"]:
            if fam in per_family:
                c, n, a = per_family[fam]
                print(f"  {fam:<18} {c:>4}/{n:<4} correct   abstains={a}")
        print(f"  {'TOTAL':<18} {correct:>4}/{len(self.results):<4}")
        print("  methods:", dict(methods))

        for row, sol in (wrong + abstained)[:10]:
            print(
                f"  FAIL r{row['round_id']}q{row['q_index']} key={row['key']} "
                f"got={sol.letter} {sol.method} {sol.detail}\n    {row['text']}\n    {row['options']}"
            )
        self.assertEqual(wrong, [], f"{len(wrong)} confidently wrong answers")
        self.assertEqual(abstained, [], f"{len(abstained)} abstains")
        self.assertEqual(correct, 1200)

    def test_every_family_exercised(self) -> None:
        seen = {
            sol.method.split(":", 1)[1].split("+", 1)[0]
            for _, sol in self.results
            if sol.method.startswith("rule:")
        }
        self.assertEqual(seen, set(FAMILIES))

    def test_no_tolerance_matches_needed_on_practice(self) -> None:
        """Practice keys all follow plain half-up rounding; tolerance fallback unused."""
        tol = [(r["round_id"], r["q_index"]) for r, s in self.results if s.method.endswith("+tol")]
        self.assertEqual(tol, [])


if __name__ == "__main__":
    unittest.main()
