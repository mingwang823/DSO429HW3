"""Unit tests for the deterministic family solvers (``quizbot/solvers.py``).

Every expected value below was computed independently with exact fractions (not with the
code under test) and every option set contains realistic *wrong-formula* distractors
(simple vs compound interest, markup vs margin, unweighted mean, reversed FX direction,
off-by-one dates ...).  The rules these tests protect:

* a solver must never be confidently wrong -> anything not understood must abstain;
* every family + documented variant answers correctly on at least two cases;
* every adversarial finding that was ever reproduced stays fixed (``Regression*``).

Python 3.9 compatible (no ``match``, no ``X | Y`` at runtime).
"""
from __future__ import annotations

import time
import unittest

from quizbot.solvers import FAMILIES, Solution, solve_question


def _opts(values):
    return dict(zip("ABCD", values))


class SolverCase(unittest.TestCase):
    """Shared assertion helpers."""

    def solve(self, question, options):
        return solve_question(question, _opts(options) if isinstance(options, (list, tuple)) else options)

    def assertSolves(self, question, options, letter, family=None, tol_ok=False):
        sol = self.solve(question, options)
        self.assertEqual(
            sol.letter, letter,
            f"expected {letter}, got {sol.letter} ({sol.method}: {sol.detail})\n  Q: {question}\n  O: {options}",
        )
        if family is not None:
            self.assertEqual(sol.method.split("+")[0], f"rule:{family}", sol.detail)
        if not tol_ok:
            self.assertNotIn("+tol", sol.method, f"needed the tolerance path: {sol.detail}")
        return sol

    def assertAbstains(self, question, options, reason=None):
        sol = self.solve(question, options)
        self.assertIsNone(
            sol.letter,
            f"must abstain but answered {sol.letter} ({sol.method}: {sol.detail})\n  Q: {question}\n  O: {options}",
        )
        self.assertTrue(sol.method.startswith("abstain:"), sol.method)
        if reason is not None:
            self.assertEqual(sol.method, f"abstain:{reason}", sol.detail)
        return sol

    def run_cases(self, family, cases):
        for i, (question, options, letter) in enumerate(cases):
            with self.subTest(family=family, case=i, q=question[:70]):
                self.assertSolves(question, options, letter, family)


# --------------------------------------------------------------------------
# 1. compound interest
# --------------------------------------------------------------------------

class CompoundInterestTest(SolverCase):
    FAMILY = "compound_interest"

    def test_annually(self):
        self.run_cases(self.FAMILY, [
            # 20000 * 1.05^3 = 23,152.50 ; distractors: simple, 2 years, 4 years
            ("An investment of $20,000.00 grows at 5% compounded annually. What is its value after 3 years?",
             ["$23,000.00", "$22,050.00", "$23,152.50", "$24,310.13"], "C"),
            # 31500 * 1.12^7 = 69,636.46 ; distractors: simple, 6 years, 8 years
            ("An investment of $31,500.00 grows at 12% compounded annually. What is its value after 7 years?",
             ["$57,960.00", "$69,636.46", "$62,175.41", "$77,992.84"], "B"),
        ])

    def test_yearly_and_rate_phrase(self):
        self.run_cases(self.FAMILY, [
            ("An investment of $20,000.00 grows at an annual interest rate of 5% compounded yearly. "
             "What is its value after 3 years?",
             ["$23,152.50", "$23,000.00", "$22,050.00", "$24,310.13"], "A"),
            ("An investment of $20,000.00 grows at 5% per year, compounded annually. "
             "What will it be worth after 3 years?",
             ["$23,000.00", "$23,152.50", "$22,050.00", "$24,310.13"], "B"),
        ])

    def test_semiannually_and_semi_annually(self):
        self.run_cases(self.FAMILY, [
            # 10000 * 1.03^4 = 11,255.09 ; distractors annual / simple / monthly
            ("An investment of $10,000.00 grows at 6% compounded semiannually. What is its value after 2 years?",
             ["$11,236.00", "$11,200.00", "$11,271.60", "$11,255.09"], "D"),
            ("An account with $10,000.00 earns interest at 6% compounded semi-annually. "
             "What will it be worth after 2 years?",
             ["$11,255.09", "$11,236.00", "$11,200.00", "$11,271.60"], "A"),
        ])

    def test_quarterly(self):
        self.run_cases(self.FAMILY, [
            # 5000 * 1.02^12 = 6,341.21
            ("An investment of $5,000.00 grows at 8% compounded quarterly. What is its value after 3 years?",
             ["$6,298.56", "$6,200.00", "$6,351.19", "$6,341.21"], "D"),
            # 12000 * 1.01^8 = 12,994.28 (GBP in another test); here USD 4000 6 months 8% quarterly = 4161.60
            ("An investment of $4,000.00 grows at 8% compounded quarterly. What is its value after 6 months?",
             ["$4,160.00", "$4,161.60", "$4,163.20", "$4,320.00"], "B"),
        ])

    def test_monthly(self):
        self.run_cases(self.FAMILY, [
            # 2500 * 1.01^12 = 2,817.06
            ("An investment of $2,500.00 grows at 12% compounded monthly. What is its value after 1 year?",
             ["$2,800.00", "$2,817.06", "$2,500.00", "$3,136.00"], "B"),
            # 500 * 1.005^60 = 674.43
            ("An investment of $500.00 grows at 6% compounded monthly. What is its value after 5 years?",
             ["$650.00", "$674.43", "$680.00", "$671.00"], "B"),
        ])

    def test_weekly_and_daily(self):
        self.run_cases(self.FAMILY, [
            # 1000 * (1 + 0.052/52)^52 = 1,053.35
            ("An investment of $1,000.00 grows at 5.2% compounded weekly. What is its value after 1 year?",
             ["$1,052.00", "$1,053.35", "$1,050.00", "$1,051.20"], "B"),
            # daily = 365 periods: 1000 * (1 + 0.05/365)^730 = 1,105.16
            ("An investment of $1,000.00 grows at 5% compounded daily. What is its value after 2 years?",
             ["$1,102.50", "$1,104.94", "$1,105.16", "$1,100.00"], "C"),
        ])

    def test_continuously(self):
        self.run_cases(self.FAMILY, [
            # 1000 e^0.5 = 1,648.72
            ("An investment of $1,000.00 grows at 5% compounded continuously. What is its value after 10 years?",
             ["$1,628.89", "$1,647.01", "$1,648.72", "$1,500.00"], "C"),
            # 2000 e^0.15 = 2,323.67
            ("An investment of $2,000.00 grows at 3% compounded continuously. What is its value after 5 years?",
             ["$2,323.67", "$2,318.55", "$2,300.00", "$2,321.47"], "A"),
        ])

    def test_period_in_months(self):
        self.run_cases(self.FAMILY, [
            # 8000 * 1.06^1.5 = 8,730.69 (t = 18/12)
            ("An investment of $8,000.00 grows at 6% compounded annually. What is its value after 18 months?",
             ["$8,720.00", "$8,730.69", "$8,480.00", "$8,988.80"], "B"),
            # 10000 * 1.005^12 = 10,616.78 ; 12 months at 6% monthly
            ("An investment of $10,000.00 grows at 6% compounded monthly. What is its value after 12 months?",
             ["$10,600.00", "$10,616.78", "$10,500.00", "$10,630.00"], "B"),
        ])

    def test_simple_interest_value_and_interest_amount(self):
        self.run_cases(self.FAMILY, [
            # 5000 * (1 + 0.04*3) = 5,600.00 ; compound would be 5,624.32
            ("An investment of $5,000.00 earns 4% simple interest. What is its value after 3 years?",
             ["$5,624.32", "$5,600.00", "$5,200.00", "$5,400.00"], "B"),
            # interest only: 600.00
            ("An investment of $5,000.00 earns simple interest at 4%. How much interest is earned after 3 years?",
             ["$624.32", "$5,600.00", "$600.00", "$1,800.00"], "C"),
        ])

    def test_compound_interest_amount(self):
        self.run_cases(self.FAMILY, [
            # 10000 * 1.05^2 - 10000 = 1,025.00
            ("An investment of $10,000.00 grows at 5% compounded annually. How much interest is earned after 2 years?",
             ["$1,000.00", "$1,025.00", "$1,102.50", "$11,025.00"], "B"),
            # 12000 * 1.01^8 - 12000 = 994.28 (quarterly 4% over 2 years)
            ("An investment of $12,000.00 grows at 4% compounded quarterly. What is the total interest earned after 2 years?",
             ["$960.00", "$979.20", "$994.28", "$12,994.28"], "C"),
        ])

    def test_lump_sum_adjectives_that_are_allowed(self):
        self.run_cases(self.FAMILY, [
            ("An initial investment of $20,000.00 grows at 5% compounded annually. What is its value after 3 years?",
             ["$23,152.50", "$23,000.00", "$22,050.00", "$24,310.13"], "A"),
            ("A savings account with $20,000.00 earns interest at 5% compounded annually. "
             "What is its value after 3 years?",
             ["$23,000.00", "$22,050.00", "$24,310.13", "$23,152.50"], "D"),
        ])

    def test_invested_sentence_form(self):
        self.run_cases(self.FAMILY, [
            ("$20,000.00 is invested at 5% compounded annually. What is its value after 3 years?",
             ["$23,000.00", "$23,152.50", "$22,050.00", "$24,310.13"], "B"),
            ("$5,000.00 is deposited at 8%, compounded quarterly. What is its value after 3 years?",
             ["$6,298.56", "$6,200.00", "$6,341.21", "$6,351.19"], "C"),
        ])

    def test_extra_wordings(self):
        # p.a., twice a year, "earns X% interest, compounded", "percent"
        self.run_cases(self.FAMILY, [
            ("An investment of $20,000.00 grows at 5% p.a. compounded annually. What is its value after 3 years?",
             ["$23,152.50", "$23,000.00", "$22,050.00", "$24,310.13"], "A"),
            ("An investment of $20,000.00 grows at 5% compounded twice a year. What is its value after 3 years?",
             ["$23,152.50", "$23,000.00", "$23,229.44", "$23,193.87"], "D"),
            ("An investment of $20,000.00 earns 5% interest, compounded annually. What is its value after 3 years?",
             ["$23,000.00", "$23,152.50", "$22,050.00", "$24,310.13"], "B"),
            ("An investment of $20,000.00 grows at 5 percent compounded annually. What is its value after 3 years?",
             ["$23,000.00", "$22,050.00", "$23,152.50", "$24,310.13"], "C"),
        ])

    def test_pound_symbol_with_matching_options(self):
        self.run_cases(self.FAMILY, [
            ("An investment of £12,000.00 grows at 4% compounded quarterly. What is its value after 2 years?",
             ["£12,960.00", "£12,979.20", "£13,000.00", "£12,994.28"], "D"),
            ("An investment of €20,000.00 grows at 5% compounded annually. What is its value after 3 years?",
             ["€23,000.00", "€22,050.00", "€24,310.13", "€23,152.50"], "D"),
        ])

    def test_option_formats(self):
        # code prefix, code suffix and bare numbers all identify the same value
        q = "An investment of $20,000.00 grows at 5% compounded annually. What is its value after 3 years?"
        self.assertSolves(q, ["USD 23,000.00", "USD 22,050.00", "USD 23,152.50", "USD 24,310.13"], "C", self.FAMILY)
        self.assertSolves(q, ["23,000.00 USD", "23,152.50 USD", "22,050.00 USD", "24,310.13 USD"], "B", self.FAMILY)
        self.assertSolves(q, ["23,152.50", "23,000.00", "22,050.00", "24,310.13"], "A", self.FAMILY)


# --------------------------------------------------------------------------
# 2. gross margin / profit / markup
# --------------------------------------------------------------------------

class GrossMarginTest(SolverCase):
    FAMILY = "gross_margin"

    def test_gross_margin(self):
        self.run_cases(self.FAMILY, [
            # (59000 - 52862.47) / 59000 = 10.40% ; markup 11.6% ; C/R 89.6%
            ("A product line generates $59,000.00 in revenue against $52,862.47 in cost of goods sold. "
             "What is the gross margin?",
             ["11.6%", "10.4%", "5.5%", "89.6%"], "B"),
            # (120000 - 87654.32)/120000 = 26.95 -> 27.0% ; markup 36.9% ; 100 - 36.9 = 63.1
            ("A product line generates $120,000.00 in revenue against $87,654.32 in cost of goods sold. "
             "What is the gross margin?",
             ["36.9%", "73.0%", "27.0%", "63.1%"], "C"),
        ])

    def test_gross_margin_percentage_wording(self):
        self.run_cases(self.FAMILY, [
            ("A product line generates $2,500.00 in revenue with $1,800.00 in cost of goods sold. "
             "What is the gross margin percentage?",
             ["38.9%", "72.0%", "28.0%", "12.0%"], "C"),
            ("A division generates $2,500.00 in revenue against $1,800.00 in cost of goods sold. "
             "What is its gross profit margin?",
             ["28.0%", "38.9%", "72.0%", "70.0%"], "A"),
        ])

    def test_gross_profit_is_money(self):
        self.run_cases(self.FAMILY, [
            ("A product line generates $59,000.00 in revenue against $52,862.47 in cost of goods sold. "
             "What is the gross profit?",
             ["$59,000.00", "$6,137.53", "$10.40", "$52,862.47"], "B"),
            ("A product line generates $2,500.00 in revenue against $1,800.00 in cost of goods sold. "
             "What is the gross profit?",
             ["$4,300.00", "$1,800.00", "$700.00", "$28.00"], "C"),
        ])

    def test_markup_on_cost(self):
        self.run_cases(self.FAMILY, [
            # 6137.53 / 52862.47 = 11.6% ; margin would be 10.4
            ("A product line generates $59,000.00 in revenue against $52,862.47 in cost of goods sold. "
             "What is the markup?",
             ["10.4%", "11.6%", "89.6%", "111.6%"], "B"),
            ("A product line generates $2,500.00 in revenue against $1,800.00 in cost of goods sold. "
             "What is the markup percentage on cost?",
             ["28.0%", "72.0%", "138.9%", "38.9%"], "D"),
        ])

    def test_revenue_of_and_cost_of_goods_sold_of_phrasing(self):
        self.run_cases(self.FAMILY, [
            ("A product line has a revenue of $2,500.00 and a cost of goods sold of $1,800.00. "
             "What is the gross margin?",
             ["38.9%", "72.0%", "28.0%", "12.0%"], "C"),
            ("A product line reports revenue of $120,000.00 and cost of goods sold of $87,654.32. "
             "What is the gross margin?",
             ["27.0%", "36.9%", "73.0%", "63.1%"], "A"),
        ])

    def test_sales_and_cogs_synonyms(self):
        self.run_cases(self.FAMILY, [
            ("A product line generates $120,000.00 in sales against $87,654.32 in COGS. What is the gross margin?",
             ["36.9%", "73.0%", "27.0%", "63.1%"], "C"),
            ("A product line generates $2,500.00 in sales against $1,800.00 in cost of goods sold. "
             "What is the gross margin?",
             ["28.0%", "38.9%", "72.0%", "12.0%"], "A"),
        ])

    def test_negative_margin_keeps_its_sign(self):
        self.run_cases(self.FAMILY, [
            # (40000 - 50000)/40000 = -25.0% ; markup -20.0%
            ("A product line generates $40,000.00 in revenue against $50,000.00 in cost of goods sold. "
             "What is the gross margin?",
             ["25.0%", "-20.0%", "-25.0%", "125.0%"], "C"),
            ("A product line generates $40,000.00 in revenue against $50,000.00 in cost of goods sold. "
             "What is the markup?",
             ["-25.0%", "25.0%", "-20.0%", "80.0%"], "C"),
        ])

    def test_options_without_percent_sign(self):
        self.run_cases(self.FAMILY, [
            ("A product line generates $120,000.00 in revenue against $87,654.32 in cost of goods sold. "
             "What is the gross margin?",
             ["36.9", "73.0", "27.0", "63.1"], "C"),
            ("A product line generates $40,000.00 in revenue against $50,000.00 in cost of goods sold. "
             "What is the gross margin?",
             ["25.0", "-20.0", "-25.0", "125.0"], "C"),
        ])


# --------------------------------------------------------------------------
# 3. sequence
# --------------------------------------------------------------------------

class SequenceTest(SolverCase):
    FAMILY = "sequence_next"

    def test_arithmetic(self):
        self.run_cases(self.FAMILY, [
            ("A monthly headcount series runs 13, 18, 23, 28, 33. If the pattern continues, what is the next value?",
             ["36", "38", "43", "37"], "B"),
            ("A daily inventory series runs 50, 43, 36, 29. If the pattern continues, what is the next value?",
             ["21", "15", "22", "29"], "C"),
        ])

    def test_geometric(self):
        self.run_cases(self.FAMILY, [
            ("A weekly sales series runs 3, 6, 12, 24. If the pattern continues, what is the next value?",
             ["30", "36", "48", "72"], "C"),
            # ratio 1/2
            ("A weekly sales series runs 80, 40, 20, 10. If the pattern continues, what is the next value?",
             ["5", "0", "-10", "2.5"], "A"),
        ])

    def test_constant_second_difference(self):
        self.run_cases(self.FAMILY, [
            ("A quarterly orders sequence runs 1, 4, 9, 16, 25. If the pattern continues, what is the next value?",
             ["34", "35", "36", "41"], "C"),
            ("A monthly headcount series runs 2, 5, 10, 17, 26. If the pattern continues, what is the next value?",
             ["35", "36", "37", "38"], "C"),
        ])

    def test_decimals_and_negatives(self):
        self.run_cases(self.FAMILY, [
            ("A monthly headcount series runs 1.5, 3.0, 4.5, 6.0. If the pattern continues, what is the next value?",
             ["7.5", "7.0", "8.0", "9.0"], "A"),
            ("A monthly headcount series runs -4, -1, 2, 5. If the pattern continues, what is the next value?",
             ["7", "8", "9", "-7"], "B"),
        ])

    def test_and_separator_and_three_terms(self):
        self.run_cases(self.FAMILY, [
            ("A monthly headcount series runs 5, 10 and 15. If the pattern continues, what is the next value?",
             ["25", "20", "17", "30"], "B"),
            ("A monthly headcount series runs 7, 14, and 28. If the pattern continues, what is the next value?",
             ["56", "35", "42", "21"], "A"),
        ])

    def test_dollar_terms_and_trend_wording(self):
        self.run_cases(self.FAMILY, [
            ("A weekly revenue series runs $120, $135, $150, $165. If the trend continues, what is the next value?",
             ["$175", "$180", "$195", "$170"], "B"),
            ("A weekly revenue series runs $2, $4, $8, $16. If the pattern continues, what is the next number?",
             ["$24", "$32", "$18", "$20"], "B"),
        ])

    def test_other_nouns_and_question_forms(self):
        self.run_cases(self.FAMILY, [
            ("A units sequence goes 4, 8, 12, 16. Assuming the pattern holds, what comes next?",
             ["20", "18", "24", "32"], "A"),
            ("The call volume series reads 100, 90, 80, 70. If the pattern continues, what is the next term?",
             ["60", "50", "65", "0"], "A"),
        ])


# --------------------------------------------------------------------------
# 4. discounts / markups
# --------------------------------------------------------------------------

class DiscountTest(SolverCase):
    FAMILY = "discount_price"

    def test_sale_price(self):
        self.run_cases(self.FAMILY, [
            # 45.33 * 0.8 = 36.264 -> 36.26
            ("An item listed at $45.33 is discounted by 20%. What is the sale price?",
             ["$54.40", "$9.07", "$36.26", "$25.33"], "C"),
            ("An item priced at $120.00 is reduced by 15%. What is the final price?",
             ["$102.00", "$105.00", "$18.00", "$138.00"], "A"),
        ])

    def test_discount_amount_and_savings(self):
        self.run_cases(self.FAMILY, [
            ("An item listed at $45.33 is discounted by 20%. What is the discount amount?",
             ["$36.26", "$9.07", "$54.40", "$45.33"], "B"),
            ("An item marked at $89.99 is marked down by 20%. How much is saved?",
             ["$71.99", "$18.00", "$17.99", "$107.99"], "B"),
            ("An item listed at $120.00 is discounted by 15%. How much does the customer save?",
             ["$102.00", "$105.00", "$138.00", "$18.00"], "D"),
        ])

    def test_markup(self):
        self.run_cases(self.FAMILY, [
            ("An item costing $80.00 is marked up by 25%. What is the selling price?",
             ["$20.00", "$60.00", "$100.00", "$105.00"], "C"),
            ("An item bought for $80.00 is marked up by 25%. What is the markup amount?",
             ["$100.00", "$20.00", "$60.00", "$25.00"], "B"),
        ])

    def test_successive_discounts(self):
        self.run_cases(self.FAMILY, [
            # 200 * 0.9 * 0.95 = 171.00 ; naive 15% -> 170.00 ; first only -> 180
            ("An item listed at $200.00 is discounted by 10% and then by a further 5%. What is the sale price?",
             ["$170.00", "$180.00", "$171.00", "$190.00"], "C"),
            # 150 * 0.8 * 0.9 = 108.00 ; naive 30% -> 105.00
            ("An item priced at $150.00 is reduced by 20% and then by an additional 10%. What is the final price?",
             ["$105.00", "$108.00", "$120.00", "$112.50"], "B"),
        ])

    def test_on_sale_and_has_discount_forms(self):
        self.run_cases(self.FAMILY, [
            ("An item listed at $2,500.00 is on sale for 35% off. What is the sale price?",
             ["$875.00", "$1,625.00", "$2,465.00", "$3,375.00"], "B"),
            ("An item priced at $120.00 has a 25% discount. What is the sale price?",
             ["$150.00", "$30.00", "$95.00", "$90.00"], "D"),
        ])

    def test_percent_word_and_pound_symbol(self):
        self.run_cases(self.FAMILY, [
            ("An item listed at $45.33 is discounted by 20 percent. What is the sale price?",
             ["$54.40", "$9.07", "$25.33", "$36.26"], "D"),
            ("An item listed at £120.00 is discounted by 15%. What is the sale price?",
             ["£102.00", "£105.00", "£18.00", "£138.00"], "A"),
        ])

    def test_zero_discount_and_full_discount(self):
        self.run_cases(self.FAMILY, [
            ("An item listed at $45.33 is discounted by 100%. What is the sale price?",
             ["$45.33", "$0.00", "$4.53", "$90.66"], "B"),
            ("An item listed at $45.33 is discounted by 0%. What is the sale price?",
             ["$0.00", "$45.33", "$4.53", "$90.66"], "B"),
        ])


# --------------------------------------------------------------------------
# 5. percentage change
# --------------------------------------------------------------------------

class PercentChangeTest(SolverCase):
    FAMILY = "percent_change"

    def test_decrease(self):
        self.run_cases(self.FAMILY, [
            # (297 - 350) / 350 = -15.14 -> -15.1% ; B/A 84.9 ; (A-B)/B 17.8
            ("Quarterly revenue moved from $350.00 to $297.00. What is the percentage change?",
             ["84.9%", "15.1%", "-17.8%", "-15.1%"], "D"),
            ("Quarterly revenue fell from $200.00 to $145.00. What is the percentage change?",
             ["-27.5%", "-37.9%", "27.5%", "72.5%"], "A"),
        ])

    def test_increase(self):
        self.run_cases(self.FAMILY, [
            ("Monthly sales rose from $80.00 to $100.00. What is the percentage change?",
             ["20.0%", "-20.0%", "125.0%", "25.0%"], "D"),
            ("Prices increased from $1,250.00 to $1,500.00. What is the percent change?",
             ["120.0%", "20.0%", "-16.7%", "16.7%"], "B"),
        ])

    def test_plain_numbers(self):
        self.run_cases(self.FAMILY, [
            ("Headcount fell from 60 to 45. What is the percent change?",
             ["-25.0%", "25.0%", "75.0%", "-33.3%"], "A"),
            ("Orders grew from 40 to 50. What is the percentage change?",
             ["-20.0%", "20.0%", "25.0%", "125.0%"], "C"),
        ])

    def test_verb_variants(self):
        self.run_cases(self.FAMILY, [
            ("Prices went up from $1,250.00 to $1,500.00. What is the percentage change?",
             ["20.0%", "-16.7%", "120.0%", "25.0%"], "A"),
            ("Prices went down from $60.00 to $45.00. What is the percentage change?",
             ["33.3%", "-25.0%", "75.0%", "-33.3%"], "B"),
            ("Quarterly revenue went from $1,250.00 to $1,500.00. What is the percentage change?",
             ["20.0%", "-16.7%", "120.0%", "25.0%"], "A"),
            ("Quarterly revenue falls from $200.00 to $145.00. What is the percentage change?",
             ["-27.5%", "-37.9%", "27.5%", "72.5%"], "A"),
            ("Monthly sales rises from $80.00 to $100.00. What is the percentage change?",
             ["20.0%", "-20.0%", "125.0%", "25.0%"], "D"),
        ])

    def test_options_without_percent_sign(self):
        self.run_cases(self.FAMILY, [
            ("Quarterly revenue moved from $350.00 to $297.00. What is the percentage change?",
             ["84.9", "15.1", "-17.8", "-15.1"], "D"),
            ("Monthly sales rose from $80.00 to $100.00. What is the percentage change?",
             ["20.0", "-20.0", "125.0", "25.0"], "D"),
        ])

    def test_pound_symbol(self):
        self.run_cases(self.FAMILY, [
            ("Quarterly revenue moved from £350.00 to £297.00. What is the percentage change?",
             ["84.9%", "15.1%", "-17.8%", "-15.1%"], "D"),
            ("Quarterly revenue moved from €200.00 to €145.00. What is the percentage change?",
             ["-27.5%", "-37.9%", "27.5%", "72.5%"], "A"),
        ])


# --------------------------------------------------------------------------
# 6. settlement date
# --------------------------------------------------------------------------

class SettlementDateTest(SolverCase):
    FAMILY = "settlement_date"

    def test_calendar_days_hyphenated(self):
        self.run_cases(self.FAMILY, [
            # 12 May + 36 days = 17 June 2026
            ("A contract signed on 12 May 2026 has a 36-day settlement period. On what date does it settle?",
             ["16 June 2026", "18 June 2026", "17 June 2026", "19 June 2026"], "C"),
            # 3 March + 90 days = 1 June 2026
            ("A trade executed on 3 March 2026 has a 90-day settlement period. On what date does it settle?",
             ["31 May 2026", "1 June 2026", "2 June 2026", "30 May 2026"], "B"),
        ])

    def test_month_year_and_leap_boundaries(self):
        self.run_cases(self.FAMILY, [
            ("A trade executed on 30 December 2026 has a 5-day settlement period. On what date does it settle?",
             ["3 January 2027", "4 January 2027", "5 January 2027", "4 December 2026"], "B"),
            # 2028 is a leap year: 28 Feb + 2 = 1 March
            ("An invoice dated on 28 February 2028 has a settlement period of 2 days. On what date does it settle?",
             ["29 February 2028", "2 March 2028", "1 March 2028", "28 March 2028"], "C"),
            # 2026 is not: 28 Feb + 2 = 2 March
            ("An invoice dated on 28 February 2026 has a settlement period of 2 days. On what date does it settle?",
             ["1 March 2026", "2 March 2026", "29 February 2026", "3 March 2026"], "B"),
        ])

    def test_business_days_skip_weekends_only(self):
        self.run_cases(self.FAMILY, [
            # Tue 12 May 2026 + 10 business days = Tue 26 May 2026 (calendar +10 would be 22 May)
            ("A trade placed on 12 May 2026 has a 10-business-day settlement period. On what date does it settle?",
             ["22 May 2026", "26 May 2026", "25 May 2026", "27 May 2026"], "B"),
            # Fri 15 May 2026 + 5 working days = Fri 22 May 2026
            ("A contract signed on 15 May 2026 has a settlement period of 5 working days. On what date does it settle?",
             ["20 May 2026", "21 May 2026", "22 May 2026", "25 May 2026"], "C"),
            # Fri + 1 business day = Mon 18 May 2026
            ("A contract signed on 15 May 2026 has a 1-business-day settlement period. On what date does it settle?",
             ["16 May 2026", "17 May 2026", "19 May 2026", "18 May 2026"], "D"),
            # Sat 17 Oct 2026 + 3 working days = Wed 21 Oct 2026
            ("An order placed on 17 October 2026 has a settlement period of 3 working days. On what date does it settle?",
             ["20 October 2026", "21 October 2026", "22 October 2026", "19 October 2026"], "B"),
        ])

    def test_explicit_calendar_keyword(self):
        self.run_cases(self.FAMILY, [
            ("A contract signed on 12 May 2026 has a 36-calendar-day settlement period. On what date does it settle?",
             ["16 June 2026", "18 June 2026", "17 June 2026", "19 June 2026"], "C"),
            ("A contract signed on 12 May 2026 has a settlement period of 36 calendar days. On what date does it settle?",
             ["17 June 2026", "16 June 2026", "18 June 2026", "19 June 2026"], "A"),
        ])

    def test_date_formats_in_question_and_options(self):
        self.run_cases(self.FAMILY, [
            ("A contract signed on May 12, 2026 has a 36-day settlement period. On what date does it settle?",
             ["June 16, 2026", "June 18, 2026", "June 17, 2026", "June 19, 2026"], "C"),
            ("A contract signed on 2026-05-12 has a 36-day settlement period. On what date does it settle?",
             ["2026-06-16", "2026-06-17", "2026-06-18", "2026-06-19"], "B"),
            ("A contract signed on 12 May 2026 has a 36-day settlement period. On what date does it settle?",
             ["16 Jun 2026", "18 Jun 2026", "19 Jun 2026", "17 Jun 2026"], "D"),
            # options in a different format than the question
            ("A contract signed on 12 May 2026 has a 36-day settlement period. On what date does it settle?",
             ["2026-06-16", "June 17, 2026", "2026-06-18", "2026-06-19"], "B"),
            # an unambiguous numeric date (day > 12 so it can only be dd/mm/yyyy)
            ("A contract signed on 13/06/2026 has a 10-day settlement period. On what date does it settle?",
             ["22 June 2026", "23 June 2026", "24 June 2026", "3 July 2026"], "B"),
        ])

    def test_settles_n_days_after_signing(self):
        self.run_cases(self.FAMILY, [
            ("A contract signed on 12 May 2026 settles 36 days after signing. When does it settle?",
             ["16 June 2026", "17 June 2026", "18 June 2026", "19 June 2026"], "B"),
            ("A trade executed on 3 March 2026 settles 90 days after the signing date. What is the settlement date?",
             ["31 May 2026", "30 May 2026", "2 June 2026", "1 June 2026"], "D"),
        ])

    def test_unhyphenated_and_question_variants(self):
        self.run_cases(self.FAMILY, [
            ("A contract signed on 12 May 2026 has a 36 day settlement period. On what date does it settle?",
             ["16 June 2026", "18 June 2026", "17 June 2026", "19 June 2026"], "C"),
            ("A contract signed on 12 May 2026 has a 36-day settlement period. When will it settle?",
             ["16 June 2026", "17 June 2026", "18 June 2026", "19 June 2026"], "B"),
            ("A contract signed on 12 May 2026 has a 36-day settlement period. On what date will it settle?",
             ["17 June 2026", "16 June 2026", "18 June 2026", "19 June 2026"], "A"),
        ])

    def test_weekday_prefix_must_agree(self):
        self.assertSolves(
            "A contract signed on Tuesday, 12 May 2026 has a 36-day settlement period. On what date does it settle?",
            ["16 June 2026", "17 June 2026", "18 June 2026", "19 June 2026"], "B", self.FAMILY)


# --------------------------------------------------------------------------
# 7. break-even
# --------------------------------------------------------------------------

_BE = "Fixed costs are $12,000.00. Each unit sells for $31.00 with a variable cost of $18.00. "


class BreakEvenTest(SolverCase):
    FAMILY = "break_even"

    def test_nearest_unit(self):
        self.run_cases(self.FAMILY, [
            # 12000 / 13 = 923.08 -> 923 ; ceil 924 is a distractor
            (_BE + "How many units must be sold to break even, to the nearest unit?",
             ["924 units", "923 units", "387 units", "667 units"], "B"),
            # 60000 / 15 = 4,000
            ("Fixed costs are $60,000.00. Each unit sells for $25.00 with a variable cost of $10.00. "
             "How many units must be sold to break even, to the nearest unit?",
             ["2,400 units", "6,000 units", "4,000 units", "2,000 units"], "C"),
            ("Fixed costs are $12,000.00. Each unit sells for $31.00 with a variable cost of $18.00. "
             "How many units must be sold to break even, rounded to the nearest unit?",
             ["923 units", "924 units", "387 units", "667 units"], "A"),
        ])

    def test_round_up_wordings(self):
        for tail in (
            "How many units must be sold to break even, rounded up to the next whole unit?",
            "What is the minimum number of whole units that must be sold to break even?",
            "At least how many units must be sold to break even?",
            "What is the minimum number of units needed to break even?",
            "How many units must be sold to break even? Round up to the nearest whole unit.",
        ):
            with self.subTest(tail=tail):
                self.assertSolves(_BE + tail, ["923 units", "924 units", "387 units", "667 units"], "B", self.FAMILY)
        # 10000 / 13 = 769.23 -> ceil 770
        self.assertSolves(
            "Fixed costs are $10,000.00. Each unit sells for $30.00 with a variable cost of $17.00. "
            "At least how many units must be sold to break even?",
            ["769 units", "770 units", "7,692 units", "1,300 units"], "B", self.FAMILY)

    def test_target_profit(self):
        self.run_cases(self.FAMILY, [
            # (12000 + 5000) / 13 = 1,307.69 -> 1,308
            (_BE + "How many units must be sold to earn a target profit of $5,000.00, to the nearest unit?",
             ["923 units", "385 units", "1,308 units", "1,307 units"], "C"),
            # (12000 + 5000) / 13 rounded up is 1,308 as well
            (_BE + "How many units must be sold to achieve a profit of $5,000.00, rounded up to the next whole unit?",
             ["1,307 units", "1,308 units", "923 units", "385 units"], "B"),
        ])

    def test_break_even_revenue(self):
        self.run_cases(self.FAMILY, [
            # 12000 / 13 * 31 = 28,615.38
            (_BE + "What is the break-even revenue?",
             ["$28,615.38", "$12,000.00", "$923.08", "$16,615.38"], "A"),
            # 60000 / 15 * 25 = 100,000.00
            ("Fixed costs are $60,000.00. Each unit sells for $25.00 with a variable cost of $10.00. "
             "What is the break-even point, in sales dollars?",
             ["$60,000.00", "$150,000.00", "$100,000.00", "$4,000.00"], "C"),
        ])

    def test_contribution_margin_wording(self):
        self.run_cases(self.FAMILY, [
            # 10000 / 13 = 769.23 -> 769
            ("Fixed costs are $10,000.00. Each unit has a contribution margin of $13.00. "
             "How many units must be sold to break even, to the nearest unit?",
             ["769 units", "770 units", "1,300 units", "7,692 units"], "A"),
            # (8000 + 2500) / 12.5 = 840
            ("Fixed costs are $8,000.00. Each unit has a contribution margin of $12.50. "
             "How many units must be sold to earn a target profit of $2,500.00, to the nearest unit?",
             ["640 units", "200 units", "840 units", "1,312 units"], "C"),
        ])

    def test_no_rounding_phrase_exact_integer(self):
        self.run_cases(self.FAMILY, [
            ("Fixed costs are $60,000.00. Each unit sells for $25.00 with a variable cost of $10.00. "
             "How many units must be sold to break even?",
             ["2,400 units", "6,000 units", "4,000 units", "2,000 units"], "C"),
            ("Fixed costs are $9,000.00. Each unit sells for $20.00 with a variable cost of $11.00. "
             "How many units must be sold to break even?",
             ["1,000 units", "450 units", "900 units", "818 units"], "A"),
        ])

    def test_no_rounding_phrase_non_integer_needs_no_competing_neighbour(self):
        # 923.08: only the half-up neighbour (923) is offered -> unambiguous
        self.run_cases(self.FAMILY, [
            (_BE + "How many units must be sold to break even?",
             ["923 units", "387 units", "667 units", "1,533 units"], "A"),
            (_BE + "How many units must be sold to break even?",
             ["1,533 units", "387 units", "923 units", "667 units"], "C"),
        ])

    def test_rounding_tie_with_single_neighbour(self):
        # (8000 + 2500) / 8 = 1,312.5 exactly: half-up is 1,313
        q = ("Fixed costs are $8,000.00. Each unit sells for $12.00 with a variable cost of $4.00. "
             "How many units must be sold to earn a target profit of $2,500.00, to the nearest unit?")
        self.assertSolves(q, ["1,313 units", "2,625 units", "1,000 units", "875 units"], "A", self.FAMILY)


# --------------------------------------------------------------------------
# 8. ratio split
# --------------------------------------------------------------------------

_RS2 = "A $900.00 marketing budget is split between two channels in the ratio 2:5. "
_RS3 = "A $1,200.00 marketing budget is split between three channels in the ratio 2:3:5. "


class RatioSplitTest(SolverCase):
    FAMILY = "ratio_split"

    def test_first_and_second_share_two_way(self):
        self.run_cases(self.FAMILY, [
            # 900 * 2/7 = 257.14 ; 900 * 5/7 = 642.86 ; wrong: 900/2 = 450
            (_RS2 + "What does the first channel receive?",
             ["$128.57", "$642.86", "$257.14", "$450.00"], "C"),
            (_RS2 + "What does the second channel receive?",
             ["$257.14", "$450.00", "$128.57", "$642.86"], "D"),
        ])

    def test_three_and_four_way_ordinals(self):
        self.run_cases(self.FAMILY, [
            ("A $1,200.00 bonus pool is divided among three employees in the ratio 2:3:5. "
             "What does the third employee receive?",
             ["$400.00", "$360.00", "$600.00", "$240.00"], "C"),
            ("A $1,200.00 bonus pool is divided among three employees in the ratio 2:3:5. "
             "What does the second employee receive?",
             ["$400.00", "$360.00", "$600.00", "$240.00"], "B"),
            ("A $1,200.00 bonus pool is divided among three employees in the ratio 2:3:5. "
             "What does the first employee receive?",
             ["$400.00", "$360.00", "$600.00", "$240.00"], "D"),
            # 5000 * 4/10 = 2,000
            ("A $5,000.00 profit is shared among four partners in the ratio 1:2:3:4. "
             "How much will the fourth partner get?",
             ["$1,250.00", "$2,500.00", "$2,000.00", "$1,500.00"], "C"),
            ("A $5,000.00 profit is shared among four partners in the ratio 1:2:3:4. "
             "How much will the first partner get?",
             ["$500.00", "$1,250.00", "$2,000.00", "$1,000.00"], "A"),
        ])

    def test_larger_and_smaller(self):
        self.run_cases(self.FAMILY, [
            (_RS2 + "What does the larger channel receive?",
             ["$642.86", "$128.57", "$2,250.00", "$257.14"], "A"),
            (_RS2 + "What does the smaller channel receive?",
             ["$642.86", "$128.57", "$2,250.00", "$257.14"], "D"),
            (_RS2 + "What is the larger share?",
             ["$257.14", "$642.86", "$450.00", "$128.57"], "B"),
            (_RS2 + "What is the smaller share?",
             ["$257.14", "$642.86", "$450.00", "$128.57"], "A"),
            ("A $750.00 prize is split between two winners in the ratio 3:7. What does the larger winner receive?",
             ["$225.00", "$525.00", "$375.00", "$250.00"], "B"),
        ])

    def test_largest_and_smallest_with_three_parts(self):
        self.run_cases(self.FAMILY, [
            (_RS3 + "What does the largest channel receive?",
             ["$120.00", "$1,200.00", "$600.00", "$240.00"], "C"),
            (_RS3 + "What does the smallest channel receive?",
             ["$240.00", "$1,200.00", "$600.00", "$120.00"], "A"),
            (_RS3 + "What is the largest share?",
             ["$600.00", "$360.00", "$240.00", "$400.00"], "A"),
            (_RS3 + "What is the smallest share?",
             ["$600.00", "$360.00", "$240.00", "$400.00"], "C"),
        ])

    def test_share_question_forms(self):
        self.run_cases(self.FAMILY, [
            (_RS2 + "What is the first share?",
             ["$642.86", "$128.57", "$257.14", "$450.00"], "C"),
            (_RS2 + "What is the second allocation?",
             ["$642.86", "$128.57", "$257.14", "$450.00"], "A"),
            (_RS2 + "How much will the first channel get?",
             ["$642.86", "$128.57", "$450.00", "$257.14"], "D"),
        ])

    def test_equal_parts_and_other_nouns(self):
        self.run_cases(self.FAMILY, [
            ("A $2,400.00 reserve is allocated among three teams in the ratio 1:1:2. What does the first team receive?",
             ["$800.00", "$600.00", "$1,200.00", "$2,400.00"], "B"),
            ("A $2,400.00 reserve is allocated among three teams in the ratio 1:1:2. What does the largest team receive?",
             ["$800.00", "$600.00", "$1,200.00", "$2,400.00"], "C"),
        ])

    def test_ratio_written_with_to_and_a_ratio_noun(self):
        self.run_cases(self.FAMILY, [
            ("A $900.00 marketing budget is split between two channels in the ratio of 2 to 5. "
             "What does the first channel receive?",
             ["$128.57", "$642.86", "$257.14", "$450.00"], "C"),
            ("A $900.00 marketing budget is split between two channels in a 2:5 ratio. "
             "What does the second channel receive?",
             ["$128.57", "$642.86", "$257.14", "$450.00"], "B"),
            ("A $900.00 marketing budget is split between two channels in the ratio 2 : 5. "
             "What does the first channel receive?",
             ["$128.57", "$642.86", "$257.14", "$450.00"], "C"),
        ])

    def test_euro_symbol(self):
        self.run_cases(self.FAMILY, [
            ("A €900.00 marketing budget is split between two channels in the ratio 2:5. "
             "What does the first channel receive?",
             ["€128.57", "€642.86", "€257.14", "€450.00"], "C"),
            ("A €900.00 marketing budget is split between two channels in the ratio 2:5. "
             "What does the larger channel receive?",
             ["€128.57", "€642.86", "€257.14", "€450.00"], "B"),
        ])


# --------------------------------------------------------------------------
# 9. weighted average
# --------------------------------------------------------------------------

_WA = "Region A had 65 orders averaging $65.00; region B had 189 orders averaging $49.00. "


class WeightedAverageTest(SolverCase):
    FAMILY = "weighted_average"

    def test_two_groups(self):
        self.run_cases(self.FAMILY, [
            # (65*65 + 189*49) / 254 = 53.09 ; unweighted 57.00
            (_WA + "What is the overall average order value?",
             ["57.00", "60.91", "53.09", "65.00"], "C"),
            # (30*81.5 + 20*74) / 50 = 78.50 ; unweighted 77.75
            ("Region A had 30 orders averaging $81.50; region B had 20 orders averaging $74.00. "
             "What is the combined average order value?",
             ["77.75", "78.50", "81.50", "74.00"], "B"),
        ])

    def test_three_groups_and_separators(self):
        self.run_cases(self.FAMILY, [
            # (10*20 + 20*30 + 30*40)/60 = 33.33
            ("Store A had 10 orders averaging $20.00, store B had 20 orders averaging $30.00 and "
             "store C had 30 orders averaging $40.00. What is the weighted average order value?",
             ["30.00", "33.33", "35.00", "28.00"], "B"),
            # (120*4.75 + 80*6.2 + 50*9.1)/250 = 6.08 ; unweighted 6.68
            ("Branch A had 120 orders averaging $4.75; branch B had 80 orders averaging $6.20; and "
             "branch C had 50 orders averaging $9.10. What is the overall average order value?",
             ["6.68", "6.20", "6.08", "7.50"], "C"),
        ])

    def test_options_with_and_without_dollar_sign(self):
        self.run_cases(self.FAMILY, [
            (_WA + "What is the overall average order value?",
             ["$57.00", "$60.91", "$53.09", "$65.00"], "C"),
            (_WA + "What is the overall average order value?",
             ["USD 57.00", "USD 60.91", "USD 65.00", "USD 53.09"], "D"),
        ])

    def test_plain_numbers_without_dollar(self):
        self.run_cases(self.FAMILY, [
            # (40*12.5 + 60*18.25)/100 = 15.95 ; unweighted 15.375 (15.38)
            ("Class 1 had 40 students averaging 12.50; Class 2 had 60 students averaging 18.25. "
             "What is the overall average score?",
             ["15.38", "15.95", "16.00", "18.25"], "B"),
            ("Region A had 30 students averaging 81.50; Region B had 20 students averaging 74.00. "
             "What is the combined average score?",
             ["78.50", "77.75", "77.00", "82.00"], "A"),
        ])

    def test_tail_variants(self):
        for tail, nice in (
            ("What is the overall average?", True),
            ("What is the blended average order value?", True),
            ("What is the weighted mean?", True),
            ("What is the combined average order value across both regions?", True),
            ("What is the overall average order value across all regions?", True),
        ):
            with self.subTest(tail=tail):
                q = ("Region A had 30 orders averaging $81.50; region B had 20 orders averaging $74.00. " + tail)
                self.assertSolves(q, ["77.75", "81.50", "78.50", "74.00"], "C", self.FAMILY)

    def test_euro_symbol(self):
        self.assertSolves(
            "Region A had 30 orders averaging €81.50; region B had 20 orders averaging €74.00. "
            "What is the overall average order value?",
            ["€77.75", "€78.50", "€81.50", "€74.00"], "B", self.FAMILY)


# --------------------------------------------------------------------------
# 10. currency conversion
# --------------------------------------------------------------------------

class CurrencyTest(SolverCase):
    FAMILY = "currency"

    def test_usd_to_eur_multiplies_when_rate_is_eur_per_usd(self):
        self.run_cases(self.FAMILY, [
            # 2450.05 * 1.5726 = 3,852.95 ; wrong direction 2450.05 / 1.5726 = 1,557.96
            ("An invoice for $2,450.05 is converted to euros at 1.5726 euros per US dollar. How many euros is that?",
             ["1,557.96", "2,450.05", "3,852.95", "4,052.95"], "C"),
            # 2450.05 * 0.9164 = 2,245.23 ; wrong 2,673.56
            ("An invoice for $2,450.05 is converted to euros at 0.9164 euros per US dollar. How many euros is that?",
             ["2,673.56", "2,245.23", "2,450.05", "2,045.23"], "B"),
        ])

    def test_eur_to_usd_with_inverse_rate(self):
        self.run_cases(self.FAMILY, [
            # 2450.05 * 1.0873 = 2,663.94 ; wrong direction 2,253.33
            ("An invoice for €2,450.05 is converted to US dollars at 1.0873 US dollars per euro. "
             "How many US dollars is that?",
             ["2,253.33", "2,663.94", "2,450.05", "2,930.33"], "B"),
            # amount given in words
            ("An invoice for 2,450.05 euros is converted to US dollars at 1.0873 US dollars per euro. "
             "How many US dollars is that?",
             ["2,253.33", "2,450.05", "2,930.33", "2,663.94"], "D"),
        ])

    def test_rate_stated_the_other_way_divides(self):
        self.run_cases(self.FAMILY, [
            # $ -> euros at 0.9164 US dollars per euro : 2450.05 / 0.9164 = 2,673.56
            ("An invoice for $2,450.05 is converted to euros at 0.9164 US dollars per euro. How many euros is that?",
             ["2,245.23", "2,673.56", "2,450.05", "3,000.00"], "B"),
            # 5000 yen / 151.2 yen per USD = 33.07
            ("5,000 yen is converted to US dollars at 151.2 yen per US dollar. How much is that in US dollars?",
             ["756,000.00", "33.07", "5,000.00", "0.03"], "B"),
        ])

    def test_one_unit_equals_form(self):
        self.run_cases(self.FAMILY, [
            ("An invoice for $2,450.05 is converted to euros at 1 US dollar = 0.9164 euros. How many euros is that?",
             ["2,673.56", "2,450.05", "2,245.23", "2,045.23"], "C"),
            ("An invoice for €2,450.05 is converted to US dollars at 1 euro = 1.0873 US dollars. "
             "How many US dollars is that?",
             ["2,253.33", "2,450.05", "2,930.33", "2,663.94"], "D"),
        ])

    def test_symbol_and_name_sources_pounds(self):
        self.run_cases(self.FAMILY, [
            ("An invoice for £1,800.00 is converted to US dollars at 1.27 US dollars per pound. "
             "How many US dollars is that?",
             ["1,417.32", "2,286.00", "1,800.00", "2,287.00"], "B"),
            ("1,800 British pounds are converted to US dollars at 1.27 US dollars per British pound. "
             "How many US dollars is that?",
             ["2,286.00", "1,417.32", "1,800.00", "2,287.00"], "A"),
        ])

    def test_question_tails(self):
        base = "An invoice for $2,450.05 is converted to euros at 1.5726 euros per US dollar. "
        for tail in (
            "How many euros is that?",
            "What is that in euros?",
            "How much is that in euros?",
            "What is the converted amount in euros?",
            "What is the equivalent in euros?",
            "How many euros does that make?",
        ):
            with self.subTest(tail=tail):
                self.assertSolves(base + tail, ["1,557.96", "3,852.95", "2,450.05", "4,052.95"], "B", self.FAMILY)

    def test_option_formats_with_currency_codes_and_names(self):
        q = "An invoice for $2,450.05 is converted to euros at 1.5726 euros per US dollar. How many euros is that?"
        self.assertSolves(q, ["EUR 1,557.96", "EUR 2,450.05", "EUR 3,852.95", "EUR 4,052.95"], "C", self.FAMILY)
        self.assertSolves(q, ["1,557.96 euros", "3,852.95 euros", "2,450.05 euros", "4,052.95 euros"], "B", self.FAMILY)
        self.assertSolves(q, ["€1,557.96", "€2,450.05", "€3,852.95", "€4,052.95"], "C", self.FAMILY)
        q2 = ("An invoice for €2,450.05 is converted to US dollars at 1.0873 US dollars per euro. "
              "How many US dollars is that?")
        self.assertSolves(q2, ["USD 2,253.33", "USD 2,663.94", "USD 2,236.16", "USD 2,930.33"], "B", self.FAMILY)


# --------------------------------------------------------------------------
# 11. probability
# --------------------------------------------------------------------------

_PQ = "A support queue holds 25 billing tickets, 6 technical tickets and 10 account tickets. If one is drawn at random, "


class ProbabilityTest(SolverCase):
    FAMILY = "probability"

    def test_single_category(self):
        self.run_cases(self.FAMILY, [
            # 25/41 = 60.98 -> 61.0% ; 25/(6+10) = 156.3%
            (_PQ + "what is the probability it is a billing ticket?",
             ["156.3%", "61.0%", "75.6%", "39.0%"], "B"),
            # 6/41 = 14.6% ; 6/35 = 17.1%
            (_PQ + "what is the probability it is a technical ticket?",
             ["14.6%", "17.1%", "24.4%", "85.4%"], "A"),
            # 10/41 = 24.4%
            (_PQ + "what is the probability it is an account ticket?",
             ["14.6%", "17.1%", "85.4%", "24.4%"], "D"),
        ])

    def test_not(self):
        self.run_cases(self.FAMILY, [
            (_PQ + "what is the probability it is not a billing ticket?",
             ["61.0%", "39.0%", "156.3%", "14.6%"], "B"),
            # (25 + 6) / 41 = 75.6%
            (_PQ + "what is the probability it is not an account ticket?",
             ["24.4%", "85.4%", "14.6%", "75.6%"], "D"),
        ])

    def test_or(self):
        self.run_cases(self.FAMILY, [
            # (25+6)/41 = 75.6%
            (_PQ + "what is the probability it is a billing or technical ticket?",
             ["75.6%", "61.0%", "14.6%", "24.4%"], "A"),
            # (25+10)/41 = 85.4%
            (_PQ + "what is the probability it is a billing ticket or an account ticket?",
             ["24.4%", "61.0%", "85.4%", "75.6%"], "C"),
            (_PQ + "what is the probability it is either a billing or a technical ticket?",
             ["24.4%", "61.0%", "85.4%", "75.6%"], "D"),
        ])

    def test_other_category_counts_and_nouns(self):
        self.run_cases(self.FAMILY, [
            # 17/57 = 29.8% ; 9/57 = 15.8% ; 31/57 = 54.4%
            ("A support queue holds 17 billing tickets, 9 technical tickets and 31 account tickets. "
             "If one is drawn at random, what is the probability it is an account ticket?",
             ["29.8%", "15.8%", "54.4%", "45.6%"], "C"),
            # 4/25 = 16.0%
            ("A bag holds 5 red marbles, 6 blue marbles, 10 green marbles and 4 yellow marbles. "
             "If one is drawn at random, what is the probability it is a yellow marble?",
             ["4.0%", "16.0%", "19.0%", "84.0%"], "B"),
            # 12/30 = 40.0% ; 12/18 = 66.7%
            ("A box contains 12 red balls and 18 blue balls. "
             "If one is chosen at random, what is the probability it is a red ball?",
             ["60.0%", "66.7%", "40.0%", "30.0%"], "C"),
        ])

    def test_verbs_and_draw_phrasing(self):
        self.run_cases(self.FAMILY, [
            ("A support queue contains 25 billing tickets, 6 technical tickets and 10 account tickets. "
             "If one is picked randomly, what is the probability it is a billing ticket?",
             ["156.3%", "39.0%", "75.6%", "61.0%"], "D"),
            ("A support queue has 25 billing tickets, 6 technical tickets and 10 account tickets. "
             "If a ticket is selected at random, what is the probability that it is a billing ticket?",
             ["61.0%", "156.3%", "75.6%", "39.0%"], "A"),
        ])

    def test_chances_and_bare_category(self):
        self.run_cases(self.FAMILY, [
            (_PQ + "what are the chances it is a billing ticket?",
             ["156.3%", "61.0%", "75.6%", "39.0%"], "B"),
            (_PQ + "what is the probability it is billing?",
             ["156.3%", "39.0%", "75.6%", "61.0%"], "D"),
        ])

    def test_options_without_percent_sign(self):
        self.run_cases(self.FAMILY, [
            (_PQ + "what is the probability it is a billing ticket?",
             ["156.3", "61.0", "75.6", "39.0"], "B"),
            (_PQ + "what is the probability it is not a billing ticket?",
             ["61.0", "156.3", "39.0", "14.6"], "C"),
        ])

    def test_zero_count_category(self):
        self.run_cases(self.FAMILY, [
            ("A support queue holds 25 billing tickets, 0 technical tickets and 10 account tickets. "
             "If one is drawn at random, what is the probability it is a technical ticket?",
             ["0.0%", "28.6%", "71.4%", "100.0%"], "A"),
            ("A support queue holds 25 billing tickets, 0 technical tickets and 10 account tickets. "
             "If one is drawn at random, what is the probability it is not a technical ticket?",
             ["0.0%", "28.6%", "71.4%", "100.0%"], "D"),
        ])


# --------------------------------------------------------------------------
# 12. total revenue
# --------------------------------------------------------------------------

class TotalRevenueTest(SolverCase):
    FAMILY = "total_revenue"

    def test_basic(self):
        self.run_cases(self.FAMILY, [
            # 301 * 22.71 = 6,835.71
            ("A retailer sells 301 units at $22.71 each. What is total revenue?",
             ["$6,835.71", "$6,812.29", "$22.71", "$6,858.42"], "A"),
            # 412 * 17.35 = 7,148.20
            ("A retailer sold 412 units at $17.35 per unit. What is the total revenue?",
             ["$7,130.85", "$7,148.20", "$7,165.55", "$17.35"], "B"),
            # 7 * 1999.99 = 13,999.93
            ("A retailer sells 7 units at $1,999.99 each. What is the total revenue?",
             ["$14,000.00", "$13,999.93", "$13,999.92", "$2,000.00"], "B"),
        ])

    def test_other_nouns_and_price_wording(self):
        self.run_cases(self.FAMILY, [
            ("A bakery sells 1,250 muffins at $8.40 each. What is the total revenue?",
             ["$10,500.00", "$1,250.00", "$8.40", "$10,508.40"], "A"),
            ("A shop sold 1,250 red shirts at $8.40 apiece. What is the total sales revenue?",
             ["$10,508.40", "$10,500.00", "$1,250.00", "$8.40"], "B"),
        ])

    def test_other_question_forms(self):
        self.run_cases(self.FAMILY, [
            ("A retailer sells 412 units at $17.35 each. How much revenue does it generate?",
             ["$7,130.85", "$7,148.20", "$7,165.55", "$17.35"], "B"),
            ("A retailer sells 412 units at $17.35 each. What is the total revenue in dollars?",
             ["$7,148.20", "$7,130.85", "$7,165.55", "$17.35"], "A"),
        ])

    def test_euro_symbol(self):
        self.assertSolves(
            "A retailer sells 301 units at €22.71 each. What is total revenue?",
            ["€6,835.71", "€6,812.29", "€22.71", "€6,858.42"], "A", self.FAMILY)


# --------------------------------------------------------------------------
# families / registry
# --------------------------------------------------------------------------

class RegistryTest(unittest.TestCase):
    def test_twelve_families(self):
        self.assertEqual(len(FAMILIES), 12)
        self.assertEqual(len(set(FAMILIES)), 12)

    def test_solution_is_a_frozen_dataclass(self):
        sol = Solution("A", "rule:x", "detail")
        with self.assertRaises(Exception):
            sol.letter = "B"  # type: ignore[misc]


# --------------------------------------------------------------------------
# regression tests: every reproduced adversarial finding
# --------------------------------------------------------------------------

class RegressionCriticalTest(SolverCase):
    """Confident wrong letters reported by the adversarial review: must now abstain."""

    def test_probability_stated_population_total_is_not_swallowed(self):
        cases = [
            ("A support queue of 100 tickets holds 25 billing tickets, 6 technical tickets and 10 account tickets. "
             "If one is drawn at random, what is the probability it is a billing ticket?",
             ["25.0%", "61.0%", "156.3%", "75.6%"]),
            ("A bag of 50 marbles contains 5 red marbles, 3 blue marbles and 2 green marbles. "
             "If one is drawn at random, what is the probability it is a red marble?",
             ["10.0%", "50.0%", "30.0%", "20.0%"]),
            ("The box with 40 balls holds 10 red balls, 8 blue balls and 12 green balls. "
             "If one is drawn at random, what is the probability it is a red ball?",
             ["33.3%", "40.0%", "66.7%", "25.0%"]),
            ("A support queue with 100 tickets holds 25 billing tickets, 6 technical tickets and 10 account tickets. "
             "If one is drawn at random, what is the probability it is a billing ticket?",
             ["25.0%", "61.0%", "156.3%", "75.6%"]),
            ("A fifty-ticket queue holds 25 billing tickets, 6 technical tickets and 10 account tickets. "
             "If one is drawn at random, what is the probability it is a billing ticket?",
             ["50.0%", "61.0%", "156.3%", "75.6%"]),
        ]
        for q, o in cases:
            with self.subTest(q=q[:60]):
                self.assertAbstains(q, o)

    def test_compound_recurring_deposits_are_not_lump_sums(self):
        cases = [
            ("A monthly deposit of $500.00 grows at 6% compounded monthly. What is its value after 5 years?",
             ["$34,885.01", "$674.43", "$30,000.00", "$34,000.00"]),
            ("An annual deposit of $1,000.00 grows at 5% compounded annually. What is its value after 10 years?",
             ["$1,628.89", "$13,206.79", "$10,000.00", "$12,577.89"]),
            ("A recurring deposit of $1,000.00 grows at 5% compounded annually. What is its value after 10 years?",
             ["$1,628.89", "$10,000.00", "$13,206.79", "$12,577.89"]),
            ("A quarterly deposit of $1,000.00 grows at 4% compounded quarterly. What is its value after 3 years?",
             ["$1,126.83", "$12,000.00", "$12,682.50", "$12,809.33"]),
            ("A weekly investment of $100.00 grows at 5% compounded weekly. What is its value after 2 years?",
             ["$10,400.00", "$110.51", "$5,400.00", "$100.00"]),
            ("A regular deposit of $100.00 grows at 5% compounded monthly. What is its value after 2 years?",
             ["$110.51", "$2,500.00", "$2,525.00", "$100.00"]),
        ]
        for q, o in cases:
            with self.subTest(q=q[:60]):
                self.assertAbstains(q, o, "no_family")

    def test_weighted_average_qualifier_tail_is_not_ignored(self):
        tails = [
            ("excluding region A", ["53.09", "49.00", "57.00", "60.91"]),
            ("for region A", ["65.00", "53.09", "57.00", "60.91"]),
            ("without region B", ["53.09", "57.00", "60.91", "65.00"]),
            ("value excluding returns", ["53.09", "57.00", "60.91", "47.50"]),
            ("if region A doubles", ["53.09", "57.00", "60.91", "55.00"]),
            ("in 2025", ["53.09", "52.09", "57.00", "60.91"]),
            ("order value for region B only", ["49.00", "53.09", "57.00", "65.00"]),
        ]
        for tail, o in tails:
            for lead in ("overall", "combined"):
                with self.subTest(tail=tail, lead=lead):
                    self.assertAbstains(f"{_WA}What is the {lead} average {tail}?", o, "no_family")

    def test_weighted_average_median_is_not_a_mean(self):
        self.assertAbstains(
            _WA + "What is the overall median order value?", ["57.00", "60.91", "53.09", "56.59"])


class RegressionMajorTest(SolverCase):
    def test_ratio_comparatives_no_longer_crash(self):
        for q, o, want in [
            (_RS2 + "What does the larger channel receive?", ["$642.86", "$128.57", "$2,250.00", "$257.14"], "A"),
            (_RS2 + "What does the smaller channel receive?", ["$642.86", "$128.57", "$2,250.00", "$257.14"], "D"),
            (_RS2 + "What is the larger share?", ["$642.86", "$128.57", "$2,250.00", "$257.14"], "A"),
            (_RS3 + "What does the largest channel receive?", ["$120.00", "$1,200.00", "$600.00", "$240.00"], "C"),
            (_RS3 + "What does the smallest channel receive?", ["$240.00", "$1,200.00", "$600.00", "$120.00"], "A"),
        ]:
            with self.subTest(q=q[-45:]):
                sol = self.assertSolves(q, o, want, "ratio_split")
                self.assertNotIn("error", sol.method)

    def test_break_even_without_rounding_phrase_abstains_when_floor_and_ceil_compete(self):
        q = "Fixed costs are $12,000.00. Each unit sells for $31.00 with a variable cost of $18.00. " \
            "How many units must be sold to break even?"
        self.assertAbstains(q, ["924 units", "923 units", "387 units", "667 units"], "ambiguous_rounding")
        self.assertAbstains(q, ["923 units", "924 units", "925 units", "667 units"], "ambiguous_rounding")

    def test_break_even_ceil_only_answer_is_not_guessed_without_a_phrase(self):
        # only 924 offered, 923 absent: could be "round up" - not provable, abstain
        q = "Fixed costs are $12,000.00. Each unit sells for $31.00 with a variable cost of $18.00. " \
            "How many units must be sold to break even?"
        self.assertAbstains(q, ["924 units", "387 units", "667 units", "1,533 units"])

    def test_break_even_with_explicit_phrase_still_answers_with_both_neighbours(self):
        q = "Fixed costs are $12,000.00. Each unit sells for $31.00 with a variable cost of $18.00. "
        self.assertSolves(q + "How many units must be sold to break even, to the nearest unit?",
                          ["924 units", "923 units", "387 units", "667 units"], "B", "break_even")
        self.assertSolves(q + "At least how many units must be sold to break even?",
                          ["924 units", "923 units", "387 units", "667 units"], "A", "break_even")


class RegressionMinorTest(SolverCase):
    def test_exact_rounding_tie_with_both_neighbours_abstains(self):
        # 1/80 = 1.25% exactly: half-up 1.3, half-even / float formatting 1.2
        q = ("A support queue holds 1 billing tickets, 76 technical tickets and 3 account tickets. "
             "If one is drawn at random, what is the probability it is a billing ticket?")
        self.assertAbstains(q, ["98.8%", "1.3%", "95.0%", "1.2%"], "tie_ambiguous")

    def test_exact_rounding_tie_with_one_neighbour_still_answers(self):
        q = ("A support queue holds 1 billing tickets, 76 technical tickets and 3 account tickets. "
             "If one is drawn at random, what is the probability it is a billing ticket?")
        self.assertSolves(q, ["98.8%", "1.3%", "95.0%", "97.5%"], "B", "probability")
        # only the half-even neighbour present -> accepted via the rounding slack
        sol = self.assertSolves(q, ["98.8%", "1.2%", "95.0%", "97.5%"], "B", "probability", tol_ok=True)
        self.assertTrue(sol.method.endswith("+tol"), sol.method)

    def test_subject_clauses_in_other_families_abstain(self):
        gm = ("generates $59,000.00 in revenue against $52,862.47 in cost of goods sold. What is the gross margin?")
        opts = ["11.6%", "10.4%", "5.5%", "89.6%"]
        for subj in (
            "A product line adjusted for inflation",
            "A product line after a 10 percent fee",
            "A product line excluding returns",
            "A product line in thousands of dollars",
            "A product line of 100 items",
            "A product line with 5 stores",
            "A net product line",
        ):
            with self.subTest(subj=subj):
                self.assertAbstains(f"{subj} {gm}", opts)
        self.assertAbstains(
            "An item of 100 items listed at $45.33 is discounted by 20%. What is the sale price?",
            ["$54.40", "$9.07", "$36.26", "$25.33"])
        self.assertAbstains(
            "A retailer with 5 stores sells 301 units at $22.71 each. What is total revenue?",
            ["$6,835.71", "$6,812.29", "$22.71", "$6,858.42"])
        self.assertAbstains(
            "A $900.00 marketing budget in thousands of dollars is split between two channels in the ratio 2:5. "
            "What does the first channel receive?",
            ["$128.57", "$642.86", "$257.14", "$450.00"])
        self.assertAbstains(
            "A $900.00 budget after fees is split between two channels in the ratio 2:5. "
            "What does the first channel receive?",
            ["$128.57", "$642.86", "$257.14", "$450.00"])

    def test_mixed_currency_symbols_abstain(self):
        self.assertAbstains(
            "A product line generates $59,000.00 in revenue against €52,862.47 in cost of goods sold. "
            "What is the gross margin?", ["11.6%", "10.4%", "5.5%", "89.6%"], "mixed_currency")
        self.assertAbstains(
            "Fixed costs are $12,000.00. Each unit sells for €31.00 with a variable cost of $18.00. "
            "How many units must be sold to break even, to the nearest unit?",
            ["924 units", "923 units", "387 units", "667 units"], "mixed_currency")
        self.assertAbstains(
            "Quarterly revenue moved from $350.00 to €297.00. What is the percentage change?",
            ["84.9%", "15.1%", "-17.8%", "-15.1%"], "mixed_units")

    def test_symbol_options_must_match_the_question_currency(self):
        # euro question, dollar-sign options -> do not guess
        self.assertAbstains(
            "An investment of €20,000.00 grows at 5% compounded annually. What is its value after 3 years?",
            ["$23,000.00", "$22,050.00", "$23,152.50", "$24,310.13"])
        self.assertAbstains(
            "An item listed at £120.00 is discounted by 15%. What is the sale price?",
            ["$102.00", "$105.00", "$18.00", "$138.00"])

    def test_percent_change_subject_cannot_hide_a_second_change_or_a_qualifier(self):
        o = ["84.9%", "15.1%", "-17.8%", "-15.1%"]
        for q in (
            "Revenue went from 100 to 200 then moved from 350 to 297. What is the percentage change?",
            "Revenue fell by half and then moved from $350.00 to $297.00. What is the percentage change?",
            "Costs fell by 5 then moved from $350.00 to $297.00. What is the percentage change?",
            "Quarterly revenue after tax moved from $350.00 to $297.00. What is the percentage change?",
            "Quarterly revenue excluding returns moved from $350.00 to $297.00. What is the percentage change?",
        ):
            with self.subTest(q=q[:50]):
                self.assertAbstains(q, o, "no_family")
        # plain subjects, with harmless prepositions or digits in a label, still work
        for q in (
            "Quarterly revenue moved from $350.00 to $297.00. What is the percentage change?",
            "Q3 revenue moved from $350.00 to $297.00. What is the percentage change?",
            "Monthly active users in Q3 fell from 350 to 297. What is the percentage change?",
        ):
            with self.subTest(q=q[:50]):
                self.assertSolves(q, o, "D", "percent_change")

    def test_bi_annually_is_ambiguous(self):
        for word in ("bi-annually", "biannually"):
            with self.subTest(word=word):
                self.assertAbstains(
                    f"An investment of $20,000.00 grows at 5% compounded {word}. What is its value after 3 years?",
                    ["$23,152.50", "$23,000.00", "$23,229.44", "$23,193.87"])


# --------------------------------------------------------------------------
# must-abstain: wording / numbers the solvers must not guess
# --------------------------------------------------------------------------

_ABSTAIN_CASES = [
    # (label, question, options)
    ("old-style fx with no direction",
     "An invoice for $2,450.05 is converted to euros at an exchange rate of 1.57. How many euros is that?",
     ["1,560.00", "3,846.58", "2,450.05", "1,560.54"]),
    ("fx asks for a different currency than the target",
     "An invoice for $2,450.05 is converted to euros at 1.5726 euros per US dollar. How many US dollars is that?",
     ["1,557.96", "3,852.95", "2,450.05", "4,052.95"]),
    ("fx rate between unrelated currencies",
     "An invoice for $2,450.05 is converted to euros at 1.2 pounds per Canadian dollar. How many euros is that?",
     ["2,940.06", "2,041.71", "2,450.05", "1,500.00"]),
    ("fx source equals target",
     "An invoice for $2,450.05 is converted to US dollars at 1.5726 euros per US dollar. How many US dollars is that?",
     ["1,557.96", "3,852.95", "2,450.05", "4,052.95"]),
    ("fx with a fee clause",
     "An invoice for $2,450.05 is converted to euros at 1.5726 euros per US dollar after a 2% fee. How many euros is that?",
     ["1,557.96", "3,852.95", "3,775.89", "4,052.95"]),
    ("numeric date that could be dd/mm or mm/dd",
     "A contract signed on 05/06/2026 has a 10-day settlement period. On what date does it settle?",
     ["15 June 2026", "16 May 2026", "15 May 2026", "16 June 2026"]),
    ("weekday that disagrees with the date",
     "A contract signed on Monday, 12 May 2026 has a 36-day settlement period. On what date does it settle?",
     ["16 June 2026", "17 June 2026", "18 June 2026", "19 June 2026"]),
    ("impossible calendar date",
     "A contract signed on 31 June 2026 has a 36-day settlement period. On what date does it settle?",
     ["16 July 2026", "17 July 2026", "18 July 2026", "19 July 2026"]),
    ("settlement with holiday clause",
     "A contract signed on 12 May 2026 has a 10-business-day settlement period excluding holidays. "
     "On what date does it settle?",
     ["22 May 2026", "26 May 2026", "25 May 2026", "27 May 2026"]),
    ("discount above 100 percent",
     "An item listed at $45.33 is discounted by 120%. What is the sale price?",
     ["$54.40", "$-9.07", "$36.26", "$25.33"]),
    ("markup sentence with a discount question",
     "An item costing $80.00 is marked up by 25%. How much is saved?",
     ["$100.00", "$20.00", "$60.00", "$25.00"]),
    ("discount sentence with a markup question",
     "An item listed at $80.00 is discounted by 25%. What is the markup amount?",
     ["$100.00", "$20.00", "$60.00", "$25.00"]),
    ("discount with tax clause",
     "An item listed at $45.33 is discounted by 20% plus 8% tax. What is the sale price?",
     ["$54.40", "$9.07", "$39.16", "$25.33"]),
    ("percent change that contradicts its verb",
     "Quarterly revenue rose from $350.00 to $297.00. What is the percentage change?",
     ["84.9%", "15.1%", "-17.8%", "-15.1%"]),
    ("percent change mixing a $ amount with a bare number",
     "Quarterly revenue moved from $350.00 to 297. What is the percentage change?",
     ["84.9%", "15.1%", "-17.8%", "-15.1%"]),
    ("percent change from zero",
     "Quarterly revenue moved from $0.00 to $297.00. What is the percentage change?",
     ["84.9%", "15.1%", "-17.8%", "-15.1%"]),
    ("sequence that is neither arithmetic nor geometric",
     "A monthly headcount series runs 2, 3, 5, 8, 13. If the pattern continues, what is the next value?",
     ["19", "20", "21", "22"]),
    ("four squares is not proof of a quadratic",
     "A monthly headcount series runs 1, 4, 9, 16. If the pattern continues, what is the next value?",
     ["23", "25", "24", "26"]),
    ("sequence mixing $ and bare terms",
     "A weekly revenue series runs $120, 135, $150, 165. If the pattern continues, what is the next value?",
     ["$175", "$180", "$195", "$170"]),
    ("sequence that asks for something else",
     "A monthly headcount series runs 13, 18, 23, 28, 33. If the pattern continues, what is the sum of the next two values?",
     ["38", "81", "43", "76"]),
    ("break-even with price below variable cost",
     "Fixed costs are $12,000.00. Each unit sells for $18.00 with a variable cost of $31.00. "
     "How many units must be sold to break even, to the nearest unit?",
     ["924 units", "923 units", "387 units", "667 units"]),
    ("break-even with an unknown extra clause",
     "Fixed costs are $12,000.00. Each unit sells for $31.00 with a variable cost of $18.00. "
     "How many units must be sold to break even after a 10% price cut, to the nearest unit?",
     ["924 units", "923 units", "387 units", "667 units"]),
    ("ratio count mismatch (three recipients, two parts)",
     "A $900.00 marketing budget is split between three channels in the ratio 2:5. What does the first channel receive?",
     ["$128.57", "$642.86", "$257.14", "$450.00"]),
    ("ratio: larger with three parts",
     "A $1,200.00 marketing budget is split between three channels in the ratio 2:3:5. What does the larger channel receive?",
     ["$120.00", "$1,200.00", "$600.00", "$240.00"]),
    ("ratio: ordinal beyond the number of parts",
     "A $900.00 marketing budget is split between two channels in the ratio 2:5. What does the third channel receive?",
     ["$128.57", "$642.86", "$257.14", "$450.00"]),
    ("ratio: contradictory selector",
     "A $900.00 marketing budget is split between two channels in the ratio 2:5. What does the largest first channel receive?",
     ["$128.57", "$642.86", "$257.14", "$450.00"]),
    ("ratio: named recipients instead of a count",
     "A $900.00 marketing budget is split between partners A and B in the ratio 2:5. What does the first partner receive?",
     ["$128.57", "$642.86", "$257.14", "$450.00"]),
    ("weighted average: groups count different things",
     "Region A had 65 orders averaging $65.00; region B had 189 tickets averaging $49.00. What is the overall average order value?",
     ["57.00", "60.91", "53.09", "65.00"]),
    ("weighted average: only one group has a $",
     "Region A had 65 orders averaging $65.00; region B had 189 orders averaging 49.00. What is the overall average order value?",
     ["57.00", "60.91", "53.09", "65.00"]),
    ("weighted average: 'both' with three groups",
     "Region A had 30 orders averaging $81.50; region B had 20 orders averaging $74.00; region C had 10 orders averaging $60.00. "
     "What is the combined average order value across both regions?",
     ["77.75", "78.50", "75.00", "71.83"]),
    ("probability: category not in the list",
     "A support queue holds 25 billing tickets, 6 technical tickets and 10 account tickets. "
     "If one is drawn at random, what is the probability it is a sales ticket?",
     ["61.0%", "14.6%", "24.4%", "0.0%"]),
    ("probability: duplicate category",
     "A support queue holds 25 billing tickets, 6 billing tickets and 10 account tickets. "
     "If one is drawn at random, what is the probability it is a billing ticket?",
     ["61.0%", "14.6%", "24.4%", "75.6%"]),
    ("probability: categories use different nouns",
     "A support queue holds 25 billing tickets, 6 technical calls and 10 account tickets. "
     "If one is drawn at random, what is the probability it is a billing ticket?",
     ["61.0%", "14.6%", "24.4%", "75.6%"]),
    ("probability: two draws",
     "A support queue holds 25 billing tickets, 6 technical tickets and 10 account tickets. "
     "If two are drawn at random, what is the probability both are billing tickets?",
     ["61.0%", "37.2%", "24.4%", "75.6%"]),
    ("probability: empty population",
     "A support queue holds 0 billing tickets and 0 technical tickets. "
     "If one is drawn at random, what is the probability it is a billing ticket?",
     ["0.0%", "50.0%", "100.0%", "25.0%"]),
    ("total revenue with an extra clause",
     "A retailer sells 301 units at $22.71 each after a 10% discount. What is total revenue?",
     ["$6,835.71", "$6,152.14", "$22.71", "$6,858.42"]),
    ("total revenue: profit instead of revenue",
     "A retailer sells 301 units at $22.71 each. What is the total profit?",
     ["$6,835.71", "$6,152.14", "$22.71", "$6,858.42"]),
    ("compound: monthly interest rate",
     "An investment of $20,000.00 grows at 5% per month compounded monthly. What is its value after 3 years?",
     ["$23,152.50", "$23,000.00", "$23,229.44", "$117,000.00"]),
    ("compound: interest rate above 1000 percent",
     "An investment of $20,000.00 grows at 5000% compounded annually. What is its value after 3 years?",
     ["$23,152.50", "$23,000.00", "$23,229.44", "$117,000.00"]),
    ("compound: zero years",
     "An investment of $20,000.00 grows at 5% compounded annually. What is its value after 0 years?",
     ["$20,000.00", "$23,000.00", "$23,229.44", "$117,000.00"]),
    ("compound: unknown compounding word",
     "An investment of $20,000.00 grows at 5% compounded fortnightly. What is its value after 3 years?",
     ["$23,152.50", "$23,000.00", "$23,229.44", "$23,193.87"]),
    ("gross margin: unknown extra cost line",
     "A product line generates $59,000.00 in revenue against $52,862.47 in cost of goods sold and $1,000.00 in overhead. "
     "What is the gross margin?",
     ["11.6%", "10.4%", "5.5%", "89.6%"]),
    ("gross margin with zero revenue",
     "A product line generates $0.00 in revenue against $52,862.47 in cost of goods sold. What is the gross margin?",
     ["11.6%", "10.4%", "5.5%", "89.6%"]),
    ("unknown family: median",
     "A monthly headcount series runs 13, 18, 23, 28, 33. What is the median value?",
     ["23", "28", "18", "33"]),
    ("unknown family: loan payment",
     "A loan of $10,000.00 is repaid over 5 years at 6% interest. What is the monthly payment?",
     ["$193.33", "$200.00", "$166.67", "$188.71"]),
]


class MustAbstainTest(SolverCase):
    def test_there_are_at_least_twenty_abstain_cases(self):
        self.assertGreaterEqual(len(_ABSTAIN_CASES), 20)

    def test_every_abstain_case(self):
        for label, question, options in _ABSTAIN_CASES:
            with self.subTest(label):
                sol = self.assertAbstains(question, options)
                self.assertIsNone(sol.letter)

    def test_options_that_do_not_contain_the_answer_abstain(self):
        q = "An investment of $20,000.00 grows at 5% compounded annually. What is its value after 3 years?"
        # correct value 23,152.50 is missing from the options
        self.assertAbstains(q, ["$23,000.00", "$22,050.00", "$24,310.13", "$25,000.00"], "no_option_match")
        # options of the wrong type (dates / percentages / words)
        self.assertAbstains(q, ["12 May 2026", "13 May 2026", "14 May 2026", "15 May 2026"], "no_option_match")
        self.assertAbstains(q, ["23.2%", "22.0%", "24.3%", "23.0%"], "no_option_match")
        self.assertAbstains(q, ["about twenty-three thousand", "less", "more", "unchanged"], "no_option_match")

    def test_two_options_with_the_same_value_are_ambiguous(self):
        q = "An investment of $20,000.00 grows at 5% compounded annually. What is its value after 3 years?"
        self.assertAbstains(q, ["$23,152.50", "$23,152.50", "$22,050.00", "$24,310.13"], "ambiguous")
        self.assertAbstains(q, ["$23,152.50", "USD 23,152.50", "$22,050.00", "$24,310.13"], "ambiguous")

    def test_settlement_options_in_a_wrong_year_do_not_match(self):
        q = "A contract signed on 12 May 2026 has a 36-day settlement period. On what date does it settle?"
        self.assertAbstains(q, ["17 June 2025", "17 June 2027", "17 July 2026", "18 June 2026"], "no_option_match")

    def test_an_unreadable_option_does_not_stop_a_clear_match(self):
        q = "An investment of $20,000.00 grows at 5% compounded annually. What is its value after 3 years?"
        self.assertSolves(q, ["$23,152.50", "n/a", "$22,050.00", "$24,310.13"], "A", "compound_interest")


# --------------------------------------------------------------------------
# robustness: never raises, bounded time
# --------------------------------------------------------------------------

class RobustnessTest(unittest.TestCase):
    Q = "An investment of $20,000.00 grows at 5% compounded annually. What is its value after 3 years?"
    OK_OPTS = {"A": "$23,000.00", "B": "$22,050.00", "C": "$23,152.50", "D": "$24,310.13"}

    def test_reference_question_works(self):
        self.assertEqual(solve_question(self.Q, self.OK_OPTS).letter, "C")

    def test_bad_inputs_abstain_and_never_raise(self):
        bad = [
            (None, self.OK_OPTS),
            (123, self.OK_OPTS),
            (self.Q, None),
            (self.Q, "A B C D"),
            (self.Q, {}),
            (self.Q, {"A": "$23,152.50"}),
            (self.Q, {"A": "x", "B": None, "C": 3, "D": ["list"]}),
            (self.Q, {"E": "$23,152.50", "F": "$23,000.00"}),
            ("", self.OK_OPTS),
            ("   \n\t  ", self.OK_OPTS),
            ("\x00\x01\x02", self.OK_OPTS),
            ("‮" * 50, self.OK_OPTS),
            (self.Q * 500, self.OK_OPTS),
            ("A " * 5000, self.OK_OPTS),
            (self.Q, {k: "9" * 5000 for k in "ABCD"}),
            (self.Q, {"A": "\ud800", "B": "nan", "C": "inf", "D": "-inf"}),
            (self.Q, {"A": "$1e400", "B": "$" + "9" * 400, "C": "0.0000", "D": "-0"}),
        ]
        for q, o in bad:
            with self.subTest(q=str(q)[:30], o=str(o)[:30]):
                sol = solve_question(q, o)  # type: ignore[arg-type]
                self.assertIsInstance(sol, Solution)
                if sol.letter is None:
                    self.assertTrue(sol.method.startswith("abstain:"), sol.method)
                else:
                    self.assertIn(sol.letter, "ABCD")

    def test_lowercase_option_keys_are_accepted(self):
        sol = solve_question(self.Q, {k.lower(): v for k, v in self.OK_OPTS.items()})
        self.assertEqual(sol.letter, "C")

    def test_unicode_variants_are_normalised(self):
        q = ("An investment of $20,000.00 grows at 5 % compounded annually. "
             "What is its value after 3 years?")
        self.assertEqual(solve_question(q, self.OK_OPTS).letter, "C")
        # unicode minus in options
        q2 = "Quarterly revenue moved from $350.00 to $297.00. What is the percentage change?"
        o2 = {"A": "84.9%", "B": "15.1%", "C": "−17.8%", "D": "−15.1%"}
        self.assertEqual(solve_question(q2, o2).letter, "D")

    def test_case_and_spacing_do_not_matter(self):
        q = "AN INVESTMENT OF $20,000.00   GROWS AT 5% COMPOUNDED ANNUALLY.   WHAT IS ITS VALUE AFTER 3 YEARS?"
        self.assertEqual(solve_question(q, self.OK_OPTS).letter, "C")

    def test_deterministic(self):
        first = solve_question(self.Q, self.OK_OPTS)
        for _ in range(5):
            self.assertEqual(solve_question(self.Q, self.OK_OPTS), first)

    def test_near_miss_inputs_are_fast(self):
        """No catastrophic regex backtracking on long near-miss text."""
        words = " ".join(["word"] * 200)
        samples = [
            f"A {words} generates $5.00 in revenue against $4.00 in cost of goods sold. What is the gross margin?",
            f"A monthly headcount series runs {', '.join(['1'] * 28)}, x. If the pattern continues, what is the next value?",
            "Region A had " + "5 orders averaging $5.00; region B had " * 40 + "What is the overall average order value?",
            "A support queue holds " + ", ".join(f"{i} cat{i} tickets" for i in range(100)) + ".",
            "An investment of $20,000.00 grows at " + "5% " * 300 + "compounded annually.",
            "A $900.00 " + "budget " * 250 + "is split between two channels in the ratio 2:5.",
            "Quarterly revenue " + "moved " * 250 + "from $350.00 to $297.00. What is the percentage change?",
            "A contract signed on " + "12 May 2026 " * 100 + "has a 36-day settlement period.",
        ]
        start = time.perf_counter()
        for text in samples:
            sol = solve_question(text, self.OK_OPTS)
            self.assertIsNone(sol.letter)
        self.assertLess(time.perf_counter() - start, 5.0, "solver is too slow on near-miss input")

    def test_exact_wording_numbers_in_extreme_ranges(self):
        # tiny and huge magnitudes must not raise or misbehave
        q = "A retailer sells 123,456,789 units at $98,765.43 each. What is total revenue?"
        total = "12,193,262,852,004.27"  # 123456789 * 98765.43, exact
        sol = solve_question(q, {"A": f"${total}", "B": "$12,000,000,000.00", "C": "$98,765.43", "D": "$1.00"})
        self.assertEqual(sol.letter, "A", sol.detail)


if __name__ == "__main__":
    unittest.main()
