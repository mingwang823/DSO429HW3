"""Unit tests for ``quizbot/matching.py`` (computed value -> option letter) and the
option / date parsing helpers in ``quizbot/parsing.py`` that feed it.

The matcher's contract: pick the ONE option equal to the computed value at the precision
that option displays (round half up), compare only values of the same type, and abstain
(``letter is None``) whenever picking would be a guess.

Python 3.9 compatible (no ``match``, no ``X | Y`` at runtime).
"""
from __future__ import annotations

import datetime as dt
import re
import unittest
from decimal import Decimal
from fractions import Fraction

from quizbot.matching import Computed, MatchResult, format_value, match_options, to_decimal
from quizbot.parsing import (
    SUBJ,
    count_decimals,
    money_g,
    normalize_text,
    parse_date,
    parse_date_ex,
    parse_option,
    pct_g,
    phrase,
)


def opts(*values):
    return dict(zip("ABCD", values))


def money(value, currency="USD", note=""):
    return Computed(value, "money", currency, note)


def percent(value):
    return Computed(value, "percent")


def number(value):
    return Computed(value, "number")


def units(value, rivals=()):
    return Computed(value, "units", None, "", tuple(rivals))


def when(y, m, d):
    return Computed(dt.date(y, m, d), "date")


class MatchCase(unittest.TestCase):
    def assertPicks(self, computed, options, letter, status="exact"):
        res = match_options(computed, options)
        self.assertEqual(
            (res.letter, res.status), (letter, status),
            f"{computed} vs {options} -> {res}",
        )
        return res

    def assertNoPick(self, computed, options, status=None):
        res = match_options(computed, options)
        self.assertIsNone(res.letter, f"{computed} vs {options} -> {res}")
        if status is not None:
            self.assertEqual(res.status, status, str(res))
        return res


# --------------------------------------------------------------------------
# rounding boundaries
# --------------------------------------------------------------------------

class RoundingTest(MatchCase):
    def test_rounds_to_the_precision_the_option_displays(self):
        # 10.4026 -> 10.4 at one decimal, 10.40 at two, 10 at none, 10.403 at three
        self.assertPicks(percent(Decimal("10.4026")), opts("11.6%", "10.4%", "5.5%", "89.6%"), "B")
        self.assertPicks(percent(Decimal("10.4026")), opts("11.60%", "10.40%", "5.50%", "89.60%"), "B")
        self.assertPicks(percent(Decimal("10.4026")), opts("12%", "10%", "5%", "90%"), "B")
        self.assertPicks(percent(Decimal("10.4026")), opts("10.402%", "10.403%", "10.404%", "10.405%"), "B")

    def test_half_up_not_bankers_rounding_at_zero_decimals(self):
        # 0.5 -> 1, 1.5 -> 2, 2.5 -> 3 (banker's would give 0, 2, 2)
        for value, want in ((Decimal("0.5"), "1"), (Decimal("1.5"), "2"), (Decimal("2.5"), "3")):
            with self.subTest(value=value):
                self.assertPicks(number(value), opts(want, "100", "200", "300"), "A")

    def test_half_up_at_two_decimals(self):
        self.assertPicks(money(Decimal("1.005")), opts("$1.01", "$2.00", "$3.00", "$4.00"), "A")
        self.assertPicks(money(Decimal("2.675")), opts("$2.00", "$2.68", "$3.00", "$4.00"), "B")
        self.assertPicks(money(Decimal("36.264")), opts("$54.40", "$9.07", "$36.26", "$25.33"), "C")
        self.assertPicks(money(Decimal("36.266")), opts("$36.26", "$9.07", "$36.27", "$25.33"), "C")

    def test_negative_half_rounds_away_from_zero(self):
        # Decimal ROUND_HALF_UP: -2.5 -> -3, -1.005 -> -1.01
        self.assertPicks(number(Decimal("-2.5")), opts("-3", "9", "8", "7"), "A")
        self.assertPicks(money(Decimal("-1.005")), opts("$-1.01", "$5.00", "$6.00", "$7.00"), "A")

    def test_just_below_and_just_above_a_tie(self):
        self.assertPicks(money(Decimal("1.00499")), opts("$1.00", "$1.01", "$3.00", "$4.00"), "A")
        self.assertPicks(money(Decimal("1.00501")), opts("$1.00", "$1.01", "$3.00", "$4.00"), "B")

    def test_float_noise_is_cleaned(self):
        # 2.675 as a float is 2.67499999...; the matcher treats it as the decimal 2.675
        self.assertPicks(money(2.675), opts("$2.68", "$5.00", "$6.00", "$7.00"), "A")
        self.assertPicks(money(0.1 + 0.2), opts("$0.30", "$5.00", "$6.00", "$7.00"), "A")
        self.assertPicks(percent(1.1 * 3), opts("$3.30", "3.3%", "$6.00", "$7.00"), "B")

    def test_exact_fractions_and_ints(self):
        self.assertPicks(percent(Fraction(100, 3)), opts("33.3%", "66.7%", "50.0%", "25.0%"), "A")
        self.assertPicks(percent(Fraction(200, 3)), opts("33.3%", "66.7%", "50.0%", "25.0%"), "B")
        self.assertPicks(number(7), opts("6", "7", "8", "9"), "B")
        self.assertPicks(money(Fraction(1, 8)), opts("$0.13", "$0.30", "$0.20", "$0.10"), "A")  # 0.125 tie

    def test_very_large_and_very_small_values(self):
        self.assertPicks(money(Decimal("12193262852004.27")),
                         opts("$12,193,262,852,004.27", "$12,000,000,000,000.00", "$1.00", "$2.00"), "A")
        self.assertPicks(percent(Decimal("0.0004")), opts("0.000%", "0.001%", "0.01%", "0.1%"), "A")

    def test_no_value_near_any_option_abstains(self):
        self.assertNoPick(money(Decimal("23152.50")), opts("$23,000.00", "$22,050.00", "$24,310.13", "$25,000.00"),
                          "no_option_match")
        # 0.51 away at 0 decimals is more than half a unit
        self.assertNoPick(number(Decimal("2.51")), opts("2", "9", "8", "7"), "no_option_match")

    def test_near_miss_of_one_unit_in_the_last_place_is_not_accepted(self):
        # computed 36.264 vs option 36.27: off by 0.006 > half unit(0.005)*1.001
        self.assertNoPick(money(Decimal("36.264")), opts("$36.27", "$9.07", "$54.40", "$25.33"), "no_option_match")


# --------------------------------------------------------------------------
# exact rounding ties
# --------------------------------------------------------------------------

class TieTest(MatchCase):
    def test_tie_with_only_the_half_up_neighbour_is_answered(self):
        self.assertPicks(percent(Decimal("1.25")), opts("98.8%", "1.3%", "95.0%", "97.5%"), "B")

    def test_tie_with_only_the_half_down_neighbour_uses_the_slack(self):
        res = self.assertPicks(percent(Decimal("1.25")), opts("98.8%", "1.2%", "95.0%", "97.5%"), "B",
                               status="tolerance")
        self.assertIn("slack", res.detail)

    def test_tie_with_both_neighbours_abstains(self):
        res = self.assertNoPick(percent(Decimal("1.25")), opts("98.8%", "1.3%", "95.0%", "1.2%"), "tie_ambiguous")
        self.assertIn("tie", res.detail)
        self.assertNoPick(money(Decimal("0.015")), opts("$0.02", "$0.01", "$0.05", "$0.10"), "tie_ambiguous")
        self.assertNoPick(number(Decimal("2.5")), opts("2", "3", "7", "9"), "tie_ambiguous")
        self.assertNoPick(number(Decimal("-2.5")), opts("-2", "-3", "7", "9"), "tie_ambiguous")

    def test_tie_neighbours_of_a_different_precision_do_not_count(self):
        # "1.20%" is a different display precision from "1.3%"; only same-precision neighbours conflict
        self.assertPicks(percent(Decimal("1.25")), opts("98.8%", "1.3%", "95.0%", "1.24%"), "B")

    def test_not_a_tie_is_unaffected(self):
        self.assertPicks(percent(Decimal("1.2501")), opts("98.8%", "1.3%", "95.0%", "1.2%"), "B")
        self.assertPicks(percent(Decimal("1.2499")), opts("98.8%", "1.3%", "95.0%", "1.2%"), "D")

    def test_exact_fraction_tie(self):
        # 1/80 = 1.25 exactly
        self.assertNoPick(percent(Fraction(100, 80)), opts("1.3%", "1.2%", "2.0%", "3.0%"), "tie_ambiguous")
        # 1/16 = 6.25
        self.assertNoPick(percent(Fraction(100, 16)), opts("6.3%", "6.2%", "2.0%", "3.0%"), "tie_ambiguous")


# --------------------------------------------------------------------------
# rival interpretations (break-even without a rounding phrase)
# --------------------------------------------------------------------------

class RivalTest(MatchCase):
    def test_rival_equal_to_another_option_abstains(self):
        v = Decimal("923.0769230769")
        res = self.assertNoPick(units(v, rivals=(923, 924)), opts("924 units", "923 units", "387 units", "667 units"),
                                "ambiguous_rounding")
        self.assertIn("alternative", res.detail)

    def test_rival_equal_to_the_chosen_option_is_fine(self):
        v = Decimal("923.6")  # half-up 924, floor 923, ceil 924
        self.assertPicks(units(v, rivals=(923, 924)), opts("924 units", "387 units", "667 units", "1,533 units"), "A")

    def test_rival_absent_from_options_is_fine(self):
        v = Decimal("923.0769230769")
        self.assertPicks(units(v, rivals=(923, 924)), opts("923 units", "387 units", "667 units", "1,533 units"), "A")

    def test_no_rivals_means_no_extra_check(self):
        v = Decimal("923.0769230769")
        self.assertPicks(units(v), opts("924 units", "923 units", "387 units", "667 units"), "B")

    def test_rival_check_also_guards_the_slack_path(self):
        # 2.5 exactly: only "2" (via slack) matches; rival 3 is also offered as a different option
        self.assertNoPick(units(Decimal("2.5"), rivals=(2, 3)), opts("2 units", "3 units", "9 units", "8 units"))

    def test_bad_rival_values_are_ignored(self):
        self.assertPicks(units(Decimal("5"), rivals=("not a number", None)), opts("5 units", "6 units", "7 units", "8 units"), "A")


# --------------------------------------------------------------------------
# sign handling
# --------------------------------------------------------------------------

class SignTest(MatchCase):
    def test_negative_percent_is_not_its_absolute_value(self):
        self.assertPicks(percent(Decimal("-15.1428")), opts("84.9%", "15.1%", "-17.8%", "-15.1%"), "D")
        self.assertPicks(percent(Decimal("15.1428")), opts("84.9%", "15.1%", "-17.8%", "-15.1%"), "B")
        self.assertNoPick(percent(Decimal("-15.1428")), opts("84.9%", "15.1%", "17.8%", "85.1%"), "no_option_match")

    def test_unicode_minus_and_parentheses(self):
        self.assertPicks(percent(Decimal("-17.8")), opts("17.8%", "−17.8%", "84.9%", "15.1%"), "B")
        self.assertPicks(percent(Decimal("-17.8")), opts("17.8%", "(17.8%)", "84.9%", "15.1%"), "B")
        self.assertPicks(money(Decimal("-9.6")), opts("$9.60", "($9.60)", "$96.00", "$0.96"), "B")

    def test_money_sign_positions(self):
        for neg in ("$-9.60", "-$9.60", "($9.60)", "$ -9.60", "- $9.60"):
            with self.subTest(option=neg):
                self.assertPicks(money(Decimal("-9.6")), opts("$9.60", neg, "$96.00", "$0.96"), "B")

    def test_plus_sign(self):
        self.assertPicks(percent(Decimal("25")), opts("+25.0%", "-25.0%", "75.0%", "125.0%"), "A")

    def test_zero_and_negative_zero_are_one_value(self):
        self.assertNoPick(percent(Decimal("-0.04")), opts("0.0%", "-0.0%", "1.0%", "-1.0%"), "ambiguous")
        self.assertPicks(percent(Decimal("0")), opts("0.0%", "1.0%", "-1.0%", "2.0%"), "A")


# --------------------------------------------------------------------------
# percent vs number
# --------------------------------------------------------------------------

class PercentVsNumberTest(MatchCase):
    def test_percent_value_prefers_percent_options(self):
        self.assertPicks(percent(Decimal("61.0")), opts("61.0", "61.0%", "75.6", "39.0"), "B")

    def test_percent_value_falls_back_to_all_bare_numbers(self):
        self.assertPicks(percent(Decimal("-15.1428")), opts("84.9", "15.1", "-17.8", "-15.1"), "D")

    def test_percent_value_does_not_fall_back_for_a_mixed_option_set(self):
        # "x%" options exist (none match) and a bare number does: do NOT use the bare one
        self.assertNoPick(percent(Decimal("61.0")), opts("60.0%", "62.0%", "63.0%", "61.0"), "no_option_match")

    def test_percent_fallback_requires_every_option_to_be_a_bare_number(self):
        self.assertNoPick(percent(Decimal("61.0")), opts("61.0", "75.6", "39.0", "n/a"), "no_option_match")
        self.assertNoPick(percent(Decimal("61.0")), opts("61.0", "75.6", "39.0", "$61.00"), "no_option_match")

    def test_number_value_never_matches_percent_options(self):
        self.assertNoPick(number(Decimal("61.0")), opts("61.0%", "75.6%", "39.0%", "14.6%"), "no_option_match")

    def test_percent_value_is_in_percent_units_not_fractions(self):
        # 0.61 as a fraction must NOT be mistaken for 0.61 percent or 61 percent
        self.assertNoPick(percent(Decimal("61.0")), opts("0.61", "0.756", "0.39", "0.146"), "no_option_match")
        self.assertPicks(percent(Decimal("0.61")), opts("0.61%", "0.756%", "0.39%", "0.146%"), "A")

    def test_percent_word_variants(self):
        self.assertPicks(percent(Decimal("61.0")), opts("61.0 percent", "75.6 percent", "39.0 percent", "14.6 percent"), "A")
        self.assertPicks(percent(Decimal("61.0")), opts("75.6 %", "61.0 %", "39.0 %", "14.6 %"), "B")


# --------------------------------------------------------------------------
# money: symbols, codes, currency compatibility
# --------------------------------------------------------------------------

class MoneyTest(MatchCase):
    def test_with_and_without_symbols_and_separators(self):
        v = money(Decimal("3852.9486"))
        self.assertPicks(v, opts("$3,852.95", "$1,557.96", "$2,450.05", "$4,052.95"), "A")
        self.assertPicks(v, opts("3,852.95", "1,557.96", "2,450.05", "4,052.95"), "A")
        self.assertPicks(v, opts("3852.95", "1557.96", "2450.05", "4052.95"), "A")

    def test_currency_symbols_and_codes_for_the_same_amount(self):
        v = money(Decimal("3852.9486"), "EUR")
        self.assertPicks(v, opts("€1,557.96", "€3,852.95", "€2,450.05", "€4,052.95"), "B")
        self.assertPicks(v, opts("EUR 1,557.96", "EUR 3,852.95", "EUR 2,450.05", "EUR 4,052.95"), "B")
        self.assertPicks(v, opts("1,557.96 EUR", "3,852.95 EUR", "2,450.05 EUR", "4,052.95 EUR"), "B")
        self.assertPicks(v, opts("1,557.96 euros", "3,852.95 euros", "2,450.05 euros", "4,052.95 euros"), "B")
        self.assertPicks(v, opts("3,852.95 euro", "1,557.96 euro", "2,450.05 euro", "4,052.95 euro"), "A")

    def test_dollar_sign_is_compatible_with_dollar_currencies_only(self):
        usd = money(Decimal("100"), "USD")
        cad = money(Decimal("100"), "CAD")
        eur = money(Decimal("100"), "EUR")
        self.assertPicks(usd, opts("$100.00", "$101.00", "$102.00", "$103.00"), "A")
        self.assertPicks(cad, opts("$101.00", "$100.00", "$102.00", "$103.00"), "B")
        self.assertNoPick(eur, opts("$100.00", "$101.00", "$102.00", "$103.00"), "no_option_match")

    def test_wrong_currency_options_do_not_match(self):
        usd = money(Decimal("100"), "USD")
        self.assertNoPick(usd, opts("€100.00", "€101.00", "€102.00", "€103.00"), "no_option_match")
        self.assertNoPick(usd, opts("EUR 100.00", "EUR 101.00", "EUR 102.00", "EUR 103.00"), "no_option_match")
        self.assertNoPick(usd, opts("£100.00", "£101.00", "£102.00", "£103.00"), "no_option_match")
        self.assertNoPick(usd, opts("100.00 yen", "101.00 yen", "102.00 yen", "103.00 yen"), "no_option_match")

    def test_mixed_currency_options_only_the_compatible_one_counts(self):
        usd = money(Decimal("100"), "USD")
        self.assertPicks(usd, opts("€100.00", "$100.00", "£100.00", "100.00 yen"), "B")

    def test_unknown_currency_on_the_computed_value_accepts_any_symbol(self):
        v = Computed(Decimal("100"), "money", None)
        self.assertPicks(v, opts("€100.00", "$101.00", "£102.00", "103.00"), "A")

    def test_money_value_falls_back_to_bare_numbers_only_if_all_options_are_bare(self):
        self.assertPicks(money(Decimal("53.0945")), opts("57.00", "60.91", "53.09", "65.00"), "C")
        # one option has a $ so the money kind exists -> the bare ones are ignored
        self.assertNoPick(money(Decimal("53.0945")), opts("$57.00", "$60.91", "53.09", "$65.00"), "no_option_match")

    def test_option_decimals_may_differ(self):
        # one option shows no decimals: rounding is per option
        self.assertPicks(money(Decimal("1625.0")), opts("$875", "$1,625.00", "$2,465.00", "$3,375.00"), "B")
        self.assertPicks(money(Decimal("1625.0")), opts("$875", "$1,625", "$2,465", "$3,375"), "B")

    def test_duplicate_equal_options_are_ambiguous(self):
        self.assertNoPick(money(Decimal("1625.0")), opts("$1,625.00", "$1,625.00", "$2,465.00", "$3,375.00"), "ambiguous")
        self.assertNoPick(money(Decimal("1625.0")), opts("$1,625.00", "USD 1,625.00", "$2,465.00", "$3,375.00"), "ambiguous")
        # same value shown at two precisions is also a collision
        self.assertNoPick(money(Decimal("1625.0")), opts("$1,625", "$1,625.00", "$2,465.00", "$3,375.00"), "ambiguous")


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

class UnitsTest(MatchCase):
    def test_units_options(self):
        self.assertPicks(units(Decimal("923.0769")), opts("924 units", "923 units", "387 units", "667 units"), "B")
        self.assertPicks(units(Decimal("1532.6")), opts("1,532 units", "1,533 units", "1,534 units", "667 units"), "B")
        self.assertPicks(units(Decimal("4000")), opts("2,400 units", "6,000 units", "4,000 units", "2,000 units"), "C")

    def test_units_value_falls_back_to_bare_numbers(self):
        self.assertPicks(units(Decimal("923.0769")), opts("924", "923", "387", "667"), "B")

    def test_units_value_ignores_mixed_sets(self):
        self.assertNoPick(units(Decimal("923.0769")), opts("924 units", "923", "387 units", "667 units"), "no_option_match")

    def test_other_unit_words(self):
        self.assertPicks(units(Decimal("923")), opts("924 widgets", "923 widgets", "387 widgets", "667 widgets"), "B")
        self.assertPicks(units(Decimal("923")), opts("924 units sold", "923 units sold", "387 units sold", "667 units sold"), "B")

    def test_month_names_are_not_units(self):
        self.assertEqual(parse_option("12 May").kind, "text")
        self.assertEqual(parse_option("5 days").kind, "units")


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------

class DateMatchTest(MatchCase):
    def test_every_supported_format_matches(self):
        target = when(2026, 6, 17)  # a Wednesday
        formats = [
            "17 June 2026", "17 Jun 2026", "17th June 2026", "June 17, 2026", "June 17 2026", "Jun 17, 2026",
            "2026-06-17", "2026/06/17", "2026/6/17", "17/06/2026", "06/17/2026", "17.06.2026",
            "Wednesday, 17 June 2026", "Wed 17 June 2026", "Wed, June 17, 2026", "17 JUNE 2026", "17 june 2026",
        ]
        for fmt in formats:
            with self.subTest(fmt=fmt):
                self.assertPicks(target, opts("1 January 2026", fmt, "2 February 2026", "3 March 2026"), "B")

    def test_mixed_formats_in_one_option_set(self):
        self.assertPicks(when(2026, 6, 17), opts("2026-06-16", "June 17, 2026", "2026-06-18", "19 June 2026"), "B")

    def test_one_day_off_is_not_a_match(self):
        self.assertNoPick(when(2026, 6, 17), opts("16 June 2026", "18 June 2026", "17 July 2026", "17 June 2025"),
                          "no_option_match")

    def test_ambiguous_numeric_option_never_matches(self):
        # 05/06/2026 could be 5 June or 6 May; it must not be matched either way
        self.assertNoPick(when(2026, 6, 5), opts("05/06/2026", "06/05/2026", "04/06/2026", "07/06/2026"),
                          "no_option_match")
        self.assertPicks(when(2026, 6, 5), opts("05/06/2026", "5 June 2026", "04/06/2026", "07/06/2026"), "B")

    def test_equal_dates_in_two_formats_are_ambiguous(self):
        self.assertNoPick(when(2026, 6, 17), opts("17 June 2026", "2026-06-17", "18 June 2026", "19 June 2026"),
                          "ambiguous")

    def test_wrong_weekday_makes_an_option_unusable(self):
        # 17 June 2026 is a Wednesday
        self.assertNoPick(when(2026, 6, 17), opts("Monday, 17 June 2026", "Tuesday 17 June 2026",
                                                  "Thursday, 17 June 2026", "18 June 2026"), "no_option_match")

    def test_non_date_options_abstain(self):
        self.assertNoPick(when(2026, 6, 17), opts("June 2026", "17 June", "mid June", "$17.06"), "no_option_match")
        self.assertNoPick(when(2026, 6, 17), opts("31 June 2026", "29 February 2026", "0 June 2026", "2026-13-01"),
                          "no_option_match")

    def test_leap_day(self):
        self.assertPicks(when(2028, 2, 29), opts("28 February 2028", "29 February 2028", "1 March 2028", "2 March 2028"), "B")
        self.assertNoPick(when(2028, 2, 29), opts("28 February 2026", "29 February 2026", "1 March 2028", "2 March 2028"),
                          "no_option_match")

    def test_date_value_is_not_matched_to_numbers(self):
        self.assertNoPick(when(2026, 6, 17), opts("17", "2026", "6/17", "17.06"), "no_option_match")

    def test_computed_date_must_be_a_date(self):
        res = match_options(Computed("2026-06-17", "date"), opts("17 June 2026", "x", "y", "z"))
        self.assertIsNone(res.letter)
        self.assertEqual(res.status, "bad_computed")


# --------------------------------------------------------------------------
# abstain paths
# --------------------------------------------------------------------------

class AbstainTest(MatchCase):
    def test_no_options(self):
        self.assertNoPick(money(Decimal("1")), {}, "no_options")

    def test_all_text_options(self):
        self.assertNoPick(money(Decimal("1")), opts("one", "two", "three", "four"), "no_option_match")
        self.assertNoPick(percent(Decimal("1")), opts("about one", "n/a", "", "   "), "no_option_match")

    def test_bad_computed_values(self):
        for bad in (float("nan"), float("inf"), float("-inf"), "12.5", None, [1], True, object()):
            with self.subTest(value=bad):
                res = match_options(Computed(bad, "money", "USD"), opts("$1.00", "$2.00", "$3.00", "$4.00"))
                self.assertIsNone(res.letter)
                self.assertEqual(res.status, "bad_computed")

    def test_unknown_kind_abstains(self):
        res = match_options(Computed(Decimal("1"), "weird"), opts("1", "2", "3", "4"))
        self.assertIsNone(res.letter)

    def test_two_near_options_when_none_is_exact_is_ambiguous(self):
        # 2.5 at 0 decimals with options 2 and 3 (and no way to know the convention)
        self.assertNoPick(number(Decimal("2.5")), opts("2", "3", "8", "9"))

    def test_result_type(self):
        res = match_options(money(Decimal("1.00")), opts("$1.00", "$2.00", "$3.00", "$4.00"))
        self.assertIsInstance(res, MatchResult)
        self.assertEqual(res.letter, "A")

    def test_detail_mentions_the_computed_value(self):
        res = match_options(money(Decimal("36.264")), opts("$54.40", "$9.07", "$36.26", "$25.33"))
        self.assertIn("36.264", res.detail)
        res = match_options(money(Decimal("36.264")), opts("$1.00", "$2.00", "$3.00", "$4.00"))
        self.assertIn("36.264", res.detail)

    def test_extra_option_letters_are_matched_in_sorted_order(self):
        # options can be any dict; the letter returned is the dict key
        self.assertPicks(number(Decimal("3")), {"D": "3", "A": "1", "C": "2", "B": "9"}, "D")


# --------------------------------------------------------------------------
# helpers: to_decimal / format_value
# --------------------------------------------------------------------------

class HelperTest(unittest.TestCase):
    def test_to_decimal_types(self):
        self.assertEqual(to_decimal(5), Decimal("5.0000000000"))
        self.assertEqual(to_decimal(Decimal("1.5")), Decimal("1.5000000000"))
        self.assertEqual(to_decimal(Fraction(1, 3)), Decimal("0.3333333333"))
        self.assertEqual(to_decimal(2.675), Decimal("2.6750000000"))
        self.assertEqual(to_decimal(Fraction(2, 3)), Decimal("0.6666666667"))

    def test_to_decimal_rejects_junk(self):
        for bad in (float("nan"), float("inf"), "1", None, True, [1], Decimal("NaN"), Decimal("Infinity")):
            with self.subTest(value=bad):
                with self.assertRaises((TypeError, ValueError)):
                    to_decimal(bad)

    def test_format_value(self):
        self.assertEqual(format_value(percent(Decimal("10.4026"))), "10.4026%")
        self.assertEqual(format_value(money(Decimal("36.264"), "USD")), "36.264 USD")
        self.assertEqual(format_value(number(Decimal("38.000"))), "38")
        self.assertEqual(format_value(units(Decimal("923.0769230769"))), "923.0769230769")
        self.assertEqual(format_value(when(2026, 6, 17)), "2026-06-17")
        self.assertEqual(format_value(Computed(object(), "money")).startswith("<object"), True)  # never raises


# --------------------------------------------------------------------------
# parsing: option values
# --------------------------------------------------------------------------

class ParseOptionTest(unittest.TestCase):
    CASES = [
        # raw, kind, value, decimals, currency
        ("$1,234.50", "money", "1234.50", 2, "$"),
        ("$-9.60", "money", "-9.60", 2, "$"),
        ("-$9.60", "money", "-9.60", 2, "$"),
        ("($9.60)", "money", "-9.60", 2, "$"),
        ("€3,852.95", "money", "3852.95", 2, "EUR"),
        ("£1,800", "money", "1800", 0, "GBP"),
        ("¥1,000", "money", "1000", 0, "JPY"),
        ("USD 2,663.94", "money", "2663.94", 2, "USD"),
        ("EUR 2,245.23", "money", "2245.23", 2, "EUR"),
        ("eur 5", "money", "5", 0, "EUR"),
        ("2,663.94 USD", "money", "2663.94", 2, "USD"),
        ("3,852.95 euros", "money", "3852.95", 2, "EUR"),
        ("1,000 yen", "money", "1000", 0, "JPY"),
        ("12.5 US dollars", "money", "12.5", 1, "USD"),
        ("-15.1%", "percent", "-15.1", 1, None),
        ("15.1 %", "percent", "15.1", 1, None),
        ("15.1 percent", "percent", "15.1", 1, None),
        ("(15.1%)", "percent", "-15.1", 1, None),
        ("−15.1%", "percent", "-15.1", 1, None),
        (".5%", "percent", "0.5", 1, None),
        ("1,533 units", "units", "1533", 0, None),
        ("923.08 units", "units", "923.08", 2, None),
        ("5 days", "units", "5", 0, None),
        ("53.09", "number", "53.09", 2, None),
        ("-17.8", "number", "-17.8", 1, None),
        ("1,000", "number", "1000", 0, None),
        ("  38  ", "number", "38", 0, None),
        ('"38"', "number", "38", 0, None),
        ("+7", "number", "7", 0, None),
    ]

    def test_table(self):
        for raw, kind, value, decimals, cur in self.CASES:
            with self.subTest(raw=raw):
                o = parse_option(raw)
                self.assertEqual(o.kind, kind, o)
                self.assertEqual(o.value, Decimal(value), o)
                self.assertEqual(o.decimals, decimals, o)
                self.assertEqual(o.currency, cur, o)

    def test_text(self):
        for raw in ("", "   ", "n/a", "none of the above", "$", "%", "--5", "$$5", "5%%", "1.2.3", "12,34", "$5 USD EUR",
                    "ABC 5", "USDD 5", "12 May", "May", "$ 5 5"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_option(raw).kind, "text", parse_option(raw))

    def test_dates(self):
        self.assertEqual(parse_option("17 June 2026").kind, "date")
        self.assertEqual(parse_option("17 June 2026").value, dt.date(2026, 6, 17))
        self.assertEqual(parse_option("05/06/2026").kind, "ambiguous_date")

    def test_conflicting_dollar_codes_are_text(self):
        self.assertEqual(parse_option("$5.00 EUR").kind, "text")
        self.assertEqual(parse_option("€5.00 USD").kind, "text")
        self.assertEqual(parse_option("$5.00 CAD").currency, "CAD")

    def test_never_raises_on_junk(self):
        for raw in ("\x00", "\ud800", "9" * 3000, "1e400", "‮5", "5" + "," * 500, "$" * 100, "-" * 100):
            with self.subTest(raw=raw[:20]):
                parse_option(raw)  # must not raise

    def test_value_types_are_decimal_or_date(self):
        self.assertIsInstance(parse_option("$5").value, Decimal)
        self.assertIsInstance(parse_option("2026-06-17").value, dt.date)
        self.assertIsNone(parse_option("n/a").value)


# --------------------------------------------------------------------------
# parsing: dates
# --------------------------------------------------------------------------

class ParseDateTest(unittest.TestCase):
    def test_ok_formats(self):
        d = dt.date(2026, 5, 12)
        for text in ("12 May 2026", "May 12, 2026", "12th May 2026", "2026-05-12", "2026/05/12", "12/05/2026",
                     "05/12/2026", "12.05.2026", "Tuesday, 12 May 2026", "Tue 12 May 2026", "12 MAY 2026",
                     "  12   May   2026  "):
            with self.subTest(text=text):
                if text in ("12/05/2026", "05/12/2026", "12.05.2026"):
                    self.assertEqual(parse_date_ex(text), (None, "ambiguous"))
                else:
                    self.assertEqual(parse_date_ex(text), (d, "ok"))

    def test_numeric_dates_are_resolved_only_when_unambiguous(self):
        self.assertEqual(parse_date_ex("13/06/2026"), (dt.date(2026, 6, 13), "ok"))   # dd/mm
        self.assertEqual(parse_date_ex("06/13/2026"), (dt.date(2026, 6, 13), "ok"))   # mm/dd
        self.assertEqual(parse_date_ex("05/05/2026"), (dt.date(2026, 5, 5), "ok"))    # same either way
        self.assertEqual(parse_date_ex("05/06/2026"), (None, "ambiguous"))
        self.assertEqual(parse_date_ex("13/13/2026"), (None, "invalid"))

    def test_invalid(self):
        for text in ("", "   ", "31 June 2026", "29 February 2026", "0 May 2026", "32 May 2026", "12 Mayy 2026",
                     "12 May 26", "May 2026", "2026-13-01", "2026-00-10", "12 May 1800", "12 May 2300",
                     "Monday, 12 May 2026", "x" * 100, "12 May 2026 extra"):
            with self.subTest(text=text):
                self.assertEqual(parse_date_ex(text)[1], "invalid", text)

    def test_parse_date_wrapper(self):
        self.assertEqual(parse_date("12 May 2026"), dt.date(2026, 5, 12))
        self.assertIsNone(parse_date("05/06/2026"))
        self.assertIsNone(parse_date("nonsense"))

    def test_leap_years(self):
        self.assertEqual(parse_date("29 February 2028"), dt.date(2028, 2, 29))
        self.assertIsNone(parse_date("29 February 2027"))
        self.assertEqual(parse_date("29 February 2000"), dt.date(2000, 2, 29))
        self.assertIsNone(parse_date("29 February 2100"))


# --------------------------------------------------------------------------
# parsing: normalisation and regex fragments
# --------------------------------------------------------------------------

class NormalizeTest(unittest.TestCase):
    def test_whitespace_dashes_quotes_width(self):
        self.assertEqual(normalize_text("  a \n\t b c d  "), "a b c d")
        self.assertEqual(normalize_text("5– 3— 2− 1"), "5- 3- 2- 1")
        self.assertEqual(normalize_text("‘x’ “y”"), "'x' \"y\"")
        self.assertEqual(normalize_text("＄100．50"), "$100.50")  # full-width forms (NFKC)
        self.assertEqual(normalize_text("a​b­c"), "abc")  # zero-width / soft hyphen removed
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(12345), "12345")

    def test_count_decimals(self):
        self.assertEqual(count_decimals("12"), 0)
        self.assertEqual(count_decimals("12.50"), 2)
        self.assertEqual(count_decimals("1,234.5"), 1)


class RegexFragmentTest(unittest.TestCase):
    FLAGS = re.IGNORECASE | re.ASCII

    def subj_ok(self, text):
        return re.fullmatch(rf"{SUBJ} generates", text + " generates", self.FLAGS) is not None

    def test_subject_accepts_short_plain_noun_phrases(self):
        for text in ("A product line", "The retailer", "An item", "A support queue", "A monthly headcount",
                     "A regional sales team", "A retailer's store", "A well-known brand"):
            with self.subTest(text=text):
                self.assertTrue(self.subj_ok(text), text)

    def test_subject_rejects_digits_clauses_and_long_phrases(self):
        for text in ("A support queue of 100 tickets", "The box with 40 balls", "A retailer with 5 stores",
                     "A product line adjusted for inflation", "A product line after a 10 percent fee",
                     "A product line excluding returns", "A net product line", "A product line in thousands",
                     "A fifty-ticket queue", "A dozen item box", "A very big old red shiny product line",
                     "Product line", "A", "A 5-ticket queue", "An item that is discounted"):
            with self.subTest(text=text):
                self.assertFalse(self.subj_ok(text), text)

    def test_phrase_helper(self):
        p = re.compile(rf"(?:{phrase(1)})", self.FLAGS)
        self.assertIsNotNone(p.fullmatch("marketing budget"))
        self.assertIsNone(p.fullmatch("big marketing budget"))
        self.assertIsNone(p.fullmatch("budget of 5"))

    def test_money_and_percent_groups(self):
        m = re.fullmatch(rf"{money_g('p')} at {pct_g('r')}", "$1,234.50 at 5.5%", self.FLAGS)
        self.assertEqual((m["p"], m["p_c"], m["r"]), ("1,234.50", "$", "5.5"))
        m = re.fullmatch(rf"{money_g('p')} at {pct_g('r')}", "€10 at 5 percent", self.FLAGS)
        self.assertEqual((m["p"], m["p_c"], m["r"]), ("10", "€", "5"))
        m = re.fullmatch(rf"{money_g('p')} at {pct_g('r')}", "£10.25 at 5 %", self.FLAGS)
        self.assertEqual((m["p"], m["p_c"], m["r"]), ("10.25", "£", "5"))
        self.assertIsNone(re.fullmatch(rf"{money_g('p')}", "10.00", self.FLAGS))
        self.assertIsNone(re.fullmatch(rf"{money_g('p')}", "¥10", self.FLAGS))  # yen sign is ambiguous: not accepted


if __name__ == "__main__":
    unittest.main()
