"""Deterministic family solvers for the hourly quiz.

Design rule: a solver must never be confidently wrong.  Every family recognises its
questions with ``re.fullmatch`` on the *whole* normalised text (case-insensitive,
ASCII), so any unexpected clause, keyword or reworded question simply fails to match
and the question is left to the fallback.  A solver either

* returns ``None``       - "this is not my kind of question";
* returns a ``Computed`` - the value, its type and a short formula note;
* raises ``Abstain``     - "this looks like mine but cannot be answered safely"
                            (ambiguous date, reversed FX direction, bad numbers ...).

``solve_question`` then maps the computed value onto the options with the
type- and precision-aware matcher in ``matching.py``.  It never raises.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from typing import Callable

from .matching import Computed, format_value, match_options
from .parsing import (
    CUR_RE,
    CURRENCY_SYMBOLS,
    DATE_RE,
    INT,
    MAX_TEXT_LEN,
    NUM,
    SUBJ,
    WORD,
    WORD_D,
    currency_code,
    money_g,
    normalize_text,
    parse_date_ex,
    pct_g,
    phrase,
    to_decimal,
)

_FLAGS = re.IGNORECASE | re.ASCII


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, _FLAGS)


@dataclass(frozen=True)
class Solution:
    letter: str | None      # "A".."D" or None (abstain)
    method: str             # e.g. "rule:compound_interest" | "abstain:<reason>"
    detail: str = ""        # short human-readable explanation (computed value etc.)


class Abstain(Exception):
    """Raised by a family solver that recognised the question but cannot answer it safely."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _first_match(patterns: list[re.Pattern[str]], text: str) -> re.Match[str] | None:
    for p in patterns:
        m = p.fullmatch(text)
        if m:
            return m
    return None


def _fd(fr: Fraction) -> Decimal:
    """Fraction -> Decimal (current context precision)."""
    return Decimal(fr.numerator) / Decimal(fr.denominator)


def _fraction(d: Decimal) -> Fraction:
    return Fraction(d)


_SYM_ISO = {"$": "USD", "€": "EUR", "£": "GBP"}


def _cur(m: re.Match[str], *names: str) -> str:
    """ISO code of the currency symbol(s) in front of the named ``money_g`` groups.

    All named amounts that took part in the match must use the same symbol; mixed
    symbols ("$100 ... €50") are not safe to combine, so abstain."""
    gd = m.groupdict()
    syms = {gd[f"{n}_c"] for n in names if gd.get(f"{n}_c")}
    if len(syms) != 1:
        raise Abstain("mixed_currency", "amounts use different currency symbols")
    return _SYM_ISO[next(iter(syms))]


def _names_for_group(m: re.Match[str], *names: str) -> str | None:
    for n in names:
        v = m.groupdict().get(n)
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------
# 1. compound interest
# --------------------------------------------------------------------------

_FREQ_M: dict[str, int | None] = {
    "annually": 1, "yearly": 1, "everyyear": 1,
    "semiannually": 2, "twiceayear": 2, "twiceyearly": 2, "halfyearly": 2, "everysixmonths": 2,
    "quarterly": 4, "everyquarter": 4, "monthly": 12, "everymonth": 12,
    "weekly": 52, "everyweek": 52, "daily": 365, "everyday": 365, "continuously": None,
}
# note: "bi-annually" / "biannually" is deliberately absent (twice a year OR every 2 years)
_FREQ = (
    r"(?P<freq>annually|yearly|every year|semi-?annually|semi annually|twice a year"
    r"|twice yearly|half-yearly|every six months|quarterly|every quarter|monthly"
    r"|every month|weekly|every week|daily|every day|continuously)"
)
_TIME = rf"(?P<t>{NUM}) (?P<tu>years?|months?)"
_RATE_PHRASE = r"(?:an? (?:annual |yearly )?(?:interest )?rate of )?"
_PER_YEAR = r"(?: per (?:year|annum)| a year| annually| p\.a\.)?"
# Only adjectives that clearly mean a one-off lump sum.  A free adjective slot let
# "a monthly / annual / recurring deposit of $500" (an annuity) through as a lump sum.
_INV_ADJ = (
    r"(?:initial |opening |starting |original |one-time |lump-sum |lump sum |savings |bank )?"
)
_LEAD_INV = (
    rf"(?:an?|the) {_INV_ADJ}(?:investment|deposit|principal|balance|account) "
    rf"(?:of|with) {money_g('p')}"
)
_CI_SENTENCES = [
    rf"{_LEAD_INV} (?:grows|earns interest|earns|accrues interest|accrues) at "
    rf"{_RATE_PHRASE}{pct_g('r')}(?: interest)?{_PER_YEAR},? (?:and is |and )?compounded {_FREQ}",
    rf"{money_g('p')} is (?:invested|deposited) at {_RATE_PHRASE}{pct_g('r')}"
    rf"(?: interest)?{_PER_YEAR},? (?:and is |and )?compounded {_FREQ}",
    rf"{_LEAD_INV} (?:earns|accrues) {_RATE_PHRASE}{pct_g('r')} interest{_PER_YEAR},? "
    rf"(?:and is |and )?compounded {_FREQ}",
]
_SIMPLE_SENTENCES = [
    rf"{_LEAD_INV} (?:earns|grows at|accrues) {_RATE_PHRASE}{pct_g('r')}{_PER_YEAR} "
    rf"simple interest{_PER_YEAR}",
    rf"{_LEAD_INV} (?:earns|accrues) simple interest at {_RATE_PHRASE}{pct_g('r')}{_PER_YEAR}",
    rf"{_LEAD_INV} grows at {pct_g('r')} per year (?:in|with|using) simple interest",
    rf"{money_g('p')} is (?:invested|deposited) at {_RATE_PHRASE}{pct_g('r')}{_PER_YEAR} "
    rf"simple interest{_PER_YEAR}",
]
_CI_Q = (
    r"(?:(?P<qv>what (?:is|will be) (?:its|the) "
    r"(?:(?:future |final |total |ending |accumulated )?value|(?:final |ending |accumulated )?balance)"
    r"|what will it be worth)"
    r"|(?P<qi>how much interest (?:is|will be) (?:earned|accrued|accumulated)"
    r"|what is the (?:total )?interest (?:earned|accrued)))"
    rf" (?:after|over|in|at the end of) {_TIME}\?"
)
_COMPOUND_RX = [_rx(rf"{s}\. {_CI_Q}") for s in _CI_SENTENCES]
_SIMPLE_RX = [_rx(rf"{s}\. {_CI_Q}") for s in _SIMPLE_SENTENCES]


def _solve_compound(text: str) -> Computed | None:
    m = _first_match(_COMPOUND_RX, text)
    simple = False
    if m is None:
        m = _first_match(_SIMPLE_RX, text)
        simple = m is not None
    if m is None:
        return None
    principal = to_decimal(m["p"])
    cur = _cur(m, "p")
    rate = to_decimal(m["r"]) / 100
    amount = _fraction(to_decimal(m["t"]))
    years = amount if m["tu"].lower().startswith("year") else amount / 12
    if not (0 < years <= 200) or rate > 10:
        raise Abstain("unsafe_range", "time or rate outside the supported range")
    if simple:
        fv = principal * (1 + rate * _fd(years))
        how = f"simple {principal}*(1+{rate}*{_fd(years):.6g})"
    else:
        key = m["freq"].lower().replace("-", "").replace(" ", "")
        per_year = _FREQ_M[key]
        if per_year is None:
            fv = principal * (rate * _fd(years)).exp()
            how = f"continuous {principal}*e^({rate}*{_fd(years):.6g})"
        else:
            periods = Fraction(per_year) * years
            if periods > 40000:
                raise Abstain("unsafe_range", "too many compounding periods")
            fv = principal * (1 + rate / per_year) ** _fd(periods)
            how = f"compound m={per_year} {principal}*(1+{rate}/{per_year})^{_fd(periods):.6g}"
    if m["qi"] is not None:
        return Computed(fv - principal, "money", cur, f"interest earned, {how}")
    return Computed(fv, "money", cur, f"future value, {how}")


# --------------------------------------------------------------------------
# 2. gross margin / gross profit / markup
# --------------------------------------------------------------------------

_REV = r"(?:revenue|sales)"
_COGS = r"(?:cost of goods sold|cogs)"
_GM_SENTENCES = [
    rf"{SUBJ} generates {money_g('rev')} in {_REV} against {money_g('cogs')} in {_COGS}",
    rf"{SUBJ} generates {money_g('rev')} in {_REV} with {money_g('cogs')} in {_COGS}",
    rf"{SUBJ} (?:has|reports|records|posts) (?:a )?{_REV} of {money_g('rev')} and "
    rf"(?:a )?{_COGS} of {money_g('cogs')}",
    rf"{SUBJ} with (?:a )?{_REV} of {money_g('rev')} and (?:a )?{_COGS} of {money_g('cogs')}",
    rf"{SUBJ} (?:has|reports|records) {money_g('rev')} in {_REV} and {money_g('cogs')} "
    rf"in {_COGS}",
]
_GM_Q = (
    r"what is (?:the|its) (?:(?P<gm>gross margin(?: percentage| percent)?"
    r"|gross profit margin(?: percentage| percent)?)"
    r"|(?P<gp>gross profit)"
    r"|(?P<mk>markup(?: percentage| percent)?(?: on cost)?))\?"
)
_GM_RX = [_rx(rf"{s}\. {_GM_Q}") for s in _GM_SENTENCES]


def _solve_gross_margin(text: str) -> Computed | None:
    m = _first_match(_GM_RX, text)
    if m is None:
        return None
    rev, cogs = to_decimal(m["rev"]), to_decimal(m["cogs"])
    cur = _cur(m, "rev", "cogs")
    if m["gp"] is not None:
        return Computed(rev - cogs, "money", cur, f"gross profit {rev}-{cogs}")
    if m["mk"] is not None:
        if cogs == 0:
            raise Abstain("zero_denominator", "markup with zero cost")
        return Computed((rev - cogs) / cogs * 100, "percent", None, f"markup ({rev}-{cogs})/{cogs}")
    if rev == 0:
        raise Abstain("zero_denominator", "margin with zero revenue")
    return Computed((rev - cogs) / rev * 100, "percent", None, f"gross margin ({rev}-{cogs})/{rev}")


# --------------------------------------------------------------------------
# 3. next term of a sequence
# --------------------------------------------------------------------------

_SEQ_NUM = rf"(?:-?\$?{NUM}|\$-{NUM})"
_SEQ_RX = _rx(
    rf"{SUBJ} (?:series|sequence) (?:runs|goes|reads|is) "
    rf"(?P<nums>{_SEQ_NUM}(?:(?:, and |, | and ){_SEQ_NUM}){{2,29}})\. "
    r"(?:if|assuming) (?:the|this) (?:pattern|trend) (?:continues|holds), "
    r"what (?:is the next (?:value|number|term|figure|item)|comes next)\?"
)
_SEQ_SPLIT = re.compile(r"(?:, and |, | and )(?=-?\$?-?\d)")


def _predict_sequence(terms: list[Fraction]) -> set[Fraction]:
    n = len(terms)
    preds: set[Fraction] = set()
    diffs = [b - a for a, b in zip(terms, terms[1:])]
    if n >= 3 and len(set(diffs)) == 1:
        preds.add(terms[-1] + diffs[0])
    if n >= 3 and all(t != 0 for t in terms):
        ratios = {b / a for a, b in zip(terms, terms[1:])}
        if len(ratios) == 1:
            preds.add(terms[-1] * next(iter(ratios)))
    if n >= 5:
        second = [b - a for a, b in zip(diffs, diffs[1:])]
        if len(set(second)) == 1:
            preds.add(terms[-1] + diffs[-1] + second[0])
    return preds


def _solve_sequence(text: str) -> Computed | None:
    m = _SEQ_RX.fullmatch(text)
    if m is None:
        return None
    pieces = _SEQ_SPLIT.split(m["nums"])
    dollars = {"$" in p for p in pieces}
    if len(dollars) != 1:
        raise Abstain("mixed_units", "some terms carry a $ and some do not")
    terms = [_fraction(to_decimal(p.replace("$", ""))) for p in pieces]
    preds = _predict_sequence(terms)
    if not preds:
        raise Abstain("unknown_pattern", "not arithmetic, geometric or constant 2nd difference")
    if len(preds) > 1:
        raise Abstain("ambiguous_pattern", "several patterns fit with different next values")
    nxt = next(iter(preds))
    if dollars.pop():
        return Computed(_fd(nxt), "money", "USD", f"next term of {len(terms)}-term $ sequence")
    return Computed(_fd(nxt), "number", None, f"next term of {len(terms)}-term sequence")


# --------------------------------------------------------------------------
# 4. discounts / markups
# --------------------------------------------------------------------------

_DISC_Q = (
    r"(?:(?P<qp>what is the (?:sale|discounted|final|new|reduced|selling|marked-up|marked up) price"
    r"|what is the (?:new )?price after (?:the )?(?:discount|markup|mark-up|markdown)"
    r"|what does it cost after (?:the )?(?:discount|markup|mark-up)))"
    r"|(?P<qs>what is the (?:discount amount|amount of the discount|savings|amount saved"
    r"|amount of savings|markup amount|amount of the markup)"
    r"|how much is saved"
    r"|how much (?:does|will) (?:the customer|a customer|the buyer|a buyer|you) save)"
)
_DISC_Q_RX = rf"(?:{_DISC_Q})\?"
_LISTED = r"(?:listed|priced|marked) at"
_DISC_SENTENCES = [
    ("discount", rf"{SUBJ} {_LISTED} {money_g('p')} is (?:discounted|reduced|marked down) by {pct_g('d')}"),
    ("discount", rf"{SUBJ} {_LISTED} {money_g('p')} is on sale for {pct_g('d')} off"),
    ("discount", rf"{SUBJ} {_LISTED} {money_g('p')} has an? {pct_g('d')} discount"),
    (
        "successive",
        rf"{SUBJ} {_LISTED} {money_g('p')} is (?:discounted|reduced|marked down) by {pct_g('d1')} "
        rf"and then by (?:a further|an additional|another) {pct_g('d2')}",
    ),
    (
        "markup",
        rf"{SUBJ} (?:costing|that costs|priced at|listed at|bought for|purchased for) "
        rf"{money_g('p')} is (?:marked up|increased) by {pct_g('d')}",
    ),
]
_DISC_RX = [(kind, _rx(rf"{s}\. {_DISC_Q_RX}")) for kind, s in _DISC_SENTENCES]


def _solve_discount(text: str) -> Computed | None:
    kind, m = None, None
    for k, p in _DISC_RX:
        m = p.fullmatch(text)
        if m:
            kind = k
            break
    if m is None:
        return None
    price = to_decimal(m["p"])
    cur = _cur(m, "p")
    q_text = (m["qp"] or m["qs"] or "").lower()
    if kind == "markup":
        if "discount" in q_text or "sav" in q_text or "reduced" in q_text or "markdown" in q_text:
            raise Abstain("inconsistent_question", "markup sentence with a discount question")
        rate = to_decimal(m["d"]) / 100
        final = price * (1 + rate)
        how = f"markup {price}*(1+{rate})"
    else:
        if "markup" in q_text or "mark-up" in q_text or "marked" in q_text:
            raise Abstain("inconsistent_question", "discount sentence with a markup question")
        if kind == "successive":
            d1, d2 = to_decimal(m["d1"]) / 100, to_decimal(m["d2"]) / 100
            if d1 > 1 or d2 > 1:
                raise Abstain("unsafe_range", "discount above 100%")
            final = price * (1 - d1) * (1 - d2)
            how = f"successive {price}*(1-{d1})*(1-{d2})"
        else:
            d = to_decimal(m["d"]) / 100
            if d > 1:
                raise Abstain("unsafe_range", "discount above 100%")
            final = price * (1 - d)
            how = f"discount {price}*(1-{d})"
    if m["qs"] is not None:
        amount = abs(price - final)
        return Computed(amount, "money", cur, f"amount, {how}")
    return Computed(final, "money", cur, f"price, {how}")


# --------------------------------------------------------------------------
# 5. percentage change
# --------------------------------------------------------------------------

_UP_VERBS = ("rose", "increased", "grew", "climbed", "jumped", "went up",
             "rises", "increases", "grows", "climbs", "jumps")
_DOWN_VERBS = ("fell", "decreased", "dropped", "declined", "shrank", "went down",
               "falls", "decreases", "drops", "declines", "shrinks")
# Subject of the change: a short phrase.  Clause words (from / to / then / and / after /
# excluding ...) are rejected, so "Revenue went from 100 to 200 then moved from ..." and
# "Revenue after tax ..." are left alone; harmless prepositions are allowed.
_PC_SUBJ = rf"(?:{WORD_D}|of|in|for|at|on)(?: (?:{WORD_D}|of|in|for|at|on)){{0,5}}"
_PC_RX = _rx(
    rf"(?P<subj>{_PC_SUBJ}) "
    r"(?P<verb>moved|changed|went|rose|fell|increased|decreased|grew|dropped|climbed|declined"
    r"|jumped|shrank|went up|went down|rises|falls|increases|decreases|grows|drops|climbs"
    r"|declines|jumps|shrinks) from "
    rf"(?:(?P<sa>[$€£])(?P<a1>{NUM})|(?P<a2>{NUM})) to "
    rf"(?:(?P<sb>[$€£])(?P<b1>{NUM})|(?P<b2>{NUM}))\. "
    r"what (?:is|was) the (?:percentage|percent) change\?"
)


def _solve_percent_change(text: str) -> Computed | None:
    m = _PC_RX.fullmatch(text)
    if m is None:
        return None
    if (m["a1"] is None) != (m["b1"] is None) or m["sa"] != m["sb"]:
        raise Abstain("mixed_units", "the two amounts do not carry the same currency symbol")
    a = to_decimal(m["a1"] or m["a2"])
    b = to_decimal(m["b1"] or m["b2"])
    if a == 0:
        raise Abstain("zero_denominator", "percentage change from zero")
    verb = m["verb"].lower()
    if (verb in _UP_VERBS and b < a) or (verb in _DOWN_VERBS and b > a):
        raise Abstain("inconsistent_direction", f"'{verb}' contradicts the numbers")
    return Computed((b - a) / a * 100, "percent", None, f"percent change ({b}-{a})/{a}")


# --------------------------------------------------------------------------
# 6. settlement date
# --------------------------------------------------------------------------

_SET_KIND = r"(?:business|working|calendar)"
_SET_SENT = [
    rf"(?:an?) (?:contract|trade|agreement|invoice|order|lease|loan|deal|purchase) "
    rf"(?:signed|executed|made|placed|issued|opened|dated) on (?P<date>{DATE_RE}) "
    rf"has an? (?P<n>\d+)[- ](?:(?P<k>business|working|calendar)[- ])?day settlement period",
    rf"(?:an?) (?:contract|trade|agreement|invoice|order|lease|loan|deal|purchase) "
    rf"(?:signed|executed|made|placed|issued|opened|dated) on (?P<date>{DATE_RE}) "
    rf"has a settlement period of (?P<n>\d+) (?:(?P<k>business|working|calendar) )?days?",
    rf"(?:an?) (?:contract|trade|agreement|invoice|order|lease|loan|deal|purchase) "
    rf"(?:signed|executed|made|placed|issued|opened|dated) on (?P<date>{DATE_RE}) "
    rf"settles (?P<n>\d+) (?:(?P<k>business|working|calendar) )?days? after "
    rf"(?:signing|the signing date|being signed)",
]
_SET_Q = (
    r"(?:on what date (?:does|will) (?:it|the (?:contract|trade|agreement|invoice|order|lease|loan"
    r"|deal|purchase)) settle|what date (?:does|will) it settle|when (?:does|will) it settle"
    r"|what is the settlement date)\?"
)
_SET_RX = [_rx(rf"{s}\. {_SET_Q}") for s in _SET_SENT]


def _solve_settlement(text: str) -> Computed | None:
    m = _first_match(_SET_RX, text)
    if m is None:
        return None
    start, status = parse_date_ex(m["date"])
    if start is None:
        raise Abstain("ambiguous_date" if status == "ambiguous" else "invalid_date", m["date"])
    n = int(m["n"])
    kind = (m["k"] or "calendar").lower()
    try:
        if kind == "calendar":
            if n > 36500:
                raise Abstain("unsafe_range", "settlement period too long")
            result = start + dt.timedelta(days=n)
            how = f"{start.isoformat()} + {n} calendar days"
        else:
            if not (1 <= n <= 20000):
                raise Abstain("unsafe_range", "business-day count out of range")
            result, count = start, 0
            while count < n:
                result += dt.timedelta(days=1)
                if result.weekday() < 5:
                    count += 1
            how = f"{start.isoformat()} + {n} business days (Sat/Sun skipped only)"
    except OverflowError as exc:
        raise Abstain("date_overflow", str(exc)) from exc
    return Computed(result, "date", None, how)


# --------------------------------------------------------------------------
# 7. break-even
# --------------------------------------------------------------------------

_BE_FIXED = rf"(?:the )?fixed costs (?:are|total|equal) {money_g('f')}\. "
_BE_SENT = (
    _BE_FIXED
    + rf"each unit (?:sells|is sold) for {money_g('price')} (?:with|and has|and carries) "
    + rf"a variable cost of {money_g('v')}(?: per unit)?\. "
)
_BE_SENT_CM = (
    _BE_FIXED
    + rf"each unit (?:has|carries|earns) a contribution margin of {money_g('cm')}(?: per unit)?\. "
)
_GOAL_BE = r"break even"
_GOAL_TP = rf"(?:achieve|earn|reach|make|generate) a (?:target )?profit of {money_g('tp')}"
_BE_UNIT_TAILS = [
    (r"how many units must be sold to {G}, to the nearest unit\?", "nearest"),
    (r"how many units must be sold to {G}, rounded to the nearest unit\?", "nearest"),
    (r"how many units must be sold to {G}\?", "none"),
    (r"how many units must be sold to {G}\? round to the nearest (?:whole )?unit\.?", "nearest"),
    (r"how many units must be sold to {G}, rounded up to the (?:next|nearest) whole unit\?", "ceil"),
    (r"how many units must be sold to {G}\? round up to the (?:next|nearest) whole unit\.?", "ceil"),
    (r"what is the minimum (?:whole )?number of (?:whole )?units that must be sold to {G}\?", "ceil"),
    (r"what is the minimum number of (?:whole )?units (?:needed|required) to {G}\?", "ceil"),
    (r"at least how many units must be sold to {G}\?", "ceil"),
]
_BE_REV_TAILS = [
    r"what is the break-even (?:revenue|sales|sales revenue|sales dollars)\?",
    r"what is the break-even point,? in (?:sales )?dollars\?",
    r"what sales revenue is needed to break even\?",
    r"how much (?:sales )?revenue (?:is needed|is required|must be generated) to break even\?",
    r"what is the break-even point in terms of (?:sales )?revenue\?",
]
_BE_RX: list[tuple[str, str, re.Pattern[str]]] = []
for _head in (_BE_SENT, _BE_SENT_CM):
    for _goal_name, _goal in (("be", _GOAL_BE), ("tp", _GOAL_TP)):
        for _tail, _mode in _BE_UNIT_TAILS:
            _BE_RX.append((_goal_name, _mode, _rx(_head + _tail.replace("{G}", _goal))))
    for _tail in _BE_REV_TAILS:
        _BE_RX.append(("be", "revenue", _rx(_head + _tail)))


def _solve_break_even(text: str) -> Computed | None:
    goal, mode, m = None, None, None
    for g, md, p in _BE_RX:
        m = p.fullmatch(text)
        if m:
            goal, mode = g, md
            break
    if m is None:
        return None
    fixed = to_decimal(m["f"])
    if m.groupdict().get("cm") is not None:
        cur = _cur(m, "f", "cm", *(("tp",) if goal == "tp" else ()))
        margin = to_decimal(m["cm"])
        price = None
        how = f"({fixed}+{{t}})/{margin}"
    else:
        cur = _cur(m, "f", "price", "v", *(("tp",) if goal == "tp" else ()))
        price, var = to_decimal(m["price"]), to_decimal(m["v"])
        margin = price - var
        how = f"({fixed}+{{t}})/({price}-{var})"
    if margin <= 0:
        raise Abstain("no_contribution_margin", "price must exceed variable cost")
    target = to_decimal(m["tp"]) if goal == "tp" else Decimal(0)
    units = (fixed + target) / margin
    how = how.replace("{t}", str(target))
    if mode == "revenue":
        if price is None:
            raise Abstain("revenue_needs_price", "break-even revenue needs the unit price")
        return Computed(units * price, "money", cur, f"break-even revenue {how}*{price}")
    if mode == "nearest":
        return Computed(units.quantize(Decimal(1), rounding=ROUND_HALF_UP), "units", None,
                        f"units rounded half-up {how}")
    if mode == "ceil":
        return Computed(units.to_integral_value(rounding=ROUND_CEILING), "units", None,
                        f"units rounded up {how}")
    # No rounding phrase: the whole-unit answer is ambiguous (round vs. round up).  Accept
    # only when no other option equals floor / ceil of the exact value.
    if units == units.to_integral_value():
        return Computed(units, "units", None, f"exact whole units {how}")
    rivals = (units.to_integral_value(rounding=ROUND_FLOOR), units.to_integral_value(rounding=ROUND_CEILING))
    return Computed(units, "units", None, f"units (no rounding phrase) {how}", rivals)


# --------------------------------------------------------------------------
# 8. ratio split
# --------------------------------------------------------------------------

_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "sixth": 5}
_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
_RS_RATIO = rf"{NUM}(?: ?: ?{NUM})+|{NUM}(?: to {NUM})+"
_RS_HEAD = (
    rf"an? {money_g('t')} (?P<what>{phrase(3)}) (?:is|are) "
    r"(?:split|divided|shared|allocated|distributed) (?:between|among) "
    rf"(?P<who>{WORD_D}(?: {WORD_D}){{0,3}}) "
    rf"(?:(?:in|at) (?:the|a) ratio (?:of )?(?P<ratio>{_RS_RATIO})"
    rf"|(?:in|at) an? (?P<ratio2>{_RS_RATIO}) ratio)\. "
)
# The noun after "the first ..." must be a plain noun: not another ordinal / comparative /
# selector ("the largest first channel", "the second other channel" are contradictory).
_SELECTOR_WORDS = (
    "first|second|third|fourth|fifth|sixth|last|larger|smaller|largest|smallest|other|"
    "remaining|rest|combined|total|bigger|biggest|lesser|least|most|top|bottom|middle|"
    "one|two|three|four|five|six"
)
_NOUN_W = rf"(?!(?:{_SELECTOR_WORDS})(?![a-z0-9\-'])){WORD}"
_WORDS2 = rf"{_NOUN_W}(?: {_NOUN_W})?"
_RS_TAILS = [
    rf"(?:what|how much) (?:does|will) the (?P<ord>first|second|third|fourth|fifth|sixth) {_WORDS2} (?:receive|get)\?",
    rf"(?:what|how much) (?:does|will) the (?P<cmp>larger|smaller|largest|smallest) {_WORDS2} (?:receive|get)\?",
    r"what is the (?P<cmp>larger|smaller|largest|smallest) (?:share|amount|allocation|portion)\?",
    r"what is the (?P<ord>first|second|third|fourth|fifth|sixth) (?:share|amount|allocation|portion)\?",
]
_RS_RX = [_rx(_RS_HEAD + t) for t in _RS_TAILS]


def _solve_ratio_split(text: str) -> Computed | None:
    m = _first_match(_RS_RX, text)
    if m is None:
        return None
    total = to_decimal(m["t"])
    cur = _cur(m, "t")
    ratio_text = (m["ratio"] or m["ratio2"]).lower().replace(" to ", ":")
    parts = [to_decimal(p.strip()) for p in ratio_text.split(":")]
    gd = m.groupdict()
    first_word = m["who"].split()[0].lower()
    if first_word not in _COUNT_WORDS:
        raise Abstain("unclear_recipients", "cannot map ordinals onto the recipients")
    if _COUNT_WORDS[first_word] != len(parts):
        raise Abstain("ratio_count_mismatch", "number of recipients differs from ratio parts")
    whole = sum(parts)
    if whole <= 0:
        raise Abstain("zero_denominator", "ratio parts sum to zero")
    if gd.get("ord") is not None:
        idx = _ORDINALS[gd["ord"].lower()]
        if idx >= len(parts):
            raise Abstain("ordinal_out_of_range", gd["ord"])
        share = parts[idx]
        label = gd["ord"].lower()
    else:
        cmp_word = (gd.get("cmp") or "").lower()
        if cmp_word not in ("larger", "smaller", "largest", "smallest"):
            raise Abstain("no_selector", "neither an ordinal nor a comparative was matched")
        if cmp_word in ("larger", "smaller") and len(parts) != 2:
            raise Abstain("comparative_with_many", "larger/smaller needs exactly two parts")
        share = max(parts) if cmp_word in ("larger", "largest") else min(parts)
        label = cmp_word
    return Computed(total * share / whole, "money", cur, f"{label} share {total}*{share}/{whole}")


# --------------------------------------------------------------------------
# 9. weighted average
# --------------------------------------------------------------------------

# Group label ("Region A", "the north store"): up to three words, no clause words.
_WA_LABEL = rf"{WORD}(?: {WORD_D}){{0,2}}"
_WA_GROUP_NC = (
    rf"{_WA_LABEL} had "
    rf"{INT} [a-z]+(?: [a-z]+)? (?:averaging|with an average of|at an average of) [$€£]?{NUM}"
)
_WA_GROUP_NAMED = _rx(
    r"[a-z][a-z0-9\-]*(?: [a-z0-9\-]+){0,2} had "
    rf"(?P<n>{INT}) (?P<u>[a-z]+(?: [a-z]+)?) "
    rf"(?:averaging|with an average of|at an average of) (?P<d>[$€£])?(?P<a>{NUM})"
)
_WA_SEP = r"(?:; and |, and |; |, | and )"
# The question tail is a whitelist, NOT free words: "excluding region A", "for region A",
# "without region B", "excluding returns" ... each change what is being averaged.
_WA_NOUNS = (
    r"(?:order|purchase|sale|sales|ticket|transaction|basket|invoice|customer|student|employee"
    r"|item|unit|product|test|exam|value|size|amount|spend|price|cost|total|score|grade|rating"
    r"|salary|wage|age|weight|time|duration)"
)
_WA_TAIL = (
    r"what is the (?:overall|combined|weighted|blended) (?:average|mean)"
    rf"(?: {_WA_NOUNS}){{0,2}}"
    r"(?P<scope> across (?:both|all|the two|all the) [a-z]+)?\?"
)
_WA_RX = _rx(
    rf"(?P<body>{_WA_GROUP_NC}(?:{_WA_SEP}{_WA_GROUP_NC}){{1,5}})\. {_WA_TAIL}"
)


def _solve_weighted_average(text: str) -> Computed | None:
    m = _WA_RX.fullmatch(text)
    if m is None:
        return None
    groups = [g for g in _WA_GROUP_NAMED.finditer(m["body"])]
    if len(groups) < 2 or len(groups) != m["body"].lower().count(" had "):
        raise Abstain("group_parse", "could not isolate the groups")
    scope = (m["scope"] or "").lower()
    if (" both " in scope or " the two " in scope) and len(groups) != 2:
        raise Abstain("scope_mismatch", "'both' used with a number of groups other than two")
    units = {g["u"].lower() for g in groups}
    dollars = {g["d"] for g in groups}
    if len(units) != 1:
        raise Abstain("mixed_units", "groups count different things")
    if len(dollars) != 1:
        raise Abstain("mixed_units", "averages carry different (or only some) currency symbols")
    weights = [to_decimal(g["n"]) for g in groups]
    avgs = [to_decimal(g["a"]) for g in groups]
    total_w = sum(weights)
    if total_w <= 0:
        raise Abstain("zero_denominator", "total weight is zero")
    value = sum(w * a for w, a in zip(weights, avgs)) / total_w
    symbol = dollars.pop()
    return Computed(value, "money" if symbol else "number", _SYM_ISO[symbol] if symbol else None,
                    f"weighted mean of {len(groups)} groups")


# --------------------------------------------------------------------------
# 10. currency conversion
# --------------------------------------------------------------------------

_SYMS = "".join(re.escape(s) for s in CURRENCY_SYMBOLS)
_CUR_AMOUNT = (
    rf"(?:(?P<sym>[{_SYMS}])(?P<a1>{NUM})|(?P<a2>{NUM}) (?P<c2>{CUR_RE}))"
)
_CUR_RATE = (
    rf"(?:(?P<r1>{NUM}) (?P<x1>{CUR_RE}) per (?P<y1>{CUR_RE})"
    rf"|(?:1|one) (?P<y2>{CUR_RE}) (?:=|equals|buys|is worth|is equal to) "
    rf"(?P<r2>{NUM}) (?P<x2>{CUR_RE}))"
)
_CUR_VERB = (
    r"(?:is|are|was|were) (?:converted|exchanged|changed|translated) (?:to|into|for) "
    rf"(?P<tgt>{CUR_RE}) at (?:an? exchange rate of |the exchange rate of |a rate of "
    rf"|the rate of |a rate |the rate )?{_CUR_RATE}\. "
)
_CUR_HEADS = [
    rf"{SUBJ} (?:of|for) {_CUR_AMOUNT} {_CUR_VERB}",
    rf"{_CUR_AMOUNT} {_CUR_VERB}",
]
_CUR_QS = [
    rf"how many (?P<q>{CUR_RE}) (?:is|are) (?:that|this|it)\?",
    rf"what is (?:that|this) in (?P<q>{CUR_RE})\?",
    rf"how much is (?:that|this) in (?P<q>{CUR_RE})\?",
    rf"what is the (?:converted|resulting) amount in (?P<q>{CUR_RE})\?",
    rf"what is the equivalent in (?P<q>{CUR_RE})\?",
    rf"how many (?P<q>{CUR_RE}) does (?:that|this|it) (?:make|come to|equal|amount to|total)\?",
]
_CUR_RX = [_rx(h + q) for h in _CUR_HEADS for q in _CUR_QS]


def _solve_currency(text: str) -> Computed | None:
    m = _first_match(_CUR_RX, text)
    if m is None:
        return None
    amount = to_decimal(_names_for_group(m, "a1", "a2"))
    if m["sym"] is not None:
        src = CURRENCY_SYMBOLS[m["sym"]]
    else:
        src = currency_code(m["c2"])
    tgt = currency_code(m["tgt"])
    asked = currency_code(m["q"])
    if m["r1"] is not None:
        rate, x, y = to_decimal(m["r1"]), currency_code(m["x1"]), currency_code(m["y1"])
    else:
        rate, x, y = to_decimal(m["r2"]), currency_code(m["x2"]), currency_code(m["y2"])
    if None in (src, tgt, asked, x, y):
        raise Abstain("unknown_currency", "unrecognised currency name")
    if asked != tgt:
        raise Abstain("currency_mismatch", "question asks for a different currency")
    if x == y or src == tgt:
        raise Abstain("currency_mismatch", "degenerate conversion")
    if rate <= 0:
        raise Abstain("unsafe_range", "non-positive rate")
    if src == y and tgt == x:
        return Computed(amount * rate, "money", tgt, f"{amount} {src} * {rate} {x}/{y}")
    if src == x and tgt == y:
        return Computed(amount / rate, "money", tgt, f"{amount} {src} / {rate} {x}/{y}")
    raise Abstain("currency_direction", f"rate is {x} per {y} but converting {src} to {tgt}")


# --------------------------------------------------------------------------
# 11. probability
# --------------------------------------------------------------------------

_PR_CAT = r"\d+(?: [a-z\-]+){1,3}"
_PR_RX = _rx(
    rf"{SUBJ} (?:holds|contains|has|includes) "
    rf"(?P<cats>{_PR_CAT}(?:(?:, and |, | and ){_PR_CAT}){{1,5}})\. "
    r"if (?:one|one item|one ticket|a single item|an item|a ticket) is "
    r"(?:drawn|chosen|picked|selected|pulled) (?:at random|randomly), "
    r"what (?:is the (?:probability|chance)|are the chances) (?:that )?"
    r"(?:it|the item|the ticket|the item drawn) is "
    r"(?P<neg>not )?(?:either )?(?P<tgt>(?:an? )?[a-z\- ,]+?)\?"
)
_PR_SPLIT = re.compile(r"(?:, and |, | and )(?=\d)")
_PR_PIECE = re.compile(r"(\d+) ((?:[a-z\-]+ )*[a-z\-]+)", re.I | re.A)
_PR_OR = re.compile(r"\s*,\s*(?:or\s+)?|\s+or\s+", re.I | re.A)


def _singular(word: str) -> str:
    w = word.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _solve_probability(text: str) -> Computed | None:
    m = _PR_RX.fullmatch(text)
    if m is None:
        return None
    counts: dict[str, int] = {}
    nouns: set[str] = set()
    for piece in _PR_SPLIT.split(m["cats"]):
        pm = _PR_PIECE.fullmatch(piece.strip())
        if pm is None:
            raise Abstain("category_parse", piece)
        words = pm.group(2).lower().split()
        if len(words) < 2:
            raise Abstain("category_parse", piece)
        name = " ".join(words[:-1])
        nouns.add(_singular(words[-1]))
        if name in counts:
            raise Abstain("duplicate_category", name)
        counts[name] = int(pm.group(1))
    if len(nouns) != 1:
        raise Abstain("mixed_nouns", "categories use different nouns")
    noun = next(iter(nouns))
    total = sum(counts.values())
    if total <= 0:
        raise Abstain("zero_denominator", "empty population")
    tgt = re.sub(r"^an? ", "", m["tgt"].strip(), flags=re.I)
    chosen: list[str] = []
    for part in _PR_OR.split(tgt):
        part = re.sub(r"^an? ", "", part.strip(), flags=re.I)
        part = re.sub(rf"\s+{re.escape(noun)}s?$", "", part, flags=re.I).lower()
        if part not in counts:
            raise Abstain("unknown_category", part)
        if part in chosen:
            raise Abstain("duplicate_category", part)
        chosen.append(part)
    hit = sum(counts[c] for c in chosen)
    if m["neg"] is not None:
        hit = total - hit
    pct = Decimal(hit) * 100 / Decimal(total)
    return Computed(pct, "percent", None, f"probability {hit}/{total}")


# --------------------------------------------------------------------------
# 12. total revenue
# --------------------------------------------------------------------------

_TR_RX = _rx(
    rf"{SUBJ} (?:sells|sold) (?P<n>{INT}) [a-z]+(?: [a-z]+)? at {money_g('p')} "
    r"(?:each|per unit|apiece|per item|a piece|per piece)\. "
    r"(?:what is (?:the )?(?:total )?(?:sales )?revenue(?: in dollars)?"
    r"|how much (?:total )?(?:sales )?revenue does (?:it|the [a-z]+) (?:generate|earn|make|bring in))\?"
)


def _solve_total_revenue(text: str) -> Computed | None:
    m = _TR_RX.fullmatch(text)
    if m is None:
        return None
    units, price = to_decimal(m["n"]), to_decimal(m["p"])
    return Computed(units * price, "money", _cur(m, "p"), f"{units}*{price}")


# --------------------------------------------------------------------------
# registry + entry point
# --------------------------------------------------------------------------

SOLVERS: list[tuple[str, Callable[[str], Computed | None]]] = [
    ("compound_interest", _solve_compound),
    ("gross_margin", _solve_gross_margin),
    ("sequence_next", _solve_sequence),
    ("discount_price", _solve_discount),
    ("percent_change", _solve_percent_change),
    ("settlement_date", _solve_settlement),
    ("break_even", _solve_break_even),
    ("ratio_split", _solve_ratio_split),
    ("weighted_average", _solve_weighted_average),
    ("currency", _solve_currency),
    ("probability", _solve_probability),
    ("total_revenue", _solve_total_revenue),
]
FAMILIES = tuple(name for name, _ in SOLVERS)


def _solve(question_text: str, options: dict[str, str]) -> Solution:
    if not isinstance(question_text, str) or not isinstance(options, dict):
        return Solution(None, "abstain:bad_input", "question or options have the wrong type")
    clean_options = {
        str(k).strip().upper(): v for k, v in options.items() if isinstance(v, str)
    }
    if not clean_options or not set(clean_options) <= set("ABCD") or len(clean_options) < 2:
        return Solution(None, "abstain:bad_options", "options must be A-D strings")
    text = normalize_text(question_text)
    if not text or len(text) > MAX_TEXT_LEN:
        return Solution(None, "abstain:bad_input", "empty or oversized question")

    claims: list[tuple[str, Computed | None, Abstain | None]] = []
    with localcontext() as ctx:
        ctx.prec = 60
        for family, solver in SOLVERS:
            try:
                computed = solver(text)
            except Abstain as ab:
                claims.append((family, None, ab))
                continue
            if computed is not None:
                claims.append((family, computed, None))
    if not claims:
        return Solution(None, "abstain:no_family", "wording not recognised by any family")
    if len(claims) > 1:
        names = ",".join(f for f, _, _ in claims)
        return Solution(None, "abstain:multiple_families", f"question claimed by {names}")
    family, computed, ab = claims[0]
    if ab is not None:
        return Solution(None, f"abstain:{ab.reason}", f"{family}: {ab.detail}".strip())
    assert computed is not None
    result = match_options(computed, clean_options)
    if result.letter is None:
        return Solution(None, f"abstain:{result.status}", f"{family}: {computed.note}; {result.detail}")
    method = f"rule:{family}" + ("+tol" if result.status == "tolerance" else "")
    return Solution(result.letter, method, f"{computed.note} = {format_value(computed)}; {result.detail}")


def solve_question(question_text: str, options: dict[str, str]) -> Solution:
    """Solve one question deterministically.  options = {"A": "...", ... "D": "..."}.

    Never raises: any unexpected problem becomes an abstain with the reason in ``method``.
    """
    try:
        return _solve(question_text, options)
    except Exception as exc:  # noqa: BLE001 - contract: solve_question never raises
        return Solution(None, "abstain:error", f"{type(exc).__name__}: {exc}")
