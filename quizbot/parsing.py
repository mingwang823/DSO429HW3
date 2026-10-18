"""Text normalisation and number / money / percent / date parsing helpers.

Pure functions, standard library only.  The family regexes themselves live in
``solvers.py``; this module owns the small building blocks they share:

* ``normalize_text``  - unify whitespace, dashes, quotes, width variants.
* regex fragments (``NUM``, ``money_g`` ...) used to build full-match grammars.
* ``parse_date_ex``   - strict date parser that reports ambiguity instead of guessing.
* ``parse_option``    - turn an option string ("$1,234.50", "-15.1%", "1,533 units",
  "17 June 2026") into a typed ``OptionValue`` for ``matching.py``.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_DASHES = dict.fromkeys(
    map(ord, "‐‑‒–—―−﹘﹣－"), "-"
)
_QUOTES = {ord(c): "'" for c in "‘’‚‛′ʼ"}
_QUOTES.update({ord(c): '"' for c in "“”„‟″"})

MAX_TEXT_LEN = 2000


def normalize_text(text: str) -> str:
    """Collapse whitespace, unify unicode dashes / quotes / width variants, strip."""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.translate(_ZERO_WIDTH).translate(_DASHES).translate(_QUOTES)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------------------------------------------------------
# Regex fragments (all used with re.IGNORECASE | re.ASCII)
# --------------------------------------------------------------------------

#: A non-negative number, optionally with US thousands separators and decimals.
NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
#: A non-negative integer (thousands separators allowed).
INT = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
#: Words that may never appear inside a "subject" or other free noun phrase.  They are
#: prepositions / conjunctions (a clause such as "after a fee", "excluding returns", "of
#: 100 tickets"), quantifiers, financial qualifiers that change what a figure means, and
#: the verbs the family grammars use to end a subject.  A phrase containing one of them
#: simply does not match, so the question is left to the fallback.
_STOP_WORDS = (
    "after before during until since of for with without within across over under between "
    "among in on at by from to into onto as than then and or but if when while whereas per "
    "each every all both any no not only either neither "
    "excluding including excludes includes minus plus less except net gross pretax "
    "pre-tax aftertax after-tax adjusted tax taxes taxed fee fees returns returned "
    "refund refunds refunded discount discounted reduced "
    "is are was were be been has have had holds contains generates generated sells sold "
    "listed priced marked costing"
)
_STOP_RE = "|".join(sorted(set(_STOP_WORDS.split()), key=len, reverse=True))
#: Spelled-out numbers ("fifty-ticket queue", "a dozen items") hide a stated quantity just
#: like digits do, so a word that *starts* with one is not a plain noun either.
_NUM_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety|hundred|thousand|million|billion|dozen|half|double|triple|single"
)
#: Selector adjectives that point at a *subset* or a derived figure ("the remaining
#: stores", "the largest line").  Not allowed in a plain subject noun phrase.
_SELECTORS = (
    "other|others|remaining|rest|additional|further|extra|combined|overall|median|largest|"
    "smallest|biggest|negative|positive|twice|last|next|previous"
)
_NOT_PLAIN = rf"(?!(?:{_STOP_RE}|{_SELECTORS})(?![a-z0-9\-'])|(?:{_NUM_WORDS})(?![a-z]))"
#: One plain lower-case word (letters, hyphen, apostrophe; no digits, no stop word).
WORD = rf"{_NOT_PLAIN}[a-z][a-z\-']*"
#: Like ``WORD`` but may contain / start with digits ("Region 3", "Q3").  Group labels
#: and recipient lists (``two channels``) use this one, so number words are allowed there.
WORD_D = rf"(?!(?:{_STOP_RE})(?![a-z0-9\-']))[a-z0-9][a-z0-9\-']*"


def phrase(max_extra: int = 3) -> str:
    """A short digit-free noun phrase of 1..max_extra+1 words without stop words."""
    return rf"{WORD}(?: {WORD}){{0,{max_extra}}}"


#: Generic subject noun phrase: "A product line", "The retailer" ...  Digit-free, at most
#: four words, no prepositions / qualifiers (so "A support queue of 100 tickets" and
#: "A product line adjusted for inflation" do NOT match).
SUBJ = rf"(?:an?|the) {phrase(3)}"

#: Currency symbols the money grammars accept in front of an amount.
MONEY_SYMBOLS = "$€£"


def money_g(name: str) -> str:
    """An amount with a currency symbol; groups ``<name>_c`` (symbol) and ``<name>``."""
    return rf"(?P<{name}_c>[$€£])(?P<{name}>{NUM})"


def pct_g(name: str) -> str:
    return rf"(?P<{name}>{NUM})(?: ?%| percent| per cent)"


def num_g(name: str) -> str:
    return rf"(?P<{name}>{NUM})"


def to_decimal(s: str) -> Decimal:
    """'1,234.50' -> Decimal('1234.50').  Raises InvalidOperation on junk."""
    return Decimal(s.replace(",", ""))


def count_decimals(num_text: str) -> int:
    """Number of digits displayed after the decimal point."""
    return len(num_text.split(".", 1)[1]) if "." in num_text else 0


# --------------------------------------------------------------------------
# Currencies
# --------------------------------------------------------------------------

#: Lower-case currency names / codes -> ISO code.  Used by the currency solver and
#: by option parsing ("3,852.95 euros").
CURRENCY_NAMES: dict[str, str] = {
    "us dollars": "USD", "us dollar": "USD", "u.s. dollars": "USD", "u.s. dollar": "USD",
    "american dollars": "USD", "american dollar": "USD", "dollars": "USD", "dollar": "USD",
    "usd": "USD",
    "euros": "EUR", "euro": "EUR", "eur": "EUR",
    "british pounds": "GBP", "british pound": "GBP", "pounds sterling": "GBP",
    "pound sterling": "GBP", "pounds": "GBP", "pound": "GBP", "gbp": "GBP",
    "japanese yen": "JPY", "yen": "JPY", "jpy": "JPY",
    "canadian dollars": "CAD", "canadian dollar": "CAD", "cad": "CAD",
    "australian dollars": "AUD", "australian dollar": "AUD", "aud": "AUD",
    "new zealand dollars": "NZD", "new zealand dollar": "NZD", "nzd": "NZD",
    "singapore dollars": "SGD", "singapore dollar": "SGD", "sgd": "SGD",
    "hong kong dollars": "HKD", "hong kong dollar": "HKD", "hkd": "HKD",
    "swiss francs": "CHF", "swiss franc": "CHF", "chf": "CHF",
    "mexican pesos": "MXN", "mexican peso": "MXN", "mxn": "MXN",
    "chinese yuan": "CNY", "yuan": "CNY", "renminbi": "CNY", "cny": "CNY",
    "indian rupees": "INR", "indian rupee": "INR", "inr": "INR",
    "brazilian reals": "BRL", "brazilian real": "BRL", "brl": "BRL",
    "south korean won": "KRW", "korean won": "KRW", "krw": "KRW",
    "swedish kronor": "SEK", "swedish krona": "SEK", "sek": "SEK",
    "norwegian kroner": "NOK", "norwegian krone": "NOK", "nok": "NOK",
    "danish kroner": "DKK", "danish krone": "DKK", "dkk": "DKK",
}

#: Currency symbol -> ISO code ("$" is USD in question text; options treat "$" loosely).
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

#: ISO codes that are written with a dollar sign.
DOLLAR_CODES = {"USD", "CAD", "AUD", "NZD", "SGD", "HKD"}

#: Regex alternation matching any currency name (longest first).
CUR_RE = "(?:" + "|".join(
    re.escape(k) for k in sorted(CURRENCY_NAMES, key=len, reverse=True)
) + ")"


def currency_code(name: str) -> str | None:
    return CURRENCY_NAMES.get(normalize_text(name).lower())


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2,
    "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4,
    "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_WEEKDAY_RE = "|".join(sorted(WEEKDAYS, key=len, reverse=True))

#: Grammar (no capture groups) for every date format we accept, used inside question
#: regexes.  Semantics (and ambiguity) are decided afterwards by ``parse_date_ex``.
DATE_RE = (
    rf"(?:(?:{_WEEKDAY_RE})\.?,? )?(?:"
    rf"\d{{1,2}}(?:st|nd|rd|th)? (?:{_MONTH_RE})\.?,? \d{{4}}"
    rf"|(?:{_MONTH_RE})\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|\d{{4}}/\d{{1,2}}/\d{{1,2}}"
    rf"|\d{{1,2}}[/.]\d{{1,2}}[/.]\d{{4}})"
)

_RE_WEEKDAY_PREFIX = re.compile(rf"({_WEEKDAY_RE})\.?,? +(.+)", re.I | re.A)
_RE_DMY = re.compile(r"(\d{1,2})(?:st|nd|rd|th)? +([a-z]+)\.?,? +(\d{4})", re.I | re.A)
_RE_MDY = re.compile(r"([a-z]+)\.? +(\d{1,2})(?:st|nd|rd|th)?,? +(\d{4})", re.I | re.A)
_RE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})", re.A)
_RE_YMD_SLASH = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})", re.A)
_RE_NUMERIC = re.compile(r"(\d{1,2})([/.])(\d{1,2})\2(\d{4})", re.A)

_MIN_YEAR, _MAX_YEAR = 1900, 2200


def _make_date(y: int, m: int, d: int) -> dt.date | None:
    if not (_MIN_YEAR <= y <= _MAX_YEAR):
        return None
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def parse_date_ex(text: str) -> tuple[dt.date | None, str]:
    """Parse a date string.

    Returns ``(date, "ok")``, ``(None, "ambiguous")`` for numeric ``a/b/yyyy`` where
    day-first and month-first readings differ, or ``(None, "invalid")``.
    Accepts ``12 May 2026``, ``May 12, 2026``, ``12th May 2026``, ``2026-05-12``,
    ``2026/05/12``, ``12/05/2026`` (only when unambiguous) with an optional leading
    weekday that must agree with the date.
    """
    s = normalize_text(text)
    if not s or len(s) > 60:
        return None, "invalid"
    weekday: int | None = None
    mw = _RE_WEEKDAY_PREFIX.fullmatch(s)
    if mw:
        weekday = WEEKDAYS[mw.group(1).lower()]
        s = mw.group(2)

    result: dt.date | None = None
    status = "invalid"
    m = _RE_DMY.fullmatch(s)
    if m:
        month = MONTHS.get(m.group(2).lower())
        if month:
            result = _make_date(int(m.group(3)), month, int(m.group(1)))
    else:
        m = _RE_MDY.fullmatch(s)
        if m:
            month = MONTHS.get(m.group(1).lower())
            if month:
                result = _make_date(int(m.group(3)), month, int(m.group(2)))
        else:
            m = _RE_ISO.fullmatch(s) or _RE_YMD_SLASH.fullmatch(s)
            if m:
                result = _make_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            else:
                m = _RE_NUMERIC.fullmatch(s)
                if m:
                    a, b, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
                    if a == b:
                        result = _make_date(y, a, b)
                    elif a > 12 and b <= 12:
                        result = _make_date(y, b, a)  # dd/mm/yyyy
                    elif b > 12 and a <= 12:
                        result = _make_date(y, a, b)  # mm/dd/yyyy
                    elif a <= 12 and b <= 12:
                        return None, "ambiguous"
    if result is None:
        return None, status
    if weekday is not None and result.weekday() != weekday:
        return None, "invalid"
    return result, "ok"


def parse_date(text: str) -> dt.date | None:
    d, status = parse_date_ex(text)
    return d if status == "ok" else None


# --------------------------------------------------------------------------
# Option values
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionValue:
    """A multiple-choice option parsed into a typed value.

    kind: money | number | percent | units | date | ambiguous_date | text
    ``decimals`` is the number of decimals *displayed* (drives rounding precision).
    ``currency`` is an ISO code, ``"$"`` for a bare dollar sign, or None.
    """

    kind: str
    value: Decimal | dt.date | None
    decimals: int = 0
    currency: str | None = None
    raw: str = ""


_ON = rf"(?:{NUM}|\.\d+)"  # option number: allow ".5"
_RE_MONEY_SYM = re.compile(
    rf"(?P<s1>[-+])? *(?P<sym>[$€£¥]) *(?P<s2>[-+])? *(?P<num>{_ON})"
    rf"(?: *(?P<code>[a-z]{{3}}))?",
    re.I | re.A,
)
_RE_MONEY_SUFFIX = re.compile(
    rf"(?P<s1>[-+])? *(?P<num>{_ON}) *(?P<cur>[a-z][a-z. ]*)", re.I | re.A
)
_RE_MONEY_CODE_PREFIX = re.compile(
    rf"(?P<code>[a-z]{{3}}) *(?P<s1>[-+])? *(?P<num>{_ON})", re.I | re.A
)
_RE_PERCENT = re.compile(
    rf"(?P<s1>[-+])? *(?P<num>{_ON}) *(?:%|percent|per cent|pct\.?)", re.I | re.A
)
_RE_UNITS = re.compile(
    rf"(?P<s1>[-+])? *(?P<num>{_ON}) +(?P<unit>[a-z][a-z\-.' ]*)", re.I | re.A
)
_RE_NUMBER = re.compile(rf"(?P<s1>[-+])? *(?P<num>{_ON})", re.I | re.A)

_NOT_UNITS = set(MONTHS) | set(WEEKDAYS)


def _signed(sign_chars: list[str | None], num_text: str, negate: bool) -> Decimal | None:
    signs = [c for c in sign_chars if c]
    if len(signs) > 1:
        return None
    try:
        v = to_decimal(num_text)
    except InvalidOperation:
        return None
    explicit_minus = bool(signs) and signs[0] == "-"
    if explicit_minus != negate:
        v = -v
    return v


def parse_option(raw: str) -> OptionValue:
    """Parse one option string into a typed ``OptionValue`` (kind ``text`` if unknown)."""
    t = normalize_text(raw).strip(" \"'")
    text = OptionValue("text", None, 0, None, raw)
    if not t:
        return text

    d, status = parse_date_ex(t)
    if status == "ok":
        return OptionValue("date", d, 0, None, raw)
    if status == "ambiguous":
        return OptionValue("ambiguous_date", None, 0, None, raw)

    negate = False
    body = t
    if body.startswith("(") and body.endswith(")"):
        negate, body = True, body[1:-1].strip()

    m = _RE_MONEY_SYM.fullmatch(body)
    if m:
        code = m.group("code")
        sym = m.group("sym")
        cur: str | None = "$" if sym == "$" else CURRENCY_SYMBOLS[sym]
        if code:
            iso = code.upper()
            if iso not in CURRENCY_NAMES.values():
                return text
            if sym == "$" and iso not in DOLLAR_CODES:
                return text
            if sym != "$" and CURRENCY_SYMBOLS[sym] != iso:
                return text
            cur = iso
        v = _signed([m.group("s1"), m.group("s2")], m.group("num"), negate)
        if v is None:
            return text
        return OptionValue("money", v, count_decimals(m.group("num")), cur, raw)

    m = _RE_MONEY_CODE_PREFIX.fullmatch(body)
    if m:
        iso = m.group("code").upper()
        if iso in CURRENCY_NAMES.values():
            v = _signed([m.group("s1")], m.group("num"), negate)
            if v is None:
                return text
            return OptionValue("money", v, count_decimals(m.group("num")), iso, raw)

    m = _RE_PERCENT.fullmatch(body)
    if m:
        v = _signed([m.group("s1")], m.group("num"), negate)
        if v is None:
            return text
        return OptionValue("percent", v, count_decimals(m.group("num")), None, raw)

    m = _RE_MONEY_SUFFIX.fullmatch(body)
    if m:
        iso = CURRENCY_NAMES.get(m.group("cur").strip().lower())
        if iso:
            v = _signed([m.group("s1")], m.group("num"), negate)
            if v is None:
                return text
            return OptionValue("money", v, count_decimals(m.group("num")), iso, raw)

    m = _RE_UNITS.fullmatch(body)
    if m:
        unit = m.group("unit").strip().lower().rstrip(".")
        if unit in _NOT_UNITS or unit in CURRENCY_NAMES:
            return text
        v = _signed([m.group("s1")], m.group("num"), negate)
        if v is None:
            return text
        return OptionValue("units", v, count_decimals(m.group("num")), None, raw)

    m = _RE_NUMBER.fullmatch(body)
    if m:
        v = _signed([m.group("s1")], m.group("num"), negate)
        if v is None:
            return text
        return OptionValue("number", v, count_decimals(m.group("num")), None, raw)

    return text
