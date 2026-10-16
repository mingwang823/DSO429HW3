"""Computed value -> option letter.

The solver computes a typed value (money, percent, number, units, date).  This module
parses every option into a typed value and picks the one option that equals the computed
value *at the precision the option displays*.  If that is not possible without
guessing (no option matches, or more than one does) it abstains.

Algorithm (numbers):
  1. Round the computed value half-up (Decimal) to d places, where d is the number of
     decimals displayed in the option.  Options equal to that rounding -> if exactly
     one, choose it.
  2. Otherwise options within ``0.5 * 10**-d * 1.001 + 1e-9 * max(1, |v|)`` of the
     unrounded value -> if exactly one, choose it (method flagged as tolerance).
  3. Otherwise abstain (``no_option_match`` or ``ambiguous``).
Only same-type options are compared.  Dates must be equal.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Any

from .parsing import DOLLAR_CODES, OptionValue, parse_option

#: kinds an option may have when no option of the primary kind exists
FALLBACK_KINDS: dict[str, tuple[str, ...]] = {
    "money": ("number",),
    "number": ("money", "units"),
    "units": ("number",),
    "percent": (),
    "date": (),
}


@dataclass(frozen=True)
class Computed:
    """A value computed by a family solver.

    ``value`` is a Decimal / Fraction / int / float (numeric kinds) or a ``datetime.date``.
    ``kind`` is one of money, number, percent, units, date.  Percent values are in
    percent units (10.4 means 10.4%).  ``currency`` is an ISO code for money, if known.
    """

    value: Any
    kind: str
    currency: str | None = None
    note: str = ""


@dataclass(frozen=True)
class MatchResult:
    letter: str | None
    #: exact | tolerance | no_option_match | ambiguous | no_options | bad_computed
    status: str
    detail: str = ""


def to_decimal(value: Any) -> Decimal:
    """Convert int / float / Decimal / Fraction to Decimal; floats are cleaned via a
    10-place rounded string repr so 2.675 stays 2.675 (not 2.67499999...)."""
    with localcontext() as ctx:
        ctx.prec = 80
        if isinstance(value, Decimal):
            d = value
        elif isinstance(value, bool):
            raise TypeError("bool is not a numeric value")
        elif isinstance(value, int):
            d = Decimal(value)
        elif isinstance(value, Fraction):
            d = Decimal(value.numerator) / Decimal(value.denominator)
        elif isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("non-finite float")
            d = Decimal(repr(round(value, 10)))
        else:
            raise TypeError(f"unsupported value type {type(value).__name__}")
        if not d.is_finite():
            raise ValueError("non-finite value")
        return d.quantize(Decimal("1e-10"), rounding=ROUND_HALF_UP)


def format_value(computed: Computed) -> str:
    """Short human-readable rendering for ``Solution.detail``."""
    try:
        if computed.kind == "date":
            return computed.value.isoformat()
        d = to_decimal(computed.value)
        text = format(d.normalize(), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        suffix = {"percent": "%", "money": f" {computed.currency or ''}".rstrip()}.get(
            computed.kind, ""
        )
        return f"{text}{suffix}"
    except Exception:  # noqa: BLE001 - detail text must never fail
        return str(computed.value)


def _currency_compatible(want: str | None, have: str | None) -> bool:
    if want is None or have is None:
        return True
    if want == have:
        return True
    if have == "$" and want in DOLLAR_CODES:
        return True
    if want == "$" and have in DOLLAR_CODES:
        return True
    return False


def _round_half_up(v: Decimal, places: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 80
        return v.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def match_options(computed: Computed, options: dict[str, str]) -> MatchResult:
    """Pick the unique option matching ``computed`` or abstain."""
    if not options:
        return MatchResult(None, "no_options", "no options supplied")
    parsed = {letter: parse_option(text) for letter, text in sorted(options.items())}
    shown = format_value(computed)

    if computed.kind == "date":
        return _match_dates(computed, parsed, shown)
    try:
        v = to_decimal(computed.value)
    except (TypeError, ValueError, InvalidOperation) as exc:
        return MatchResult(None, "bad_computed", f"cannot use computed value: {exc}")

    primary = computed.kind
    usable = {k: o for k, o in parsed.items() if o.kind == primary}
    if not usable:
        fallbacks = FALLBACK_KINDS.get(primary, ())
        usable = {k: o for k, o in parsed.items() if o.kind in fallbacks}
    if primary in ("money",):
        usable = {
            k: o for k, o in usable.items()
            if o.kind != "money" or _currency_compatible(computed.currency, o.currency)
        }
    if not usable:
        return MatchResult(
            None, "no_option_match", f"computed {shown}; no option of type {primary}"
        )

    # step 1: equal after rounding to the option's own displayed precision
    exact = [
        k for k, o in usable.items() if _round_half_up(v, o.decimals) == o.value
    ]
    if len(exact) == 1:
        return MatchResult(exact[0], "exact", f"computed {shown}")
    if len(exact) > 1:
        return MatchResult(
            None, "ambiguous", f"computed {shown}; options {','.join(exact)} all match"
        )

    # step 2: within half a unit of the displayed precision (rounding-convention slack)
    near = []
    for k, o in usable.items():
        with localcontext() as ctx:
            ctx.prec = 80
            tol = (
                Decimal("0.5") * Decimal(10) ** (-o.decimals) * Decimal("1.001")
                + Decimal("1e-9") * max(Decimal(1), abs(v))
            )
            if abs(o.value - v) <= tol:
                near.append(k)
    if len(near) == 1:
        return MatchResult(near[0], "tolerance", f"computed {shown} (within rounding slack)")
    if len(near) > 1:
        return MatchResult(
            None, "ambiguous", f"computed {shown}; options {','.join(near)} all within slack"
        )
    return MatchResult(None, "no_option_match", f"computed {shown}; no option matches")


def _match_dates(
    computed: Computed, parsed: dict[str, OptionValue], shown: str
) -> MatchResult:
    target = computed.value
    if not isinstance(target, dt.date):
        return MatchResult(None, "bad_computed", "computed value is not a date")
    hits = [k for k, o in parsed.items() if o.kind == "date" and o.value == target]
    if len(hits) == 1:
        return MatchResult(hits[0], "exact", f"computed {shown}")
    if len(hits) > 1:
        return MatchResult(
            None, "ambiguous", f"computed {shown}; options {','.join(hits)} all match"
        )
    return MatchResult(None, "no_option_match", f"computed {shown}; no option matches")
