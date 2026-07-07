"""Field-type-aware correctness: does an extracted value match the gold answer?

Naive string equality is wrong here -- "1240", "$1,240.00" and "1,240.00" are
the same number. Too strict and we mark good extractions as errors and poison
the calibrator; too loose and we miss real errors. So we normalize per field
type before comparing. Field dtypes:
  number : compared numerically (currency symbols / commas stripped)
  date   : parsed to a canonical date, many input formats accepted
  exact  : normalized exact match only (identifiers, codes -- no fuzz)
  str    : normalized, then fuzzy (names, free text)
"""
from __future__ import annotations

import re
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any, Optional


def _norm_str(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower())


def _norm_number(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[^0-9.,\-]", "", str(v))          # keep digits, separators, sign
    if not s or s in ("-", ".", ",", "-.", "-,"):
        return None
    # decide which separator is the decimal point
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):            # EU: 1.234,56 -> comma is decimal
            s = s.replace(".", "").replace(",", ".")
        else:                                       # US: 1,234.56 -> comma is thousands
            s = s.replace(",", "")
    elif "," in s:
        if s.count(",") == 1 and re.search(r",\d{2}$", s):   # 66,00 -> decimal
            s = s.replace(",", ".")
        else:                                                 # 1,240 -> thousands
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    "%d-%m-%y", "%d/%m/%y", "%m/%d/%y", "%m-%d-%y",     # two-digit years (real receipts)
]


def _norm_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _presence_verdict(predicted: Any, gold: Any) -> Optional[bool]:
    """Decide the absent-field cases shared by every type. Returns True/False if
    presence alone settles it, else None (both present -> compare by type)."""
    pred_empty = predicted is None or (isinstance(predicted, str) and not predicted.strip())
    gold_empty = gold is None or (isinstance(gold, str) and not gold.strip())
    if gold_empty and pred_empty:
        return True                    # correctly produced nothing
    if gold_empty and not pred_empty:
        return False                   # hallucinated a value for an absent field
    if pred_empty and not gold_empty:
        return False                   # missed a real value
    return None                        # both present


def is_correct(predicted: Any, gold: Any, dtype: str = "str",
               *, fuzzy_threshold: float = 0.9) -> bool:
    verdict = _presence_verdict(predicted, gold)
    if verdict is not None:
        return verdict

    if dtype == "number":
        p, g = _norm_number(predicted), _norm_number(gold)
        return p is not None and g is not None and abs(p - g) < 1e-6
    if dtype == "date":
        p, g = _norm_date(predicted), _norm_date(gold)
        return p is not None and g is not None and p == g

    ps, gs = _norm_str(predicted), _norm_str(gold)
    if ps == gs:
        return True
    if dtype == "exact":
        return False
    return SequenceMatcher(None, ps, gs).ratio() >= fuzzy_threshold


# Public normalization helpers, reused by the §3 signal generators so all
# normalization logic lives in one place.
def to_number(v: Any) -> Optional[float]:
    return _norm_number(v)


def to_date(v: Any):
    return _norm_date(v)