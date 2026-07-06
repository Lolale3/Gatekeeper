"""Synthetic freight-email dataset (§1b, the demo path). Real freight tenders arrive
as messy broker prose -- values buried in human language, varied phrasing, informal
formats, ambiguity -- which is exactly where an LLM is necessary and regex is not.

We generate emails with real variety and KNOWN ground truth, so a real model makes
real mistakes on them, giving the calibrator a real error distribution (unlike the
regex-easy invoice fixtures). Implements the same DatasetLoader interface, so the
whole pipeline runs on it unchanged -- only the schema swaps.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from ..schema import Schema, FieldSpec
from .example import GroundTruth, LabeledExample
from .loader import DatasetLoader
from .matching import to_number, to_date

EQUIPMENT = ["dry van", "reefer", "flatbed", "step deck"]


def _positive_rate(value, fields=None) -> bool:
    if value is None:
        return True
    v = to_number(value)
    return v is not None and 100 <= v <= 20000


def _plausible_weight(value, fields=None) -> bool:
    if value is None:
        return True
    v = to_number(value)
    return v is not None and 0 < v <= 48000            # ~legal max gross


def _known_equipment(value, fields=None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in EQUIPMENT


def _plausible_year(value, fields=None) -> bool:
    if value is None:
        return True
    d = to_date(value)
    return d is None or (2024 <= d.year <= 2030)


def freight_schema() -> Schema:
    return Schema(doc_type="freight_load", fields=[
        FieldSpec("origin",      dtype="str",    critical=True,  error_cost=3.0),
        FieldSpec("destination", dtype="str",    critical=True,  error_cost=3.0),
        FieldSpec("weight",      dtype="number", critical=False, error_cost=2.0, rules=[_plausible_weight]),
        FieldSpec("rate",        dtype="number", critical=True,  error_cost=5.0, rules=[_positive_rate]),
        FieldSpec("pickup_date", dtype="date",   critical=False, error_cost=1.0, rules=[_plausible_year]),
        FieldSpec("equipment",   dtype="str",    critical=False, error_cost=1.0, rules=[_known_equipment]),
    ])


_LANES = [  # (origin city, nearby metro)
    ("Grand Prairie", "Dallas"), ("Ontario", "Inland Empire"), ("Joliet", "Chicago"),
    ("Plainfield", "Indianapolis"), ("Carlisle", "Harrisburg"), ("Byhalia", "Memphis"),
    ("Fontana", "Riverside"), ("Groveport", "Columbus"),
]
_DESTS = ["Memphis, TN", "Atlanta, GA", "Columbus, OH", "Phoenix, AZ",
          "Savannah, GA", "Laredo, TX", "Kansas City, MO", "Charlotte, NC"]


def _fmt_weight(rng, w):
    return rng.choice([f"{w // 1000}k lbs", f"{w:,} lbs", f"~{w} pounds", f"{w // 1000},000 lbs"])


def _fmt_rate(rng, r):
    return rng.choice([f"${r:,}", f"${r:,} all-in", f"${r:,}, flat", f"${r / 1000:.1f}k"])


def _fmt_date(rng, d):
    return rng.choice([f"{d.month}/{d.day}", d.strftime("%B %d"),
                       d.strftime("%m/%d"), d.strftime("%b %d")])


def _templates(v):
    o, metro, dest = v["origin"], v["metro"], v["destination"]
    eq, wt, rt, dt = v["equipment"], v["weight_txt"], v["rate_txt"], v["date_txt"]
    return [
        f"Hey -- got a {eq} load we need covered. About {wt} of palletized freight, "
        f"out of the {metro} area (specifically {o}) heading to {dest}. Looking at {rt}, "
        f"pickup {dt}. LMK if you can cover it.",
        f"{o} -> {dest} | {eq} | {wt} | {dt} | {rt}. Truck available?",
        f"We have a shipment tendering from {o} to {dest}. Equipment: {eq}. "
        f"Weight approx {wt}. Target rate {rt}. Pickup requested {dt}.",
        f"need a truck asap -- {wt} {eq}, {metro} area ({o}) down to {dest}. "
        f"can do {rt}, might flex a little. gotta pick up by {dt}.",
        f"Hi team, following up on capacity. Load: {o} to {dest}, {eq}, {wt}. "
        f"Rate {rt}. PU {dt}. Thanks!",
    ]


class SyntheticFreightLoader(DatasetLoader):
    name = "synthetic_freight"

    def __init__(self, n: int = 12, seed: int = 7):
        self.n = n
        self.seed = seed

    def load(self):
        rng = random.Random(self.seed)
        base = date(2026, 3, 2)
        out = []
        for i in range(self.n):
            origin, metro = rng.choice(_LANES)
            dest = rng.choice(_DESTS)
            weight = rng.choice([26000, 30000, 38000, 42000, 44000, 45000])
            rate = rng.choice([950, 1600, 1850, 2400, 2750, 3100, 3400])
            d = base + timedelta(days=rng.randint(3, 40))
            equip = rng.choice(EQUIPMENT)
            v = {
                "origin": origin, "metro": metro, "destination": dest, "equipment": equip,
                "weight_txt": _fmt_weight(rng, weight),
                "rate_txt": _fmt_rate(rng, rate),
                "date_txt": _fmt_date(rng, d),
            }
            text = rng.choice(_templates(v))
            gold = GroundTruth(values={
                "origin": origin,
                "destination": dest,
                "weight": str(weight),
                "rate": str(rate),
                "pickup_date": d.isoformat(),
                "equipment": equip,
            })
            out.append(LabeledExample(id=f"load-{i + 1:03d}", source_text=text, ground_truth=gold))
        return out