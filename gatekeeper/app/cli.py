"""The reference-app CLI: fit (or load) the calibrator, run documents through the
gated pipeline, and print / save a report. Defaults to a live Ollama extractor;
--offline uses a canned extractor so it runs (and is testable) without Ollama.

    python -m gatekeeper.app                 # gate the built-in fixtures (needs Ollama)
    python -m gatekeeper.app doc.txt --html report.html
    python -m gatekeeper.app --offline       # no Ollama; canned extraction (smoke)
"""
from __future__ import annotations

import argparse
import os

from ..schema import Schema
from ..types import Record
from ..data import FixtureInvoiceLoader, invoice_schema, split_examples
from ..calibration import build_training_data, LogisticCalibrator
from ..extract import LLMExtractor, OllamaClient, FakeClient
from ..signals import GroundingSignal, RuleSignal
from ..policy import CostAwarePolicy
from .report import build_report, render_text, render_html
from .persistence import save_calibrator, load_calibrator

_CANNED = ('{"invoice_number":"INV-2025-0088","invoice_date":"2025-11-05",'
           '"total_amount":"2500","vendor_name":"Blue Ridge Logistics LLC"}')


def _extractor(offline: bool, model: str):
    client = FakeClient(_CANNED) if offline else OllamaClient(model=model)
    return LLMExtractor(client)


def _signals():
    return [GroundingSignal(), RuleSignal()]


def fit_calibrator(extractor, signal_generators, schema: Schema, examples):
    records, gts = [], []
    for ex in examples:
        rec = extractor.extract(ex.to_record(), schema)
        for gen in signal_generators:
            gen.generate(rec, schema)
        records.append(rec)
        gts.append(ex.ground_truth)
    X, y = build_training_data(records, gts, schema)
    return LogisticCalibrator().fit(X, y)


def _load_inputs(path, schema):
    if path is None:                                  # default: gate a few fixtures
        return [ex.to_record() for ex in FixtureInvoiceLoader().load()[:4]]
    records = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".txt"):
                with open(os.path.join(path, name)) as fh:
                    records.append(Record(id=name, source_text=fh.read()))
    else:
        with open(path) as fh:
            records.append(Record(id=os.path.basename(path), source_text=fh.read()))
    return records


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gatekeeper", description="gate documents by extraction confidence")
    p.add_argument("input", nargs="?", help="a .txt file, a directory of .txt, or omit for fixtures")
    p.add_argument("--offline", action="store_true", help="canned extractor, no Ollama")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--review-cost", type=float, default=1.0)
    p.add_argument("--html", metavar="PATH", help="also write an HTML report")
    p.add_argument("--load-calibrator", metavar="PATH")
    p.add_argument("--save-calibrator", metavar="PATH")
    args = p.parse_args(argv)

    schema = invoice_schema()
    extractor = _extractor(args.offline, args.model)
    signal_generators = _signals()

    if args.load_calibrator:
        calibrator = load_calibrator(args.load_calibrator)
    else:
        cal_split = split_examples(FixtureInvoiceLoader().load()).calibration
        calibrator = fit_calibrator(extractor, signal_generators, schema, cal_split)
    if args.save_calibrator:
        save_calibrator(calibrator, args.save_calibrator)

    policy = CostAwarePolicy(review_cost=args.review_cost)

    results = []
    for rec in _load_inputs(args.input, schema):
        rec = extractor.extract(rec, schema)
        for gen in signal_generators:
            gen.generate(rec, schema)
        calibrator.calibrate(rec, schema)
        policy.decide(rec, schema)
        results.append(rec)

    report = build_report(results, schema)
    print(render_text(report))
    if args.html:
        with open(args.html, "w") as fh:
            fh.write(render_html(report))
        print(f"\nHTML report written to {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())