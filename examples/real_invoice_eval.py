"""THE honest evaluation. Runs the real Ollama extractor over real invoice-OCR
documents (mychen76/invoices-and-receipts_ocr_v1), fits the calibrator on the real
calibration split, and measures the risk-coverage curve on the untouched real test
split. Neither the OCR noise nor the model's errors are authored by us, so the
numbers are honest.

    pip install -e ".[real,eval]"
    python examples/real_invoice_eval.py --limit 200 --consistency-n 3
    python examples/real_invoice_eval.py --limit 120 --consistency-n 3 --two-temp   # + ablation
    python examples/analyze_saved.py     # re-analyze thresholds later, no re-extraction

--consistency-n N adds self-consistency (N samples). --two-temp adds the two-temperature
stability signal. With >1 signal, a per-signal ablation prints (does each earn its keep?).
Both multiply run time. --offline sanity-checks plumbing without Ollama (degenerate).
"""
import argparse
import json

from gatekeeper import invoice_schema
from gatekeeper.data import RealInvoiceLoader, split_examples
from gatekeeper.extract import LLMExtractor, OllamaClient, FakeClient
from gatekeeper.signals import (
    GroundingSignal, RuleSignal, SelfConsistencySignal, TwoTemperatureSignal,
)
from gatekeeper.calibration import (
    LogisticCalibrator, build_training_data, expected_calibration_error,
)
from gatekeeper.evaluation import (
    collect_risk_labels, selective_error_at_coverage, aurc, ablate_signals,
)
from gatekeeper.policy import provable_risk_controlled_threshold

_CANNED = '{"invoice_number":"X","invoice_date":"2012-10-15","total_amount":"66","vendor_name":"Acme"}'


def _run(extractor, signal_gens, schema, examples):
    records, gts = [], []
    for ex in examples:
        rec = extractor.extract(ex.to_record(), schema)
        for gen in signal_gens:
            gen.generate(rec, schema)
        records.append(rec)
        gts.append(ex.ground_truth)
    return records, gts


def coverage_at_precision(risks, errors, target):
    best = 0.0
    for cov in [i / 100 for i in range(5, 101, 5)]:
        if 1 - selective_error_at_coverage(risks, errors, cov) >= target:
            best = cov
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--consistency-n", type=int, default=0, help="self-consistency samples (0=off)")
    p.add_argument("--two-temp", action="store_true", help="add the two-temperature stability signal")
    p.add_argument("--dump-records", metavar="PATH",
                   help="save gated test records (values + real risks) to JSON for the demo")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    schema = invoice_schema()

    def make_extractor(temp=None):
        if args.offline:
            return LLMExtractor(FakeClient(_CANNED))
        return LLMExtractor(OllamaClient(model=args.model)) if temp is None \
            else LLMExtractor(OllamaClient(model=args.model), temperature=temp)

    extractor = make_extractor()
    signal_gens = [GroundingSignal(), RuleSignal()]
    if args.consistency_n > 0:
        signal_gens.insert(0, SelfConsistencySignal(make_extractor(0.7), n=args.consistency_n))
    if args.two_temp:
        signal_gens.insert(0, TwoTemperatureSignal(make_extractor(0.2), make_extractor(0.8), n=1))
    extra = sum([args.consistency_n if args.consistency_n else 0, 2 if args.two_temp else 0])
    if extra:
        print(f"extra sampling on: extraction runs ~{extra + 1}x, this will be slow.\n")

    print(f"loading up to {args.limit} real invoices...")
    examples = RealInvoiceLoader(limit=args.limit).load()
    splits = split_examples(examples)
    print(f"loaded {len(examples)}  ->  {splits.summary()}\n")

    print("extracting + fitting calibrator on the real calibration split...")
    cal_recs, cal_gts = _run(extractor, signal_gens, schema, splits.calibration)
    X, y = build_training_data(cal_recs, cal_gts, schema)
    calibrator = LogisticCalibrator().fit(X, y)
    print(f"  fit on {len(X)} fields ({sum(y)} errors, {sum(y)/max(len(y),1):.0%} error rate)\n")

    print("extracting + gating the real TEST split...")
    test_recs, test_gts = _run(extractor, signal_gens, schema, splits.test)
    for rec in test_recs:
        calibrator.calibrate(rec, schema)
    risks, errors = collect_risk_labels(test_recs, test_gts, schema)

    with open("real_eval_risks.json", "w") as fh:
        json.dump({"risks": risks, "errors": errors}, fh)

    if args.dump_records:
        from gatekeeper import is_correct
        dump = []
        for rec, gt in zip(test_recs, test_gts):
            fields = {}
            for spec in schema.fields:
                f = rec.fields[spec.name]
                fields[spec.name] = {
                    "value": f.value,
                    "risk": round(f.risk, 4) if f.risk is not None else None,
                    "gold": gt.get(spec.name),
                    "correct": is_correct(f.value, gt.get(spec.name), spec.dtype),
                }
            dump.append({"id": rec.id, "fields": fields})
        with open(args.dump_records, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=2)
        print(f"\ndumped {len(dump)} gated records to {args.dump_records}")

    print("\n=== HONEST RESULTS (real invoice OCR, unauthored errors) ===")
    print(f"test fields: {len(errors)}, base error rate: {sum(errors)/max(len(errors),1):.1%}")
    print(f"AURC: {aurc(risks, errors):.3f}   ECE: {expected_calibration_error(risks, errors):.3f}")
    for cov in (0.5, 0.7, 0.9):
        e = selective_error_at_coverage(risks, errors, cov)
        print(f"  at {cov:.0%} coverage -> {e:.1%} error ({1 - e:.1%} precision)")
    print(f"  coverage at >=97% precision: {coverage_at_precision(risks, errors, 0.97):.0%}")

    print("\nprovable operating points (guarantee holds with probability >= 90%):")
    for alpha in (0.10, 0.25, 0.40):
        t = provable_risk_controlled_threshold(risks, errors, alpha=alpha, delta=0.1)
        appr = [(r, e) for r, e in zip(risks, errors) if r <= t]
        cov = len(appr) / len(risks)
        act = (sum(e for _, e in appr) / len(appr)) if appr else 0.0
        print(f"  guarantee error <= {alpha:.0%}  ->  auto-approve {cov:.0%} (actual {act:.0%})")

    if len(signal_gens) > 1:
        print("\nper-signal ablation (AURC when a signal is removed; higher = it mattered):")
        X_test, y_test = build_training_data(test_recs, test_gts, schema)
        names = [g.name for g in signal_gens]
        res = ablate_signals(X, y, X_test, y_test, names)
        print(f"  full model: {res['full']:.3f}")
        for nm in names:
            print(f"  without {nm:16}: {res['without_' + nm]:.3f}  (+{res['without_' + nm] - res['full']:.3f})")

    try:
        from gatekeeper.evaluation import plot_risk_coverage, plot_reliability
        plot_risk_coverage(risks, errors, "real_risk_coverage.png")
        plot_reliability(risks, errors, "real_reliability.png")
        print("\nplots saved: real_risk_coverage.png, real_reliability.png")
    except ImportError:
        print("\n(install matplotlib for plots)")


if __name__ == "__main__":
    main()