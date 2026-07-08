"""THE honest evaluation. Runs the real Ollama extractor over real invoice-OCR
documents (mychen76/invoices-and-receipts_ocr_v1), fits the calibrator on the real
calibration split, and measures the risk-coverage curve on the untouched real test
split. Neither the OCR noise nor the model's errors are authored by us, so these
numbers are honest -- comparable to production trust-gate reports.

Needs Ollama running and the datasets library:
    pip install -e ".[real,eval]"     (datasets + matplotlib)
    python examples/real_invoice_eval.py --limit 120

Note: downloads ~282MB the first time and runs the model over the split (a few
minutes). Use --offline to sanity-check plumbing without Ollama (degenerate).
"""
import argparse

from gatekeeper import invoice_schema, is_correct
from gatekeeper.data import RealInvoiceLoader, split_examples
from gatekeeper.extract import LLMExtractor, OllamaClient, FakeClient
from gatekeeper.signals import GroundingSignal, RuleSignal, SelfConsistencySignal
from gatekeeper.calibration import (
    LogisticCalibrator, build_training_data, expected_calibration_error,
)
from gatekeeper.evaluation import (
    collect_risk_labels, selective_error_at_coverage, aurc,
)


def _run(extractor, signal_gens, schema, examples):
    records, gts = [], []
    for ex in examples:
        rec = extractor.extract(ex.to_record(), schema)
        for gen in signal_gens:
            gen.generate(rec, schema)
        records.append(rec)
        gts.append(ex.ground_truth)
    return records, gts


def coverage_at_precision(risks, errors, target_precision):
    """Largest auto-approval coverage whose precision (1 - error) >= target."""
    best = 0.0
    for cov in [i / 100 for i in range(5, 101, 5)]:
        if 1 - selective_error_at_coverage(risks, errors, cov) >= target_precision:
            best = cov
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--consistency-n", type=int, default=0,
                   help="add self-consistency with N samples (0 = off). Multiplies run time by N+1.")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    schema = invoice_schema()
    extractor = (LLMExtractor(FakeClient('{"invoice_number":"X","invoice_date":"2012-10-15","total_amount":"66","vendor_name":"Acme"}'))
                 if args.offline else LLMExtractor(OllamaClient(model=args.model)))
    signal_gens = [GroundingSignal(), RuleSignal()]
    if args.consistency_n > 0:
        sampler = (extractor if args.offline
                   else LLMExtractor(OllamaClient(model=args.model), temperature=0.7))
        signal_gens.insert(0, SelfConsistencySignal(sampler, n=args.consistency_n))
        print(f"self-consistency ON (n={args.consistency_n}) -- extraction runs "
              f"{args.consistency_n + 1}x, this will be slow.\n")

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

    import json
    with open("real_eval_risks.json", "w") as fh:
        json.dump({"risks": risks, "errors": errors}, fh)

    print("\n=== HONEST RESULTS (real invoice OCR, unauthored errors) ===")
    print(f"test fields: {len(errors)}, base error rate: {sum(errors)/max(len(errors),1):.1%}")
    print(f"AURC: {aurc(risks, errors):.3f}   ECE: {expected_calibration_error(risks, errors):.3f}")
    for cov in (0.5, 0.7, 0.9):
        print(f"  at {cov:.0%} coverage -> {selective_error_at_coverage(risks, errors, cov):.1%} error "
              f"({1 - selective_error_at_coverage(risks, errors, cov):.1%} precision)")
    cov97 = coverage_at_precision(risks, errors, 0.97)
    print(f"  coverage at >=97% precision: {cov97:.0%}   (compare to production trust-gate reports)")

    from gatekeeper.policy import provable_risk_controlled_threshold
    print("\nprovable operating points (guarantee holds with probability >= 90%):")
    for alpha in (0.10, 0.25, 0.40):
        t = provable_risk_controlled_threshold(risks, errors, alpha=alpha, delta=0.1)
        approved = [(r, e) for r, e in zip(risks, errors) if r <= t]
        cov = len(approved) / len(risks)
        actual = (sum(e for _, e in approved) / len(approved)) if approved else 0.0
        print(f"  guarantee error <= {alpha:.0%}  ->  auto-approve {cov:.0%}  "
              f"(actual error on approved: {actual:.0%})")

    try:
        from gatekeeper.evaluation import plot_risk_coverage, plot_reliability
        plot_risk_coverage(risks, errors, "real_risk_coverage.png")
        plot_reliability(risks, errors, "real_reliability.png")
        print("\nplots saved: real_risk_coverage.png, real_reliability.png")
    except ImportError:
        print("\n(install matplotlib for plots: pip install matplotlib)")


if __name__ == "__main__":
    main()