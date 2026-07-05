"""Per-signal ablation: drop one signal, refit the calibrator on the calibration
split without it, remeasure on test, and see how much the risk-coverage area
(AURC) degrades. A signal that earns its keep makes the metric noticeably worse
when removed; a redundant or noisy one barely moves it -- an ablation study in the
diff-in-diff spirit, applied to confidence signals (answers 'is self-consistency's
N-call cost justified?')."""
from __future__ import annotations

from ..calibration import LogisticCalibrator
from .curves import aurc


def _subset(feature_dicts, keep):
    return [{k: v for k, v in d.items() if k in keep} for d in feature_dicts]


def ablate_signals(X_cal, y_cal, X_test, y_test, signal_names, *,
                   metric=aurc, calibrator_factory=LogisticCalibrator):
    """Returns {'full': m, 'without_<signal>': m, ...}. Lower metric = better; a
    larger value for 'without_s' than 'full' means signal s was helping."""
    names = list(signal_names)

    def evaluate(keep):
        cal = calibrator_factory().fit(_subset(X_cal, keep), y_cal)
        risks = [cal.predict_proba(x) for x in _subset(X_test, keep)]
        return metric(risks, y_test)

    results = {"full": evaluate(set(names))}
    for s in names:
        results[f"without_{s}"] = evaluate(set(names) - {s})
    return results