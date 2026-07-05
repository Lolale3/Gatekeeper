"""Optional plotting (needs matplotlib: pip install 'gatekeeper[eval]'). Renders
the risk-coverage curve and the reliability diagram to PNG. Importing this module
does not require matplotlib -- the import happens lazily inside each function, so
the rest of the package stays dependency-free."""
from __future__ import annotations

from .curves import risk_coverage_curve
from ..calibration import reliability_table


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("plotting needs matplotlib: pip install 'gatekeeper[eval]'") from e


def plot_risk_coverage(risks, errors, path: str) -> str:
    plt = _plt()
    points = risk_coverage_curve(risks, errors)
    xs = [c for c, _ in points]
    ys = [e for _, e in points]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, marker="o", ms=3, color="#534AB7")
    ax.set_xlabel("coverage (fraction auto-approved)")
    ax.set_ylabel("error rate among auto-approved")
    ax.set_title("risk-coverage curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_reliability(predictions, labels, path: str, *, n_bins: int = 10) -> str:
    plt = _plt()
    rows = [r for r in reliability_table(predictions, labels, n_bins) if r["count"]]
    xs = [r["mean_pred"] for r in rows]
    ys = [r["actual"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="#888780", label="perfect")
    ax.plot(xs, ys, marker="o", ms=4, color="#1D9E75", label="calibrated")
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("actual error rate")
    ax.set_title("reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path