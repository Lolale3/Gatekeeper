"""gatekeeper.app -- §8: the reference app. A CLI that gates documents through the
full pipeline and renders a text or standalone-HTML report."""
from .report import build_report, render_text, render_html
from .persistence import save_calibrator, load_calibrator

__all__ = ["build_report", "render_text", "render_html", "save_calibrator", "load_calibrator"]