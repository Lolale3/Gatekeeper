# gatekeeper

A confidence-gating / selective-escalation layer for LLM extraction: given an
extracted output, decide -- unattended -- whether to trust it or route it to a
human. This is the escalation-decision layer, not another eval framework; it is
designed to sit on top of your existing extraction and observability.

## Status: §0 -- foundations (walking skeleton)

The data model, the four swappable contracts (`Extractor`, `SignalGenerator`,
`Calibrator`, `Policy`), and a dumb orchestrator are in place, wired with stub
components so a `Record` can travel the entire pipeline end to end. Real
components replace the stubs one section at a time.

## Try the skeleton

    pip install -e .
    python examples/walking_skeleton.py
    pytest -q