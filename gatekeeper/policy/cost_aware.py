"""The cost-aware decision policy. Each field's escalation threshold is derived
from economics rather than guessed:

    threshold = review_cost / error_cost   (clamped to [0, 1])

Escalate a field when its calibrated risk exceeds that threshold -- so a costlier
field escalates at a lower risk. The record escalates if any CRITICAL field
escalates (a low-confidence critical field routes the whole document to a human).
"""
from __future__ import annotations

from ..types import Record, Decision, Action
from ..schema import Schema
from ..interfaces import Policy


class CostAwarePolicy(Policy):
    name = "cost_aware"

    def __init__(self, *, review_cost: float = 1.0):
        self.review_cost = review_cost

    def field_threshold(self, error_cost: float) -> float:
        if error_cost <= 0:
            return 1.0                      # no error cost -> never escalates on its own
        return min(1.0, self.review_cost / error_cost)

    def decide(self, record: Record, schema: Schema) -> Record:
        escalate_record = False
        for spec in schema.fields:
            field = record.fields[spec.name]
            risk = field.risk if field.risk is not None else 1.0
            threshold = self.field_threshold(spec.error_cost)
            action = Action.ESCALATE if risk > threshold else Action.APPROVE
            field.decision = Decision(
                action=action,
                reason=(f"risk {risk:.2f} vs threshold {threshold:.2f} "
                        f"(review {self.review_cost:g} / error_cost {spec.error_cost:g})"),
                threshold=threshold,
                risk=risk,
            )
            if action is Action.ESCALATE and spec.critical:
                escalate_record = True
        record.decision = Decision(
            action=Action.ESCALATE if escalate_record else Action.APPROVE,
            reason=("a critical field was escalated" if escalate_record
                    else "all critical fields approved"),
        )
        return record