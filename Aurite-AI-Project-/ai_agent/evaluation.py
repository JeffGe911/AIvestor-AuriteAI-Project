"""Deterministic evaluation gates for AIvestor agent outputs.

These checks are intentionally model-independent. They make it possible to measure
agent reliability in CI and during experiments instead of judging outputs only by
whether they look plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

from .contracts import StockSignal


@dataclass(frozen=True)
class EvalSummary:
    total: int
    schema_valid: int
    evidence_backed: int
    with_warnings: int
    contradictory_pairs: int

    @property
    def schema_valid_rate(self) -> float:
        return self.schema_valid / self.total if self.total else 1.0

    @property
    def evidence_coverage_rate(self) -> float:
        return self.evidence_backed / self.total if self.total else 1.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["schema_valid_rate"] = self.schema_valid_rate
        result["evidence_coverage_rate"] = self.evidence_coverage_rate
        return result


def _direction(signal: str) -> int:
    return {
        "Strong Buy": 2,
        "Buy": 1,
        "Hold": 0,
        "Sell": -1,
        "Strong Sell": -2,
    }[signal]


def count_material_contradictions(signals: Sequence[StockSignal]) -> int:
    """Count same-ticker recommendations that disagree by >= 3 direction steps."""
    by_ticker: dict[str, list[StockSignal]] = {}
    for signal in signals:
        by_ticker.setdefault(signal.ticker, []).append(signal)

    contradictions = 0
    for ticker_signals in by_ticker.values():
        for index, left in enumerate(ticker_signals):
            for right in ticker_signals[index + 1 :]:
                if abs(_direction(left.signal) - _direction(right.signal)) >= 3:
                    contradictions += 1
    return contradictions


def evaluate_stock_signals(signals: Iterable[StockSignal]) -> EvalSummary:
    materialized = list(signals)
    return EvalSummary(
        total=len(materialized),
        schema_valid=len(materialized),
        evidence_backed=sum(bool(signal.evidence) for signal in materialized),
        with_warnings=sum(bool(signal.warnings) for signal in materialized),
        contradictory_pairs=count_material_contradictions(materialized),
    )
