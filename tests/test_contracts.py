from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

PROJECT_DIR = Path(__file__).resolve().parents[1] / "Aurite-AI-Project-"
sys.path.insert(0, str(PROJECT_DIR))

from ai_agent.contracts import AgentResult, EvidenceRef, StockRanking, StockSignal


def test_stock_signal_normalizes_ticker_and_validates_ranges():
    signal = StockSignal(
        ticker="aapl",
        signal="Buy",
        confidence=0.82,
        investment_score=8.1,
        thesis="Revenue quality and margin resilience support the thesis.",
        primary_risk="Valuation compression.",
        evidence=[EvidenceRef(source="market_api", identifier="AAPL")],
    )
    assert signal.ticker == "AAPL"
    assert signal.confidence == 0.82


def test_stock_signal_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        StockSignal(
            ticker="MSFT",
            signal="Hold",
            confidence=1.2,
            investment_score=5.0,
            thesis="Neutral setup.",
            primary_risk="Multiple compression.",
        )


def test_ranking_rejects_overallocated_portfolio():
    with pytest.raises(ValidationError):
        StockRanking(
            ranking=[],
            market_outlook="Mixed",
            portfolio_allocation={"AAPL": 0.7, "MSFT": 0.5},
        )


def test_agent_result_has_run_identity_and_can_finish():
    result = AgentResult(agent="stock_analysis", payload={"ticker": "NVDA"}).finish()
    assert result.run_id
    assert result.completed_at is not None
    assert result.status.value == "success"
