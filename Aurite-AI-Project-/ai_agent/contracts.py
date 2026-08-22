"""Typed contracts for AIvestor agent handoffs.

The legacy project exchanged loosely shaped dictionaries and timestamp-selected JSON
files. These Pydantic models make agent outputs explicit, validateable, and easier
to evaluate or expose through MCP tools.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class RunStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


class EvidenceRef(BaseModel):
    """A minimal provenance record for a fact used by an agent."""

    source: str = Field(min_length=1)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    as_of: Optional[datetime] = None
    identifier: Optional[str] = None
    fields: List[str] = Field(default_factory=list)


class StockSignal(BaseModel):
    """Structured output for one equity-analysis decision."""

    ticker: str = Field(min_length=1, max_length=16)
    signal: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    confidence: float = Field(ge=0.0, le=1.0)
    investment_score: float = Field(ge=1.0, le=10.0)
    thesis: str = Field(min_length=1)
    primary_risk: str = Field(min_length=1)
    secondary_risk: Optional[str] = None
    target_price: Optional[float] = Field(default=None, gt=0)
    time_horizon: Optional[str] = None
    evidence: List[EvidenceRef] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class RankingItem(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    rank: int = Field(ge=1)
    score: float = Field(ge=1.0, le=10.0)
    rationale: str = Field(min_length=1)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class StockRanking(BaseModel):
    ranking: List[RankingItem]
    market_outlook: str
    portfolio_allocation: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    @field_validator("portfolio_allocation")
    @classmethod
    def validate_allocations(cls, values: Dict[str, float]) -> Dict[str, float]:
        normalized = {ticker.upper(): float(weight) for ticker, weight in values.items()}
        if any(weight < 0 or weight > 1 for weight in normalized.values()):
            raise ValueError("portfolio weights must be between 0 and 1")
        if normalized and sum(normalized.values()) > 1.000001:
            raise ValueError("portfolio weights cannot sum to more than 1")
        return normalized


class AgentResult(BaseModel):
    """Common envelope for agent-to-agent and MCP handoffs."""

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    agent: str = Field(min_length=1)
    status: RunStatus = RunStatus.SUCCESS
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[EvidenceRef] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    def finish(self, *, status: RunStatus = RunStatus.SUCCESS) -> "AgentResult":
        self.status = status
        self.completed_at = datetime.now(timezone.utc)
        return self
