# AIvestor 2026 Modernization

This branch upgrades AIvestor from a 2025 multi-script agent demo toward an evaluation-driven agent system. The goal is not to make every step agentic. Deterministic data processing, portfolio constraints, validation, and routing should stay deterministic; LLM agents should be used where synthesis and judgment add value.

## What is already strong

- Real FRED and market-data integrations.
- Separate macro, equity, bond, gold, and portfolio components.
- Parallel asset analysis in the MCP server.
- Existing MCP tool surface for analysis and health checks.
- ML feature engineering and time-series-oriented market modeling.
- Human-readable and machine-readable report artifacts.

## P0 correctness findings

These items should be fixed before publishing new performance claims.

1. **Next-quarter target alignment**
   - The current macro classifier derives `Target` from the same quarter's `NASDAQ100_Return` while also constructing same-quarter price-derived features.
   - For a next-quarter task, the label must be shifted forward and the final unlabeled row excluded from training.

2. **Scaler leakage**
   - `train_unified_model.py` fits the scaler before the final chronological train/test split.
   - Fit preprocessing only on the training window, ideally inside a time-series-validation pipeline.

3. **Synthetic-target fallback**
   - Training can fall back to macro-derived or random synthetic market returns when market data is unavailable.
   - Synthetic data can be useful for smoke tests, but it must never be mixed into reported out-of-sample performance.

4. **Agent 1 non-interactive syntax defect**
   - The current non-interactive risk default contains a malformed string/parenthesis and should be repaired before whole-repository CI is enabled.

5. **Dependency drift**
   - `stock_analysis_agent.py` imports LangChain packages that were not declared in the original requirements file.

## Architecture direction

### Before

`script -> timestamped JSON -> scan for latest file -> next component`

This makes hidden state and stale-artifact selection possible.

### Target

`RunContext -> deterministic fan-out -> typed AgentResult contracts -> deterministic validation/evaluation -> portfolio aggregation -> persisted artifacts`

Files should be outputs of a run, not the source of truth for handoffs.

## Agent contracts

`ai_agent/contracts.py` adds typed Pydantic contracts for:

- run identity and timestamps,
- agent status and error state,
- evidence/provenance references,
- stock signals,
- rankings and allocation constraints.

The next integration step is to have LLM calls return these schemas directly instead of prompting for JSON and then cleaning markdown fences manually.

## Evaluation gates

`ai_agent/evaluation.py` starts a model-independent evaluation layer with:

- schema validity,
- evidence coverage,
- warning rate,
- material cross-agent contradiction count.

Recommended next metrics:

- stale-data rate,
- source freshness and source-availability rate,
- tool-call success rate,
- retry/fallback rate,
- latency and token/cost budget,
- deterministic replay consistency,
- recommendation change rate when evidence is held constant,
- portfolio-constraint violation rate.

## LLM modernization

The legacy repo mixes `chat.completions`, LangChain free-form text generation, prompt-requested JSON, and manual `json.loads` fallbacks. The target pattern is:

1. typed structured outputs validated against Pydantic,
2. explicit tool schemas for data access,
3. low-temperature evidence-grounded synthesis,
4. retry only on well-defined validation/tool failures,
5. no keyword-based Buy/Sell inference when structured parsing fails,
6. trace IDs attached to every agent result.

## MCP

Keep MCP. It is one of the strongest parts of the original project. The server already exposes market-analysis and health-check tools and performs parallel asset analysis. The modernization should make the MCP responses use the same typed contracts and run IDs as the internal workflow.

## Recruiter-facing project story after integration

> Built an evaluation-driven multi-agent financial research system with MCP tools, typed Pydantic handoffs, parallel market-analysis agents, evidence/provenance tracking, deterministic portfolio constraints, and automated reliability checks for schema validity, source coverage, and cross-agent contradictions.

That statement should only be used after the typed contracts and evaluation gates are wired into the live workflow, not merely because the files exist on this branch.
