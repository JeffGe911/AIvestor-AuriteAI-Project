"""
Portfolio Agent 
==================================

Generates a professional, investor-facing portfolio report by:
1) Consolidating outputs from prior agents (analysis_outputs/)
2) Mapping them into portfolio metrics & signals
3) Producing a Markdown report (+ JSON + optional charts)

Requirements
------------
- Python 3+
- Optional: matplotlib (for charts)
  pip install matplotlib

Typical Usage
-------------
def profile_header(profile: Profile) -> str:
    return (
        f"# Portfolio Strategy Report\n\n"
        f"**Client:** {profile.profile_id or 'Client'}  \n"
        f"**Objective:** {(profile.investment_goal or 'wealth building').title()}  \n"
        f"**Risk Tolerance:** {(profile.risk_level or 'moderate').capitalize()}  \n"
        f"**Time Horizon:** {profile.time_horizon or 5} years  \n\n"
        f"*For discussion purposes only; not investment advice.*\n\n---\n\n"
    )ysis outputs
python portfolio_agent.py \
  --profile portfolio_outputs/session_XXXX/profile/user_YYYY.json \
  --analysis-dir analysis_outputs \
  --out portfolio_outputs/session_XXXX/portfolio/report.md \
  --json-out portfolio_outputs/session_XXXX/portfolio/report.json \
  --plots portfolio_outputs/session_XXXX/portfolio/plots

# Or pass inputs explicitly to override analysis:
python portfolio_agent.py \
  --profile portfolio_outputs/session_XXXX/profile/user_YYYY.json \
  --stats '{"annual_return":0.072,"annual_vol":0.153,"sharpe":0.37,"max_drawdown":-0.10}' \
  --benchmark '{"annual_return":0.065,"annual_vol":0.148,"sharpe":0.33}' \
  --current-alloc '{"Equity":0.45,"Bond":0.50,"Cash":0.05}' \
  --sector-alloc '{"Technology":0.20,"Healthcare":0.10,"Energy":0.03}' \
  --signals '{"Technology":0.8,"Energy":-0.6}' \
  --out portfolio_outputs/session_XXXX/portfolio/report.md

Flags
-----
--test : run built-in tests and exit
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import glob
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Optional plotting
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str):
    if not p:
        return
    os.makedirs(p, exist_ok=True)

def _dig(obj: Any, *paths, default=None):
    """Safely traverse nested dicts/lists by trying multiple possible paths."""
    for path in paths:
        cur = obj
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            elif isinstance(cur, list) and isinstance(p, int) and 0 <= p < len(cur):
                cur = cur[p]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default

def _normalize_weights(weights: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not weights or not isinstance(weights, dict):
        return None
    out = {}
    for k, v in weights.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    s = sum(out.values()) if out else 0.0
    if s > 0 and abs(s - 1.0) > 1e-6:
        out = {k: v / s for k, v in out.items()}
    return out or None


# -----------------------------
# Consolidate analysis_outputs/
# -----------------------------
def consolidate_analysis_outputs(analysis_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Scan analysis_dir for JSON/CSV outputs from prior agents and
    consolidate into a single dict: performance, benchmark, weights,
    sector weights, sector signals, macro context.
    """
    if not analysis_dir or not os.path.isdir(analysis_dir):
        return None

    consolidated: Dict[str, Any] = {
        "performance": {},
        "benchmark": {},
        "portfolio": {"weights": {}},
        "sectors": {"current": {}},
        "signals": {"sectors": {}},
        "macro_context": {},
        "asset_recommendations": {
            "stocks": [],
            "bonds": [],
            "gold": [],
            "etfs": []
        },
        "raw_files": [],
    }

    # JSONs - use only the most recent filtered stock analysis for stocks
    all_json_files = glob.glob(os.path.join(analysis_dir, "**", "*.json"), recursive=True)
    
    # Find the most recent filtered stock analysis file
    filtered_stock_files = [f for f in all_json_files if "stock_analysis_filtered_" in f]
    most_recent_filtered = None
    if filtered_stock_files:
        # Sort by modification time to get the most recent
        most_recent_filtered = max(filtered_stock_files, key=lambda f: os.path.getmtime(f))
    
    # Process all files, but only use filtered file for stock recommendations
    for path in all_json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            consolidated["raw_files"].append(path)

            perf = _dig(data, ('performance',), ('metrics',), default=None)
            if isinstance(perf, dict):
                for k in ("annual_return", "annual_vol", "sharpe", "max_drawdown"):
                    if k in perf and isinstance(perf[k], (int, float)):
                        consolidated["performance"][k] = perf[k]

            bench = _dig(data, ('benchmark',), ('bench',), ('reference','benchmark'), default=None)
            if isinstance(bench, dict):
                for k in ("annual_return", "annual_vol", "sharpe"):
                    if k in bench and isinstance(bench[k], (int, float)):
                        consolidated["benchmark"][k] = bench[k]

            weights = _dig(data,
                           ('current_alloc',),
                           ('portfolio','weights'),
                           ('portfolio','current','asset_classes'),
                           ('allocations','current','asset_classes'),
                           default=None)
            if isinstance(weights, dict):
                consolidated["portfolio"]["weights"].update(weights)

            sectors = _dig(data,
                           ('sectors','current'),
                           ('portfolio','current','sectors'),
                           ('allocations','current','sectors'),
                           ('sectors','weights'),
                           default=None)
            if isinstance(sectors, dict):
                consolidated["sectors"]["current"].update(sectors)

            sec_sig = _dig(data,
                           ('signals','sectors'),
                           ('ranking','sector_signal'),
                           ('sectors','signals'),
                           default=None)
            if isinstance(sec_sig, dict):
                for k, v in sec_sig.items():
                    try:
                        consolidated["signals"]["sectors"][str(k)] = float(v)
                    except Exception:
                        pass

            macro = _dig(data, ('macro_context',), ('macro',), default=None)
            if isinstance(macro, dict):
                consolidated["macro_context"].update(macro)

            # Extract asset recommendations
            # Stock recommendations - ONLY extract from filtered stock analysis files
            is_filtered_stock_file = "stock_analysis_filtered_" in path
            
            if is_filtered_stock_file:
                # For filtered stock files, check signals_summary first, then other patterns
                signals_summary = _dig(data, ('signals_summary',), default=None)
                if isinstance(signals_summary, list):
                    # Clear any previous stock recommendations when we find a filtered file
                    consolidated["asset_recommendations"]["stocks"] = []
                    for stock in signals_summary[:10]:  # Top 10 from filtered file
                        if isinstance(stock, dict) and ('ticker' in stock or 'symbol' in stock):
                            ticker = stock.get("ticker") or stock.get("symbol")
                            consolidated["asset_recommendations"]["stocks"].append({
                                "ticker": ticker,
                                "rank": stock.get("rank"),
                                "score": stock.get("score") or stock.get("expected_return"),
                                "signal": stock.get("signal", "Hold") or stock.get("sentiment", "Hold"),
                                "reason": stock.get("reason") or stock.get("summary", "No reason provided")
                            })
                
                # Also check horizons pattern for filtered files
                stock_ranking = _dig(data, ('horizons', 'next_quarter'), default=None)
                if isinstance(stock_ranking, list) and not consolidated["asset_recommendations"]["stocks"]:
                    for stock in stock_ranking[:10]:  # Top 10
                        if isinstance(stock, dict) and ('ticker' in stock or 'symbol' in stock):
                            ticker = stock.get("ticker") or stock.get("symbol")
                            consolidated["asset_recommendations"]["stocks"].append({
                                "ticker": ticker,
                                "rank": stock.get("rank"),
                                "score": stock.get("score") or stock.get("expected_return"),
                                "signal": stock.get("signal", "Hold") or stock.get("sentiment", "Hold"),
                                "reason": stock.get("reason") or stock.get("summary", "No reason provided")
                            })
            
            # For non-filtered files, only extract stocks if no filtered file has been processed
            elif not any("stock_analysis_filtered_" in f for f in consolidated["raw_files"] if f != path):
                stock_ranking = _dig(data, ('stock_ranking', 'ranking'), ('ranking',), ('horizons', 'next_quarter'), default=None)
                if isinstance(stock_ranking, list):
                    for stock in stock_ranking[:5]:  # Top 5
                        if isinstance(stock, dict) and ('ticker' in stock or 'symbol' in stock):
                            ticker = stock.get("ticker") or stock.get("symbol")
                            consolidated["asset_recommendations"]["stocks"].append({
                                "ticker": ticker,
                                "rank": stock.get("rank"),
                                "score": stock.get("score") or stock.get("expected_return"),
                                "signal": stock.get("signal", "Hold") or stock.get("sentiment", "Hold"),
                                "reason": stock.get("reason") or stock.get("summary", "No reason provided")
                            })

            # Bond recommendations
            bond_horizons = _dig(data, ('horizons', 'next_quarter'), default=None)
            if isinstance(bond_horizons, list):
                for bond in bond_horizons[:5]:  # Top 5
                    if isinstance(bond, dict) and 'ticker' in bond:
                        consolidated["asset_recommendations"]["bonds"].append({
                            "ticker": bond.get("ticker"),
                            "label": bond.get("label"),
                            "bond_type": bond.get("bond_type"),
                            "expected_return": bond.get("expected_return"),
                            "rank": bond.get("rank"),
                            "sentiment": bond.get("sentiment"),
                            "confidence": bond.get("confidence"),
                            "summary": bond.get("summary", "")[:100] + "..." if bond.get("summary") else ""
                        })

            # Gold/commodity recommendations
            gold_horizons = _dig(data, ('horizons', 'next_quarter'), default=None)
            if isinstance(gold_horizons, list) and data.get("analysis_type") == "Quantitative gold analysis":
                for gold in gold_horizons[:3]:  # Top 3
                    if isinstance(gold, dict) and 'ticker' in gold:
                        consolidated["asset_recommendations"]["gold"].append({
                            "ticker": gold.get("ticker"),
                            "label": gold.get("label"),
                            "gold_type": gold.get("gold_type"),
                            "expected_return": gold.get("expected_return"),
                            "rank": gold.get("rank"),
                            "sentiment": gold.get("sentiment"),
                            "confidence": gold.get("confidence"),
                            "summary": gold.get("summary", "")[:100] + "..." if gold.get("summary") else ""
                        })

            # infer from stock/etf picks if present
            for key in ("stocks", "equities", "etfs", "etf"):
                arr = data.get(key)
                if isinstance(arr, list):
                    counts: Dict[str, int] = {}
                    for it in arr:
                        sec = it.get("sector")
                        if sec:
                            counts[sec] = counts.get(sec, 0) + 1
                    if counts:
                        top = max(counts, key=counts.get)
                        consolidated["signals"]["sectors"][top] = max(
                            consolidated["signals"]["sectors"].get(top, 0.0), 0.4
                        )
        except Exception:
            continue

    # CSV fallback for sector weights
    for path in glob.glob(os.path.join(analysis_dir, "**", "*sector*.csv"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sec = row.get("Sector") or row.get("sector")
                    wt = row.get("Weight") or row.get("weight")
                    if sec and wt:
                        try:
                            consolidated["sectors"]["current"][str(sec)] = float(wt)
                        except Exception:
                            pass
            consolidated["raw_files"].append(path)
        except Exception:
            continue

    return consolidated


# -----------------------------
# Data model & policy logic
# -----------------------------
@dataclass
class Profile:
    risk_level: str = "moderate"
    time_horizon: int = 5
    investment_amount: float = 0.0
    investment_goal: str = "wealth building"
    monthly_contribution: float = 0.0
    preferred_sectors: List[str] = None
    avoid_sectors: List[str] = None
    prefers_esg: bool = False
    needs_liquidity: bool = False
    tax_sensitive: bool = False
    profile_id: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict) -> "Profile":
        return Profile(
            risk_level=str(d.get("risk_level", "moderate")).lower(),
            time_horizon=int(d.get("time_horizon", 5) or 5),
            investment_amount=float(d.get("investment_amount", 0) or 0),
            investment_goal=d.get("investment_goal") or "wealth building",
            monthly_contribution=float(d.get("monthly_contribution", 0) or 0),
            preferred_sectors=list(d.get("preferred_sectors", []) or []),
            avoid_sectors=list(d.get("avoid_sectors", []) or []),
            prefers_esg=bool(d.get("prefers_esg", False)),
            needs_liquidity=bool(d.get("needs_liquidity", False)),
            tax_sensitive=bool(d.get("tax_sensitive", False)),
            profile_id=d.get("profile_id"),
        )

def target_allocation_from_risk_horizon(risk_level: str, horizon_years: int) -> Dict[str, float]:
    risk = risk_level.lower()
    h = int(horizon_years)
    col = "short" if h <= 5 else ("medium" if h <= 15 else "long")
    matrix = {
        "conservative": {
            "short":  {"Equity": 0.35, "Bond": 0.55, "Cash": 0.10},
            "medium": {"Equity": 0.40, "Bond": 0.50, "Cash": 0.10},
            "long":   {"Equity": 0.45, "Bond": 0.45, "Cash": 0.10},
        },
        "moderate": {
            "short":  {"Equity": 0.50, "Bond": 0.45, "Cash": 0.05},
            "medium": {"Equity": 0.60, "Bond": 0.35, "Cash": 0.05},
            "long":   {"Equity": 0.65, "Bond": 0.30, "Cash": 0.05},
        },
        "aggressive": {
            "short":  {"Equity": 0.65, "Bond": 0.30, "Cash": 0.05},
            "medium": {"Equity": 0.75, "Bond": 0.20, "Cash": 0.05},
            "long":   {"Equity": 0.85, "Bond": 0.10, "Cash": 0.05},
        },
    }
    base = matrix.get(risk, matrix["moderate"])[col]
    s = sum(base.values())
    if abs(s - 1.0) > 1e-6:
        base = {k: v / s for k, v in base.items()}
    return base

def magnitude_bucket(delta: float) -> Tuple[str, str]:
    ad = abs(delta)
    if ad < 0.05:  return "small", "a small adjustment of approximately"
    if ad < 0.15:  return "medium", "a measured adjustment of roughly"
    return "large", "a decisive adjustment of around"

def suggest_rebalance(current: Optional[Dict[str, float]],
                      target: Dict[str, float]) -> List[Dict[str, str]]:
    suggestions = []
    current = current or {}
    for k in set(target) | set(current):
        cur = float(current.get(k, 0.0))
        tgt = float(target.get(k, 0.0))
        delta = tgt - cur
        if abs(delta) < 1e-3:
            continue
        band, tone = magnitude_bucket(delta)
        direction = "increase" if delta > 0 else "reduce"
        suggestions.append({
            "asset_class": k,
            "current": f"{cur:.2%}",
            "target": f"{tgt:.2%}",
            "delta": f"{delta:+.2%}",
            "band": band,
            "narrative": f"{direction.capitalize()} {k} with {tone} {abs(delta):.0%}."
        })
    return sorted(suggestions, key=lambda x: x["delta"], reverse=True)

def apply_signal_bias(prefer: List[str], avoid: List[str],
                      signals: Optional[Dict[str, float]]) -> Tuple[List[str], List[str], List[str]]:
    overweight, underweight, highlights = [], [], []
    if signals:
        for sector, strength in sorted(signals.items(), key=lambda kv: kv[1], reverse=True):
            if strength >= 0.25:
                overweight.append(sector); highlights.append(f"{sector} (+{strength:.2f})")
            elif strength <= -0.25:
                underweight.append(sector); highlights.append(f"{sector} ({strength:.2f})")
    for s in (prefer or []):
        if s not in overweight:
            overweight.append(s)
    for s in (avoid or []):
        if s not in underweight:
            underweight.append(s)
    return overweight, underweight, highlights

def scenario_projection(profile: Profile,
                        stats: Optional[Dict[str, float]],
                        signals: Optional[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    risk_map = {"conservative": 0.7, "moderate": 1.0, "aggressive": 1.3}
    rf = risk_map.get(profile.risk_level, 1.0)
    base_ret = float(stats["annual_return"]) if stats and stats.get("annual_return") is not None else \
               {"conservative": 0.04, "moderate": 0.06, "aggressive": 0.08}.get(profile.risk_level, 0.06)
    signal_strength = (sum(signals.values())/len(signals)) if signals else 0.0
    bull_boost = (0.08 + 0.04 * rf) + max(0.0, 0.5 * signal_strength)
    bear_drag = (0.08 + 0.04 * rf) + max(0.0, 0.25 * (-signal_strength))
    if profile.risk_level == "conservative": bear_drag *= 0.8
    return {"Base": {"expected_return": base_ret},
            "Bull": {"expected_return": base_ret + bull_boost},
            "Bear": {"expected_return": base_ret - bear_drag}}


# -----------------------------
# Narrative builders
# -----------------------------
def cover_page(profile: Profile) -> str:
    return (
        f"# Portfolio Strategy Report\n\n"
        f"**Client:** {profile.profile_id or 'Client'}  \n"
        f"**Objective:** {(profile.investment_goal or 'wealth building').title()}  \n"
        f"**Risk Tolerance:** {(profile.risk_level or 'moderate').capitalize()}  \n"
        f"**Time Horizon:** {profile.time_horizon or 5} years  \n\n"
        f"*For discussion purposes only; not investment advice.*\n\n---\n\n"
    )

def executive_summary(profile: Profile,
                      stats: Optional[Dict[str, float]],
                      benchmark: Optional[Dict[str, float]],
                      target_alloc: Dict[str, float],
                      scenarios: Dict[str, Dict[str, float]]) -> str:
    lines = ["## Executive Summary"]
    r, b = stats or {}, benchmark or {}
    if r.get("annual_return") is not None:
        if b.get("annual_return") is not None:
            diff = r["annual_return"] - b["annual_return"]
            comp = "outperformed" if diff > 0 else ("underperformed" if diff < 0 else "matched")
            lines.append(f"Annualized return **{r['annual_return']:.2%}**, {comp} benchmark by **{abs(diff):.2%}**.")
        else:
            lines.append(f"Annualized return **{r['annual_return']:.2%}**.")
    if r.get("sharpe") is not None:
        lines.append(f"Sharpe ratio **{r['sharpe']:.2f}**.")
    if r.get("max_drawdown") is not None:
        lines.append(f"Max drawdown **{r['max_drawdown']:.2%}**.")
    lines.append(f"Strategic mix aligns with your **{profile.risk_level}** risk tolerance and **{profile.time_horizon}-year** horizon.")
    s_base, s_bull, s_bear = (scenarios["Base"]["expected_return"],
                              scenarios["Bull"]["expected_return"],
                              scenarios["Bear"]["expected_return"])
    lines.append(f"1Y scenario range: **{s_bear:.0%} → {s_bull:.0%}**; base case **{s_base:.0%}**.")
    lines.append("\n**Policy Weights:**")
    for k, v in target_alloc.items():
        lines.append(f"- **{k}**: {v:.0%}")
    lines.append("")
    return "\n".join(lines)

def macro_market_view(analysis: Optional[Dict]) -> str:
    lines = ["## Macro & Market View"]
    if analysis and analysis.get("macro_context"):
        mc = analysis["macro_context"]
        for k, v in mc.items():
            lines.append(f"- **{k.replace('_',' ').title()}**: {v}")
    else:
        lines.append("- Macro conditions remain mixed; policy rates stay restrictive while inflation trends lower.")
        lines.append("- Earnings leadership skews to quality growth; breadth is gradually improving.")
        lines.append("- Diversification across equities, duration and alternatives remains beneficial.")
    lines.append("")
    return "\n".join(lines)

def sector_theme_positioning(over: List[str], under: List[str], highlights: List[str]) -> str:
    lines = ["## Sector & Theme Positioning"]
    if over:  lines.append(f"- **Overweight:** {', '.join(over)}")
    if under: lines.append(f"- **Underweight:** {', '.join(under)}")
    if highlights: lines.append(f"- _Signal highlights_: {', '.join(highlights)}")
    if not (over or under):
        lines.append("- Neutral sector stance pending clearer signals.")
    lines.append("")
    return "\n".join(lines)

def asset_recommendations(analysis: Optional[Dict], target_alloc: Dict[str, float], profile: Profile) -> str:
    lines = ["## Top Asset Recommendations"]
    
    if not analysis or not analysis.get("asset_recommendations"):
        lines.append("- Asset recommendations not available from analysis.")
        lines.append("")
        return "\n".join(lines)
    
    recs = analysis["asset_recommendations"]
    
    # Helper function to check if a ticker should be avoided based on sector
    def is_ticker_avoided(ticker: str, avoid_sectors: List[str]) -> bool:
        if not avoid_sectors:
            return False
        
        # Map common tech tickers to technology sector
        tech_tickers = {'MSFT', 'GOOGL', 'GOOG', 'AAPL', 'NVDA', 'META', 'AMZN', 'TSLA', 'CRM', 'ADBE', 'NFLX', 'AMD', 'INTC', 'AVGO', 'ORCL'}
        energy_tickers = {'XOM', 'CVX', 'COP', 'PSX', 'VLO', 'MPC', 'EOG', 'PXD', 'SLB', 'OKE'}
        healthcare_tickers = {'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD'}
        
        for sector in avoid_sectors:
            sector_lower = sector.lower()
            if sector_lower in ['technology', 'tech'] and ticker.upper() in tech_tickers:
                return True
            elif sector_lower in ['energy'] and ticker.upper() in energy_tickers:
                return True
            elif sector_lower in ['healthcare', 'health'] and ticker.upper() in healthcare_tickers:
                return True
        return False
    
    # Equity recommendations (if equity allocation > 0)
    if target_alloc.get("Equity", 0) > 0 and recs.get("stocks"):
        # Filter out avoided sectors
        filtered_stocks = [
            stock for stock in recs["stocks"] 
            if not is_ticker_avoided(stock.get("ticker", ""), profile.avoid_sectors or [])
        ]
        
        if filtered_stocks:
            lines.append("### Equity Holdings")
            lines.append("| Ticker | Rank | Signal | Score | Rationale |")
            lines.append("|--------|------|--------|-------|-----------|")
            for stock in filtered_stocks[:5]:
                ticker = stock.get("ticker", "")
                rank = stock.get("rank", "-")
                signal = stock.get("signal", "Hold")
                score = f"{stock.get('score', 0):.1f}" if stock.get("score") else "-"
                reason = stock.get("reason", "No reason provided")[:50] + "..." if len(stock.get("reason", "")) > 50 else stock.get("reason", "No reason provided")
                lines.append(f"| {ticker} | {rank} | {signal} | {score} | {reason} |")
            lines.append("")
        elif profile.avoid_sectors:
            lines.append("### Equity Holdings")
            lines.append(f"_No suitable equity recommendations found (avoiding {', '.join(profile.avoid_sectors)} sectors)._")
            lines.append("")
    
    # Bond recommendations (if bond allocation > 0)
    if target_alloc.get("Bond", 0) > 0 and recs.get("bonds"):
        lines.append("### Fixed Income Holdings")
        lines.append("| Ticker | Type | Expected Return | Sentiment | Rationale |")
        lines.append("|--------|------|----------------|-----------|-----------|")
        for bond in recs["bonds"][:5]:
            ticker = bond.get("ticker", "")
            bond_type = (bond.get("bond_type") or "").title()
            exp_ret = f"{bond.get('expected_return', 0):.1%}" if bond.get("expected_return") else "-"
            sentiment = (bond.get("sentiment") or "Neutral").title()
            summary = bond.get("summary", "No summary available")[:50] + "..." if len(bond.get("summary", "") or "") > 50 else bond.get("summary", "No summary available")
            lines.append(f"| {ticker} | {bond_type} | {exp_ret} | {sentiment} | {summary} |")
        lines.append("")
    
    # Alternative investments (Gold/Commodities)
    if recs.get("gold"):
        lines.append("### Alternative Investments")
        lines.append("| Ticker | Type | Expected Return | Sentiment | Rationale |")
        lines.append("|--------|------|----------------|-----------|-----------|")
        for gold in recs["gold"][:3]:
            ticker = gold.get("ticker", "")
            gold_type = (gold.get("gold_type") or "").replace("/", " / ")
            exp_ret = f"{gold.get('expected_return', 0):.1%}" if gold.get("expected_return") else "-"
            sentiment = (gold.get("sentiment") or "Neutral").title()
            summary = gold.get("summary", "No summary available")[:50] + "..." if len(gold.get("summary", "") or "") > 50 else gold.get("summary", "No summary available")
            lines.append(f"| {ticker} | {gold_type} | {exp_ret} | {sentiment} | {summary} |")
        lines.append("")
    
    lines.append("_Recommendations based on current market analysis and risk-adjusted scoring._")
    lines.append("")
    return "\n".join(lines)

def risk_and_scenarios(scenarios: Dict[str, Dict[str, float]]) -> str:
    lines = ["## Risk & Scenario Analysis",
             "| Scenario | 1Y Expected Return |",
             "|---|---:|"]
    for n, o in scenarios.items():
        lines.append(f"| {n} | {o['expected_return']:.0%} |")
    lines.append("")
    lines.append("Key risks: policy surprises, inflation shocks, geopolitics, liquidity stress. Mitigation: diversification, disciplined rebalancing, cash sleeve.")
    lines.append("")
    return "\n".join(lines)

def implementation_roadmap(monthly: float) -> str:
    lines = ["## Implementation Roadmap",
             "**0–30 Days**",
             "- Rebalance toward policy weights; stage trades if large.",
             "- Fund cash sleeve for near-term needs/opportunities."]
    if monthly and monthly > 0:
        lines.append(f"- Set up automated contributions of **${monthly:,.0f}** per month.")
    lines += ["",
              "**3–6 Months**",
              "- Review performance vs. policy; adjust tactical tilts as needed.",
              "- Reassess macro drivers and sector earnings revisions.",
              "",
              "**12 Months**",
              "- Policy review; reset target weights and guardrails if required.",
              ""]
    return "\n".join(lines)

def key_metrics_table(stats: Optional[Dict[str, float]], benchmark: Optional[Dict[str, float]]) -> str:
    r, b = stats or {}, benchmark or {}
    lines = ["## Key Metrics (Snapshot)"]
    if r:
        if r.get("annual_return") is not None: lines.append(f"- Return: {r['annual_return']:.2%}")
        if r.get("annual_vol") is not None:    lines.append(f"- Volatility: {r['annual_vol']:.2%}")
        if r.get("sharpe") is not None:        lines.append(f"- Sharpe: {r['sharpe']:.2f}")
        if r.get("max_drawdown") is not None:  lines.append(f"- Max Drawdown: {r['max_drawdown']:.2%}")
    else:
        lines.append("- Metrics not available.")
    if b:
        parts = []
        if b.get("annual_return") is not None: parts.append(f"Return {b['annual_return']:.2%}")
        if b.get("annual_vol") is not None:    parts.append(f"Vol {b['annual_vol']:.2%}")
        if b.get("sharpe") is not None:        parts.append(f"Sharpe {b['sharpe']:.2f}")
        if parts: lines.append(f"- Benchmark — " + ", ".join(parts))
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# Plotting
# -----------------------------
def plot_allocations(plots_dir: str,
                     current_alloc: Optional[Dict[str, float]],
                     target_alloc: Dict[str, float]) -> List[str]:
    if not _HAS_MPL: return []
    ensure_dir(plots_dir)
    keys = sorted(set((current_alloc or {}).keys()) | set(target_alloc.keys()))
    cur = [float((current_alloc or {}).get(k, 0.0)) for k in keys]
    tgt = [float(target_alloc.get(k, 0.0)) for k in keys]
    plt.figure()
    x = range(len(keys)); w = 0.4
    plt.bar([i - w/2 for i in x], cur, width=w, label="Current")
    plt.bar([i + w/2 for i in x], tgt, width=w, label="Target")
    plt.xticks(list(x), keys); plt.title("Allocation: Current vs Target"); plt.legend(); plt.tight_layout()
    out = os.path.join(plots_dir, "allocation_current_vs_target.png")
    plt.savefig(out, dpi=160); plt.close()
    return [out]

def plot_sectors(plots_dir: str,
                 sector_alloc: Optional[Dict[str, float]],
                 overweight: List[str], underweight: List[str]) -> List[str]:
    if not _HAS_MPL or not sector_alloc: return []
    ensure_dir(plots_dir)
    keys = list(sorted(sector_alloc.keys()))
    vals = [float(sector_alloc[k]) for k in keys]
    plt.figure()
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
    plt.title("Current Sector Allocation"); plt.tight_layout()
    out = os.path.join(plots_dir, "sector_current.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close()
    return [out]

def plot_scenarios(plots_dir: str, scenarios: Dict[str, Dict[str, float]]) -> List[str]:
    if not _HAS_MPL: return []
    ensure_dir(plots_dir)
    labels = list(scenarios.keys())
    vals = [scenarios[k]["expected_return"] for k in labels]
    plt.figure()
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels); plt.title("One-Year Scenario Returns"); plt.tight_layout()
    out = os.path.join(plots_dir, "scenarios_1y.png")
    plt.savefig(out, dpi=160); plt.close()
    return [out]


# -----------------------------
# Report builder
# -----------------------------
def make_report_markdown(profile: Profile,
                         analysis: Optional[Dict],
                         target_alloc: Dict[str, float],
                         rebalance: List[Dict[str, str]],
                         overweight: List[str],
                         underweight: List[str],
                         highlights: List[str],
                         scenarios: Dict[str, Dict[str, float]],
                         stats: Optional[Dict[str, float]] = None,
                         benchmark: Optional[Dict[str, float]] = None,
                         plot_paths: Optional[List[str]] = None) -> str:
    md = []
    md.append(cover_page(profile))
    md.append(executive_summary(profile, stats, benchmark, target_alloc, scenarios))
    md.append(macro_market_view(analysis))

    md.append("## Strategic Asset Allocation")
    md.append("**Policy Weights:**")
    for k, v in target_alloc.items():
        md.append(f"- **{k}**: {v:.0%}")
    if rebalance:
        md.append("\n**Rebalancing Instructions:**")
        for s in rebalance:
            md.append(f"- {s['narrative']} (current {s['current']} → target {s['target']})")
    md.append("")

    md.append(sector_theme_positioning(overweight, underweight, highlights))
    md.append(asset_recommendations(analysis, target_alloc, profile))
    md.append(risk_and_scenarios(scenarios))
    md.append(key_metrics_table(stats, benchmark))

    if plot_paths:
        md.append("## Figures")
        for p in plot_paths:
            md.append(f"- {os.path.basename(p)}  \n  `./{os.path.relpath(p)}`")
        md.append("")

    md.append(implementation_roadmap(profile.monthly_contribution))
    md.append("## Preferences, Constraints & Disclosures")
    if profile.preferred_sectors: md.append(f"- Sector preferences: {', '.join(profile.preferred_sectors)}")
    if profile.avoid_sectors:     md.append(f"- Sector exclusions: {', '.join(profile.avoid_sectors)}")
    if profile.prefers_esg:       md.append("- ESG preference: apply positive screening where feasible.")
    if profile.needs_liquidity:   md.append("- Liquidity needs: maintain a dedicated cash sleeve.")
    if profile.tax_sensitive:     md.append("- Tax sensitivity: consider tax-efficient wrappers and asset location.")
    md.append("\n*Illustrative only; not investment advice.*\n")
    return "\n".join(md)


# -----------------------------
# CLI + main flow
# -----------------------------
def parse_json_arg(s: Optional[str]) -> Optional[Dict]:
    if not s: return None
    try:
        return json.loads(s)
    except Exception:
        raise SystemExit("Invalid JSON for CLI flag. Check --stats/--benchmark/--current-alloc/--sector-alloc/--signals.")

def run(profile_path: str,
        analysis_dir: Optional[str],
        out_md: Optional[str],
        out_json: Optional[str],
        plots_dir: Optional[str],
        overrides: Dict[str, Optional[Dict]]) -> Dict[str, str]:
    # Load profile
    with open(profile_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    profile = Profile.from_dict(raw)

    # Consolidate analysis (optional)
    analysis = consolidate_analysis_outputs(analysis_dir)

    # Derive inputs from analysis
    a_stats = _dig(analysis or {}, ('performance',), default=None)
    a_bench = _dig(analysis or {}, ('benchmark',), default=None)
    a_curr  = _normalize_weights(_dig(analysis or {}, ('portfolio','weights'), default=None))
    a_sect  = _normalize_weights(_dig(analysis or {}, ('sectors','current'), default=None))
    a_sign  = _dig(analysis or {}, ('signals','sectors'), default=None)
    if isinstance(a_sign, dict):
        a_sign = {str(k): float(v) for k, v in a_sign.items() if isinstance(v, (int, float))}
    else:
        a_sign = {}

    # Merge explicit user preferences into signals (soft bias)
    for s in (profile.preferred_sectors or []): a_sign.setdefault(str(s), 0.6)
    for s in (profile.avoid_sectors or []):     a_sign.setdefault(str(s), -0.6)

    # Apply override flags if provided
    stats        = overrides.get("stats")        or a_stats
    benchmark    = overrides.get("benchmark")    or a_bench
    current_alloc= overrides.get("current")      or a_curr
    sector_alloc = overrides.get("sector")       or a_sect
    signals      = overrides.get("signals")      or (a_sign if a_sign else None)

    # Build portfolio recommendations
    target_alloc = target_allocation_from_risk_horizon(profile.risk_level, profile.time_horizon)
    rebalance = suggest_rebalance(current_alloc, target_alloc)
    overweight, underweight, highlights = apply_signal_bias(profile.preferred_sectors or [],
                                                            profile.avoid_sectors or [],
                                                            signals)
    scenarios = scenario_projection(profile, stats, signals)

    # Visuals
    plot_paths = []
    if plots_dir:
        ensure_dir(plots_dir)
        plot_paths += plot_allocations(plots_dir, current_alloc, target_alloc)
        plot_paths += plot_sectors(plots_dir, sector_alloc, overweight, underweight)
        plot_paths += plot_scenarios(plots_dir, scenarios)

    # Build report
    md = make_report_markdown(profile, analysis, target_alloc, rebalance,
                              overweight, underweight, highlights, scenarios,
                              stats=stats, benchmark=benchmark, plot_paths=plot_paths)

    if out_md:
        ensure_dir(os.path.dirname(out_md))
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        print(md)

    if out_json:
        ensure_dir(os.path.dirname(out_json))
        obj = {
            "meta": {"profile_id": profile.profile_id, "risk_level": profile.risk_level, "time_horizon": profile.time_horizon},
            "target_allocation": target_alloc,
            "rebalance": rebalance,
            "sector_overweights": overweight,
            "sector_underweights": underweight,
            "signal_highlights": highlights,
            "scenarios": scenarios,
            "preferences": {
                "preferred_sectors": profile.preferred_sectors or [],
                "avoid_sectors": profile.avoid_sectors or [],
                "prefers_esg": profile.prefers_esg,
                "needs_liquidity": profile.needs_liquidity,
                "tax_sensitive": profile.tax_sensitive,
            },
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    return {
        "report_markdown": out_md or "<stdout>",
        "report_json": out_json or "<none>",
        "plots_dir": plots_dir or "<none>",
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Portfolio Agent – consolidate analysis and generate report")
    ap.add_argument("--profile", help="Path to Agent 1 profile JSON")
    ap.add_argument("--analysis-dir", default="analysis_outputs", help="Directory with analysis outputs (optional)")
    ap.add_argument("--out", help="Path to write Markdown report (prints to stdout if omitted)")
    ap.add_argument("--json-out", help="Path to write machine-readable JSON")
    ap.add_argument("--plots", help="Directory to write PNG charts")
    ap.add_argument("--stats", help="JSON for portfolio stats")
    ap.add_argument("--benchmark", help="JSON for benchmark stats")
    ap.add_argument("--current-alloc", help="JSON for current asset-class weights")
    ap.add_argument("--sector-alloc", help="JSON for current sector weights")
    ap.add_argument("--signals", help="JSON for sector signals")
    ap.add_argument("--test", action="store_true", help="Run tests and exit")
    
    args = ap.parse_args()
    
    # Validate required arguments when not in test mode
    if not args.test and not args.profile:
        ap.error("--profile is required when not running tests")
    
    return args


# -----------------------------
# Tests
# -----------------------------
def _run_tests():
    import unittest
    class T(unittest.TestCase):
        def test_matrix(self):
            self.assertAlmostEqual(target_allocation_from_risk_horizon("moderate", 10)["Equity"], 0.60, 5)
        def test_rebalance(self):
            s = suggest_rebalance({"Equity":0.45,"Bond":0.50,"Cash":0.05}, {"Equity":0.60,"Bond":0.35,"Cash":0.05})
            self.assertTrue(any(x["asset_class"]=="Equity" and x["delta"].startswith("+") for x in s))
            self.assertTrue(any(x["asset_class"]=="Bond" and x["delta"].startswith("-") for x in s))
        def test_scenarios(self):
            p = Profile(risk_level="moderate", time_horizon=7)
            sc = scenario_projection(p, {"annual_return":0.06}, {"Technology":0.8})
            self.assertIn("Base", sc); self.assertIn("Bull", sc); self.assertIn("Bear", sc)
    res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(T))
    if not res.wasSuccessful(): sys.exit(1)
    print("All tests passed.")

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    if args.test:
        _run_tests(); sys.exit(0)

    overrides = {
        "stats":        parse_json_arg(args.stats),
        "benchmark":    parse_json_arg(args.benchmark),
        "current":      parse_json_arg(args.current_alloc),
        "sector":       parse_json_arg(args.sector_alloc),
        "signals":      parse_json_arg(args.signals),
    }

    out = run(profile_path=args.profile,
              analysis_dir=args.analysis_dir,
              out_md=args.out,
              out_json=args.json_out,
              plots_dir=args.plots,
              overrides=overrides)
    print(json.dumps(out, indent=2))
