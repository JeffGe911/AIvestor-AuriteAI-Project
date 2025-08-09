"""
Enhanced AIvestor Agent 1 — AIvestor!
=================================================================

Typical usage
-------------
Interactive (TTY):
    python enhanced_aivestor_agent1.py

Non-interactive / CI:
    python enhanced_aivestor_agent1.py \
      --objective retirement \
      --amount 20000 \
      --horizon 7 \
      --risk moderate \
      --monthly 500 \
      --prefer Technology --avoid Energy \
      --esg yes --liquidity no --tax no

Disable saving (dry run):
    python enhanced_aivestor_agent1.py --no-save

Run tests:
    python enhanced_aivestor_agent1.py --test
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import tempfile
import unittest
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("AIvestor")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ------------------------------
# Core Agent
# ------------------------------
class EnhancedAIvestor:
    def __init__(self):
        # Original profile structure retained
        self.profile: Dict[str, Any] = {
            "investment_amount": 10000,
            "risk_level": "moderate",
            "time_horizon": 5,
            "investment_goal": "wealth building",
            "monthly_contribution": 0,
            "preferred_sectors": [],
            "avoid_sectors": [],
            "needs_liquidity": False,
            "prefers_esg": False,
            "tax_sensitive": False,
        }
        self.context = {
            "turn_number": 0,
            "extracted_info": {},
            "confidence_scores": {},
        }

    # --------- Robust input handling ---------
    @staticmethod
    def _safe_input(prompt: str, default: Optional[str] = None) -> str:
        """Attempt to read from stdin. On EOF/OSError (e.g., non-interactive
        or restricted environments), return a default string and log a notice."""
        try:
            return input(prompt)
        except (EOFError, OSError) as e:
            logger.warning("Input unavailable (%s); falling back to default.", type(e).__name__)
            return default or ""

    # ------------------ Conversation ------------------
    def start_conversation(self, interactive: bool = True, seeded: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run 3 advisor-style turns (+ confirmation). If not interactive, use
        `seeded` values and skip prompts; still run extractors for consistency."""
        print("\n" + "=" * 60)
        print("🤖 AIvestor — Your Personal Investment Advisor")
        print("=" * 60)

        if not seeded:
            seeded = {}

        # Turn 1 — objectives + amount + horizon
        if interactive:
            t1 = self._safe_input(
                "\nTo tailor an investment strategy, could you share your primary objective,\n"
                "your initial investment amount, and your time horizon?\n"
                "(e.g., 'retirement, $20,000, 7 years')\n> ",
                default=seeded.get("t1", ""),
            )
        else:
            # Compose a synthetic response from seeded values
            t1 = f"{seeded.get('objective','wealth')}, ${seeded.get('amount',10000)}, {seeded.get('horizon',5)} years"
        self._process_turn_1(t1)

        # Turn 2 — risk tolerance + contributions (single prompt)
        if interactive:
            t2 = self._safe_input(
                "\nHow would you approach a temporary 15% drawdown?\n"
                "a) Buying opportunity  b) Hold steady  c) Reduce risk  d) Exit\n"
                "And do you plan regular contributions (e.g., $500/month)?\n> ",
                default=seeded.get("t2", ""),
            )
        else:
            plan = seeded.get("monthly", 0)
            risk = seeded.get("risk", "b)")
            t2 = f"{risk} and {plan}/monthly"
        self._process_turn_2(t2)

        # Turn 3 — sector tilt + ESG/liquidity/tax (single prompt)
        if interactive:
            t3 = self._safe_input(
                "\nDo you have sectors to overweight or avoid (e.g., Technology, Energy)?\n"
                "Any preferences for ESG, liquidity availability, or tax-efficient approaches?\n> ",
                default=seeded.get("t3", ""),
            )
        else:
            prefer = seeded.get("prefer", [])
            avoid = seeded.get("avoid", [])
            esg = "yes" if str(seeded.get("esg", "no")).lower() in {"1","true","yes","y"} else "no"
            liq = "yes" if str(seeded.get("liquidity", "no")).lower() in {"1","true","yes","y"} else "no"
            tax = "yes" if str(seeded.get("tax", "no")).lower() in {"1","true","yes","y"} else "no"
            t3 = (
                f"prefer {', '.join(prefer)}; avoid {', '.join(avoid)}; "
                f"esg {esg}; liquidity {liq}; tax {tax}"
            )
        self._process_turn_3(t3)
        self._extract_booleans(t3)
        self._extract_monthly_contribution(t3)
        self._extract_sector_keywords(t3)

        # Optional Turn 4 — read-back confirmation & single-pass correction
        self._finalize_profile()
        print("\n—— Quick confirmation of your profile ——")
        print(self._profile_readback())
        if interactive:
            confirm = self._safe_input("\nDoes this look accurate? (Y to confirm / N to adjust)\n> ", default="y").strip().lower()
            if confirm == "n":
                fix = self._safe_input(
                    "Please specify updates (e.g., conservative, $300/month, avoid Energy):\n> ",
                    default="",
                )
                self._process_turn_3(fix)
                self._extract_booleans(fix)
                self._extract_monthly_contribution(fix)
                self._extract_sector_keywords(fix)
                self._finalize_profile()

        print("\n✅ Profile complete. Thank you.")
        print(f"   Profile ID: {self.profile['profile_id']}")
        return self.profile

    # ------------------ Turn processors ------------------
    def _process_turn_1(self, response: str):
        self._extract_amount(response)
        self._extract_timeline(response)
        self._extract_goals(response)
        self._analyze_sophistication(response)

    def _process_turn_2(self, response: str):
        r = response.lower()
        if any(k in r for k in ["a)", "buying", "aggressive"]):
            self.profile["risk_level"] = "aggressive"
            self.profile["behavioral_traits"] = {"contrarian": True, "patient": True}
        elif any(k in r for k in ["b)", "hold", "moderate", "balanced"]):
            self.profile["risk_level"] = "moderate"
            self.profile["behavioral_traits"] = {"disciplined": True}
        elif any(k in r for k in ["c)", "reduce", "conservative", "safety"]):
            self.profile["risk_level"] = "conservative"
            self.profile["behavioral_traits"] = {"risk_averse": True}
        elif any(k in r for k in ["d)", "exit", "capital preserve"]):
            self.profile["risk_level"] = "conservative"
            self.profile["behavioral_traits"] = {"risk_averse": True}
        self._extract_monthly_contribution(response)

    def _process_turn_3(self, response: str):
        self._extract_sector_keywords(response)
        self._extract_booleans(response)
        self._extract_monthly_contribution(response)

    # ------------------ Finalization ------------------
    def _finalize_profile(self):
        self.profile["profile_id"] = f"AIV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.profile["creation_timestamp"] = datetime.now().isoformat()
        info_completeness = sum(1 for v in self.profile.values() if v) / len(self.profile)
        self.profile["profile_quality"] = round(info_completeness, 2)
        if not self.profile.get("behavioral_traits"):
            self.profile["behavioral_traits"] = self._infer_behavioral_traits()

    def _infer_behavioral_traits(self) -> Dict:
        traits: Dict[str, Any] = {}
        if self.profile["risk_level"] == "aggressive":
            traits["risk_seeking"] = True
            traits["growth_focused"] = True
        elif self.profile["risk_level"] == "conservative":
            traits["risk_averse"] = True
            traits["stability_focused"] = True
        if self.profile.get("monthly_contribution", 0) > 0:
            traits["disciplined_saver"] = True
        if self.profile.get("time_horizon", 0) > 10:
            traits["long_term_thinker"] = True
        return traits

    # ------------------ Extraction helpers ------------------
    def _extract_amount(self, text: str):
        import re
        patterns = [
            r"\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*([kmb])?(?:illion)?",
            r"(\d+(?:,\d+)*)\s*dollars?",
        ]
        t = text.lower()
        for p in patterns:
            m = re.search(p, t)
            if m:
                amt = float(m.group(1).replace(",", ""))
                unit = m.group(2) if len(m.groups()) > 1 else None
                if unit == "k":
                    amt *= 1_000
                elif unit == "m":
                    amt *= 1_000_000
                elif unit == "b":
                    amt *= 1_000_000_000
                self.profile["investment_amount"] = amt
                break

    def _extract_timeline(self, text: str):
        import re
        t = text.lower()
        patterns = [r"(\d+)\s*years?", r"next\s*(\d+)\s*years?"]
        for p in patterns:
            m = re.search(p, t)
            if m:
                self.profile["time_horizon"] = int(m.group(1))
                break
        if "retirement" in t:
            self.profile["time_horizon"] = max(self.profile.get("time_horizon", 0), 20)
        elif any(k in t for k in ["house", "home", "property"]):
            self.profile["time_horizon"] = max(self.profile.get("time_horizon", 0), 5)

    def _extract_goals(self, text: str):
        goals_map = {
            "retirement": ["retire", "retirement"],
            "house": ["house", "home", "property"],
            "education": ["college", "education", "university"],
            "wealth": ["grow", "wealth", "rich", "build", "general"],
        }
        t = text.lower()
        goal_found = False
        for goal, kws in goals_map.items():
            if any(k in t for k in kws):
                self.profile["investment_goal"] = goal
                goal_found = True
                break
        
        # If no specific goal found, default to "wealth building"
        if not goal_found:
            self.profile["investment_goal"] = "wealth building"

    def _analyze_sophistication(self, text: str):
        adv = ["etf", "dividend", "volatility", "diversification", "duration", "factor"]
        mid = ["stocks", "bonds", "mutual fund", "index fund"]
        t = text.lower()
        if any(k in t for k in adv):
            self.profile["user_sophistication"] = "advanced"
        elif any(k in t for k in mid):
            self.profile["user_sophistication"] = "intermediate"
        else:
            self.profile["user_sophistication"] = "beginner"

    def _extract_booleans(self, text: str):
        t = text.lower()
        yes = {"yes", "y", "true", "sure", "ok"}
        no = {"no", "n", "false"}

        def guess_bool(segment: str) -> Optional[bool]:
            if any(k in segment for k in yes):
                return True
            if any(k in segment for k in no):
                return False
            return None

        if any(k in t for k in ["liquidity", "liquid"]):
            v = guess_bool(t)
            if v is not None:
                self.profile["needs_liquidity"] = v
        if any(k in t for k in ["esg", "sustainable", "responsible"]):
            v = guess_bool(t)
            if v is not None:
                self.profile["prefers_esg"] = v
        if any(k in t for k in ["tax", "tax-efficient", "taxes"]):
            v = guess_bool(t)
            if v is not None:
                self.profile["tax_sensitive"] = v

    def _extract_monthly_contribution(self, text: str):
        import re
        m = re.search(r"(?:\$)?(\d{1,3}(?:,\d{3})*|\d+)\s*(?:/month|per month|monthly)", text.lower())
        if m:
            self.profile["monthly_contribution"] = float(m.group(1).replace(",", ""))

    def _extract_sector_keywords(self, text: str):
        t = text.lower()
        mapping = {
            "tech": "Technology",
            "technology": "Technology",
            "health": "Healthcare",
            "healthcare": "Healthcare",
            "finance": "Financial",
            "financial": "Financial",
            "energy": "Energy",
            "retail": "Consumer",
            "consumer": "Consumer",
        }
        
        # Check for prefer/avoid context around each sector
        for k, v in mapping.items():
            if k in t:
                # Look for avoid context around this keyword
                pos = t.find(k)
                context_before = t[max(0, pos-20):pos]
                context_after = t[pos:pos+len(k)+20]
                full_context = context_before + context_after
                
                avoid_tokens = {"avoid", "exclude", "no ", "not "}
                prefer_tokens = {"prefer", "like", "want", "overweight"}
                
                is_avoid = any(token in full_context for token in avoid_tokens)
                is_prefer = any(token in full_context for token in prefer_tokens)
                
                if is_avoid and v not in self.profile["avoid_sectors"]:
                    self.profile["avoid_sectors"].append(v)
                elif is_prefer and v not in self.profile["preferred_sectors"]:
                    self.profile["preferred_sectors"].append(v)
                elif not is_avoid and not is_prefer:
                    # Default behavior when no clear preference signal
                    if v not in self.profile["preferred_sectors"]:
                        self.profile["preferred_sectors"].append(v)

    def _profile_readback(self) -> str:
        def yn(b: bool) -> str:
            return "Yes" if b else "No"
        return (
            f"- Amount: ${self.profile.get('investment_amount',0):,.0f}\n"
            f"- Risk: {self.profile.get('risk_level')}\n"
            f"- Horizon: {self.profile.get('time_horizon')} years\n"
            f"- Goal: {self.profile.get('investment_goal') or 'wealth'}\n"
            f"- Contribution: ${self.profile.get('monthly_contribution',0):,.0f}/month\n"
            f"- Preferred sectors: {', '.join(self.profile.get('preferred_sectors') or []) or '—'}\n"
            f"- Avoid sectors: {', '.join(self.profile.get('avoid_sectors') or []) or '—'}\n"
            f"- Needs liquidity: {yn(self.profile.get('needs_liquidity', False))}\n"
            f"- ESG preference: {yn(self.profile.get('prefers_esg', False))}\n"
            f"- Tax sensitive: {yn(self.profile.get('tax_sensitive', False))}\n"
        )


# ------------------------------
# Orchestration / IO helpers
# ------------------------------

def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.error("Failed to create directory '%s': %s", path, e)
        raise


def save_profile(profile: Dict[str, Any], outdir: str) -> Optional[str]:
    try:
        ensure_dir(outdir)
        out_path = os.path.join(outdir, f"user_{profile['profile_id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        logger.info("Saved profile to: %s", out_path)
        return out_path
    except PermissionError as e:
        logger.error("Permission denied when saving profile: %s", e)
        return None
    except OSError as e:
        logger.error("OS error when saving profile: %s", e)
        return None


def agent1_for_workflow(
    interactive: Optional[bool] = None,
    outdir: str = "data/user_preferences",
    seeded: Optional[Dict[str, Any]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Run Agent1 and optionally persist the profile to disk."""
    if interactive is None:
        interactive = sys.stdin.isatty()
    agent = EnhancedAIvestor()
    profile = agent.start_conversation(interactive=interactive, seeded=seeded)
    if save:
        save_profile(profile, outdir)
    return profile


# ------------------------------
# CLI
# ------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enhanced AIvestor Agent 1 (robust I/O)")
    p.add_argument("--outdir", default="data/user_preferences", help="Directory to save profile JSON")
    p.add_argument("--no-save", action="store_true", help="Do not write profile to disk")
    p.add_argument("--test", action="store_true", help="Run unit tests and exit")

    # Non-interactive seeding / overrides
    p.add_argument("--objective", default=None)
    p.add_argument("--amount", type=float, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--risk", default=None, help="a|b|c|d or keywords: aggressive/moderate/conservative")
    p.add_argument("--monthly", type=float, default=None)
    p.add_argument("--prefer", nargs="*", default=None)
    p.add_argument("--avoid", nargs="*", default=None)
    p.add_argument("--esg", default=None, help="yes/no")
    p.add_argument("--liquidity", default=None, help="yes/no")
    p.add_argument("--tax", default=None, help="yes/no")
    return p


def args_to_seeded(args: argparse.Namespace) -> Dict[str, Any]:
    seeded: Dict[str, Any] = {}
    if args.objective: seeded["objective"] = args.objective
    if args.amount is not None: seeded["amount"] = args.amount
    if args.horizon is not None: seeded["horizon"] = args.horizon
    if args.risk: seeded["risk"] = args.risk
    if args.monthly is not None: seeded["monthly"] = args.monthly
    if args.prefer is not None: seeded["prefer"] = args.prefer
    if args.avoid is not None: seeded["avoid"] = args.avoid
    if args.esg is not None: seeded["esg"] = args.esg
    if args.liquidity is not None: seeded["liquidity"] = args.liquidity
    if args.tax is not None: seeded["tax"] = args.tax
    return seeded


# ------------------------------
# Tests
# ------------------------------
class TestEnhancedAIvestor(unittest.TestCase):
    def setUp(self):
        self.agent = EnhancedAIvestor()

    def test_extract_amount_basic(self):
        self.agent._extract_amount("I can invest $25,000 right now")
        self.assertEqual(self.agent.profile["investment_amount"], 25000)

    def test_extract_amount_with_k(self):
        self.agent._extract_amount("around 50k dollars")
        self.assertEqual(self.agent.profile["investment_amount"], 50000)

    def test_timeline_infer(self):
        self.agent._extract_timeline("retirement goal")
        self.assertGreaterEqual(self.agent.profile["time_horizon"], 20)

    def test_goals(self):
        self.agent._extract_goals("saving for a house deposit")
        self.assertEqual(self.agent.profile["investment_goal"], "house")

    def test_booleans(self):
        self.agent._extract_booleans("ESG yes")
        self.assertTrue(self.agent.profile["prefers_esg"])
        
        self.agent._extract_booleans("tax no")
        self.assertFalse(self.agent.profile["tax_sensitive"])
        
        self.agent._extract_booleans("liquidity yes")
        self.assertTrue(self.agent.profile["needs_liquidity"])

    def test_sectors(self):
        self.agent._extract_sector_keywords("prefer Technology, avoid Energy")
        self.assertIn("Technology", self.agent.profile["preferred_sectors"])
        self.assertIn("Energy", self.agent.profile["avoid_sectors"])

    def test_save_profile_creates_dir(self):
        with tempfile.TemporaryDirectory() as td:
            self.agent._finalize_profile()
            path = save_profile(self.agent.profile, os.path.join(td, "user_prefs"))
            self.assertIsNotNone(path)
            self.assertTrue(os.path.exists(path))

    def test_non_interactive_seed(self):
        seeded = {
            "objective": "retirement",
            "amount": 20000,
            "horizon": 7,
            "risk": "b)",
            "monthly": 500,
            "prefer": ["Technology"],
            "avoid": ["Energy"],
            "esg": "yes",
            "liquidity": "no",
            "tax": "no",
        }
        profile = agent1_for_workflow(interactive=False, outdir=tempfile.gettempdir(), seeded=seeded, save=False)
        self.assertEqual(profile["investment_goal"], "retirement")  # should detect retirement objective
        self.assertEqual(profile["risk_level"], "moderate")
        self.assertIn("Technology", profile["preferred_sectors"])


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.test:
        unittest.main(argv=[sys.argv[0]])
        sys.exit(0)

    seeded = args_to_seeded(args)
    interactive = sys.stdin.isatty()
    profile = agent1_for_workflow(
        interactive=interactive,
        outdir=args.outdir,
        seeded=seeded,
        save=(not args.no_save),
    )

    # Print summary
    print("\n📊 Profile Summary:")
    print(f"   Amount: ${profile['investment_amount']:,.0f}")
    print(f"   Risk Level: {profile['risk_level']}")
    print(f"   Time Horizon: {profile['time_horizon']} years")
    print(f"   Goal: {profile.get('investment_goal', 'general wealth building')}")
    print("\n✅ Profile ready for portfolio construction!")
