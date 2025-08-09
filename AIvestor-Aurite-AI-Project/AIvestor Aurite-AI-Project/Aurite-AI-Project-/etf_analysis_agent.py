"""
Bond Analysis Agent - Bond Ranking & Prediction System with LLM Commentary
Analyzes bonds and bond ETFs with GPT-4 powered predictions
Outputs ranked bond forecasts in standardized JSON format
BONDS ONLY VERSION - Focus on fixed income investments
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import yfinance as yf
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import asyncio
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("ℹ️  python-dotenv not installed, using system environment variables")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    try:
        import yfinance
    except ImportError:
        missing_deps.append("yfinance")
    try:
        from openai import OpenAI
    except ImportError:
        missing_deps.append("openai")
    
    if missing_deps:
        raise ImportError(f"Missing required dependencies: {', '.join(missing_deps)}")
    return True

@dataclass
class LLMConfig:
    """LLM configuration for bond analysis"""
    model: str = "gpt-4o"
    temperature: float = 0.6
    max_tokens: int = 1000
    api_key: str = ""

@dataclass
class AgentConfig:
    """Agent configuration with system prompts"""
    name: str
    system_prompt: str
    llm_config: LLMConfig
    analysis_depth: str = "comprehensive"
    enable_llm_commentary: bool = True

@dataclass
class BondPrediction:
    """Individual bond prediction result"""
    ticker: str
    bond_type: str
    duration: str
    label: str
    expected_return: float
    rank: int
    sentiment: str
    confidence: float
    summary: str

@dataclass
class MarketConditions:
    """Current market conditions for bond analysis"""
    vix_level: float
    interest_rate_trend: float
    yield_curve_slope: float
    inflation_expectations: float
    economic_growth: float
    credit_spreads: float
    fed_policy_stance: float

class LLMCommentaryEngine:
    """LLM-powered bond commentary and analysis engine"""
    
    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        self.client = None
        
        if llm_config.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=llm_config.api_key)
                print("✅ OpenAI client initialized successfully")
            except ImportError:
                print("❌ OpenAI library not found. Install with: pip install openai")
            except Exception as e:
                print(f"❌ OpenAI client initialization failed: {e}")
    
    async def generate_bond_commentary(self, symbol: str, bond_type: str, duration: str,
                                     technical_data: Dict, fundamental_data: Dict,
                                     market_conditions: MarketConditions, horizon: str) -> Tuple[float, float, str, str]:
        """Generate LLM-powered commentary and predictions for bonds"""
        
        system_prompt = f"""You are an expert fixed income analyst specializing in bond investments. 
        Analyze bonds considering interest rate sensitivity, credit risk, duration risk, and yield curve dynamics.
        
        Respond ONLY with a JSON object in this exact form
        at:
        {{"expected_return": 0.XX, "confidence": 0.XX, "sentiment": "bullish/bearish/neutral", "summary": "Brief analysis"}}
        
        Consider the {horizon} investment horizon. Be realistic with bond returns (-15% to +20% range)."""
        
        user_prompt = f"""Analyze {symbol} ({bond_type}, {duration} duration) for {horizon} investment horizon:

BOND CHARACTERISTICS:
- Bond Type: {bond_type}
- Duration Category: {duration}
- Current Price: ${technical_data.get('current_price', 0):.2f}
- 1-Year Price Change: {technical_data.get('momentum_1y', 0)*100:.1f}%
- 3-Month Price Change: {technical_data.get('momentum_3m', 0)*100:.1f}%
- Price Volatility: {technical_data.get('volatility', 0.1)*100:.1f}%

FUNDAMENTAL DATA:
- 1-Year Total Return: {fundamental_data.get('returns_1y', 0)*100:.1f}%
- Expense Ratio: {fundamental_data.get('expense_ratio', 0)*100:.2f}%
- Distribution Yield: {fundamental_data.get('dividend_yield', 0)*100:.2f}%

MARKET CONDITIONS:
- VIX Level: {market_conditions.vix_level:.2f}
- Interest Rate Trend: {market_conditions.interest_rate_trend:.3f}
- Yield Curve Slope: {market_conditions.yield_curve_slope:.3f}
- Inflation Expectations: {market_conditions.inflation_expectations:.1%}
- Economic Growth: {market_conditions.economic_growth:.3f}
- Credit Spreads: {market_conditions.credit_spreads:.3f}
- Fed Policy Stance: {market_conditions.fed_policy_stance:.3f}

Focus on: interest rate sensitivity by duration, credit quality considerations, inflation protection, 
and positioning within the current yield curve environment."""

        try:
            if not self.client:
                return self._fallback_bond_analysis(symbol, bond_type, duration, technical_data, market_conditions, horizon)
                
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return self._parse_llm_response_safe(response.choices[0].message.content, symbol)
        except Exception as e:
            logger.error(f"LLM commentary failed for {symbol}: {e}")
            return self._fallback_bond_analysis(symbol, bond_type, duration, technical_data, market_conditions, horizon)
    
    def _parse_llm_response_safe(self, response: str, symbol: str) -> Tuple[float, float, str, str]:
        """Safely parse LLM response with comprehensive error handling"""
        try:
            response = response.strip()
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx <= start_idx:
                raise ValueError("No valid JSON found in response")
                
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            expected_return = float(data.get('expected_return', 0.04))
            confidence = float(data.get('confidence', 0.80))
            sentiment = str(data.get('sentiment', 'neutral')).lower()
            summary = str(data.get('summary', 'Bond analysis based on current market conditions'))
            
            # Conservative bounds for bonds
            expected_return = max(-0.15, min(0.20, expected_return))
            confidence = max(0.60, min(0.95, confidence))
            
            valid_sentiments = ['bullish', 'bearish', 'neutral']
            sentiment = sentiment if sentiment in valid_sentiments else 'neutral'
            
            if len(summary) > 200:
                summary = summary[:197] + "..."
            elif len(summary) < 10:
                summary = f"Bond analysis for {symbol} based on interest rate environment"
            
            return expected_return, confidence, sentiment, summary
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse LLM response for {symbol}: {e}")
            return self._fallback_bond_values(symbol)
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response for {symbol}: {e}")
            return self._fallback_bond_values(symbol)
    
    def _fallback_bond_values(self, symbol: str) -> Tuple[float, float, str, str]:
        """Provide safe fallback values when LLM parsing fails"""
        return (
            0.04,  # 4% expected return for bonds
            0.80,  # 80% confidence
            'neutral',  # neutral sentiment
            f'Quantitative bond analysis for {symbol} based on duration and credit factors'
        )
    
    def _fallback_bond_analysis(self, symbol: str, bond_type: str, duration: str, technical_data: Dict, 
                               market_conditions: MarketConditions, horizon: str) -> Tuple[float, float, str, str]:
        """Fallback bond analysis when LLM fails"""
        base_return = 0.04  # 4% baseline for bonds
        confidence = 0.80
        reasoning = []
        
        # Duration-based analysis
        if duration == 'short':
            base_return = 0.03
            if market_conditions.interest_rate_trend > 0.01:  # Rising rates
                base_return += 0.01  # Short bonds less affected
                reasoning.append("Short duration provides protection in rising rate environment")
        elif duration == 'intermediate':
            base_return = 0.04
            if abs(market_conditions.interest_rate_trend) < 0.005:  # Stable rates
                base_return += 0.01
                reasoning.append("Intermediate duration well-positioned in stable rate environment")
        elif duration == 'long':
            base_return = 0.05
            if market_conditions.interest_rate_trend < -0.01:  # Falling rates
                base_return += 0.03  # Long bonds benefit most
                reasoning.append("Long duration bonds benefit significantly from falling rates")
            elif market_conditions.interest_rate_trend > 0.01:  # Rising rates
                base_return -= 0.04
                reasoning.append("Long duration bonds face headwinds from rising rates")
        
        # Bond type analysis
        if bond_type == 'treasury':
            if market_conditions.vix_level > 0.6:  # High volatility
                base_return += 0.02
                reasoning.append("Flight to quality supports treasuries")
        elif bond_type == 'corporate':
            if market_conditions.economic_growth > 0.02:
                base_return += 0.01
                reasoning.append("Economic growth supports corporate credit")
            if market_conditions.credit_spreads > 0.02:
                base_return -= 0.01
                reasoning.append("Wide credit spreads pressure corporate bonds")
        elif bond_type == 'high_yield':
            base_return += 0.02  # Higher yield premium
            if market_conditions.economic_growth < 0:
                base_return -= 0.03
                reasoning.append("Economic weakness pressures high yield bonds")
            confidence -= 0.10  # Lower confidence for high yield
        elif bond_type == 'tips':
            if market_conditions.inflation_expectations > 0.035:
                base_return += 0.02
                reasoning.append("Rising inflation expectations favor TIPS")
        
        # Add momentum factor (smaller for bonds)
        momentum = technical_data.get('momentum_1y', 0)
        base_return += momentum * 0.15
        
        # Flight to quality adjustment
        if market_conditions.vix_level > 0.5 and bond_type in ['treasury', 'investment_grade']:
            base_return += 0.015
            reasoning.append("Market uncertainty supports high-quality bonds")
        
        sentiment = 'bullish' if base_return > 0.04 else ('bearish' if base_return < 0.02 else 'neutral')
        summary = f"{sentiment.title()} outlook for {bond_type} bonds. " + ". ".join(reasoning[:2])
        if not summary.endswith('.'):
            summary += "."
        
        return max(-0.15, min(0.20, base_return)), confidence, sentiment, summary

class BondAnalysisAgent:
    """Bond Analysis Agent - Bonds Only Version"""
    
    def __init__(self, config: AgentConfig = None):
        """Initialize the Bond Analysis Agent with configuration"""
        check_dependencies()
        
        if config is None:
            config = self._create_default_config()
        
        self.config = config
        self.llm_engine = LLMCommentaryEngine(config.llm_config) if config.enable_llm_commentary else None
        
        # Bond universe organized by duration and type
        self.bond_universe = {
            # Short Duration (0-3 years)
            "short_treasury": ["SHV", "SHY", "VGSH", "SCHO", "STIP"],
            "short_corporate": ["IGSB", "VCSH", "SPSB", "SLQD"],
            
            # Intermediate Duration (3-10 years)  
            "intermediate_treasury": ["IEI", "IEF", "VGIT", "SCHR"],
            "intermediate_corporate": ["IGIB", "VCIT", "SPIB"],
            "intermediate_aggregate": ["AGG", "BND", "SCHZ", "FXNAX"],
            "intermediate_tips": ["TIP", "VTIP", "SCHP"],
            
            # Long Duration (10+ years)
            "long_treasury": ["TLT", "VGLT", "SPTL", "EDV"],
            "long_corporate": ["IGLB", "VCLT", "SPLB"],
            
            # High Yield / Credit
            "high_yield": ["HYG", "JNK", "SHYG", "SJNK", "FALN"],
            "bank_loans": ["SRLN", "BKLN"],
            
            # International Bonds
            "international_treasury": ["BWX", "IAGG"],
            "emerging_market": ["EMB", "VWOB", "PCY"],
            
            # Municipal Bonds
            "municipal": ["VTEB", "MUB", "TFI", "MUNI"],
            
            # Specialty/Inflation Protected
            "tips": ["TIP", "VTIP", "SCHP", "STIP", "LTPZ"],
            "i_bonds": ["IBND", "FLOT"]
        }
        
        # Create comprehensive bond classification
        self.bond_classification = {}
        for category, symbols in self.bond_universe.items():
            for symbol in symbols:
                # Determine bond type and duration
                if 'short' in category:
                    duration = 'short'
                elif 'intermediate' in category or 'aggregate' in category:
                    duration = 'intermediate'  
                elif 'long' in category:
                    duration = 'long'
                else:
                    duration = 'mixed'
                
                if 'treasury' in category:
                    bond_type = 'treasury'
                elif 'corporate' in category:
                    bond_type = 'corporate'
                elif 'high_yield' in category:
                    bond_type = 'high_yield'
                elif 'tips' in category:
                    bond_type = 'tips'
                elif 'municipal' in category:
                    bond_type = 'municipal'
                elif 'international' in category:
                    bond_type = 'international'
                elif 'emerging' in category:
                    bond_type = 'emerging_market'
                elif 'aggregate' in category:
                    bond_type = 'aggregate'
                else:
                    bond_type = 'other'
                
                self.bond_classification[symbol] = {
                    'bond_type': bond_type,
                    'duration': duration,
                    'category': category
                }
        
        # Default analysis symbols - representative across duration and credit spectrum
        self.default_symbols = [
            # Short duration
            "SHY", "VGSH", "IGSB",
            # Intermediate duration
            "IEF", "AGG", "LQD", "TIP",
            # Long duration  
            "TLT", "VCLT",
            # Credit/High yield
            "HYG", "JNK",
            # International
            "BWX", "EMB",
            # Municipal
            "VTEB"
        ]
    
    def _create_default_config(self) -> AgentConfig:
        """Create default configuration"""
        llm_config = LLMConfig(
            model="gpt-4o",
            temperature=0.6,
            max_tokens=1000,
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        return AgentConfig(
            name="Bond Analysis Agent",
            system_prompt="""You are a senior fixed income analyst with 15+ years of experience in bond markets and portfolio management. 
            You specialize in comprehensive bond and bond ETF analysis using advanced duration modeling, credit analysis, and macroeconomic insights.
            
            Your expertise includes:
            - Duration Analysis: Interest rate sensitivity, yield curve positioning, convexity analysis
            - Credit Analysis: Credit quality assessment, default risk modeling, spread analysis
            - Sector Analysis: Government, corporate, municipal, and international bond markets
            - Macroeconomic Integration: Fed policy impact, inflation expectations, economic cycle sensitivity
            - Risk Management: Duration risk, credit risk, liquidity risk, currency risk assessment
            - Portfolio Construction: Bond laddering, barbell/bullet strategies, asset allocation optimization
            
            Provide institutional-grade bond analysis with actionable insights, clear risk assessments, and specific investment recommendations.""",
            llm_config=llm_config,
            analysis_depth="comprehensive",
            enable_llm_commentary=True
        )

    async def classify_bond_with_llm(self, symbol: str, info: Dict) -> Tuple[str, str]:
        """Use LLM to classify bond type and duration"""
        if self.llm_engine and self.llm_engine.client:
            try:
                classification_prompt = f"""Classify the bond ETF {symbol} by type and duration:

BOND TYPES:
- treasury (government bonds, treasury bills, treasury notes)
- corporate (investment grade corporate bonds)
- high_yield (high yield corporate bonds, junk bonds)
- tips (treasury inflation protected securities)
- municipal (tax-free municipal bonds)
- international (foreign government bonds)
- emerging_market (emerging market bonds)
- aggregate (broad bond market funds)

DURATION CATEGORIES:
- short (0-3 years average duration)
- intermediate (3-10 years average duration)  
- long (10+ years average duration)
- mixed (diversified duration)

Asset info: {info.get('longName', '')}, {info.get('category', '')}

Respond with JSON: {{"bond_type": "type", "duration": "duration"}}"""

                response = self.llm_engine.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": classification_prompt}],
                    temperature=0.3,
                    max_tokens=100
                )
                
                response_text = response.choices[0].message.content
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    data = json.loads(response_text[start_idx:end_idx])
                    bond_type = data.get('bond_type', 'treasury')
                    duration = data.get('duration', 'intermediate')
                    return bond_type, duration
                    
            except Exception as e:
                logger.warning(f"LLM bond classification failed for {symbol}: {e}")
        
        # Fallback to predefined classification
        if symbol in self.bond_classification:
            classification = self.bond_classification[symbol]
            return classification['bond_type'], classification['duration']
        
        return 'treasury', 'intermediate'  # Default fallback

    async def analyze_bonds(self, symbols: List[str] = None, analysis_date: str = None, macro_context: Dict = None) -> Dict:
        """Main method to analyze bonds and generate predictions with optional macro context"""
        if symbols is None:
            symbols = self.default_symbols
            
        if analysis_date is None:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Starting bond analysis for {len(symbols)} bonds on {analysis_date}")
        
        # Fetch market data
        market_data = await self._fetch_bond_data(symbols)
        
        if not market_data:
            logger.error("No bond data available")
            return {"error": "No bond data available", "prediction_date": analysis_date}
        
        # Get market conditions (enhanced with macro context if available)
        market_conditions = await self._get_bond_market_conditions(macro_context)
        
        # Analyze each horizon
        horizons = ['next_quarter', 'next_year', 'long_term']
        horizon_results = {}
        
        for horizon in horizons:
            predictions = await self._analyze_horizon(market_data, market_conditions, horizon)
            
            # Sort by expected return and add ranks
            predictions.sort(key=lambda x: x.expected_return, reverse=True)
            for i, prediction in enumerate(predictions):
                prediction.rank = i + 1
            
            # Convert to dict format
            horizon_results[horizon] = [asdict(p) for p in predictions]
        
        # Create final results
        final_results = {
            'prediction_date': analysis_date,
            'analysis_type': 'LLM-enhanced bond analysis with macro context' if macro_context and self.llm_engine and self.llm_engine.client else 'LLM-enhanced bond analysis' if self.llm_engine and self.llm_engine.client else 'Quantitative bond analysis',
            'total_bonds': len(market_data),
            'horizons': horizon_results,
            'market_conditions': asdict(market_conditions),
            'macro_context_used': macro_context is not None
        }
        
        # Add macro context if available
        if macro_context:
            final_results['macro_context'] = self._extract_macro_context(macro_context)
        
        logger.info("Bond analysis completed successfully")
        return final_results

    async def _fetch_bond_data(self, symbols: List[str]) -> Dict:
        """Fetch historical bond data with improved error handling"""
        logger.info(f"Fetching bond data for {len(symbols)} symbols...")
        
        start_date = "2020-01-01"  # 5 years of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        def fetch_bond_data(symbol):
            try:
                ticker = yf.Ticker(symbol)
                ticker.session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                })
                
                # Try different periods if full history fails
                hist_data = None
                for period_years in [5, 3, 2, 1]:
                    try:
                        period_start = (datetime.now() - timedelta(days=365*period_years)).strftime("%Y-%m-%d")
                        hist_data = ticker.history(start=period_start, end=end_date)
                        if len(hist_data) > 50:  # Need sufficient data
                            break
                    except:
                        continue
                
                if hist_data is None or hist_data.empty:
                    logger.warning(f"No data available for {symbol}")
                    return symbol, None
                
                # Get info with fallback
                try:
                    info = ticker.info
                except:
                    info = {}
                
                # Generate label
                label = info.get('longName', info.get('shortName', symbol))
                if len(label) > 35:
                    label = label[:32] + "..."
                
                return symbol, {
                    'price_data': hist_data,
                    'info': info,
                    'label': label
                }
            except Exception as e:
                logger.error(f"Error fetching bond data for {symbol}: {e}")
                return symbol, None
        
        market_data = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_symbol = {executor.submit(fetch_bond_data, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol, timeout=120):
                try:
                    symbol, data = future.result(timeout=20)
                    if data:
                        # Add bond classification
                        bond_type, duration = await self.classify_bond_with_llm(symbol, data['info'])
                        data['bond_type'] = bond_type
                        data['duration'] = duration
                        market_data[symbol] = data
                except Exception as e:
                    logger.error(f"Timeout or error fetching bond data: {e}")
        
        logger.info(f"Successfully fetched data for {len(market_data)} bonds")
        return market_data

    async def _get_bond_market_conditions(self, macro_context: Dict = None) -> MarketConditions:
        """Analyze current bond market conditions"""
        try:
            def fetch_indicator(symbol, default_value):
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="30d")
                    return data['Close'].iloc[-1] if not data.empty else default_value
                except:
                    return default_value
            
            # Market indicators
            vix_current = fetch_indicator("^VIX", 20.0)
            vix_level = min(1.0, vix_current / 40.0)
            
            # Try to get treasury yields for yield curve
            ten_year = fetch_indicator("^TNX", 4.5) / 100  # 10-year yield
            two_year = fetch_indicator("^DGS2", 4.2) / 100  # 2-year yield
            yield_curve_slope = ten_year - two_year
            
            # Base market conditions
            conditions = MarketConditions(
                vix_level=vix_level,
                interest_rate_trend=0.01,  # Default positive trend
                yield_curve_slope=yield_curve_slope,
                inflation_expectations=0.025,  # 2.5% baseline
                economic_growth=0.02,  # 2% baseline growth
                credit_spreads=0.015,  # 150 bps baseline
                fed_policy_stance=0.0  # Neutral stance
            )
            
            # Enhance with macro context if available
            if macro_context:
                conditions = self._enhance_market_conditions_with_macro(conditions, macro_context)
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error fetching bond market conditions: {e}")
            return MarketConditions(0.5, 0.01, 0.01, 0.025, 0.02, 0.015, 0.0)

    def _enhance_market_conditions_with_macro(self, conditions: MarketConditions, macro_context: Dict) -> MarketConditions:
        """Enhance market conditions with macro context"""
        try:
            # Extract macro insights
            macro_insights = self._extract_macro_insights(macro_context)
            
            # Adjust interest rate trend based on macro analysis
            if macro_insights.get('interest_rate_trend') == 'rising':
                conditions.interest_rate_trend = 0.02  # Positive trend
            elif macro_insights.get('interest_rate_trend') == 'falling':
                conditions.interest_rate_trend = -0.01  # Negative trend
            else:
                conditions.interest_rate_trend = 0.0  # Stable
            
            # Adjust inflation expectations
            if macro_insights.get('inflation_outlook') == 'inflationary':
                conditions.inflation_expectations = 0.04  # Higher inflation
            elif macro_insights.get('inflation_outlook') == 'deflationary':
                conditions.inflation_expectations = 0.01  # Lower inflation
            else:
                conditions.inflation_expectations = 0.025  # Stable
            
            # Adjust economic growth expectations
            if macro_insights.get('economic_cycle') == 'expansion':
                conditions.economic_growth = 0.03  # Higher growth
            elif macro_insights.get('economic_cycle') == 'contraction':
                conditions.economic_growth = 0.01  # Lower growth
            else:
                conditions.economic_growth = 0.02  # Stable
            
            # Adjust credit spreads based on economic outlook
            if macro_insights.get('economic_cycle') == 'contraction':
                conditions.credit_spreads = 0.025  # Wider spreads during contraction
            elif macro_insights.get('economic_cycle') == 'expansion':
                conditions.credit_spreads = 0.010  # Tighter spreads during expansion
            else:
                conditions.credit_spreads = 0.015  # Normal spreads
            
            # Adjust Fed policy stance
            if macro_insights.get('interest_rate_trend') == 'rising':
                conditions.fed_policy_stance = 0.02  # Hawkish stance
            elif macro_insights.get('interest_rate_trend') == 'falling':
                conditions.fed_policy_stance = -0.01  # Dovish stance
            else:
                conditions.fed_policy_stance = 0.0  # Neutral stance
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error enhancing market conditions with macro: {e}")
            return conditions

    def _extract_macro_insights(self, macro_context: Dict) -> Dict[str, Any]:
        """Extract key insights from macro context"""
        insights = {
            'sentiment': 'neutral',
            'confidence': 50,
            'interest_rate_trend': 'stable',
            'economic_cycle': 'unknown',
            'inflation_outlook': 'stable'
        }
        
        try:
            # Extract from macro prediction
            if 'macro_prediction' in macro_context:
                pred = macro_context['macro_prediction']
                if 'quarterly_prediction' in pred:
                    q_pred = pred['quarterly_prediction']
                    insights['sentiment'] = q_pred.get('prediction', 'neutral')
                    insights['confidence'] = q_pred.get('confidence', 50)
            
            # Extract from LLM macro analysis
            if 'llm_macro_analysis' in macro_context:
                llm_macro = macro_context['llm_macro_analysis']
                
                # Macro sentiment
                if 'macro_sentiment' in llm_macro:
                    macro_sentiment = llm_macro['macro_sentiment']
                    insights['sentiment'] = macro_sentiment.get('overall_direction', 'neutral')
                    insights['confidence'] = macro_sentiment.get('confidence_probability', 50)
                
                # Economic environment
                if 'economic_environment' in llm_macro:
                    econ_env = llm_macro['economic_environment']
                    insights['economic_cycle'] = econ_env.get('cycle_phase', 'unknown')
                    insights['inflation_outlook'] = econ_env.get('inflation_outlook', 'stable')
                    insights['interest_rate_trend'] = econ_env.get('interest_rate_trend', 'stable')
            
        except Exception as e:
            logger.error(f"Error extracting macro insights: {e}")
        
        return insights

    def _extract_macro_context(self, macro_context: Dict) -> Dict[str, Any]:
        """Extract and format macro context for bond analysis"""
        macro_insights = self._extract_macro_insights(macro_context)
        return {
            'macro_sentiment': macro_insights.get('sentiment', 'neutral'),
            'economic_cycle': macro_insights.get('economic_cycle', 'unknown'),
            'interest_rate_trend': macro_insights.get('interest_rate_trend', 'stable'),
            'inflation_outlook': macro_insights.get('inflation_outlook', 'stable'),
            'macro_analysis_timestamp': macro_context.get('timestamp', datetime.now().isoformat())
        }

    async def _analyze_horizon(self, market_data: Dict, market_conditions: MarketConditions, 
                             horizon: str) -> List[BondPrediction]:
        """Analyze bonds for specific time horizon"""
        predictions = []
        
        for symbol, data in market_data.items():
            try:
                prediction = await self._analyze_single_bond(
                    symbol, data, market_conditions, horizon
                )
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for {horizon}: {e}")
        
        return predictions

    async def _analyze_single_bond(self, symbol: str, data: Dict, 
                                 market_conditions: MarketConditions, horizon: str) -> BondPrediction:
        """Perform comprehensive analysis on single bond"""
        price_data = data['price_data']
        bond_type = data['bond_type']
        duration = data['duration']
        label = data['label']
        info = data['info']
        
        # Calculate technical indicators
        technical = self._calculate_technical_indicators(price_data)
        
        # Calculate fundamental metrics
        fundamental = self._calculate_fundamental_metrics(info, price_data)
        
        # Use LLM for prediction if available
        if self.llm_engine and self.llm_engine.client:
            expected_return, confidence, sentiment, summary = await self.llm_engine.generate_bond_commentary(
                symbol, bond_type, duration, technical, fundamental, market_conditions, horizon
            )
        else:
            expected_return, confidence, sentiment, summary = await self._predict_bond_performance_traditional(
                symbol, bond_type, duration, technical, fundamental, market_conditions, horizon
            )
        
        return BondPrediction(
            ticker=symbol,
            bond_type=bond_type,
            duration=duration,
            label=label,
            expected_return=expected_return,
            rank=0,  # Will be set later
            sentiment=sentiment,
            confidence=confidence,
            summary=summary
        )

    async def _predict_bond_performance_traditional(self, symbol: str, bond_type: str, duration: str,
                                                  technical: Dict, fundamental: Dict, 
                                                  market_conditions: MarketConditions,
                                                  horizon: str) -> Tuple[float, float, str, str]:
        """Traditional rule-based bond prediction engine"""
        
        base_return = 0.04  # 4% baseline for bonds
        confidence = 0.80
        reasoning = []
        
        # Duration-based interest rate sensitivity
        rate_impact = market_conditions.interest_rate_trend
        
        if duration == 'short':
            base_return = 0.03
            base_return -= rate_impact * 2.0  # Low sensitivity
            if rate_impact > 0:
                reasoning.append("Short duration provides rate protection")
        elif duration == 'intermediate':
            base_return = 0.04
            base_return -= rate_impact * 5.0  # Medium sensitivity
            if abs(rate_impact) < 0.005:
                reasoning.append("Intermediate duration balanced for stable rates")
        elif duration == 'long':
            base_return = 0.05
            base_return -= rate_impact * 10.0  # High sensitivity
            if rate_impact < 0:
                reasoning.append("Long duration benefits from rate declines")
            elif rate_impact > 0.01:
                reasoning.append("Long duration vulnerable to rate increases")
        
        # Bond type adjustments
        if bond_type == 'treasury':
            if market_conditions.vix_level > 0.6:
                base_return += 0.015
                reasoning.append("Flight to quality supports treasuries")
        elif bond_type == 'corporate':
            if market_conditions.economic_growth > 0.02:
                base_return += 0.01
            if market_conditions.credit_spreads > 0.02:
                base_return -= 0.015
                reasoning.append("Credit spreads impact corporate bonds")
        elif bond_type == 'high_yield':
            base_return += 0.025  # Yield premium
            if market_conditions.economic_growth < 0:
                base_return -= 0.04
                reasoning.append("Economic weakness pressures high yield")
            confidence -= 0.10
        elif bond_type == 'tips':
            if market_conditions.inflation_expectations > 0.03:
                base_return += 0.02
                reasoning.append("Inflation expectations favor TIPS")
        elif bond_type == 'municipal':
            base_return += 0.005  # Tax advantage
            reasoning.append("Tax-free income advantage")
        elif bond_type == 'emerging_market':
            base_return += 0.015  # Risk premium
            if market_conditions.vix_level > 0.5:
                base_return -= 0.02
                reasoning.append("Risk-off sentiment pressures EM bonds")
            confidence -= 0.05
        
        # Technical momentum (smaller impact for bonds)
        momentum = technical.get('momentum_1y', 0)
        base_return += momentum * 0.10
        
        # Yield curve positioning
        if market_conditions.yield_curve_slope < 0:  # Inverted curve
            if duration == 'short':
                base_return += 0.01
                reasoning.append("Inverted curve favors short duration")
            elif duration == 'long':
                base_return -= 0.01
        
        # Credit environment
        if bond_type in ['corporate', 'high_yield'] and market_conditions.credit_spreads > 0.025:
            base_return -= 0.01
        
        # Determine sentiment
        sentiment = 'bullish' if base_return > 0.045 else ('bearish' if base_return < 0.025 else 'neutral')
        
        # Generate summary
        summary = f"{sentiment.title()} outlook for {bond_type} bonds ({duration} duration). " + ". ".join(reasoning[:2])
        if not summary.endswith('.'):
            summary += "."
        
        # Ensure realistic bounds
        expected_return = max(-0.15, min(0.20, base_return))
        confidence = max(0.60, min(0.95, confidence))
        
        return expected_return, confidence, sentiment, summary

    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate technical indicators from bond price data"""
        if price_data.empty:
            return {
                'current_price': 0,
                'momentum_1y': 0,
                'momentum_3m': 0,
                'rsi': 50,
                'volatility': 0.05
            }
        
        try:
            current_price = float(price_data['Close'].iloc[-1])
            
            # Calculate momentum
            if len(price_data) >= 252:  # 1 year of data
                momentum_1y = float(current_price / price_data['Close'].iloc[-252] - 1)
            else:
                momentum_1y = 0.0
                
            if len(price_data) >= 63:  # 3 months of data
                momentum_3m = float(current_price / price_data['Close'].iloc[-63] - 1)
            else:
                momentum_3m = 0.0
            
            # Calculate RSI
            rsi = self._calculate_rsi(price_data['Close'])
            
            # Calculate volatility (annualized) - typically lower for bonds
            returns = price_data['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0.05
            
            return {
                'current_price': current_price,
                'momentum_1y': momentum_1y,
                'momentum_3m': momentum_3m,
                'rsi': rsi,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {
                'current_price': 0.0,
                'momentum_1y': 0.0,
                'momentum_3m': 0.0,
                'rsi': 50.0,
                'volatility': 0.05
            }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0
                
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception:
            return 50.0

    def _calculate_fundamental_metrics(self, info: Dict, price_data: pd.DataFrame) -> Dict:
        """Calculate fundamental metrics from bond info and price data"""
        try:
            # Calculate returns
            if len(price_data) >= 252:
                returns_1y = float(price_data['Close'].iloc[-1] / price_data['Close'].iloc[-252] - 1)
            else:
                returns_1y = 0.0
            
            # Extract bond-specific fundamental data
            expense_ratio = info.get('annualManagementExpense', info.get('totalExpenseRatio', 0))
            if expense_ratio and expense_ratio > 1:
                expense_ratio = expense_ratio / 100
                
            # For bonds, dividend yield represents distribution yield
            dividend_yield = info.get('yield', info.get('dividendYield', 0))
            if dividend_yield and dividend_yield > 1:
                dividend_yield = dividend_yield / 100
            
            # Bond-specific metrics
            duration = info.get('duration', 0)  # Modified duration if available
            credit_quality = info.get('creditQuality', 'Unknown')
            
            return {
                'returns_1y': returns_1y,
                'expense_ratio': expense_ratio or 0,
                'dividend_yield': dividend_yield or 0,  # Distribution yield for bonds
                'duration': duration,
                'credit_quality': credit_quality,
                'net_assets': info.get('totalAssets', info.get('netAssets', 0)),
                'average_maturity': info.get('averageMaturity', 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating fundamental metrics: {e}")
            return {
                'returns_1y': 0,
                'expense_ratio': 0,
                'dividend_yield': 0,
                'duration': 0,
                'credit_quality': 'Unknown',
                'net_assets': 0,
                'average_maturity': 0
            }

    def save_predictions(self, predictions: Dict, filename: str = None) -> str:
        """Save bond predictions to JSON file"""
        # Ensure analysis_outputs directory exists
        output_dir = "analysis_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bond_analysis_{timestamp}.json"
        
        # Always save to analysis_outputs folder
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            logger.info(f"Bond predictions saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving bond predictions: {e}")
            return ""

    def print_summary(self, predictions: Dict):
        """Print formatted summary of bond predictions"""
        print("\n" + "="*85)
        print("BOND ANALYSIS AGENT - FIXED INCOME PREDICTIONS")
        print("="*85)
        print(f"Analysis Date: {predictions.get('prediction_date', 'Unknown')}")
        print(f"Analysis Type: {predictions.get('analysis_type', 'Unknown')}")
        print(f"Total Bonds: {predictions.get('total_bonds', 0)}")
        
        # Print market conditions if available
        market_conditions = predictions.get('market_conditions', {})
        if market_conditions:
            print(f"\nCurrent Market Conditions:")
            print(f"  VIX Level: {market_conditions.get('vix_level', 0):.2f}")
            print(f"  Interest Rate Trend: {market_conditions.get('interest_rate_trend', 0):+.3f}")
            print(f"  Yield Curve Slope: {market_conditions.get('yield_curve_slope', 0):+.3f}")
            print(f"  Inflation Expectations: {market_conditions.get('inflation_expectations', 0):.1%}")
        
        for horizon, bonds in predictions.get('horizons', {}).items():
            print(f"\n{horizon.replace('_', ' ').title()} Bond Predictions:")
            print("-" * 85)
            print(f"{'#':>2} {'TICKER':>6} {'TYPE':>12} {'DURATION':>10} {'RETURN':>8} {'SENT':>8} {'CONF':>5} {'DESCRIPTION'}")
            print("-" * 85)
            
            for i, bond in enumerate(bonds[:15]):  # Show top 15
                return_pct = bond['expected_return'] * 100
                conf_pct = bond['confidence'] * 100
                description = bond['label'][:25] if len(bond['label']) > 25 else bond['label']
                bond_type = bond['bond_type'][:10] if len(bond['bond_type']) > 10 else bond['bond_type']
                duration = bond['duration'][:8] if len(bond['duration']) > 8 else bond['duration']
                
                print(f"{i+1:2d} {bond['ticker']:>6} {bond_type:>12} {duration:>10} {return_pct:+7.1f}% "
                      f"{bond['sentiment'][:7]:>8} {conf_pct:4.0f}% {description}")
        
        print("\n" + "="*85)

    def get_top_bond_picks(self, predictions: Dict, horizon: str = 'next_year', 
                          bond_type: str = None, duration: str = None, top_n: int = 5) -> List[Dict]:
        """Get top bond picks for specific criteria"""
        try:
            bonds = predictions.get('horizons', {}).get(horizon, [])
            
            # Filter by bond type if specified
            if bond_type:
                bonds = [bond for bond in bonds if bond.get('bond_type') == bond_type]
            
            # Filter by duration if specified
            if duration:
                bonds = [bond for bond in bonds if bond.get('duration') == duration]
            
            # Filter for positive expected returns and reasonable confidence
            good_picks = [bond for bond in bonds 
                         if bond.get('expected_return', 0) > 0.01 
                         and bond.get('confidence', 0) > 0.70]
            
            return good_picks[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting top bond picks: {e}")
            return []

    def analyze_duration_laddering(self, predictions: Dict) -> Dict:
        """Analyze bonds for duration laddering strategy"""
        try:
            bonds = predictions.get('horizons', {}).get('next_year', [])
            
            duration_analysis = {
                'short': [],
                'intermediate': [],
                'long': []
            }
            
            for bond in bonds:
                duration = bond.get('duration', 'intermediate')
                if duration in duration_analysis:
                    duration_analysis[duration].append(bond)
            
            # Get best pick from each duration bucket
            ladder_recommendations = {}
            for duration, bond_list in duration_analysis.items():
                if bond_list:
                    # Sort by expected return and pick top performer
                    sorted_bonds = sorted(bond_list, key=lambda x: x.get('expected_return', 0), reverse=True)
                    best_bond = sorted_bonds[0]
                    
                    ladder_recommendations[duration] = {
                        'recommended_bond': best_bond,
                        'avg_return': sum(b.get('expected_return', 0) for b in bond_list) / len(bond_list),
                        'count': len(bond_list)
                    }
            
            return ladder_recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing duration laddering: {e}")
            return {}

    def analyze_bond_market_sentiment(self, predictions: Dict) -> Dict:
        """Analyze overall bond market sentiment"""
        try:
            bonds = predictions.get('horizons', {}).get('next_year', [])
            
            if not bonds:
                return {}
            
            # Count sentiments
            sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            total_expected_return = 0
            total_confidence = 0
            
            # Type breakdown
            type_performance = {}
            duration_performance = {}
            
            for bond in bonds:
                sentiment = bond.get('sentiment', 'neutral')
                sentiment_counts[sentiment] += 1
                total_expected_return += bond.get('expected_return', 0)
                total_confidence += bond.get('confidence', 0)
                
                # Track by type
                bond_type = bond.get('bond_type', 'unknown')
                if bond_type not in type_performance:
                    type_performance[bond_type] = []
                type_performance[bond_type].append(bond.get('expected_return', 0))
                
                # Track by duration
                duration = bond.get('duration', 'unknown')
                if duration not in duration_performance:
                    duration_performance[duration] = []
                duration_performance[duration].append(bond.get('expected_return', 0))
            
            total_bonds = len(bonds)
            avg_expected_return = total_expected_return / total_bonds
            avg_confidence = total_confidence / total_bonds
            
            # Determine overall sentiment
            bullish_pct = sentiment_counts['bullish'] / total_bonds * 100
            bearish_pct = sentiment_counts['bearish'] / total_bonds * 100
            
            if bullish_pct > 50:
                overall_sentiment = 'bullish'
            elif bearish_pct > 50:
                overall_sentiment = 'bearish'
            else:
                overall_sentiment = 'mixed'
            
            # Calculate averages by type and duration
            type_avg = {t: sum(returns)/len(returns) for t, returns in type_performance.items()}
            duration_avg = {d: sum(returns)/len(returns) for d, returns in duration_performance.items()}
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_breakdown': {
                    'bullish': f"{bullish_pct:.1f}%",
                    'bearish': f"{bearish_pct:.1f}%",
                    'neutral': f"{sentiment_counts['neutral'] / total_bonds * 100:.1f}%"
                },
                'average_expected_return': f"{avg_expected_return * 100:+.1f}%",
                'average_confidence': f"{avg_confidence * 100:.0f}%",
                'total_bonds_analyzed': total_bonds,
                'performance_by_type': {t: f"{r*100:+.1f}%" for t, r in type_avg.items()},
                'performance_by_duration': {d: f"{r*100:+.1f}%" for d, r in duration_avg.items()}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bond market sentiment: {e}")
            return {}


# Utility functions
async def quick_bond_analysis(symbols: List[str] = None, api_key: str = None) -> Dict:
    """Quick bond analysis function for immediate use"""
    try:
        # Create agent
        if api_key:
            llm_config = LLMConfig(api_key=api_key)
            config = AgentConfig(
                name="quick_bond_agent",
                system_prompt="Quick bond analysis",
                llm_config=llm_config,
                enable_llm_commentary=True
            )
            agent = BondAnalysisAgent(config)
        else:
            agent = BondAnalysisAgent()
        
        # Run analysis
        results = await agent.analyze_bonds(symbols)
        return results
        
    except Exception as e:
        logger.error(f"Quick bond analysis failed: {e}")
        return {"error": str(e)}

def create_custom_bond_agent(openai_api_key: str = None, model: str = "gpt-4o") -> BondAnalysisAgent:
    """Create bond agent with custom configuration"""
    
    llm_config = LLMConfig(
        model=model,
        temperature=0.5,  # More conservative for bonds
        max_tokens=1500,
        api_key=openai_api_key
    )
    
    config = AgentConfig(
        name="custom_bond_agent",
        system_prompt="""You are a premium bond analysis agent with institutional-grade fixed income expertise.
        Provide sophisticated duration and credit analysis with precise return forecasts.""",
        llm_config=llm_config,
        analysis_depth="institutional",
        enable_llm_commentary=bool(openai_api_key)
    )
    
    return BondAnalysisAgent(config)

def get_sample_bond_symbols() -> Dict[str, List[str]]:
    """Get sample bond symbols by category for testing"""
    return {
        'treasury_bills': ['SHV', 'BIL'],
        'short_treasury': ['SHY', 'VGSH', 'SCHO'],
        'intermediate_treasury': ['IEF', 'IEI', 'VGIT'],
        'long_treasury': ['TLT', 'VGLT', 'EDV'],
        'corporate_bonds': ['LQD', 'VCIT', 'IGLB'],
        'high_yield': ['HYG', 'JNK', 'SHYG'],
        'tips': ['TIP', 'VTIP', 'SCHP'],
        'international': ['BWX', 'EMB', 'VWOB'],
        'municipal': ['VTEB', 'MUB', 'TFI'],
        'balanced_portfolio': ['SHY', 'IEF', 'TLT', 'LQD', 'HYG', 'TIP', 'BWX']
    }


# Main execution functions
async def main():
    """Main function demonstrating bond agent usage"""
    
    try:
        # Create bond agent
        agent = BondAnalysisAgent()
        
        # Representative bond portfolio across duration and credit spectrum
        bond_symbols = [
            # Short duration
            "SHY", "VGSH", "IGSB",
            # Intermediate duration
            "IEF", "AGG", "LQD", "TIP", "VTEB",
            # Long duration  
            "TLT", "VCLT",
            # Credit/High yield
            "HYG", "JNK",
            # International
            "BWX", "EMB"
        ]
        
        print("🏛️ Starting Bond Analysis Agent...")
        print(f"📊 Analyzing {len(bond_symbols)} bonds across duration and credit spectrum...")
        
        # Run analysis
        predictions = await agent.analyze_bonds(symbols=bond_symbols)
        
        # Display results
        agent.print_summary(predictions)
        
        # Get top picks across all bond types
        top_picks = agent.get_top_bond_picks(predictions, horizon='next_year', top_n=5)
        if top_picks:
            print(f"\n🏆 Top 5 Bond Picks for Next Year:")
            for i, pick in enumerate(top_picks, 1):
                print(f"{i}. {pick['ticker']} ({pick['bond_type']}, {pick['duration']}) - "
                      f"{pick['expected_return']*100:+.1f}% ({pick['sentiment']}, {pick['confidence']*100:.0f}% confidence)")
        
        # Duration-specific analysis
        print(f"\n📈 Best Picks by Duration:")
        for duration in ['short', 'intermediate', 'long']:
            duration_picks = agent.get_top_bond_picks(predictions, duration=duration, top_n=2)
            if duration_picks:
                print(f"{duration.title()} Duration:")
                for pick in duration_picks:
                    print(f"  {pick['ticker']} ({pick['bond_type']}) - {pick['expected_return']*100:+.1f}%")
        
        # Bond type analysis
        print(f"\n🎯 Best Picks by Bond Type:")
        for bond_type in ['treasury', 'corporate', 'high_yield', 'tips']:
            type_picks = agent.get_top_bond_picks(predictions, bond_type=bond_type, top_n=1)
            if type_picks:
                pick = type_picks[0]
                print(f"{bond_type.title()}: {pick['ticker']} - {pick['expected_return']*100:+.1f}%")
        
        # Duration laddering analysis
        ladder_analysis = agent.analyze_duration_laddering(predictions)
        if ladder_analysis:
            print(f"\n🪜 Duration Laddering Recommendations:")
            for duration, data in ladder_analysis.items():
                if 'recommended_bond' in data:
                    bond = data['recommended_bond']
                    print(f"{duration.title()}: {bond['ticker']} - {bond['expected_return']*100:+.1f}% "
                          f"(avg {duration}: {data['avg_return']*100:+.1f}%)")
        
        # Bond market sentiment analysis
        sentiment = agent.analyze_bond_market_sentiment(predictions)
        if sentiment:
            print(f"\n📊 Bond Market Sentiment Analysis:")
            print(f"Overall Sentiment: {sentiment['overall_sentiment'].title()}")
            print(f"Average Expected Return: {sentiment['average_expected_return']}")
            print(f"Performance by Duration: {sentiment['performance_by_duration']}")
            print(f"Performance by Type: {sentiment['performance_by_type']}")
        
        # Save results
        filename = agent.save_predictions(predictions)
        if filename:
            print(f"\n💾 Bond analysis results saved to: {filename}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"❌ Bond analysis failed: {e}")
        return None


async def demo_with_api_key():
    """Demo function showing usage with OpenAI API key"""
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if api_key:
        agent = create_custom_bond_agent(api_key, model="gpt-4o")
        print("✅ Using GPT-4 for enhanced bond analysis")
    else:
        agent = BondAnalysisAgent()
        print("ℹ️  Using quantitative bond analysis (no API key provided)")
    
    # Quick test with representative bonds
    test_symbols = ["SHY", "IEF", "TLT", "LQD", "HYG", "TIP"]
    
    print(f"🧪 Running demo bond analysis on {test_symbols}...")
    results = await agent.analyze_bonds(test_symbols)
    
    agent.print_summary(results)
    
    # Show top picks
    top_picks = agent.get_top_bond_picks(results, top_n=3)
    if top_picks:
        print(f"\n🎯 Demo Top Bond Picks:")
        for i, pick in enumerate(top_picks, 1):
            print(f"{i}. {pick['ticker']} ({pick['bond_type']}) - {pick['expected_return']*100:+.1f}% expected return")
    
    return results

async def duration_analysis(duration: str = None):
    """Analyze specific duration category of bonds"""
    samples = get_sample_bond_symbols()
    
    if duration and f"{duration}_treasury" in samples:
        symbols = samples[f"{duration}_treasury"] + samples.get(f"{duration}_corporate", [])
        print(f"🔍 Analyzing {duration} duration bonds: {', '.join(symbols)}")
    elif duration == 'tips':
        symbols = samples['tips']
        print(f"🔍 Analyzing TIPS bonds: {', '.join(symbols)}")
    elif duration == 'high_yield':
        symbols = samples['high_yield']
        print(f"🔍 Analyzing high yield bonds: {', '.join(symbols)}")
    else:
        # Default balanced portfolio
        symbols = samples['balanced_portfolio']
        duration = 'balanced_portfolio'
        print(f"🔍 Analyzing balanced bond portfolio: {', '.join(symbols)}")
    
    agent = BondAnalysisAgent()
    predictions = await agent.analyze_bonds(symbols)
    
    agent.print_summary(predictions)
    
    # Duration-specific insights
    top_picks = agent.get_top_bond_picks(predictions, top_n=3)
    if top_picks:
        print(f"\n🏅 Top Performers in {duration}:")
        for pick in top_picks:
            print(f"  {pick['ticker']} ({pick['bond_type']}) - {pick['expected_return']*100:+.1f}% "
                  f"expected return, {pick['confidence']*100:.0f}% confidence")
    
    return predictions


def example_usage():
    """Show different ways to use the bond agent"""
    
    print("""
🏛️ Bond Analysis Agent - Usage Examples

BASIC USAGE:
=============
# Simple analysis with default bond symbols
agent = BondAnalysisAgent()
results = await agent.analyze_bonds()

# Custom bond symbols
symbols = ["SHY", "IEF", "TLT", "LQD", "HYG"]
results = await agent.analyze_bonds(symbols)

# Quick analysis function
results = await quick_bond_analysis(["TLT", "AGG"])

LLM INTEGRATION:
================
# With OpenAI API key for enhanced analysis
agent = create_custom_bond_agent("your-api-key-here")
results = await agent.analyze_bonds()

# Using GPT-4 for institutional-grade analysis
agent = create_custom_bond_agent("your-api-key", model="gpt-4o")

ANALYSIS FEATURES:
==================
# Get top bond picks
top_picks = agent.get_top_bond_picks(results, horizon='next_year', top_n=5)

# Duration-specific picks
short_picks = agent.get_top_bond_picks(results, duration='short', top_n=3)
long_picks = agent.get_top_bond_picks(results, duration='long', top_n=3)

# Bond type-specific picks
treasury_picks = agent.get_top_bond_picks(results, bond_type='treasury', top_n=3)
corporate_picks = agent.get_top_bond_picks(results, bond_type='corporate', top_n=3)

# Duration laddering analysis  
ladder = agent.analyze_duration_laddering(results)

# Bond market sentiment analysis
sentiment = agent.analyze_bond_market_sentiment(results)

# Save results
filename = agent.save_predictions(results)

INTERACTIVE MODES:
==================
# Demo with API key input
await demo_with_api_key()

# Duration-specific analysis
await duration_analysis('short')         # Short duration bonds
await duration_analysis('intermediate')  # Intermediate duration bonds  
await duration_analysis('long')          # Long duration bonds
await duration_analysis('tips')          # TIPS bonds
await duration_analysis('high_yield')    # High yield bonds

SAMPLE BOND SYMBOLS:
====================
samples = get_sample_bond_symbols()
print(samples['treasury_bills'])    # Treasury bills
print(samples['short_treasury'])    # Short treasury bonds
print(samples['long_treasury'])     # Long treasury bonds
print(samples['corporate_bonds'])   # Corporate bonds
print(samples['high_yield'])        # High yield bonds
print(samples['tips'])              # TIPS bonds
print(samples['municipal'])         # Municipal bonds

INSTALLATION:
=============
pip install pandas numpy yfinance openai python-dotenv

ENVIRONMENT SETUP:
==================
# .env file
OPENAI_API_KEY=your_openai_api_key_here

COMMAND LINE OPTIONS:
=====================
python bond_agent.py                    # Run standard analysis
python bond_agent.py demo               # Interactive demo
python bond_agent.py short              # Analyze short duration
python bond_agent.py long               # Analyze long duration  
python bond_agent.py tips               # Analyze TIPS bonds
python bond_agent.py high_yield         # Analyze high yield bonds
python bond_agent.py examples           # Show this help
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "demo":
            asyncio.run(demo_with_api_key())
        elif command in ["short", "intermediate", "long", "tips", "high_yield"]:
            asyncio.run(duration_analysis(command))
        elif command == "examples":
            example_usage()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: demo, short, intermediate, long, tips, high_yield, examples")
    else:
        # Run standard analysis
        asyncio.run(main())