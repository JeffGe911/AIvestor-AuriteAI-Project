"""
Gold Analysis Agent - Gold Market Analysis & Prediction System with Macro Integration
Analyzes gold futures and gold ETFs with macro context integration
Outputs gold market analysis and investment recommendations
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
    """LLM configuration for gold analysis"""
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
class GoldPrediction:
    """Individual gold prediction result"""
    ticker: str
    gold_type: str
    timeframe: str
    label: str
    expected_return: float
    rank: int
    sentiment: str
    confidence: float
    summary: str

@dataclass
class GoldMarketConditions:
    """Current market conditions for gold analysis"""
    vix_level: float
    dollar_strength: float
    inflation_expectations: float
    interest_rate_trend: float
    economic_growth: float
    geopolitical_risk: float
    real_rates: float

class LLMCommentaryEngine:
    """LLM-powered gold commentary and analysis engine"""
    
    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        self.client = None
        
        if llm_config.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=llm_config.api_key)
                logger.info("✅ OpenAI client initialized for gold analysis")
            except Exception as e:
                logger.warning(f"⚠️  OpenAI client initialization failed: {e}")
    
    async def generate_gold_commentary(self, symbol: str, gold_type: str, timeframe: str,
                                     technical_data: Dict, fundamental_data: Dict,
                                     market_conditions: GoldMarketConditions, horizon: str) -> Tuple[float, float, str, str]:
        """Generate LLM-powered gold analysis commentary"""
        if not self.client:
            return self._fallback_gold_analysis(symbol, gold_type, timeframe, technical_data, market_conditions, horizon)
        
        try:
            system_prompt = f"""You are an expert gold and precious metals analyst. 
            Analyze the provided gold data and market conditions to provide:
            1. Expected return (as a decimal, e.g., 0.05 for 5%)
            2. Confidence level (0.0 to 1.0)
            3. Sentiment (bullish/bearish/neutral)
            4. Brief summary (1-2 sentences)
            
            Respond ONLY with valid JSON format:
            {{
                "expected_return": 0.05,
                "confidence": 0.85,
                "sentiment": "bullish",
                "summary": "Gold shows strong technical momentum with supportive macro conditions."
            }}"""
            
            user_prompt = f"""
            Analyze this gold investment:
            
            Symbol: {symbol}
            Type: {gold_type}
            Timeframe: {timeframe}
            Horizon: {horizon}
            
            Technical Data: {json.dumps(technical_data, indent=2)}
            Fundamental Data: {json.dumps(fundamental_data, indent=2)}
            Market Conditions: {json.dumps(asdict(market_conditions), indent=2)}
            
            Provide analysis in the exact JSON format specified.
            """
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
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
            logger.error(f"Error in LLM gold commentary: {e}")
            return self._fallback_gold_analysis(symbol, gold_type, timeframe, technical_data, market_conditions, horizon)
    
    def _parse_llm_response_safe(self, response: str, symbol: str) -> Tuple[float, float, str, str]:
        """Safely parse LLM response for gold analysis"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                expected_return = float(data.get('expected_return', 0.04))
                confidence = float(data.get('confidence', 0.8))
                sentiment = str(data.get('sentiment', 'neutral'))
                summary = str(data.get('summary', f'Analysis for {symbol}'))
                
                return expected_return, confidence, sentiment, summary
            else:
                return self._fallback_gold_values(symbol)
        except Exception as e:
            logger.error(f"Error parsing LLM response for {symbol}: {e}")
            return self._fallback_gold_values(symbol)
    
    def _fallback_gold_values(self, symbol: str) -> Tuple[float, float, str, str]:
        """Fallback values for gold analysis"""
        return 0.04, 0.7, "neutral", f"Standard analysis for {symbol}"
    
    def _fallback_gold_analysis(self, symbol: str, gold_type: str, timeframe: str, technical_data: Dict, 
                               market_conditions: GoldMarketConditions, horizon: str) -> Tuple[float, float, str, str]:
        """Fallback gold analysis when LLM is not available"""
        # Simple rule-based analysis
        base_return = 0.04  # 4% baseline for gold
        
        # Adjust based on market conditions
        if market_conditions.inflation_expectations > 0.03:
            base_return += 0.02  # Higher inflation helps gold
        if market_conditions.dollar_strength < 0.5:
            base_return += 0.015  # Weak dollar helps gold
        if market_conditions.real_rates < 0:
            base_return += 0.01  # Negative real rates help gold
        
        confidence = 0.75
        sentiment = "bullish" if base_return > 0.05 else "neutral" if base_return > 0.04 else "bearish"
        summary = f"Gold analysis for {symbol}: {sentiment} outlook based on market conditions."
        
        return base_return, confidence, sentiment, summary

class GoldAnalysisAgent:
    """Gold market analysis agent with macro integration"""
    
    def __init__(self, config: AgentConfig = None):
        """Initialize the gold analysis agent"""
        self.config = config or self._create_default_config()
        self.llm_engine = None
        
        if self.config.enable_llm_commentary:
            self.llm_engine = LLMCommentaryEngine(self.config.llm_config)
        
        # Default gold symbols
        self.default_symbols = [
            "GLD", "IAU", "SGOL", "GLDM", "BAR",  # Gold ETFs
            "GC=F", "XAUUSD=X",  # Gold futures and spot
            "GDX", "GDXJ", "NUGT", "DUST"  # Gold miners
        ]
        
        logger.info(f"🤖 {self.config.name} initialized")
    
    def _create_default_config(self) -> AgentConfig:
        """Create default configuration"""
        llm_config = LLMConfig(
            model="gpt-4o",
            temperature=0.6,
            max_tokens=1000,
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        return AgentConfig(
            name="Gold Analysis Agent",
            system_prompt="""You are a senior precious metals analyst with 15+ years of experience in gold markets and commodity trading. 
            You specialize in comprehensive gold and precious metals analysis using advanced supply-demand modeling, technical analysis, and macroeconomic insights.
            
            Your expertise includes:
            - Supply-Demand Analysis: Mining production, central bank reserves, jewelry demand, industrial usage
            - Technical Analysis: Gold price patterns, momentum indicators, support/resistance levels
            - Macroeconomic Integration: Inflation hedge analysis, dollar strength correlation, safe haven characteristics
            - Geopolitical Risk Assessment: Global uncertainty impact, currency devaluation protection
            - Market Structure: Physical vs. paper gold, ETF flows, futures market dynamics
            - Portfolio Integration: Inflation protection, diversification benefits, risk management strategies
            
            Provide institutional-grade gold analysis with actionable insights, clear risk assessments, and specific investment recommendations.""",
            llm_config=llm_config,
            analysis_depth="comprehensive",
            enable_llm_commentary=True
        )
    
    async def classify_gold_with_llm(self, symbol: str, info: Dict) -> Tuple[str, str]:
        """Classify gold type and timeframe using LLM"""
        if not self.llm_engine or not self.llm_engine.client:
            return "ETF", "long-term"
        
        try:
            system_prompt = "Classify the gold investment type and timeframe. Return JSON: {\"type\": \"ETF/Futures/Spot\", \"timeframe\": \"short-term/medium-term/long-term\"}"
            
            user_prompt = f"""
            Classify this gold investment:
            Symbol: {symbol}
            Info: {json.dumps(info, indent=2)}
            
            Return classification in JSON format.
            """
            
            response = await asyncio.to_thread(
                self.llm_engine.client.chat.completions.create,
                model=self.config.llm_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse response
            content = response.choices[0].message.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                data = json.loads(content[json_start:json_end])
                return data.get('type', 'ETF'), data.get('timeframe', 'long-term')
            
        except Exception as e:
            logger.error(f"Error classifying gold {symbol}: {e}")
        
        return "ETF", "long-term"
    
    async def analyze_gold(self, symbols: List[str] = None, analysis_date: str = None, macro_context: Dict = None) -> Dict:
        """Main method to analyze gold investments with optional macro context"""
        if symbols is None:
            symbols = self.default_symbols
            
        if analysis_date is None:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Starting gold analysis for {len(symbols)} symbols on {analysis_date}")
        
        # Fetch market data
        market_data = await self._fetch_gold_data(symbols)
        
        if not market_data:
            logger.error("No gold data available")
            return {"error": "No gold data available", "prediction_date": analysis_date}
        
        # Get market conditions (enhanced with macro context if available)
        market_conditions = await self._get_gold_market_conditions(macro_context)
        
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
            'analysis_type': 'LLM-enhanced gold analysis with macro context' if macro_context else 'Quantitative gold analysis',
            'total_gold_investments': len(market_data),
            'horizons': horizon_results,
            'market_conditions': asdict(market_conditions),
            'macro_context_used': macro_context is not None
        }
        
        # Add macro context if available
        if macro_context:
            final_results['macro_context'] = self._extract_macro_context(macro_context)
        
        logger.info("Gold analysis completed successfully")
        return final_results
    
    async def _fetch_gold_data(self, symbols: List[str]) -> Dict:
        """Fetch historical gold data with improved error handling"""
        logger.info(f"Fetching gold data for {len(symbols)} symbols...")
        
        start_date = "2020-01-01"  # 5 years of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        def fetch_gold_data(symbol):
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
                logger.error(f"Error fetching gold data for {symbol}: {e}")
                return symbol, None
        
        market_data = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_symbol = {executor.submit(fetch_gold_data, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, data = future.result()
                    if data:
                        # Classify gold type and timeframe
                        gold_type, timeframe = await self.classify_gold_with_llm(symbol, data['info'])
                        data['gold_type'] = gold_type
                        data['timeframe'] = timeframe
                        market_data[symbol] = data
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
        
        return market_data
    
    async def _get_gold_market_conditions(self, macro_context: Dict = None) -> GoldMarketConditions:
        """Analyze current gold market conditions with macro context"""
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
            
            # Dollar index
            dollar_index = fetch_indicator("DX-Y.NYB", 100.0)
            dollar_strength = dollar_index / 120.0  # Normalize
            
            # Gold spot price
            gold_spot = fetch_indicator("GC=F", 2000.0)
            
            # Base market conditions
            conditions = GoldMarketConditions(
                vix_level=vix_level,
                dollar_strength=dollar_strength,
                inflation_expectations=0.025,  # 2.5% baseline
                interest_rate_trend=0.01,  # Slight positive trend
                economic_growth=0.02,  # 2% baseline
                geopolitical_risk=0.5,  # Moderate risk
                real_rates=0.01  # Slightly positive real rates
            )
            
            # Enhance with macro context if available
            if macro_context:
                conditions = self._enhance_market_conditions_with_macro(conditions, macro_context)
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error fetching gold market conditions: {e}")
            return GoldMarketConditions(0.5, 0.5, 0.025, 0.01, 0.02, 0.5, 0.01)
    
    def _enhance_market_conditions_with_macro(self, conditions: GoldMarketConditions, macro_context: Dict) -> GoldMarketConditions:
        """Enhance market conditions with macro context"""
        try:
            # Extract macro insights
            macro_insights = self._extract_macro_insights(macro_context)
            
            # Adjust inflation expectations
            if macro_insights.get('inflation_outlook') == 'inflationary':
                conditions.inflation_expectations = 0.04
            elif macro_insights.get('inflation_outlook') == 'deflationary':
                conditions.inflation_expectations = 0.01
            
            # Adjust interest rate trend
            if macro_insights.get('interest_rate_trend') == 'rising':
                conditions.interest_rate_trend = 0.02
            elif macro_insights.get('interest_rate_trend') == 'falling':
                conditions.interest_rate_trend = -0.01
            
            # Adjust economic growth
            if macro_insights.get('economic_cycle') == 'expansion':
                conditions.economic_growth = 0.03
            elif macro_insights.get('economic_cycle') == 'contraction':
                conditions.economic_growth = 0.01
            
            # Adjust real rates
            conditions.real_rates = conditions.interest_rate_trend - conditions.inflation_expectations
            
            # Adjust geopolitical risk based on sentiment
            if macro_insights.get('sentiment') == 'bearish':
                conditions.geopolitical_risk = 0.7  # Higher risk
            elif macro_insights.get('sentiment') == 'bullish':
                conditions.geopolitical_risk = 0.3  # Lower risk
            
        except Exception as e:
            logger.error(f"Error enhancing market conditions with macro context: {e}")
        
        return conditions
    
    def _extract_macro_insights(self, macro_context: Dict) -> Dict[str, Any]:
        """Extract key insights from macro context"""
        insights = {
            'sentiment': 'neutral',
            'inflation_outlook': 'stable',
            'interest_rate_trend': 'stable',
            'economic_cycle': 'unknown'
        }
        
        try:
            # Extract from macro prediction
            if 'macro_prediction' in macro_context:
                pred = macro_context['macro_prediction']
                if 'quarterly_prediction' in pred:
                    q_pred = pred['quarterly_prediction']
                    insights['sentiment'] = q_pred.get('prediction', 'neutral')
            
            # Extract from LLM macro analysis
            if 'llm_macro_analysis' in macro_context:
                llm_macro = macro_context['llm_macro_analysis']
                
                if 'economic_environment' in llm_macro:
                    econ_env = llm_macro['economic_environment']
                    insights['economic_cycle'] = econ_env.get('cycle_phase', 'unknown')
                    insights['inflation_outlook'] = econ_env.get('inflation_outlook', 'stable')
                    insights['interest_rate_trend'] = econ_env.get('interest_rate_trend', 'stable')
            
        except Exception as e:
            logger.error(f"Error extracting macro insights: {e}")
        
        return insights
    
    def _extract_macro_context(self, macro_context: Dict) -> Dict[str, Any]:
        """Extract and format macro context for gold analysis"""
        return {
            'macro_sentiment': self._extract_macro_insights(macro_context).get('sentiment', 'neutral'),
            'inflation_outlook': self._extract_macro_insights(macro_context).get('inflation_outlook', 'stable'),
            'interest_rate_trend': self._extract_macro_insights(macro_context).get('interest_rate_trend', 'stable'),
            'economic_cycle': self._extract_macro_insights(macro_context).get('economic_cycle', 'unknown'),
            'macro_analysis_timestamp': macro_context.get('timestamp', datetime.now().isoformat())
        }
    
    async def _analyze_horizon(self, market_data: Dict, market_conditions: GoldMarketConditions, 
                             horizon: str) -> List[GoldPrediction]:
        """Analyze gold investments for specific time horizon"""
        predictions = []
        
        for symbol, data in market_data.items():
            try:
                prediction = await self._analyze_single_gold(
                    symbol, data, market_conditions, horizon
                )
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for {horizon}: {e}")
        
        return predictions
    
    async def _analyze_single_gold(self, symbol: str, data: Dict, 
                                 market_conditions: GoldMarketConditions, horizon: str) -> GoldPrediction:
        """Perform comprehensive analysis on single gold investment"""
        price_data = data['price_data']
        gold_type = data['gold_type']
        timeframe = data['timeframe']
        label = data['label']
        info = data['info']
        
        # Calculate technical indicators
        technical = self._calculate_technical_indicators(price_data)
        
        # Calculate fundamental metrics
        fundamental = self._calculate_fundamental_metrics(info, price_data)
        
        # Use LLM for prediction if available
        if self.llm_engine and self.llm_engine.client:
            expected_return, confidence, sentiment, summary = await self.llm_engine.generate_gold_commentary(
                symbol, gold_type, timeframe, technical, fundamental, market_conditions, horizon
            )
        else:
            expected_return, confidence, sentiment, summary = await self._predict_gold_performance_traditional(
                symbol, gold_type, timeframe, technical, fundamental, market_conditions, horizon
            )
        
        return GoldPrediction(
            ticker=symbol,
            gold_type=gold_type,
            timeframe=timeframe,
            label=label,
            expected_return=expected_return,
            rank=0,  # Will be set later
            sentiment=sentiment,
            confidence=confidence,
            summary=summary
        )
    
    async def _predict_gold_performance_traditional(self, symbol: str, gold_type: str, timeframe: str,
                                                  technical: Dict, fundamental: Dict, 
                                                  market_conditions: GoldMarketConditions,
                                                  horizon: str) -> Tuple[float, float, str, str]:
        """Traditional rule-based gold prediction engine"""
        
        base_return = 0.04  # 4% baseline for gold
        confidence = 0.80
        reasoning = []
        
        # Inflation impact
        if market_conditions.inflation_expectations > 0.03:
            base_return += 0.02
            reasoning.append("High inflation supports gold")
        
        # Dollar strength impact
        if market_conditions.dollar_strength < 0.5:
            base_return += 0.015
            reasoning.append("Weak dollar supports gold")
        
        # Real rates impact
        if market_conditions.real_rates < 0:
            base_return += 0.01
            reasoning.append("Negative real rates support gold")
        
        # VIX impact (fear trade)
        if market_conditions.vix_level > 0.6:
            base_return += 0.01
            reasoning.append("High volatility supports gold")
        
        # Geopolitical risk
        if market_conditions.geopolitical_risk > 0.6:
            base_return += 0.005
            reasoning.append("Geopolitical risk supports gold")
        
        # Technical indicators
        if technical.get('rsi', 50) < 30:
            base_return += 0.005
            reasoning.append("Oversold conditions")
        elif technical.get('rsi', 50) > 70:
            base_return -= 0.005
            reasoning.append("Overbought conditions")
        
        # Determine sentiment
        if base_return > 0.06:
            sentiment = "bullish"
        elif base_return < 0.02:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        # Create summary
        summary = f"Gold analysis for {symbol}: {sentiment} outlook. " + "; ".join(reasoning[:3])
        
        return base_return, confidence, sentiment, summary
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for gold"""
        try:
            close_prices = price_data['Close']
            
            # RSI
            rsi = self._calculate_rsi(close_prices)
            
            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            
            # Moving averages
            ma_20 = close_prices.rolling(20).mean()
            ma_50 = close_prices.rolling(50).mean()
            
            # Volatility
            volatility = close_prices.pct_change().rolling(20).std()
            
            return {
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0,
                'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0,
                'ma_20': float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else 0,
                'ma_50': float(ma_50.iloc[-1]) if not pd.isna(ma_50.iloc[-1]) else 0,
                'volatility': float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0.02,
                'current_price': float(close_prices.iloc[-1]) if not pd.isna(close_prices.iloc[-1]) else 0
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {'rsi': 50, 'macd': 0, 'macd_signal': 0, 'ma_20': 0, 'ma_50': 0, 'volatility': 0.02, 'current_price': 0}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for gold prices"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_fundamental_metrics(self, info: Dict, price_data: pd.DataFrame) -> Dict:
        """Calculate fundamental metrics for gold"""
        try:
            close_prices = price_data['Close']
            volume = price_data['Volume'] if 'Volume' in price_data.columns else pd.Series([0] * len(close_prices))
            
            # Price metrics
            current_price = float(close_prices.iloc[-1]) if not close_prices.empty else 0
            price_change_1m = float(((close_prices.iloc[-1] - close_prices.iloc[-22]) / close_prices.iloc[-22]) * 100) if len(close_prices) > 22 else 0
            price_change_3m = float(((close_prices.iloc[-1] - close_prices.iloc[-66]) / close_prices.iloc[-66]) * 100) if len(close_prices) > 66 else 0
            
            # Volume metrics
            avg_volume = float(volume.tail(20).mean()) if not volume.empty else 0
            current_volume = float(volume.iloc[-1]) if not volume.empty else 0
            
            return {
                'current_price': current_price,
                'price_change_1m': price_change_1m,
                'price_change_3m': price_change_3m,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'expense_ratio': info.get('expenseRatio', None)
            }
        except Exception as e:
            logger.error(f"Error calculating fundamental metrics: {e}")
            return {'current_price': 0, 'price_change_1m': 0, 'price_change_3m': 0, 'avg_volume': 0, 'current_volume': 0, 'volume_ratio': 1.0}
    
    def save_predictions(self, predictions: Dict, filename: str = None) -> str:
        """Save gold predictions to JSON file"""
        # Ensure analysis_outputs directory exists
        output_dir = "analysis_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gold_predictions_{timestamp}.json"
        
        # Always save to analysis_outputs folder
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"✅ Gold predictions saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ Error saving gold predictions: {e}")
            return ""
    
    def print_summary(self, predictions: Dict):
        """Print a summary of gold predictions"""
        if 'error' in predictions:
            print(f"❌ Error in gold analysis: {predictions['error']}")
            return
        
        print(f"\n🎯 GOLD ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"📅 Analysis Date: {predictions.get('prediction_date', 'N/A')}")
        print(f"📊 Analysis Type: {predictions.get('analysis_type', 'N/A')}")
        print(f"🏆 Total Investments: {predictions.get('total_gold_investments', 0)}")
        print(f"🔗 Macro Context Used: {predictions.get('macro_context_used', False)}")
        
        # Show top picks for next year
        if 'horizons' in predictions and 'next_year' in predictions['horizons']:
            next_year_picks = predictions['horizons']['next_year']
            print(f"\n🏆 TOP GOLD PICKS (Next Year):")
            for i, pick in enumerate(next_year_picks[:5], 1):
                print(f"{i}. {pick['ticker']}: {pick['expected_return']:.2%} return, {pick['sentiment']} sentiment")
    
    def get_top_gold_picks(self, predictions: Dict, horizon: str = 'next_year', 
                          gold_type: str = None, timeframe: str = None, top_n: int = 5) -> List[Dict]:
        """Get top gold picks based on criteria"""
        if 'error' in predictions or 'horizons' not in predictions:
            return []
        
        horizon_picks = predictions['horizons'].get(horizon, [])
        
        # Filter by criteria if specified
        if gold_type:
            horizon_picks = [pick for pick in horizon_picks if pick.get('gold_type') == gold_type]
        if timeframe:
            horizon_picks = [pick for pick in horizon_picks if pick.get('timeframe') == timeframe]
        
        # Sort by expected return and return top N
        horizon_picks.sort(key=lambda x: x.get('expected_return', 0), reverse=True)
        return horizon_picks[:top_n]
    
    def analyze_gold_market_sentiment(self, predictions: Dict) -> Dict:
        """Analyze overall gold market sentiment"""
        if 'error' in predictions or 'horizons' not in predictions:
            return {'overall_sentiment': 'neutral', 'confidence': 0.5}
        
        all_predictions = []
        for horizon_picks in predictions['horizons'].values():
            all_predictions.extend(horizon_picks)
        
        if not all_predictions:
            return {'overall_sentiment': 'neutral', 'confidence': 0.5}
        
        # Calculate sentiment distribution
        sentiment_counts = {}
        total_confidence = 0
        weighted_sentiment = 0
        
        for pred in all_predictions:
            sentiment = pred.get('sentiment', 'neutral')
            confidence = pred.get('confidence', 0.5)
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_confidence += confidence
            
            # Weight sentiment by confidence
            sentiment_value = 1 if sentiment == 'bullish' else -1 if sentiment == 'bearish' else 0
            weighted_sentiment += sentiment_value * confidence
        
        # Determine overall sentiment
        if weighted_sentiment > 0.5:
            overall_sentiment = 'bullish'
        elif weighted_sentiment < -0.5:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        avg_confidence = total_confidence / len(all_predictions) if all_predictions else 0.5
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'sentiment_distribution': sentiment_counts,
            'weighted_sentiment_score': weighted_sentiment,
            'total_analyzed': len(all_predictions)
        }

# Convenience functions
async def quick_gold_analysis(symbols: List[str] = None, api_key: str = None, macro_context: Dict = None) -> Dict:
    """Quick gold analysis with optional macro context"""
    if symbols is None:
        symbols = ["GLD", "IAU", "GC=F"]
    
    config = AgentConfig(
        name="Quick Gold Analyzer",
        system_prompt="Quick gold market analysis",
        llm_config=LLMConfig(
            model="gpt-4o",
            temperature=0.6,
            max_tokens=1000,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    )
    
    agent = GoldAnalysisAgent(config)
    return await agent.analyze_gold(symbols, macro_context=macro_context)

def create_custom_gold_agent(openai_api_key: str = None, model: str = "gpt-4o") -> GoldAnalysisAgent:
    """Create a custom gold analysis agent"""
    config = AgentConfig(
        name="Custom Gold Analyzer",
        system_prompt="Custom gold market analysis agent",
        llm_config=LLMConfig(
            model=model,
            temperature=0.6,
            max_tokens=1000,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    )
    
    return GoldAnalysisAgent(config)

def get_sample_gold_symbols() -> Dict[str, List[str]]:
    """Get sample gold symbols by category"""
    return {
        "etfs": ["GLD", "IAU", "SGOL", "GLDM", "BAR"],
        "futures": ["GC=F", "XAUUSD=X"],
        "miners": ["GDX", "GDXJ", "NUGT", "DUST"],
        "physical": ["PHYS", "SIVR", "PSLV"]
    }

# Example usage
async def main():
    """Example usage of the gold analysis agent"""
    print("🏆 Gold Analysis Agent Example")
    print("=" * 40)
    
    # Create agent
    agent = GoldAnalysisAgent()
    
    # Analyze gold investments
    symbols = ["GLD", "IAU", "GC=F", "GDX"]
    results = await agent.analyze_gold(symbols)
    
    # Print summary
    agent.print_summary(results)
    
    # Save results
    filename = agent.save_predictions(results)
    if filename:
        print(f"\n💾 Gold analysis results saved to: {filename}")
    
    # Get top picks
    top_picks = agent.get_top_gold_picks(results, horizon='next_year', top_n=3)
    print(f"\n🏆 Top 3 Gold Picks:")
    for pick in top_picks:
        print(f"  - {pick['ticker']}: {pick['expected_return']:.2%} return")

if __name__ == "__main__":
    asyncio.run(main()) 