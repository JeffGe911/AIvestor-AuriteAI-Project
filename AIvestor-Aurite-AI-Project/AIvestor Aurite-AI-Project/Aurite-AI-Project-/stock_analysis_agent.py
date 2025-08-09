import os
import sys
import json
import argparse
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import functools

# Add class-based MCP integration imports
from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

import numpy as np
import pandas as pd
import yfinance as yf

# LLM
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage
except ImportError:
    print("ERROR: Please install LangChain: pip install langchain langchain-openai")
    sys.exit(1)

# Backtesting (optional)
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    print("WARNING: VectorBT not installed, backtesting features will be disabled")
    print("   Optional install: pip install vectorbt")
    VBT_AVAILABLE = False

# =====================
# 1. Configuration and Logging
# =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory configuration
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
PRICE_DIR = DATA_DIR / 'prices'
SIGNAL_DIR = DATA_DIR / 'signals'
BT_DIR = DATA_DIR / 'backtest'

# Environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
START_DATE = os.getenv('START_DATE', '2020-01-01')
END_DATE = os.getenv('END_DATE', '2025-01-01')
FEE_BPS = float(os.getenv('FEE_BPS', '10'))
MAX_LLM_INPUT_CHARS = int(os.getenv('MAX_LLM_INPUT_CHARS', '150000'))

# =====================
# 2. LLM Prompts (Enhanced for GPT-4)
# =====================
ANALYSIS_PROMPT = """
You are a senior investment analyst with 15+ years of experience in equity research and portfolio management. 
Please provide a comprehensive, institutional-grade analysis for the following stock using GPT-4's advanced reasoning capabilities:

{stock_data}

Please structure your analysis as follows:

**1. FUNDAMENTAL ANALYSIS (300 words max)**
- Financial Health Assessment: Revenue growth, profitability trends, cash flow analysis
- Valuation Analysis: PE, PB, EV/EBITDA ratios vs. industry peers and historical averages
- Competitive Position: Market share, moat analysis, competitive advantages
- Growth Prospects: Organic growth, expansion opportunities, innovation pipeline
- Risk Factors: Business model risks, regulatory concerns, market concentration

**2. TECHNICAL ANALYSIS (200 words max)**
- Price Action: Trend analysis, key support/resistance levels, breakout patterns
- Momentum Indicators: RSI, MACD, moving averages, volume analysis
- Volatility Assessment: Historical volatility, implied volatility, risk metrics
- Market Structure: Elliott Wave patterns, Fibonacci retracements, chart patterns

**3. MACROECONOMIC CONTEXT (150 words max)**
- Sector Outlook: Industry trends, regulatory environment, technological disruption
- Economic Sensitivity: Interest rate impact, inflation sensitivity, GDP correlation
- Geopolitical Factors: Trade relations, regulatory changes, global market exposure

**4. INVESTMENT RECOMMENDATION (200 words max)**
- Clear Buy/Hold/Sell recommendation with conviction level (1-10)
- Core investment thesis with 3-5 key supporting points
- Risk-adjusted return expectations with time horizon
- Major risk warnings and mitigation strategies
- Portfolio positioning advice (core vs. satellite allocation)

Please maintain institutional standards: objective, data-driven, and actionable insights.
"""

SIGNAL_PROMPT = """
As a quantitative analyst, generate sophisticated trading signals based on the following comprehensive analysis:

{analysis_text}

Requirements:
1. **Signal Classification**: Strong Buy, Buy, Hold, Sell, Strong Sell with confidence level (1-10)
2. **Core Investment Thesis**: Concise rationale (75 words max) with key catalysts
3. **Risk Assessment**: Primary and secondary risks with probability estimates
4. **Price Targets**: 12-month target price with upside/downside scenarios
5. **Stop Loss**: Risk management levels with position sizing recommendations
6. **Investment Score**: Composite score (1-10) incorporating fundamentals, technicals, and macro factors
7. **Time Horizon**: Optimal holding period with rebalancing triggers

Output in structured JSON format:
{{
  "signal": "Strong Buy",
  "confidence": 8.5,
  "thesis": "Core investment rationale with key catalysts",
  "primary_risk": "Main risk factor with probability",
  "secondary_risk": "Secondary risk consideration",
  "target_price": 150.0,
  "upside_scenario": 180.0,
  "downside_scenario": 120.0,
  "stop_loss": 110.0,
  "position_size": "3-5% of portfolio",
  "investment_score": 8.5,
  "time_horizon": "12-18 months",
  "rebalancing_triggers": ["Earnings miss", "Technical breakdown", "Macro shift"]
}}
"""

RANKING_PROMPT = """
As a Chief Investment Officer with $50B+ AUM experience, rank these stocks using institutional-grade criteria:

{stocks_summary}

**RANKING FRAMEWORK:**

1. **FUNDAMENTAL QUALITY (35% weight)**
   - Financial Health: Revenue growth, profitability, cash flow stability
   - Valuation: Relative and absolute valuation metrics vs. peers
   - Business Model: Competitive moat, market position, innovation capability
   - Management Quality: Track record, capital allocation, governance

2. **TECHNICAL STRENGTH (25% weight)**
   - Price Momentum: Trend strength, relative strength vs. market
   - Technical Indicators: RSI, MACD, moving average alignment
   - Volume Analysis: Institutional buying/selling patterns
   - Chart Patterns: Breakout potential, support/resistance levels

3. **RISK-ADJUSTED RETURNS (25% weight)**
   - Sharpe Ratio: Risk-adjusted performance metrics
   - Downside Protection: Beta, correlation, volatility characteristics
   - Liquidity: Trading volume, bid-ask spreads, market cap
   - Sector Diversification: Portfolio balance and correlation benefits

4. **MACROECONOMIC ALIGNMENT (15% weight)**
   - Economic Cycle: Performance in current economic environment
   - Interest Rate Sensitivity: Fed policy impact assessment
   - Inflation Hedge: Real asset characteristics and pricing power
   - Geopolitical Risk: Global exposure and risk factors

**OUTPUT REQUIREMENTS:**
1. Rank by total investment value (1=highest conviction, higher numbers = lower conviction)
2. Provide composite score (1-10) with breakdown by category
3. Detailed ranking rationale (50 words max per stock)
4. Portfolio allocation recommendations with risk management
5. Market outlook and sector rotation advice

Output in comprehensive JSON format:
{{
  "ranking": [
    {{
      "ticker": "AAPL",
      "rank": 1,
      "total_score": 9.2,
      "fundamental_score": 9.5,
      "technical_score": 8.8,
      "risk_score": 9.0,
      "macro_score": 8.5,
      "rationale": "Exceptional fundamentals, strong technical momentum, defensive characteristics"
    }}
  ],
  "portfolio_allocation": {{
    "AAPL": 0.25,
    "MSFT": 0.20,
    "GOOGL": 0.15,
    "NVDA": 0.15,
    "CASH": 0.25
  }},
  "risk_management": {{
    "max_position_size": 0.25,
    "stop_loss_levels": "15% below entry",
    "rebalancing_frequency": "Quarterly"
  }},
  "market_outlook": "Comprehensive market analysis and investment strategy"
}}
"""

# =====================
# 3. NASDAQ-100 Components
# =====================
def get_nasdaq100_tickers() -> List[str]:
    """Get current NASDAQ-100 component tickers"""
    try:
        # Method 1: Try Wikipedia (most reliable)
        try:
            import pandas as pd
            tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            # Find the table with stock tickers
            for table in tables:
                if 'Symbol' in table.columns or 'Ticker' in table.columns:
                    ticker_col = 'Symbol' if 'Symbol' in table.columns else 'Ticker'
                    tickers = table[ticker_col].tolist()
                    # Clean up tickers
                    tickers = [str(t).strip().upper() for t in tickers if pd.notna(t)]
                    if len(tickers) >= 90:  # Should have ~100 stocks
                        print(f"Retrieved {len(tickers)} NASDAQ-100 tickers from Wikipedia")
                        return tickers[:100]  # Limit to 100
        except Exception as e:
            print(f"Failed to get tickers from Wikipedia: {e}")
        
        # Method 2: Try getting from QQQ ETF (NASDAQ-100 tracking ETF)
        try:
            import yfinance as yf
            qqq = yf.Ticker("QQQ")
            info = qqq.info
            if 'holdings' in info:
                tickers = [h['symbol'] for h in info['holdings']]
                print(f"Retrieved {len(tickers)} tickers from QQQ ETF")
                return tickers
        except Exception as e:
            print(f"Failed to get tickers from QQQ: {e}")
        
        # Method 3: Fallback to hardcoded list (as of 2025)
        nasdaq100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
            'ASML', 'ADBE', 'PEP', 'CSCO', 'CMCSA', 'NFLX', 'TMO', 'INTC', 'AMD', 'TMUS',
            'INTU', 'ISRG', 'TXN', 'QCOM', 'AMGN', 'HON', 'AMAT', 'SBUX', 'BKNG', 'VRTX',
            'GILD', 'ADI', 'PANW', 'MU', 'ADP', 'LRCX', 'MDLZ', 'REGN', 'MELI', 'PYPL',
            'KLAC', 'SNPS', 'CDNS', 'PDD', 'CRWD', 'MAR', 'MRVL', 'NXPI', 'CSX', 'ORLY',
            'FTNT', 'CTAS', 'ADSK', 'PCAR', 'MNST', 'ROP', 'PAYX', 'WDAY', 'CPRT', 'ROST',
            'ODFL', 'AEP', 'FAST', 'KDP', 'EA', 'CTSH', 'VRSK', 'BKR', 'GEHC', 'CHTR',
            'KHC', 'EXC', 'DDOG', 'XEL', 'IDXX', 'CCEP', 'MCHP', 'FANG', 'TTWO', 'CSGP',
            'ON', 'ANSS', 'CDW', 'BIIB', 'ILMN', 'ZS', 'DXCM', 'WBD', 'GFS', 'MDB',
            'MRNA', 'DLTR', 'TEAM', 'WBA', 'ENPH', 'SIRI', 'ALGN', 'DOCU', 'OKTA', 'LCID'
        ]
        
        print(f"Using hardcoded list of {len(nasdaq100_tickers)} NASDAQ-100 tickers")
        return nasdaq100_tickers
        
    except Exception as e:
        print(f"Failed to get NASDAQ-100 tickers: {e}")
        # Return a smaller default list if all methods fail
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST', 'ADBE']

# =====================
# 4. Utility Functions
# =====================
def ensure_dirs():
    """Create required directories"""
    for d in [DATA_DIR, PRICE_DIR, SIGNAL_DIR, BT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def validate_config():
    """Validate configuration"""
    if not OPENAI_API_KEY:
        logger.error("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set environment variable: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    logger.info("Configuration validated")

def save_json(obj: dict, path: Path):
    """Save JSON file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        logger.debug(f"Saved file: {path}")
    except Exception as e:
        logger.error(f"Failed to save file {path}: {e}")

def slice_text_for_llm(text: str, max_chars: int = MAX_LLM_INPUT_CHARS) -> str:
    """Intelligent text slicing"""
    if len(text) <= max_chars:
        return text
    
    logger.info(f"Text too long ({len(text)} chars), performing intelligent slicing...")
    
    # Keep beginning, key parts, and end
    start_size = max_chars // 3
    end_size = max_chars // 3
    middle_size = max_chars - start_size - end_size - 200
    
    start_part = text[:start_size]
    end_part = text[-end_size:]
    
    # Extract key data from middle (parts with numbers and important keywords)
    middle_start = len(text) // 2 - middle_size // 2
    middle_end = len(text) // 2 + middle_size // 2
    middle_part = text[middle_start:middle_end]
    
    result = f"{start_part}\n\n... [Data optimized, original length {len(text)} chars] ...\n\n{middle_part}\n\n... [Data continues] ...\n\n{end_part}"
    
    logger.info(f"Slicing complete: {len(result)} chars")
    return result

# =====================
# 5. Financial Data Provider
# =====================
class YFinanceDataProvider:
    """YFinance data provider"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        logger.info(f"Initialized data provider: {self.ticker}")
    
    def get_company_info(self) -> dict:
        """Get company basic information"""
        try:
            info = self.stock.info
            if not info:
                return {"error": "Unable to retrieve company information"}
            
            return {
                "symbol": info.get('symbol', self.ticker),
                "name": info.get('longName', 'Unknown'),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "market_cap": info.get('marketCap', 0),
                "employees": info.get('fullTimeEmployees', 0),
                "description": info.get('longBusinessSummary', '')[:300] + '...' if info.get('longBusinessSummary') else '',
                "website": info.get('website', ''),
                "currency": info.get('currency', 'USD')
            }
        except Exception as e:
            logger.error(f"Failed to get company info: {e}")
            return {"error": str(e)}
    
    def get_financial_data(self) -> dict:
        """Get financial data"""
        try:
            data = {}
            
            # Get financial statements (last 4 periods)
            try:
                data['income_stmt'] = self.stock.financials.iloc[:, :4]
                data['balance_sheet'] = self.stock.balance_sheet.iloc[:, :4]
                data['cash_flow'] = self.stock.cashflow.iloc[:, :4]
                
                # Quarterly data
                data['quarterly_income'] = self.stock.quarterly_financials.iloc[:, :4]
                data['quarterly_balance'] = self.stock.quarterly_balance_sheet.iloc[:, :4]
                
            except Exception as e:
                logger.warning(f"Failed to get some financial statements: {e}")
            
            logger.info(f"Financial data retrieved: {len(data)} statements")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get financial data: {e}")
            return {}
    
    def get_key_metrics(self) -> dict:
        """Get key financial metrics"""
        try:
            info = self.stock.info
            if not info:
                return {}
            
            # Extract key metrics
            metrics = {}
            
            # Valuation metrics
            valuation_metrics = {
                'pe_ratio': 'trailingPE',
                'forward_pe': 'forwardPE', 
                'pb_ratio': 'priceToBook',
                'ps_ratio': 'priceToSalesTrailing12Months',
                'peg_ratio': 'pegRatio',
                'ev_revenue': 'enterpriseToRevenue',
                'ev_ebitda': 'enterpriseToEbitda'
            }
            
            # Profitability
            profitability_metrics = {
                'roe': 'returnOnEquity',
                'roa': 'returnOnAssets', 
                'gross_margin': 'grossMargins',
                'operating_margin': 'operatingMargins',
                'net_margin': 'netIncomeToCommon'
            }
            
            # Financial health
            financial_health = {
                'debt_to_equity': 'debtToEquity',
                'current_ratio': 'currentRatio',
                'quick_ratio': 'quickRatio',
                'interest_coverage': 'interestCoverage'
            }
            
            # Growth
            growth_metrics = {
                'revenue_growth': 'revenueGrowth',
                'earnings_growth': 'earningsGrowth',
                'earnings_quarterly_growth': 'earningsQuarterlyGrowth'
            }
            
            # Dividends and returns
            dividend_metrics = {
                'dividend_yield': 'dividendYield',
                'payout_ratio': 'payoutRatio',
                'dividend_rate': 'dividendRate'
            }
            
            # Other important metrics
            other_metrics = {
                'beta': 'beta',
                'book_value': 'bookValue',
                'price_to_book': 'priceToBook',
                '52w_high': 'fiftyTwoWeekHigh',
                '52w_low': 'fiftyTwoWeekLow',
                'market_cap': 'marketCap',
                'enterprise_value': 'enterpriseValue'
            }
            
            # Combine all metrics
            all_metrics = {
                **valuation_metrics,
                **profitability_metrics, 
                **financial_health,
                **growth_metrics,
                **dividend_metrics,
                **other_metrics
            }
            
            # Extract data
            for key, info_key in all_metrics.items():
                value = info.get(info_key)
                if value is not None:
                    metrics[key] = value
            
            logger.info(f"Key metrics extracted: {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get key metrics: {e}")
            return {}
    
    def get_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data"""
        try:
            df = self.stock.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True,
                back_adjust=True
            )
            
            if df.empty:
                raise ValueError("No price data retrieved")
            
            logger.info(f"Price data retrieved: {len(df)} trading days")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get price data: {e}")
            raise

# =====================
# 6. Technical Indicators
# =====================
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    try:
        if df.empty or 'Close' not in df.columns:
            logger.error("Invalid price data")
            return df
        
        df = df.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean() 
        df['SMA_200'] = df['Close'].rolling(200, min_periods=1).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = df['Close'].rolling(bb_period, min_periods=1).mean()
        std = df['Close'].rolling(bb_period, min_periods=1).std()
        df['BB_Upper'] = sma + (std * bb_std)
        df['BB_Lower'] = sma - (std * bb_std)
        df['BB_Middle'] = sma
        
        # Bollinger Band position
        bb_range = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / bb_range.replace(0, np.nan)
        df['BB_Position'] = df['BB_Position'].clip(0, 1)
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5D'] = df['Close'].pct_change(5)
        df['Price_Change_20D'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility_20D'] = df['Price_Change'].rolling(20, min_periods=1).std() * np.sqrt(252)
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
        
        logger.info("Technical indicators calculated")
        return df
        
    except Exception as e:
        logger.error(f"Failed to calculate technical indicators: {e}")
        return df

def generate_technical_summary(df: pd.DataFrame, ticker: str) -> str:
    """Generate technical analysis summary"""
    try:
        if df.empty:
            return f"{ticker}: No technical data"
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Basic price info
        price = latest['Close']
        change = price - prev['Close'] if len(df) > 1 else 0
        change_pct = (change / prev['Close'] * 100) if len(df) > 1 and prev['Close'] != 0 else 0
        
        summary = f"Current Price: ${price:.2f} ({change_pct:+.2f}%)\n"
        
        # Trend analysis
        if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_200']):
            trend = "Bullish" if latest['SMA_20'] > latest['SMA_200'] else "Bearish"
            position = "above" if price > latest['SMA_20'] else "below"
            summary += f"Trend: {trend}, price {position} 20-day MA\n"
        
        # RSI
        if pd.notna(latest['RSI']):
            rsi = latest['RSI']
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            summary += f"RSI: {rsi:.1f} ({rsi_status})\n"
        
        # MACD
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            macd_signal = "Bullish Cross" if latest['MACD'] > latest['MACD_Signal'] else "Bearish Cross"
            summary += f"MACD: {macd_signal}\n"
        
        # Volatility
        if pd.notna(latest['Volatility_20D']):
            vol = latest['Volatility_20D'] * 100
            vol_level = "High" if vol > 30 else "Medium" if vol > 15 else "Low"
            summary += f"Annualized Volatility: {vol:.1f}% ({vol_level})\n"
        
        # Bollinger Bands
        if pd.notna(latest['BB_Position']):
            bb_pos = latest['BB_Position']
            if bb_pos > 0.8:
                bb_status = "Near upper band (overbought)"
            elif bb_pos < 0.2:
                bb_status = "Near lower band (oversold)"
            else:
                bb_status = "Middle band area"
            summary += f"Bollinger Bands: {bb_status}\n"
        
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate technical summary: {e}")
        return f"{ticker}: Technical analysis failed"

# =====================
# 7. LLM Analyzer
# =====================
class LLMAnalyzer:
    """LLM Analyzer"""
    
    def __init__(self, api_key: str, model_name: str):
        try:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0.1,
                max_tokens=2000
            )
            logger.info(f"LLM initialized successfully: {model_name}")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise
    
    @functools.lru_cache(maxsize=32)
    def _cached_analyze(self, prompt_hash: str, prompt: str) -> str:
        """Cached LLM call"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Analysis failed: {str(e)}"
    
    def analyze_stock(self, stock_data: str) -> str:
        """Comprehensive stock analysis"""
        try:
            # Apply text slicing
            sliced_data = slice_text_for_llm(stock_data, MAX_LLM_INPUT_CHARS)
            
            prompt = ANALYSIS_PROMPT.format(stock_data=sliced_data)
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            return self._cached_analyze(prompt_hash, prompt)
            
        except Exception as e:
            logger.error(f"Stock analysis failed: {e}")
            return f"Analysis failed: {str(e)}"
    
    def generate_signal(self, analysis_text: str) -> dict:
        """Generate trading signal"""
        try:
            prompt = SIGNAL_PROMPT.format(analysis_text=analysis_text)
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            raw_response = self._cached_analyze(prompt_hash, prompt)
            
            # Parse JSON response
            try:
                # Clean response
                cleaned = raw_response.strip()
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                signal_data = json.loads(cleaned)
                
                # Standardize signal
                signal = signal_data.get('signal', 'Hold').strip()
                if signal.lower() in ['buy']:
                    signal = 'Buy'
                elif signal.lower() in ['sell']:
                    signal = 'Sell'
                else:
                    signal = 'Hold'
                
                # Ensure score is in reasonable range
                score = signal_data.get('score', signal_data.get('investment_score', 5.0))
                if isinstance(score, (int, float)):
                    score = max(1.0, min(10.0, float(score)))
                else:
                    score = 5.0
                
                return {
                    "signal": signal,
                    "reason": signal_data.get('reason', 'No reason provided')[:100],
                    "risk": signal_data.get('risk', 'No risk identified')[:100],
                    "target_price": signal_data.get('target_price'),
                    "score": score,
                    "confidence": "high",
                    "raw_response": raw_response
                }
                
            except json.JSONDecodeError:
                logger.warning("JSON parsing failed, using text analysis")
                
                # Extract signal from text
                signal = 'Hold'
                score = 5.0
                if any(word in raw_response.lower() for word in ['buy', 'purchase', 'long']):
                    signal = 'Buy'
                    score = 7.0
                elif any(word in raw_response.lower() for word in ['sell', 'short']):
                    signal = 'Sell'
                    score = 3.0
                
                return {
                    "signal": signal,
                    "reason": raw_response[:100],
                    "risk": "Parse failed, check raw response",
                    "target_price": None,
                    "score": score,
                    "confidence": "low",
                    "raw_response": raw_response
                }
                
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {
                "signal": "Hold",
                "reason": f"System error: {str(e)}",
                "risk": "System unstable",
                "target_price": None,
                "score": 5.0,
                "confidence": "none",
                "raw_response": ""
            }
    
    def rank_stocks(self, stocks_analysis: List[dict]) -> dict:
        """Rank stocks comprehensively"""
        try:
            if not stocks_analysis:
                return {"error": "No valid stock analysis data"}
            
            # Prepare stock summary data
            stocks_summary = "=== Stock Analysis Summary ===\n\n"
            
            for stock in stocks_analysis:
                ticker = stock.get('ticker', 'Unknown')
                signal_data = stock.get('signal', {})
                company_info = stock.get('company_info', {})
                key_metrics = stock.get('key_metrics', {})
                
                stocks_summary += f"**{ticker} - {company_info.get('name', 'Unknown')}**\n"
                stocks_summary += f"Industry: {company_info.get('sector', 'N/A')} - {company_info.get('industry', 'N/A')}\n"
                stocks_summary += f"Market Cap: {company_info.get('market_cap', 'N/A')}\n"
                
                # Key metrics
                if key_metrics:
                    pe = key_metrics.get('pe_ratio', 'N/A')
                    roe = key_metrics.get('roe', 'N/A')
                    debt_ratio = key_metrics.get('debt_to_equity', 'N/A')
                    stocks_summary += f"PE: {pe}, ROE: {roe}, Debt Ratio: {debt_ratio}\n"
                
                # Investment signal
                signal = signal_data.get('signal', 'Hold')
                score = signal_data.get('score', 5.0)
                reason = signal_data.get('reason', 'No reason')
                risk = signal_data.get('risk', 'No risk description')
                
                stocks_summary += f"Signal: {signal} (Score: {score}/10)\n"
                stocks_summary += f"Reason: {reason}\n"
                stocks_summary += f"Risk: {risk}\n\n"
            
            # Apply text slicing
            sliced_summary = slice_text_for_llm(stocks_summary, MAX_LLM_INPUT_CHARS)
            
            # Call LLM to generate ranking
            prompt = RANKING_PROMPT.format(stocks_summary=sliced_summary)
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            raw_response = self._cached_analyze(prompt_hash, prompt)
            
            # Parse ranking results
            try:
                cleaned = raw_response.strip()
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                ranking_data = json.loads(cleaned)
                
                # Validate ranking data
                ranking_list = ranking_data.get('ranking', [])
                if not ranking_list:
                    raise ValueError("No valid ranking data found")
                
                # Ensure ranking completeness and consistency
                validated_ranking = []
                for item in ranking_list:
                    if isinstance(item, dict) and 'ticker' in item:
                        validated_ranking.append({
                            'ticker': item.get('ticker', 'Unknown'),
                            'rank': int(item.get('rank', 999)),
                            'score': float(item.get('score', 5.0)),
                            'reason': item.get('reason', 'No ranking reason')[:50]
                        })
                
                # Sort by rank
                validated_ranking.sort(key=lambda x: x['rank'])
                
                result = {
                    "ranking": validated_ranking,
                    "portfolio_allocation": ranking_data.get('portfolio_allocation', {}),
                    "market_outlook": ranking_data.get('market_outlook', 'No market outlook'),
                    "ranking_criteria": "Fundamental Quality (40%) + Technical Strength (30%) + Risk Assessment (30%)",
                    "total_stocks": len(validated_ranking),
                    "raw_response": raw_response
                }
                
                logger.info(f"Stock ranking complete: {len(validated_ranking)} stocks")
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Ranking result parsing failed: {e}")
                
                # Fallback ranking: based on signals and scores
                fallback_ranking = []
                for i, stock in enumerate(stocks_analysis):
                    signal_data = stock.get('signal', {})
                    signal = signal_data.get('signal', 'Hold')
                    score = signal_data.get('score', 5.0)
                    
                    # Simple scoring system
                    if signal == 'Buy':
                        base_score = score + 2
                    elif signal == 'Sell':
                        base_score = score - 2
                    else:
                        base_score = score
                    
                    fallback_ranking.append({
                        'ticker': stock.get('ticker', 'Unknown'),
                        'rank': i + 1,
                        'score': max(1.0, min(10.0, base_score)),
                        'reason': f"Auto-ranked based on {signal} signal"
                    })
                
                # Sort by score
                fallback_ranking.sort(key=lambda x: x['score'], reverse=True)
                
                # Reassign ranks
                for i, item in enumerate(fallback_ranking):
                    item['rank'] = i + 1
                
                return {
                    "ranking": fallback_ranking,
                    "portfolio_allocation": {},
                    "market_outlook": "System-generated fallback ranking, manual review recommended",
                    "ranking_criteria": "Simplified ranking based on signal type and score",
                    "total_stocks": len(fallback_ranking),
                    "fallback_used": True,
                    "raw_response": raw_response
                }
                
        except Exception as e:
            logger.error(f"Stock ranking failed: {e}")
            return {
                "error": str(e),
                "ranking": [],
                "portfolio_allocation": {},
                "market_outlook": "Ranking feature temporarily unavailable"
            }

# =====================
# 8. Backtesting
# =====================
def simple_backtest(signal_data: dict, price_df: pd.DataFrame) -> dict:
    """Simplified backtesting"""
    try:
        if not VBT_AVAILABLE:
            return {"error": "VectorBT not installed, backtesting unavailable"}
        
        if price_df.empty:
            return {"error": "Insufficient price data"}
        
        # Create simple buy-and-hold strategy for comparison
        initial_price = price_df['Close'].iloc[0]
        final_price = price_df['Close'].iloc[-1]
        buy_hold_return = (final_price / initial_price) - 1
        
        # Calculate basic statistics
        returns = price_df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        result = {
            "buy_hold_return": float(buy_hold_return),
            "annualized_volatility": float(volatility),
            "sharpe_ratio": float(buy_hold_return / volatility) if volatility != 0 else 0,
            "max_drawdown": float((price_df['Close'] / price_df['Close'].cummax() - 1).min()),
            "total_days": len(price_df),
            "signal_generated": signal_data.get('signal', 'Hold'),
            "note": "Simplified backtest: Based on buy-and-hold strategy"
        }
        
        logger.info("Simplified backtest complete")
        return result
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {"error": str(e)}

# =====================
# 9. Main Analysis Flow
# =====================
def analyze_single_stock(ticker: str) -> dict:
    """Analyze single stock"""
    try:
        logger.info(f"Starting analysis: {ticker}")
        
        # 1. Initialize data provider
        data_provider = YFinanceDataProvider(ticker)
        
        # 2. Get company info
        company_info = data_provider.get_company_info()
        if "error" in company_info:
            raise ValueError(f"Unable to get company info: {company_info['error']}")
        
        # 3. Get financial data
        financial_data = data_provider.get_financial_data()
        key_metrics = data_provider.get_key_metrics()
        
        # 4. Get price data
        price_df = data_provider.get_price_data(START_DATE, END_DATE)
        price_df = calculate_technical_indicators(price_df)
        
        # 5. Generate technical analysis summary
        technical_summary = generate_technical_summary(price_df, ticker)
        
        # 6. Prepare LLM input data
        stock_data = prepare_stock_data(company_info, financial_data, key_metrics, technical_summary)
        
        # 7. LLM analysis
        analyzer = LLMAnalyzer(OPENAI_API_KEY, MODEL_NAME)
        analysis_result = analyzer.analyze_stock(stock_data)
        
        # 8. Generate trading signal
        signal_result = analyzer.generate_signal(analysis_result)
        
        # 9. Backtest
        backtest_result = simple_backtest(signal_result, price_df)
        
        # 10. Save data
        save_data(ticker, {
            'company_info': company_info,
            'key_metrics': key_metrics,
            'signal': signal_result,
            'analysis': analysis_result,
            'technical_summary': technical_summary
        }, price_df)
        
        # 11. Assemble results
        result = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "company_info": company_info,
            "analysis": analysis_result,
            "technical_summary": technical_summary,
            "key_metrics": key_metrics,
            "signal": signal_result,
            "backtest": backtest_result,
            "data_stats": {
                "price_days": len(price_df),
                "metrics_count": len(key_metrics),
                "financial_reports": len(financial_data),
                "input_size": len(stock_data)
            }
        }
        
        logger.info(f"{ticker} analysis complete: {signal_result['signal']}")
        return result
        
    except Exception as e:
        logger.error(f"{ticker} analysis failed: {e}")
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def analyze_batch_stocks(tickers: List[str], batch_size: int = 20) -> dict:
    """Batch analyze stocks with progress tracking"""
    logger.info(f"Starting batch analysis: {len(tickers)} stocks")
    
    results = {}
    signals = []
    successful_analyses = []
    
    # Process in batches to avoid overwhelming the system
    for batch_start in range(0, len(tickers), batch_size):
        batch_end = min(batch_start + batch_size, len(tickers))
        batch_tickers = tickers[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
        
        for i, ticker in enumerate(batch_tickers, batch_start + 1):
            logger.info(f"Progress: {i}/{len(tickers)} - {ticker}")
            
            result = analyze_single_stock(ticker)
            results[ticker] = result
            
            if "error" not in result and "signal" in result:
                signals.append({
                    "ticker": ticker,
                    "signal": result["signal"]["signal"],
                    "reason": result["signal"]["reason"],
                    "confidence": result["signal"]["confidence"],
                    "score": result["signal"].get("score", 5.0)
                })
                # Save complete analysis for ranking
                successful_analyses.append(result)
    
    # Batch statistics
    successful = [t for t, r in results.items() if "error" not in r]
    failed = [t for t, r in results.items() if "error" in r]
    
    signal_stats = {}
    score_stats = {"total": 0, "count": 0}
    
    for signal in signals:
        sig = signal["signal"]
        signal_stats[sig] = signal_stats.get(sig, 0) + 1
        
        score = signal.get("score", 5.0)
        score_stats["total"] += score
        score_stats["count"] += 1
    
    # Calculate average score
    avg_score = score_stats["total"] / score_stats["count"] if score_stats["count"] > 0 else 0
    
    # Generate stock ranking
    ranking_result = {}
    if len(successful_analyses) >= 2:  # Need at least 2 stocks for ranking
        try:
            logger.info("Starting stock ranking generation...")
            analyzer = LLMAnalyzer(OPENAI_API_KEY, MODEL_NAME)
            ranking_result = analyzer.rank_stocks(successful_analyses)
            logger.info("Stock ranking complete")
        except Exception as e:
            logger.error(f"Stock ranking failed: {e}")
            ranking_result = {"error": f"Ranking generation failed: {str(e)}"}
    else:
        ranking_result = {"note": "At least 2 stocks needed for ranking analysis"}
    
    batch_result = {
        "timestamp": datetime.now().isoformat(),
        "total_requested": len(tickers),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(tickers) if tickers else 0,
        "signal_distribution": signal_stats,
        "average_score": round(avg_score, 2),
        "signals_summary": signals,
        "stock_ranking": ranking_result,
        "detailed_results": results,
        "successful_tickers": successful,
        "failed_tickers": failed,
        "analysis_summary": {
            "buy_signals": signal_stats.get("Buy", 0),
            "hold_signals": signal_stats.get("Hold", 0), 
            "sell_signals": signal_stats.get("Sell", 0),
            "highest_score": max([s.get("score", 0) for s in signals]) if signals else 0,
            "lowest_score": min([s.get("score", 10) for s in signals]) if signals else 0
        }
    }
    
    # Save batch results
    save_json(batch_result, SIGNAL_DIR / "batch_analysis.json")
    
    logger.info(f"Batch analysis complete: {len(successful)}/{len(tickers)} successful")
    
    # Output ranking summary to log
    if "ranking" in ranking_result and ranking_result["ranking"]:
        logger.info("Stock ranking results:")
        for rank_item in ranking_result["ranking"][:10]:  # Show top 10
            ticker = rank_item.get("ticker", "Unknown")
            rank = rank_item.get("rank", "?")
            score = rank_item.get("score", 0)
            reason = rank_item.get("reason", "No reason")
            logger.info(f"  {rank}. {ticker} (Score: {score:.1f}) - {reason}")
    
    return batch_result

# =====================
# 10. Helper Functions
# =====================
def prepare_stock_data(company_info: dict, financial_data: dict, key_metrics: dict, technical_summary: str) -> str:
    """Prepare stock analysis data"""
    try:
        data_sections = []
        
        # Company basic info
        if company_info:
            data_sections.append("=== Company Basic Information ===")
            data_sections.append(f"Company: {company_info.get('name', 'N/A')} ({company_info.get('symbol', 'N/A')})")
            data_sections.append(f"Industry: {company_info.get('sector', 'N/A')} - {company_info.get('industry', 'N/A')}")
            data_sections.append(f"Market Cap: {company_info.get('market_cap', 'N/A'):,}" if isinstance(company_info.get('market_cap'), (int, float)) else f"Market Cap: {company_info.get('market_cap', 'N/A')}")
            data_sections.append(f"Employees: {company_info.get('employees', 'N/A'):,}" if isinstance(company_info.get('employees'), (int, float)) else f"Employees: {company_info.get('employees', 'N/A')}")
            data_sections.append("")
        
        # Key financial metrics
        if key_metrics:
            data_sections.append("=== Key Financial Metrics ===")
            
            # Valuation metrics
            valuation = ["Valuation Metrics:"]
            for key in ['pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio', 'peg_ratio']:
                if key in key_metrics:
                    valuation.append(f"  {key}: {key_metrics[key]}")
            data_sections.extend(valuation)
            
            # Profitability
            profitability = ["Profitability:"]
            for key in ['roe', 'roa', 'gross_margin', 'operating_margin', 'net_margin']:
                if key in key_metrics:
                    profitability.append(f"  {key}: {key_metrics[key]}")
            data_sections.extend(profitability)
            
            # Financial health
            health = ["Financial Health:"]
            for key in ['debt_to_equity', 'current_ratio', 'quick_ratio']:
                if key in key_metrics:
                    health.append(f"  {key}: {key_metrics[key]}")
            data_sections.extend(health)
            
            # Growth
            growth = ["Growth:"]
            for key in ['revenue_growth', 'earnings_growth']:
                if key in key_metrics:
                    growth.append(f"  {key}: {key_metrics[key]}")
            data_sections.extend(growth)
            data_sections.append("")
        
        # Financial statements summary
        if financial_data:
            data_sections.append("=== Financial Statements Summary ===")
            for report_name, report_data in financial_data.items():
                if hasattr(report_data, 'head') and not report_data.empty:
                    data_sections.append(f"{report_name}:")
                    # Show only most important rows
                    summary = report_data.head(8).to_string()
                    data_sections.append(summary)
                    data_sections.append("")
        
        # Technical analysis
        data_sections.append("=== Technical Analysis ===")
        data_sections.append(technical_summary)
        
        return "\n".join(data_sections)
        
    except Exception as e:
        logger.error(f"Failed to prepare stock data: {e}")
        return f"Data preparation failed: {str(e)}"

def save_data(ticker: str, analysis_data: dict, price_df: pd.DataFrame):
    """Save analysis data"""
    try:
        # Save analysis results
        save_json(analysis_data, SIGNAL_DIR / f"{ticker}_analysis.json")
        
        # Save price data
        price_df.to_csv(PRICE_DIR / f"{ticker}_prices.csv")
        
        logger.debug(f"Data saved: {ticker}")
        
    except Exception as e:
        logger.error(f"Failed to save data: {e}")

def system_health_check() -> dict:
    """System health check"""
    health = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {},
        "recommendations": []
    }
    
    try:
        # API key check
        health["checks"]["openai_api_key"] = bool(OPENAI_API_KEY)
        if not OPENAI_API_KEY:
            health["recommendations"].append("Set OPENAI_API_KEY environment variable")
        
        # Dependency check
        dependencies = {
            "pandas": True,
            "numpy": True,
            "yfinance": True,
            "langchain": True
        }
        
        try:
            import pandas, numpy, yfinance
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            dependencies[str(e).split("'")[1]] = False
        
        health["checks"]["dependencies"] = dependencies
        
        # VectorBT check
        health["checks"]["vectorbt"] = VBT_AVAILABLE
        if not VBT_AVAILABLE:
            health["recommendations"].append("Install vectorbt for full backtesting features")
        
        # Network connection check
        try:
            test_stock = yf.Ticker("AAPL")
            test_info = test_stock.info
            health["checks"]["yfinance_api"] = bool(test_info)
        except Exception:
            health["checks"]["yfinance_api"] = False
            health["recommendations"].append("Check network connection, unable to access Yahoo Finance")
        
        # Directory check
        health["checks"]["directories"] = all(d.exists() for d in [DATA_DIR, PRICE_DIR, SIGNAL_DIR])
        
        # Overall assessment
        failed_critical = []
        if not health["checks"]["openai_api_key"]:
            failed_critical.append("OpenAI API Key")
        if not health["checks"]["yfinance_api"]:
            failed_critical.append("YFinance API")
        
        if failed_critical:
            health["status"] = "error"
            health["critical_failures"] = failed_critical
        elif health["recommendations"]:
            health["status"] = "warning"
        
    except Exception as e:
        health["status"] = "error"
        health["error"] = str(e)
    
    return health

# =====================
# 11. CLI Entry Point
# =====================
def main():
    parser = argparse.ArgumentParser(
        description="Optimized LLM Financial Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  Single stock analysis: python financial_agent.py --mode single --ticker AAPL
  Batch analysis: python financial_agent.py --mode batch --tickers AAPL MSFT GOOGL AMZN
  NASDAQ-100 analysis: python financial_agent.py --mode nasdaq100
  Health check: python financial_agent.py --health-check

Environment Variables:
  OPENAI_API_KEY: OpenAI API key (required)
  MODEL_NAME: Model name (optional, default gpt-4o)
  START_DATE: Analysis start date (optional, default 2020-01-01)
  MAX_LLM_INPUT_CHARS: LLM input character limit (optional, default 150000)

Features:
  - Fully based on YFinance, no SEC data required
  - Intelligent data slicing for LLM input limits
  - Combined fundamental and technical analysis
  - Smart stock ranking and portfolio suggestions
  - NASDAQ-100 full index analysis
  - Simplified dependencies for better stability

Batch Analysis Features:
  - Automatic comprehensive stock ranking
  - Based on Fundamentals (40%) + Technicals (30%) + Risk (30%)
  - Recommended portfolio allocation percentages
  - Market environment analysis and investment advice
        """
    )
    
    parser.add_argument('--mode', choices=['single', 'batch', 'nasdaq100'], default='single', help='Analysis mode')
    parser.add_argument('--ticker', default='AAPL', help='Single stock ticker')
    parser.add_argument('--tickers', nargs='+', help='Batch stock tickers')
    parser.add_argument('--health-check', action='store_true', help='System health check')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--show-ranking', action='store_true', help='Show stock ranking (batch mode only)')
    parser.add_argument('--limit', type=int, help='Limit number of stocks for NASDAQ-100 mode')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Health check
        if args.health_check:
            ensure_dirs()
            health = system_health_check()
            print("=" * 50)
            print("System Health Check")
            print("=" * 50)
            print(json.dumps(health, ensure_ascii=False, indent=2))
            
            if health["status"] == "error":
                print("\nERROR: System has critical issues")
                sys.exit(1)
            elif health["status"] == "warning":
                print("\nWARNING: System is usable but warnings should be addressed")
            else:
                print("\nSUCCESS: System is healthy")
            return
        
        # Validate configuration
        validate_config()
        ensure_dirs()
        
        # Execute analysis
        if args.mode == 'single':
            result = analyze_single_stock(args.ticker)
        elif args.mode == 'nasdaq100':
            # Get NASDAQ-100 tickers
            nasdaq100_tickers = get_nasdaq100_tickers()
            
            # Apply limit if specified
            if args.limit and args.limit > 0:
                nasdaq100_tickers = nasdaq100_tickers[:args.limit]
                logger.info(f"Limited to first {args.limit} NASDAQ-100 stocks")
            
            logger.info(f"Analyzing {len(nasdaq100_tickers)} NASDAQ-100 stocks")
            result = analyze_batch_stocks(nasdaq100_tickers)
            
            # Show ranking summary
            if args.show_ranking and "stock_ranking" in result:
                print_ranking_summary(result["stock_ranking"])
        else:
            tickers = args.tickers or ['AAPL', 'MSFT', 'GOOGL']
            result = analyze_batch_stocks(tickers)
            
            # Show ranking summary
            if args.show_ranking and "stock_ranking" in result:
                print_ranking_summary(result["stock_ranking"])
        
        # Output results
        output_json = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"Results saved: {output_path}")
        else:
            # Auto-save to analysis_outputs folder when no output specified
            output_dir = Path("analysis_outputs")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_filename = f"stock_analysis_{timestamp}.json"
            auto_path = output_dir / auto_filename
            
            with open(auto_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"Results auto-saved to: {auto_path}")
            print(output_json)
        
        # Status check
        if isinstance(result, dict) and "error" in result:
            logger.error("ERROR: Analysis process encountered errors")
            sys.exit(1)
        
        logger.info("Analysis complete")
        
    except KeyboardInterrupt:
        logger.info("WARNING: User interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ERROR: Program error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def print_ranking_summary(ranking_data: dict):
    """Print ranking summary to console"""
    try:
        print("\n" + "=" * 60)
        print("Stock Ranking Analysis Results")
        print("=" * 60)
        
        if "error" in ranking_data:
            print(f"ERROR: Ranking analysis failed: {ranking_data['error']}")
            return
        
        if "note" in ranking_data:
            print(f"INFO: {ranking_data['note']}")
            return
        
        # Display ranking
        ranking_list = ranking_data.get("ranking", [])
        if ranking_list:
            print("\nComprehensive Ranking (Criteria: Fundamentals 40% + Technicals 30% + Risk 30%)")
            print("-" * 60)
            
            for i, item in enumerate(ranking_list):
                rank = item.get("rank", i+1)
                ticker = item.get("ticker", "Unknown")
                score = item.get("score", 0)
                reason = item.get("reason", "No reason")
                
                # Add rank number
                print(f"{rank:2d}. {ticker:6s} | Score: {score:4.1f}/10 | {reason}")
        
        # Display portfolio recommendations
        portfolio = ranking_data.get("portfolio_allocation", {})
        if portfolio:
            print("\nRecommended Portfolio Allocation:")
            print("-" * 30)
            total_allocation = 0
            for asset, weight in portfolio.items():
                if isinstance(weight, (int, float)) and weight > 0:
                    percentage = weight * 100
                    print(f"{asset:8s}: {percentage:5.1f}%")
                    total_allocation += percentage
            
            if abs(total_allocation - 100) > 1:  # Allow 1% tolerance
                print(f"{'Cash':8s}: {100-total_allocation:5.1f}%")
        
        # Display market outlook
        outlook = ranking_data.get("market_outlook", "")
        if outlook:
            print(f"\nMarket Outlook:")
            print("-" * 20)
            print(f"{outlook}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"ERROR: Failed to display ranking summary: {e}")

# Add class-based MCP integration imports
from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration classes for MCP integration
@dataclass 
class LLMConfig:
    """LLM configuration for stock analysis"""
    model: str = "gpt-4o"
    temperature: float = 0.6
    max_tokens: int = 1000
    api_key: str = ""

@dataclass
class AgentConfig:
    """Agent configuration for stock analysis"""
    name: str = "Stock Analysis Agent"
    system_prompt: str = "You are an expert stock market analyst."
    llm_config: LLMConfig = None
    analysis_depth: str = "comprehensive"
    enable_llm_commentary: bool = True

@dataclass
class StockPrediction:
    """Stock prediction data structure"""
    symbol: str
    expected_return: float
    confidence: float
    sentiment: str
    summary: str
    rank: int = 0
    horizon: str = "next_quarter"

@dataclass
class StockMarketConditions:
    """Stock market conditions for analysis"""
    volatility_regime: str = "normal"
    trend_direction: str = "neutral"
    market_stress: float = 0.5
    sector_rotation: str = "balanced"
    
def get_sample_stock_symbols() -> List[str]:
    """Get sample stock symbols for testing"""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import yfinance
        import openai
        return True
    except ImportError:
        return False

class LLMCommentaryEngine:
    """LLM commentary engine for stock analysis"""
    
    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        self.client = None
        if self.config.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.config.api_key)
                logger.info("✅ OpenAI client initialized for stock analysis")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI client initialization failed: {e}")

    async def generate_commentary(self, symbol: str, data: Dict) -> str:
        """Generate LLM commentary for a stock"""
        if not self.client:
            return f"Basic analysis for {symbol} - LLM not available"
        
        try:
            prompt = f"Analyze {symbol} stock with market data: {str(data)[:500]}..."
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM commentary failed for {symbol}: {e}")
            return f"Standard analysis for {symbol} - Technical and fundamental metrics evaluated"

class StockAnalysisAgent:
    """Stock market analysis agent with NASDAQ-100 integration and macro context support"""
    
    def __init__(self, config: AgentConfig = None):
        """Initialize the stock analysis agent"""
        self.config = config or self._create_default_config()
        self.llm_engine = None
        
        if self.config.enable_llm_commentary and self.config.llm_config:
            self.llm_engine = LLMCommentaryEngine(self.config.llm_config)
        
        # Use NASDAQ-100 symbols as default for comprehensive analysis
        self.nasdaq100_symbols = get_nasdaq100_tickers()
        self.default_symbols = {
            "nasdaq_100": self.nasdaq100_symbols,
            "large_cap": self.nasdaq100_symbols[:30],   # Top 30 by market cap
            "mid_cap": self.nasdaq100_symbols[30:60],   # Next 30
            "growth": self.nasdaq100_symbols[60:],      # Remaining growth stocks
            "all": self.nasdaq100_symbols               # Full universe for analysis
        }
        
        logger.info(f"🤖 {self.config.name} initialized with {len(self.nasdaq100_symbols)} NASDAQ-100 stocks")
    
    def _create_default_config(self) -> AgentConfig:
        """Create default configuration"""
        llm_config = LLMConfig(
            model="gpt-4o",
            temperature=0.6,
            max_tokens=1000,
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        return AgentConfig(
            name="Stock Analysis Agent",
            system_prompt="""You are a senior investment analyst with 15+ years of experience in equity research and portfolio management. 
            You specialize in comprehensive NASDAQ-100 stock analysis using advanced financial modeling, technical analysis, and macroeconomic insights.
            
            Your expertise includes:
            - Fundamental Analysis: Financial statement analysis, valuation modeling, competitive positioning
            - Technical Analysis: Advanced chart patterns, momentum indicators, risk management
            - Macroeconomic Integration: Interest rate sensitivity, inflation impact, economic cycle analysis
            - Risk Assessment: Volatility analysis, correlation studies, downside protection strategies
            - Portfolio Construction: Asset allocation, sector rotation, risk-adjusted returns optimization
            
            Provide institutional-grade analysis with actionable insights, clear risk assessments, and specific investment recommendations.""",
            llm_config=llm_config,
            analysis_depth="comprehensive",
            enable_llm_commentary=True
        )

    async def analyze_stocks(self, symbols: List[str] = None, analysis_date: str = None, macro_context: Dict = None) -> Dict:
        """Main method to analyze stocks with optional macro context"""
        if symbols is None:
            # Use all NASDAQ-100 symbols for comprehensive analysis
            symbols = self.nasdaq100_symbols
            
        if analysis_date is None:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Starting NASDAQ-100 stock analysis for {len(symbols)} symbols on {analysis_date}")
        
        try:
            # Get market conditions first (enhanced with macro context if available)
            market_conditions = await self._get_stock_market_conditions(macro_context)
            
            # For efficiency, limit analysis to reasonable subset if analyzing full NASDAQ-100
            if len(symbols) > 50:
                logger.info(f"Large symbol set ({len(symbols)}), using sample for analysis")
                # Take a diverse sample: top 30 + random 20 from the rest
                sample_symbols = symbols[:30] + symbols[30::((len(symbols)-30)//20 + 1)][:20]
                symbols = sample_symbols[:50]  # Cap at 50 for performance
                logger.info(f"Using {len(symbols)} representative symbols for analysis")
            
            # Process symbols in parallel batches for better performance
            predictions = []
            batch_size = 10
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {batch_symbols}")
                
                # Use thread pool for parallel processing
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {
                        executor.submit(self._analyze_single_stock_sync, symbol, market_conditions): symbol 
                        for symbol in batch_symbols
                    }
                    
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            prediction = future.result(timeout=30)  # 30 second timeout per stock
                            if prediction:
                                predictions.append(prediction)
                        except Exception as e:
                            logger.warning(f"Failed to analyze {symbol}: {e}")
                            # Create a neutral prediction for failed analysis
                            predictions.append(StockPrediction(
                                symbol=symbol,
                                expected_return=5.0,  # Neutral score
                                confidence=0.3,       # Low confidence
                                sentiment="Hold",
                                summary=f"Analysis failed: {str(e)[:100]}",
                                horizon="next_quarter"
                            ))
            
            # Sort by expected return and add ranks
            predictions.sort(key=lambda x: x.expected_return, reverse=True)
            for i, prediction in enumerate(predictions):
                prediction.rank = i + 1
            
            # Apply macro context adjustments if available
            if macro_context:
                predictions = self._apply_macro_adjustments(predictions, macro_context)
            
            # Create final results compatible with MCP server
            final_results = {
                "prediction_date": analysis_date,
                "total_analyzed": len(symbols),
                "successful_predictions": len(predictions),
                "market_conditions": asdict(market_conditions),
                "horizons": {
                    "next_quarter": [asdict(p) for p in predictions]
                },
                "analysis_summary": f"Analyzed {len(symbols)} NASDAQ-100 stocks, generated {len(predictions)} predictions",
                "macro_context_used": macro_context is not None,
                "performance_metrics": {
                    "analysis_time": datetime.now().isoformat(),
                    "symbols_processed": len(symbols),
                    "success_rate": len(predictions) / len(symbols) if symbols else 0
                }
            }
            
            if macro_context:
                final_results["macro_context"] = macro_context
                
            logger.info(f"✅ Stock analysis completed: {len(predictions)} predictions generated")
            return final_results
            
        except Exception as e:
            logger.error(f"Stock analysis failed: {e}")
            return {"error": str(e), "prediction_date": analysis_date}

    def _analyze_single_stock_sync(self, symbol: str, market_conditions: StockMarketConditions) -> Optional[StockPrediction]:
        """Synchronous single stock analysis for thread pool execution"""
        try:
            # Use existing YFinanceDataProvider for data fetching
            data_provider = YFinanceDataProvider(symbol)
            
            # Get basic company info
            company_info = data_provider.get_company_info()
            if "error" in company_info:
                return None
            
            # Get key metrics (lighter than full financial data)
            key_metrics = data_provider.get_key_metrics()
            
            # Calculate a simple score based on available metrics
            score = self._calculate_stock_score(key_metrics, market_conditions)
            
            # Determine sentiment based on score
            if score >= 7.0:
                sentiment = "Buy"
                confidence = 0.8
            elif score <= 3.0:
                sentiment = "Sell" 
                confidence = 0.7
            else:
                sentiment = "Hold"
                confidence = 0.6
            
            # Generate summary
            sector = company_info.get('sector', 'Unknown')
            market_cap = company_info.get('market_cap', 0)
            pe_ratio = key_metrics.get('pe_ratio', 'N/A')
            
            summary = f"{sector} stock with P/E: {pe_ratio}, Market Cap: ${market_cap:,.0f}" if isinstance(market_cap, (int, float)) else f"{sector} stock"
            
            return StockPrediction(
                symbol=symbol,
                expected_return=score,
                confidence=confidence,
                sentiment=sentiment,
                summary=summary[:200],  # Limit summary length
                horizon="next_quarter"
            )
            
        except Exception as e:
            logger.warning(f"Single stock analysis failed for {symbol}: {e}")
            return None

    def _calculate_stock_score(self, metrics: Dict, market_conditions: StockMarketConditions) -> float:
        """Calculate a stock score based on metrics and market conditions"""
        try:
            score = 5.0  # Base neutral score
            
            # P/E ratio scoring
            pe_ratio = metrics.get('pe_ratio')
            if pe_ratio and isinstance(pe_ratio, (int, float)):
                if 10 <= pe_ratio <= 20:
                    score += 1.0  # Reasonable P/E
                elif pe_ratio < 10:
                    score += 0.5  # Low P/E (value)
                elif pe_ratio > 30:
                    score -= 0.5  # High P/E (expensive)
            
            # Growth metrics
            revenue_growth = metrics.get('revenue_growth')
            if revenue_growth and isinstance(revenue_growth, (int, float)):
                if revenue_growth > 0.1:  # 10%+ growth
                    score += 1.0
                elif revenue_growth < 0:  # Negative growth
                    score -= 1.0
            
            # Profitability metrics
            roe = metrics.get('roe')
            if roe and isinstance(roe, (int, float)):
                if roe > 0.15:  # 15%+ ROE
                    score += 0.5
                elif roe < 0:  # Negative ROE
                    score -= 1.0
            
            # Market conditions adjustment
            if market_conditions.trend_direction == "bullish":
                score += 0.5
            elif market_conditions.trend_direction == "bearish":
                score -= 0.5
            
            # Volatility adjustment
            if market_conditions.volatility_regime == "high":
                score -= 0.3  # Penalize in high volatility
            elif market_conditions.volatility_regime == "low":
                score += 0.3  # Reward in low volatility
            
            # Ensure score is within reasonable bounds
            return max(1.0, min(10.0, score))
            
        except Exception as e:
            logger.warning(f"Score calculation failed: {e}")
            return 5.0  # Return neutral score on error

    def _apply_macro_adjustments(self, predictions: List[StockPrediction], macro_context: Dict) -> List[StockPrediction]:
        """Apply macro context adjustments to predictions"""
        try:
            macro_insights = self._extract_macro_insights(macro_context)
            direction = macro_insights.get("direction", "neutral")
            confidence = macro_insights.get("confidence", 0.5)
            
            # Adjust predictions based on macro direction
            for prediction in predictions:
                if direction == "bullish" and confidence > 0.6:
                    # Boost growth/tech stocks more in bullish environment
                    if any(word in prediction.summary.lower() for word in ['technology', 'growth', 'software']):
                        prediction.expected_return = min(10.0, prediction.expected_return + 1.0)
                        prediction.confidence = min(1.0, prediction.confidence + 0.1)
                elif direction == "bearish" and confidence > 0.6:
                    # Penalize risky stocks more in bearish environment
                    if prediction.expected_return > 7.0:  # High-expectation stocks
                        prediction.expected_return = max(1.0, prediction.expected_return - 1.0)
                        prediction.confidence = max(0.1, prediction.confidence - 0.1)
            
            # Re-sort after adjustments
            predictions.sort(key=lambda x: x.expected_return, reverse=True)
            for i, prediction in enumerate(predictions):
                prediction.rank = i + 1
                
            return predictions
            
        except Exception as e:
            logger.warning(f"Failed to apply macro adjustments: {e}")
            return predictions

    async def _get_stock_market_conditions(self, macro_context: Dict = None) -> StockMarketConditions:
        """Get current market conditions with optional macro enhancement"""
        # Base market conditions
        base_conditions = StockMarketConditions(
            volatility_regime="normal",
            trend_direction="neutral", 
            market_stress=0.5,
            sector_rotation="balanced"
        )
        
        # Enhance with macro context if available
        if macro_context:
            return self._enhance_market_conditions_with_macro(base_conditions, macro_context)
        
        return base_conditions
    
    def _enhance_market_conditions_with_macro(self, base_conditions: StockMarketConditions, macro_context: Dict) -> StockMarketConditions:
        """Enhance market conditions with macro context"""
        try:
            macro_insights = self._extract_macro_insights(macro_context)
            
            # Adjust conditions based on macro insights
            if macro_insights.get("direction") == "bullish":
                base_conditions.trend_direction = "bullish"
                base_conditions.market_stress = max(0.2, base_conditions.market_stress - 0.2)
            elif macro_insights.get("direction") == "bearish":
                base_conditions.trend_direction = "bearish"
                base_conditions.market_stress = min(0.8, base_conditions.market_stress + 0.2)
            
            # Adjust volatility based on macro confidence
            confidence = macro_insights.get("confidence", 0.5)
            if confidence < 0.4:
                base_conditions.volatility_regime = "high"
            elif confidence > 0.7:
                base_conditions.volatility_regime = "low"
                
            return base_conditions
            
        except Exception as e:
            logger.warning(f"Failed to enhance conditions with macro context: {e}")
            return base_conditions
    
    def _extract_macro_insights(self, macro_context: Dict) -> Dict:
        """Extract key insights from macro context"""
        try:
            # Handle different macro context formats
            if "macro_prediction" in macro_context:
                prediction = macro_context["macro_prediction"]
                return {
                    "direction": prediction.get("direction", "neutral"),
                    "confidence": prediction.get("confidence", 0.5),
                    "environment": prediction.get("economic_environment", {})
                }
            elif "direction" in macro_context:
                return {
                    "direction": macro_context.get("direction", "neutral"),
                    "confidence": macro_context.get("confidence", 0.5),
                    "environment": macro_context.get("economic_environment", {})
                }
            else:
                return {"direction": "neutral", "confidence": 0.5, "environment": {}}
                
        except Exception as e:
            logger.warning(f"Failed to extract macro insights: {e}")
            return {"direction": "neutral", "confidence": 0.5, "environment": {}}

    def get_top_stock_picks(self, analysis: Dict, horizon: str = "next_quarter", top_n: int = 5) -> List[Dict]:
        """Get top stock picks from analysis results"""
        try:
            if "horizons" not in analysis or horizon not in analysis["horizons"]:
                logger.warning(f"No analysis found for horizon: {horizon}")
                return []
            
            predictions = analysis["horizons"][horizon]
            
            # Sort by expected return and take top N
            sorted_predictions = sorted(predictions, key=lambda x: x.get("expected_return", 0), reverse=True)
            top_picks = sorted_predictions[:top_n]
            
            logger.info(f"Selected top {len(top_picks)} stock picks for {horizon}")
            return top_picks
            
        except Exception as e:
            logger.error(f"Failed to get top stock picks: {e}")
            return []

async def quick_stock_analysis(symbols: List[str] = None, api_key: str = None, macro_context: Dict = None) -> Dict:
    """Quick stock analysis function for testing"""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    config = AgentConfig(
        llm_config=LLMConfig(api_key=api_key or os.getenv("OPENAI_API_KEY", ""))
    )
    
    agent = StockAnalysisAgent(config)
    return await agent.analyze_stocks(symbols, macro_context=macro_context)

def create_custom_stock_agent(symbols: List[str], api_key: str = None) -> StockAnalysisAgent:
    """Create a custom stock agent with specific symbols"""
    config = AgentConfig(
        name=f"Custom Stock Agent ({len(symbols)} symbols)",
        llm_config=LLMConfig(api_key=api_key or os.getenv("OPENAI_API_KEY", ""))
    )
    
    agent = StockAnalysisAgent(config)
    # Override default symbols
    agent.default_symbols["custom"] = symbols
    return agent

if __name__ == '__main__':
    main()