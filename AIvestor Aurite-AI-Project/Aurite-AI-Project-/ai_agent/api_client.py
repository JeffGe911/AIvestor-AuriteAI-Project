"""
API Client for fetching macro economic data from various financial APIs.
Replaces Neon DB functionality with real-time API calls.
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from loguru import logger
import time
import json
from dataclasses import dataclass


@dataclass
class APIConfig:
    """Configuration for API data sources."""
    fred_api_key: str = ""
    alpha_vantage_api_key: str = ""
    quandl_api_key: str = ""
    yahoo_finance_enabled: bool = True
    fred_enabled: bool = True
    alpha_vantage_enabled: bool = False
    cache_duration: int = 3600  # Cache data for 1 hour
    max_retries: int = 3
    retry_delay: float = 1.0


class MacroAPIClient:
    """Client for fetching macro economic data from various APIs."""
    
    def __init__(self, config: APIConfig):
        """Initialize API client with configuration."""
        self.config = config
        self.cache = {}
        self.cache_timestamps = {}
        
        # FRED API base URL
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Alpha Vantage API base URL
        self.alpha_vantage_base_url = "https://www.alphavantage.co/query"
        
        logger.info("🌐 Macro API Client initialized")
    
    def get_latest_macro_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve the latest macro economic data from APIs.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with macro indicators
        """
        try:
            logger.info("📊 Fetching latest macro data from APIs...")
            
            # Define the macro indicators we want to fetch
            macro_indicators = self._get_macro_indicators()
            
            # Fetch data for each indicator
            data_dict = {}
            for indicator_name, indicator_config in macro_indicators.items():
                try:
                    data = self._fetch_indicator_data(indicator_name, indicator_config)
                    if data is not None and not data.empty:
                        data_dict[indicator_name] = data
                        logger.info(f"✅ Fetched {indicator_name}: {len(data)} records")
                    else:
                        logger.warning(f"⚠️ No data for {indicator_name}")
                except Exception as e:
                    logger.error(f"❌ Error fetching {indicator_name}: {e}")
            
            # Combine all data into a single DataFrame
            if not data_dict:
                logger.warning("⚠️ No API data available, using sample data")
                return self._get_sample_macro_data(limit)
            
            # Create a unified DataFrame
            combined_df = self._combine_macro_data(data_dict)
            
            if combined_df.empty:
                logger.warning("⚠️ Combined data is empty, using sample data")
                return self._get_sample_macro_data(limit)
            
            # Limit the data as requested
            if len(combined_df) > limit:
                combined_df = combined_df.head(limit)
            
            logger.info(f"📊 Successfully fetched {len(combined_df)} macro data records")
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Error fetching macro data: {e}")
            logger.warning("⚠️ Using sample data as fallback")
            return self._get_sample_macro_data(limit)
    
    def get_quarterly_macro_data(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve quarterly aggregated macro data.
        
        Args:
            start_date: Start date for data retrieval (YYYY-MM-DD format)
            
        Returns:
            DataFrame with quarterly macro indicators
        """
        try:
            # Get daily data first
            daily_data = self.get_latest_macro_data(limit=1000)
            
            if daily_data.empty:
                logger.error("❌ No daily data available for quarterly aggregation")
                return pd.DataFrame()
            
            # Convert to quarterly data
            quarterly_data = self._aggregate_to_quarterly(daily_data, start_date)
            
            logger.info(f"📊 Aggregated {len(quarterly_data)} quarterly records")
            return quarterly_data
            
        except Exception as e:
            logger.error(f"❌ Error creating quarterly data: {e}")
            return pd.DataFrame()
    
    def get_nasdaq_data(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch NASDAQ 100 data using Yahoo Finance.
        
        Args:
            start_date: Start date for data retrieval
            
        Returns:
            DataFrame with NASDAQ 100 data
        """
        try:
            if not self.config.yahoo_finance_enabled:
                logger.warning("⚠️ Yahoo Finance is disabled")
                return self._get_sample_nasdaq_data()
            
            # Use cache if available and fresh
            cache_key = f"nasdaq_{start_date}"
            if self._is_cache_valid(cache_key):
                logger.info("📊 Using cached NASDAQ data")
                return self.cache[cache_key]
            
            # Fetch from Yahoo Finance
            ticker = "^NDX"  # NASDAQ 100 index
            start = start_date or "2008-01-01"
            
            logger.info(f"📊 Fetching NASDAQ 100 data from {start}")
            
            nasdaq = yf.Ticker(ticker)
            data = nasdaq.history(start=start)
            
            if data.empty:
                logger.warning("⚠️ No NASDAQ data available, using sample data")
                return self._get_sample_nasdaq_data()
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Rename columns for consistency
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add NASDAQ-specific columns
            data['NASDAQ100_Close'] = data['close']
            data['NASDAQ100_Return'] = data['close'].pct_change()
            data['NASDAQ100_Volatility'] = data['close'].rolling(window=20).std()
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            logger.info(f"📊 Fetched {len(data)} NASDAQ records")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching NASDAQ data: {e}")
            logger.warning("⚠️ Using sample NASDAQ data as fallback")
            return self._get_sample_nasdaq_data()
    
    def save_prediction(self, 
                       prediction: str, 
                       confidence: float, 
                       features_used: Dict,
                       model_name: str,
                       target_quarter: str) -> bool:
        """
        Save prediction to local storage (JSON file).
        
        Args:
            prediction: Prediction result (bullish/bearish)
            confidence: Confidence score
            features_used: Features used in prediction
            model_name: Name of the model used
            target_quarter: Target quarter for prediction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'confidence': confidence,
                'features_used': features_used,
                'model_name': model_name,
                'target_quarter': target_quarter
            }
            
            # Load existing predictions
            predictions_file = 'predictions.json'
            try:
                with open(predictions_file, 'r') as f:
                    predictions = json.load(f)
            except FileNotFoundError:
                predictions = []
            
            # Add new prediction
            predictions.append(prediction_record)
            
            # Save back to file
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            logger.info(f"💾 Saved prediction: {prediction} ({confidence:.1%}) for {target_quarter}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving prediction: {e}")
            return False
    
    def get_prediction_history(self, limit: int = 50) -> pd.DataFrame:
        """
        Retrieve prediction history from local storage.
        
        Args:
            limit: Maximum number of predictions to retrieve
            
        Returns:
            DataFrame with prediction history
        """
        try:
            predictions_file = 'predictions.json'
            
            try:
                with open(predictions_file, 'r') as f:
                    predictions = json.load(f)
            except FileNotFoundError:
                logger.warning("⚠️ No prediction history found")
                return pd.DataFrame()
            
            if not predictions:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(predictions)
            
            # Sort by timestamp (newest first)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            
            # Limit results
            if len(df) > limit:
                df = df.head(limit)
            
            logger.info(f"📊 Retrieved {len(df)} prediction records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error retrieving prediction history: {e}")
            return pd.DataFrame()
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health of API connections.
        
        Returns:
            Dictionary with health status for each API
        """
        health_status = {
            'yahoo_finance': False,
            'fred_api': False,
            'alpha_vantage': False
        }
        
        try:
            # Test Yahoo Finance
            if self.config.yahoo_finance_enabled:
                test_data = yf.Ticker("^GSPC").history(period="1d")
                health_status['yahoo_finance'] = not test_data.empty
            
            # Test FRED API
            if self.config.fred_enabled and self.config.fred_api_key:
                test_url = f"{self.fred_base_url}?series_id=GDP&api_key={self.config.fred_api_key}&limit=1&file_type=json"
                response = requests.get(test_url, timeout=10)
                health_status['fred_api'] = response.status_code == 200
            
            # Test Alpha Vantage
            if self.config.alpha_vantage_enabled and self.config.alpha_vantage_api_key:
                test_url = f"{self.alpha_vantage_base_url}?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=1min&apikey={self.config.alpha_vantage_api_key}"
                response = requests.get(test_url, timeout=10)
                health_status['alpha_vantage'] = response.status_code == 200
            
            logger.info(f"🔍 API Health Check: {health_status}")
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return health_status
    
    def close(self) -> None:
        """Clean up API client resources."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("🔌 API Client resources cleaned up")
    
    def _get_macro_indicators(self) -> Dict[str, Dict]:
        """Define macro indicators and their API sources."""
        return {
            'fed_funds_rate': {
                'source': 'fred',
                'series_id': 'FEDFUNDS',
                'name': 'Federal Funds Rate'
            },
            'treasury_10y': {
                'source': 'fred',
                'series_id': 'DGS10',
                'name': '10-Year Treasury Rate'
            },
            'treasury_2y': {
                'source': 'fred',
                'series_id': 'DGS2',
                'name': '2-Year Treasury Rate'
            },
            'unemployment_rate': {
                'source': 'fred',
                'series_id': 'UNRATE',
                'name': 'Unemployment Rate'
            },
            'real_gdp': {
                'source': 'fred',
                'series_id': 'GDPC1',
                'name': 'Real GDP'
            },
            'cpi': {
                'source': 'fred',
                'series_id': 'CPIAUCSL',
                'name': 'Consumer Price Index'
            },
            'core_cpi': {
                'source': 'fred',
                'series_id': 'CPILFESL',
                'name': 'Core CPI'
            },
            'pce': {
                'source': 'fred',
                'series_id': 'PCE',
                'name': 'Personal Consumption Expenditures'
            },
            'core_pce': {
                'source': 'fred',
                'series_id': 'PCEPILFE',
                'name': 'Core PCE'
            },
            'vix': {
                'source': 'yahoo',
                'symbol': '^VIX',
                'name': 'VIX Volatility Index'
            },
            'job_openings': {
                'source': 'fred',
                'series_id': 'JTSJOL',
                'name': 'Job Openings'
            },
            'job_seekers': {
                'source': 'fred',
                'series_id': 'UNEMPLOY',
                'name': 'Unemployed Persons'
            },
            'manufacturing_employment': {
                'source': 'fred',
                'series_id': 'MANEMP',
                'name': 'Manufacturing Employment'
            },
            'fed_balance_sheet': {
                'source': 'fred',
                'series_id': 'WALCL',
                'name': 'Fed Balance Sheet'
            },
            'm2_money_supply': {
                'source': 'fred',
                'series_id': 'M2SL',
                'name': 'M2 Money Supply'
            }
        }
    
    def _fetch_indicator_data(self, indicator_name: str, indicator_config: Dict) -> Optional[pd.DataFrame]:
        """Fetch data for a specific indicator."""
        source = indicator_config.get('source', 'fred')
        
        if source == 'fred':
            return self._fetch_fred_data(indicator_config)
        elif source == 'yahoo':
            return self._fetch_yahoo_data(indicator_config)
        elif source == 'alpha_vantage':
            return self._fetch_alpha_vantage_data(indicator_config)
        else:
            logger.warning(f"⚠️ Unknown data source: {source}")
            return None
    
    def _fetch_fred_data(self, indicator_config: Dict) -> Optional[pd.DataFrame]:
        """Fetch data from FRED API."""
        if not self.config.fred_enabled or not self.config.fred_api_key:
            logger.warning("⚠️ FRED API disabled or no API key")
            return None
        
        series_id = indicator_config['series_id']
        cache_key = f"fred_{series_id}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.fred_base_url}?series_id={series_id}&api_key={self.config.fred_api_key}&limit=1000&file_type=json"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'observations' not in data:
                logger.error(f"❌ Invalid FRED response for {series_id}")
                return None
            
            # Convert to DataFrame
            records = []
            for obs in data['observations']:
                try:
                    value = float(obs['value']) if obs['value'] != '.' else np.nan
                    records.append({
                        'date': obs['date'],
                        'value': value
                    })
                except (ValueError, KeyError):
                    continue
            
            if not records:
                logger.warning(f"⚠️ No valid data for {series_id}")
                return None
            
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Cache the data
            self._cache_data(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching FRED data for {series_id}: {e}")
            return None
    
    def _fetch_yahoo_data(self, indicator_config: Dict) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        if not self.config.yahoo_finance_enabled:
            return None
        
        symbol = indicator_config['symbol']
        cache_key = f"yahoo_{symbol}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="10y")
            
            if data.empty:
                logger.warning(f"⚠️ No data for {symbol}")
                return None
            
            # Reset index to make date a column
            data = data.reset_index()
            data = data.rename(columns={'Date': 'date', 'Close': 'value'})
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching Yahoo data for {symbol}: {e}")
            return None
    
    def _fetch_alpha_vantage_data(self, indicator_config: Dict) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage API."""
        if not self.config.alpha_vantage_enabled or not self.config.alpha_vantage_api_key:
            return None
        
        # Implementation for Alpha Vantage would go here
        # For now, return None as it's not enabled by default
        return None
    
    def _combine_macro_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple macro indicators into a single DataFrame."""
        try:
            # Start with the first dataset
            combined_df = None
            
            for indicator_name, df in data_dict.items():
                if df.empty:
                    continue
                
                # Ensure we have date and value columns
                if 'date' not in df.columns or 'value' not in df.columns:
                    logger.warning(f"⚠️ Skipping {indicator_name}: missing required columns")
                    continue
                
                # Rename value column to indicator name
                df_renamed = df[['date', 'value']].copy()
                df_renamed = df_renamed.rename(columns={'value': indicator_name})
                
                # Normalize timezone - convert to UTC and remove timezone info
                df_renamed['date'] = pd.to_datetime(df_renamed['date']).dt.tz_localize(None)
                
                if combined_df is None:
                    combined_df = df_renamed
                else:
                    # Merge on date
                    combined_df = combined_df.merge(df_renamed, on='date', how='outer')
            
            if combined_df is None:
                logger.error("❌ No valid data to combine")
                return pd.DataFrame()
            
            # Sort by date
            combined_df = combined_df.sort_values('date')
            
            # Fill missing values with forward fill, then backward fill
            combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"📊 Combined {len(combined_df)} records with {len(combined_df.columns)-1} indicators")
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Error combining macro data: {e}")
            return pd.DataFrame()
    
    def _aggregate_to_quarterly(self, daily_data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        """Aggregate daily data to quarterly data."""
        try:
            if daily_data.empty:
                return pd.DataFrame()
            
            # Filter by start date if provided
            if start_date:
                daily_data = daily_data[daily_data['date'] >= start_date]
            
            # Set date as index for resampling
            daily_data = daily_data.set_index('date')
            
            # Resample to quarterly (Q) and take the last value of each quarter
            quarterly_data = daily_data.resample('Q').last()
            
            # Reset index to make date a column
            quarterly_data = quarterly_data.reset_index()
            
            # Add quarter information
            quarterly_data['quarter'] = quarterly_data['date'].dt.quarter
            quarterly_data['year'] = quarterly_data['date'].dt.year
            quarterly_data['quarter_label'] = quarterly_data['year'].astype(str) + 'Q' + quarterly_data['quarter'].astype(str)
            
            logger.info(f"📊 Aggregated to {len(quarterly_data)} quarterly records")
            return quarterly_data
            
        except Exception as e:
            logger.error(f"❌ Error aggregating to quarterly: {e}")
            return pd.DataFrame()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        cache_age = time.time() - self.cache_timestamps[cache_key]
        return cache_age < self.config.cache_duration
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Cache data with timestamp."""
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = time.time() 

    def _get_sample_macro_data(self, limit: int = 100) -> pd.DataFrame:
        """Get sample macro data when APIs are not available."""
        try:
            import numpy as np
            from datetime import datetime, timedelta
            
            # Create sample data from 2008 to present
            start_date = datetime(2008, 1, 1)
            end_date = datetime.now()
            
            # Generate dates
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create sample macro indicators
            np.random.seed(42)  # For reproducible sample data
            
            sample_data = pd.DataFrame({
                'date': dates,
                'fed_funds_rate': np.random.normal(2.5, 1.5, len(dates)),
                'treasury_10y': np.random.normal(3.0, 1.0, len(dates)),
                'treasury_2y': np.random.normal(2.8, 0.8, len(dates)),
                'unemployment_rate': np.random.normal(5.5, 1.2, len(dates)),
                'real_gdp': np.random.normal(20000, 2000, len(dates)),
                'cpi': np.random.normal(250, 20, len(dates)),
                'core_cpi': np.random.normal(260, 15, len(dates)),
                'pce': np.random.normal(18000, 1500, len(dates)),
                'core_pce': np.random.normal(17000, 1200, len(dates)),
                'vix': np.random.normal(20, 8, len(dates)),
                'job_openings': np.random.normal(7000, 1000, len(dates)),
                'job_seekers': np.random.normal(8000, 1200, len(dates)),
                'manufacturing_employment': np.random.normal(13000, 800, len(dates)),
                'fed_balance_sheet': np.random.normal(8000, 2000, len(dates)),
                'm2_money_supply': np.random.normal(22000, 3000, len(dates))
            })
            
            # Add some realistic trends and patterns
            sample_data['fed_funds_rate'] = sample_data['fed_funds_rate'].clip(0, 8)
            sample_data['unemployment_rate'] = sample_data['unemployment_rate'].clip(3, 15)
            sample_data['vix'] = sample_data['vix'].clip(8, 50)
            
            # Limit to requested size
            if len(sample_data) > limit:
                sample_data = sample_data.tail(limit)
            
            logger.info(f"📊 Generated {len(sample_data)} sample macro records")
            return sample_data
            
        except Exception as e:
            logger.error(f"❌ Error generating sample data: {e}")
            return pd.DataFrame()
    
    def _get_sample_nasdaq_data(self) -> pd.DataFrame:
        """Get sample NASDAQ data when APIs are not available."""
        try:
            import numpy as np
            from datetime import datetime, timedelta
            
            # Create sample NASDAQ data
            start_date = datetime(2008, 1, 1)
            end_date = datetime.now()
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            np.random.seed(42)
            
            # Generate realistic NASDAQ prices
            base_price = 3000
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            sample_data = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
                'close': prices,
                'volume': np.random.normal(2000000, 500000, len(dates)),
                'NASDAQ100_Close': prices,
                'NASDAQ100_Return': [0] + [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))],
                'NASDAQ100_Volatility': [np.std(prices[max(0, i-20):i+1]) for i in range(len(prices))]
            })
            
            logger.info(f"📊 Generated {len(sample_data)} sample NASDAQ records")
            return sample_data
            
        except Exception as e:
            logger.error(f"❌ Error generating sample NASDAQ data: {e}")
            return pd.DataFrame() 