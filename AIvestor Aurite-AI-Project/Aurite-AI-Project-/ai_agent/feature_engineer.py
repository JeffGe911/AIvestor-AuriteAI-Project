"""
Feature engineering for macro economic data.
Implements the advanced feature engineering from your enhanced model.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger


class FeatureEngineer:
    """Advanced feature engineering for macro economic data."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_columns = []
        self.quarterly_data = None
        
    def create_quarterly_features(self, macro_data: pd.DataFrame, nasdaq_data: Optional[pd.DataFrame] = None, training_mode: bool = True) -> pd.DataFrame:
        """
        Create quarterly features from macro data and NASDAQ data.
        
        Args:
            macro_data: DataFrame with macro indicators
            nasdaq_data: Optional DataFrame with NASDAQ price data
            training_mode: Whether to create target variable (True for training, False for prediction)
            
        Returns:
            DataFrame with quarterly aggregated features
        """
        try:
            logger.info("🛠️ Starting quarterly feature engineering...")
            
            # Convert to quarterly data if not already
            quarterly_df = self._aggregate_to_quarterly(macro_data)
            
            # Add NASDAQ data if provided
            if nasdaq_data is not None and not nasdaq_data.empty:
                quarterly_df = self._merge_nasdaq_data(quarterly_df, nasdaq_data)
            else:
                # Download NASDAQ data
                quarterly_df = self._add_nasdaq_data(quarterly_df)
            
            # Create advanced features
            quarterly_df = self._create_price_features(quarterly_df)
            quarterly_df = self._create_macro_features(quarterly_df)
            quarterly_df = self._create_interaction_features(quarterly_df)
            quarterly_df = self._create_regime_features(quarterly_df)
            quarterly_df = self._create_seasonal_features(quarterly_df)
            
            # 🚀 NEW: Enhanced Time Series Features
            quarterly_df = self._create_lag_features(quarterly_df)
            quarterly_df = self._create_autoregressive_features(quarterly_df)
            quarterly_df = self._create_trend_features(quarterly_df)
            quarterly_df = self._create_cyclical_features(quarterly_df)
            quarterly_df = self._create_cross_lag_features(quarterly_df)
            quarterly_df = self._create_stationarity_features(quarterly_df)
            
            # 🎯 COMPATIBLE: Create only the 9 features the realistic model expects
            quarterly_df = self._create_compatible_features(quarterly_df)
            
            # Create target variable for training (only if in training mode and sufficient data)
            if training_mode and 'Target' not in quarterly_df.columns and len(quarterly_df) > 10:
                # Ensure we have NASDAQ return data for target creation
                if 'NASDAQ100_Return' not in quarterly_df.columns:
                    # Create synthetic target if no NASDAQ data
                    quarterly_df = self._create_synthetic_target(quarterly_df)
                
                quarterly_df = self._create_target_variable(quarterly_df)
            
            # Store feature columns
            exclude_cols = ['Quarter', 'quarter', 'quarter_id', 'NASDAQ100_Close', 'NASDAQ100_Return', 'Market_Direction', 'Target',
                           'year', 'quarter_start', 'quarter_end', 'record_count', 'date', 'created_at', 'updated_at']
            
            # Get only numeric columns and exclude string identifiers
            self.feature_columns = []
            for col in quarterly_df.columns:
                if col not in exclude_cols:
                    # Check if column is numeric and doesn't contain string values like '2020Q1'
                    try:
                        if quarterly_df[col].dtype in ['float64', 'int64', 'bool']:
                            # Additional check to ensure no string values
                            if not quarterly_df[col].astype(str).str.contains(r'Q\d', na=False).any():
                                self.feature_columns.append(col)
                    except Exception:
                        continue  # Skip problematic columns
            
            # Clean data - only clean numeric columns
            logger.info("🧹 Cleaning numeric data...")
            
            # Separate numeric and non-numeric columns
            numeric_cols = quarterly_df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = quarterly_df.select_dtypes(exclude=[np.number]).columns
            
            # Clean only numeric columns
            quarterly_df[numeric_cols] = quarterly_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            quarterly_df[numeric_cols] = quarterly_df[numeric_cols].fillna(quarterly_df[numeric_cols].median())
            
            # Log feature information
            logger.info(f"📊 Created {len(self.feature_columns)} features from {len(quarterly_df)} quarterly records")
            
            # Store quarterly data for later use
            self.quarterly_data = quarterly_df
            
            return quarterly_df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            raise
    
    def _aggregate_to_quarterly(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to quarterly frequency."""
        try:
            # Ensure date column exists
            if 'date' not in data.columns:
                if 'created_at' in data.columns:
                    data['date'] = pd.to_datetime(data['created_at'])
                else:
                    raise ValueError("No date column found in data")
            
            data['date'] = pd.to_datetime(data['date'])
            data['quarter'] = data['date'].dt.to_period('Q')
            
            # Aggregate numeric columns by quarter
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            # Exclude quarter column and any ID columns from aggregation
            agg_dict = {col: 'mean' for col in numeric_cols if col not in ['quarter', 'id']}
            
            if not agg_dict:
                logger.error("❌ No numeric columns found for aggregation")
                raise ValueError("No numeric columns available for quarterly aggregation")
            
            quarterly = data.groupby('quarter').agg(agg_dict).reset_index()
            quarterly['Quarter'] = quarterly['quarter'].astype(str)
            
            logger.info(f"📊 Aggregated columns: {list(agg_dict.keys())}")
            
            logger.info(f"📊 Aggregated to {len(quarterly)} quarters")
            return quarterly
            
        except Exception as e:
            logger.error(f"❌ Quarterly aggregation failed: {e}")
            raise
    
    def _add_nasdaq_data(self, quarterly_df: pd.DataFrame) -> pd.DataFrame:
        """Add NASDAQ 100 data to quarterly DataFrame."""
        try:
            logger.info("📈 Downloading NASDAQ 100 data...")
            
            # Download NASDAQ 100 data (using QQQ as proxy) - match macro data timeframe
            start_date = '2008-01-01'  # Post-financial crisis data to match macro data
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Try multiple NASDAQ proxies
            symbols = ['QQQ', '^NDX', '^IXIC']  # QQQ ETF, NASDAQ 100, NASDAQ Composite
            nasdaq = None
            
            for symbol in symbols:
                try:
                    logger.info(f"📈 Trying to download {symbol}...")
                    nasdaq = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not nasdaq.empty:
                        logger.info(f"✅ Successfully downloaded {symbol} data")
                        break
                except Exception as e:
                    logger.warning(f"⚠️ Failed to download {symbol}: {e}")
                    continue
            
            if nasdaq is None or nasdaq.empty:
                logger.warning("⚠️ Failed to download any NASDAQ data, creating synthetic target...")
                return self._create_synthetic_target(quarterly_df)
            
            # Flatten column names if MultiIndex
            if isinstance(nasdaq.columns, pd.MultiIndex):
                nasdaq.columns = ['_'.join(col).strip() for col in nasdaq.columns.values]
                # Clean up column names
                nasdaq.columns = [col.split('_')[0] if '_' in col else col for col in nasdaq.columns]
            
            # Reset index to get date as column
            nasdaq = nasdaq.reset_index()
            nasdaq['date'] = pd.to_datetime(nasdaq['Date'] if 'Date' in nasdaq.columns else nasdaq.index)
            nasdaq['quarter'] = nasdaq['date'].dt.to_period('Q')
            
            # Ensure we have Close price data
            close_col = None
            for col in ['Close', 'Adj Close', 'close', 'adj_close']:
                if col in nasdaq.columns:
                    close_col = col
                    break
            
            if close_col is None:
                logger.warning("⚠️ No Close price column found in NASDAQ data, creating synthetic target...")
                return self._create_synthetic_target(quarterly_df)
            
            # Aggregate NASDAQ to quarterly
            nasdaq_quarterly = nasdaq.groupby('quarter').agg({
                close_col: ['first', 'last', 'mean', 'std'],
                'Volume': 'mean' if 'Volume' in nasdaq.columns else lambda x: np.nan
            }).reset_index()
            
            # Flatten column names
            nasdaq_quarterly.columns = ['quarter', 'NASDAQ100_Open', 'NASDAQ100_Close', 
                                      'NASDAQ100_Mean', 'NASDAQ100_Volatility', 'NASDAQ100_Volume']
            
            # Calculate quarterly return
            nasdaq_quarterly['NASDAQ100_Return'] = nasdaq_quarterly['NASDAQ100_Close'].pct_change()
            
            # Merge with quarterly data
            quarterly_df['quarter'] = pd.to_period(quarterly_df['Quarter'])
            merged_df = quarterly_df.merge(nasdaq_quarterly, on='quarter', how='left')
            
            # Forward fill missing NASDAQ data for recent quarters
            nasdaq_cols = ['NASDAQ100_Open', 'NASDAQ100_Close', 'NASDAQ100_Mean', 
                          'NASDAQ100_Volatility', 'NASDAQ100_Volume', 'NASDAQ100_Return']
            for col in nasdaq_cols:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna(method='ffill')
            
            logger.info(f"📈 Added NASDAQ data for {len(merged_df)} quarters")
            return merged_df
            
        except Exception as e:
            logger.error(f"❌ NASDAQ data addition failed: {e}")
            logger.info("🔄 Creating synthetic target based on macro indicators...")
            return self._create_synthetic_target(quarterly_df)
    
    def _merge_nasdaq_data(self, quarterly_df: pd.DataFrame, nasdaq_data: pd.DataFrame) -> pd.DataFrame:
        """Merge provided NASDAQ data with quarterly DataFrame."""
        # Implementation for merging pre-existing NASDAQ data
        # This would be similar to _add_nasdaq_data but with provided data
        return quarterly_df
    
    def _create_price_features(self, data: pd.DataFrame, price_col: str = 'NASDAQ100_Close') -> pd.DataFrame:
        """Create advanced price-based technical features."""
        if price_col not in data.columns:
            logger.warning(f"⚠️ Price column {price_col} not found, skipping price features")
            return data
        
        df = data.copy()
        
        # Use minimum periods for small datasets
        min_periods_4q = min(2, len(df))
        
        # Price volatility features with minimum periods
        df['price_volatility_2q'] = df[price_col].rolling(2, min_periods=1).std().fillna(0)
        df['price_volatility_4q'] = df[price_col].rolling(4, min_periods=min_periods_4q).std().fillna(0)
        
        # Price momentum features - fill NaN with 0
        df['price_momentum_2q'] = df[price_col].pct_change(2).fillna(0)
        df['price_momentum_4q'] = df[price_col].pct_change(4).fillna(0)
        
        # Moving averages with minimum periods
        df['price_ma_2q'] = df[price_col].rolling(2, min_periods=1).mean()
        df['price_ma_4q'] = df[price_col].rolling(4, min_periods=min_periods_4q).mean()
        
        # Price position relative to moving averages
        df['price_above_ma_2q'] = (df[price_col] > df['price_ma_2q']).astype(int)
        df['price_above_ma_4q'] = (df[price_col] > df['price_ma_4q']).astype(int)
        
        logger.info("✅ Created price-based features")
        return df
    
    def _create_macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced macro economic features."""
        df = data.copy()
        
        # Key macro indicators to enhance
        key_indicators = ['vix', 'unemployment_rate', 'fed_funds_rate', 'treasury_10y', 'real_gdp']
        
        for indicator in key_indicators:
            if indicator in df.columns:
                # Use minimum periods to handle small datasets
                min_periods_2q = min(2, len(df))
                min_periods_4q = min(2, len(df))  # Use at least 2 periods for 4q calculations
                
                # Rolling averages with minimum periods
                df[f'{indicator}_ma_2q'] = df[indicator].rolling(2, min_periods=1).mean()
                df[f'{indicator}_ma_4q'] = df[indicator].rolling(4, min_periods=min_periods_4q).mean()
                
                # Momentum (quarter-over-quarter change) - fill NaN with 0
                df[f'{indicator}_momentum_1q'] = df[indicator].pct_change(1).fillna(0)
                df[f'{indicator}_momentum_2q'] = df[indicator].pct_change(2).fillna(0)
                
                # Volatility with minimum periods
                df[f'{indicator}_volatility_2q'] = df[indicator].rolling(2, min_periods=1).std().fillna(0)
                df[f'{indicator}_volatility_4q'] = df[indicator].rolling(4, min_periods=min_periods_4q).std().fillna(0)
                
                # Level indicators (above/below moving average)
                df[f'{indicator}_above_ma_2q'] = (df[indicator] > df[f'{indicator}_ma_2q']).astype(int)
                df[f'{indicator}_above_ma_4q'] = (df[indicator] > df[f'{indicator}_ma_4q']).astype(int)
        
        logger.info("✅ Created macro features")
        return df
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between indicators."""
        df = data.copy()
        
        # VIX and unemployment interaction (fear + economic stress)
        if 'vix' in df.columns and 'unemployment_rate' in df.columns:
            df['vix_unemployment_interaction'] = df['vix'] * df['unemployment_rate']
        
        # Fed funds and Treasury spread (yield curve)
        if 'fed_funds_rate' in df.columns and 'treasury_10y' in df.columns:
            df['fed_treasury_spread'] = df['treasury_10y'] - df['fed_funds_rate']
        
        # Fed policy and unemployment interaction
        if 'fed_funds_rate' in df.columns and 'unemployment_rate' in df.columns:
            df['fed_unemployment_interaction'] = df['fed_funds_rate'] * df['unemployment_rate']
        
        logger.info("✅ Created interaction features")
        return df
    
    def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create economic regime indicators."""
        df = data.copy()
        
        # High/low volatility regimes
        if 'vix' in df.columns:
            df['high_volatility_regime'] = (df['vix'] > df['vix'].quantile(0.75)).astype(int)
            df['low_volatility_regime'] = (df['vix'] < df['vix'].quantile(0.25)).astype(int)
        
        # High unemployment regime
        if 'unemployment_rate' in df.columns:
            df['high_unemployment_regime'] = (df['unemployment_rate'] > df['unemployment_rate'].quantile(0.75)).astype(int)
        
        logger.info("✅ Created regime features")
        return df
    
    def _create_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal dummy variables."""
        df = data.copy()
        
        # Create quarter dummies based on index position
        df['quarter_q1'] = (df.index % 4 == 0).astype(int)
        df['quarter_q2'] = (df.index % 4 == 1).astype(int)
        df['quarter_q3'] = (df.index % 4 == 2).astype(int)
        df['quarter_q4'] = (df.index % 4 == 3).astype(int)
        
        logger.info("✅ Created seasonal features")
        return df
    
    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lag features - previous quarter values for key indicators."""
        df = data.copy()
        
        # Key indicators to create lag features for
        key_indicators = ['vix', 'unemployment_rate', 'fed_funds_rate', 'treasury_10y', 'real_gdp']
        
        # Add NASDAQ price if available
        if 'NASDAQ100_Close' in df.columns:
            key_indicators.append('NASDAQ100_Close')
        
        for indicator in key_indicators:
            if indicator in df.columns:
                # Create lag features (1, 2, 3 quarters back)
                df[f'{indicator}_lag_1q'] = df[indicator].shift(1).fillna(df[indicator].mean())
                df[f'{indicator}_lag_2q'] = df[indicator].shift(2).fillna(df[indicator].mean())
                df[f'{indicator}_lag_3q'] = df[indicator].shift(3).fillna(df[indicator].mean())
                
                # Quarter-over-quarter change in lag
                df[f'{indicator}_lag1_change'] = (df[indicator] - df[f'{indicator}_lag_1q']).fillna(0)
                df[f'{indicator}_lag2_change'] = (df[f'{indicator}_lag_1q'] - df[f'{indicator}_lag_2q']).fillna(0)
        
        logger.info("✅ Created lag features")
        return df
    
    def _create_autoregressive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create autoregressive features - how current values relate to past values."""
        df = data.copy()
        
        key_indicators = ['vix', 'unemployment_rate', 'fed_funds_rate', 'treasury_10y', 'real_gdp']
        
        for indicator in key_indicators:
            if indicator in df.columns:
                # AR(1) - correlation with 1 quarter lag
                ar1_series = []
                for i in range(len(df)):
                    if i < 2:
                        ar1_series.append(0.0)
                    else:
                        # Calculate correlation between current and lagged values in rolling window
                        window_curr = df[indicator].iloc[max(0, i-1):i+1]
                        window_lag = df[indicator].shift(1).iloc[max(0, i-1):i+1]
                        if len(window_curr) > 1 and not window_curr.isna().all() and not window_lag.isna().all():
                            corr = window_curr.corr(window_lag)
                            ar1_series.append(corr if not np.isnan(corr) else 0.0)
                        else:
                            ar1_series.append(0.0)
                
                df[f'{indicator}_ar1'] = ar1_series
                
                # Mean reversion tendency - how far from long-term average
                long_term_mean = df[indicator].expanding().mean()
                df[f'{indicator}_mean_reversion'] = (df[indicator] - long_term_mean) / (df[indicator].std() + 1e-8)
        
        logger.info("✅ Created autoregressive features")
        return df
    
    def _create_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create longer-term trend features."""
        df = data.copy()
        
        key_indicators = ['vix', 'unemployment_rate', 'fed_funds_rate', 'treasury_10y', 'real_gdp']
        
        for indicator in key_indicators:
            if indicator in df.columns:
                # Longer-term moving averages (8Q = 2 years, 12Q = 3 years)
                min_periods_8q = min(4, len(df))
                min_periods_12q = min(6, len(df))
                
                df[f'{indicator}_ma_8q'] = df[indicator].rolling(8, min_periods=min_periods_8q).mean()
                df[f'{indicator}_ma_12q'] = df[indicator].rolling(12, min_periods=min_periods_12q).mean()
                
                # Trend direction (above/below long-term average)
                df[f'{indicator}_above_ma_8q'] = (df[indicator] > df[f'{indicator}_ma_8q']).astype(int)
                df[f'{indicator}_above_ma_12q'] = (df[indicator] > df[f'{indicator}_ma_12q']).astype(int)
                
                # Trend slope (rate of change over longer periods)
                df[f'{indicator}_trend_slope_4q'] = df[indicator].rolling(4, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
                ).fillna(0)
                
                df[f'{indicator}_trend_slope_8q'] = df[indicator].rolling(8, min_periods=4).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
                ).fillna(0)
                
                # Acceleration (change in trend)
                df[f'{indicator}_acceleration'] = df[f'{indicator}_trend_slope_4q'].diff().fillna(0)
        
        logger.info("✅ Created trend features")
        return df
    
    def _create_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical features for business cycle analysis."""
        df = data.copy()
        
        # Business cycle position (typical 6-8 year cycles)
        # 24 quarters = 6 years, 32 quarters = 8 years
        df['business_cycle_6y'] = np.sin(2 * np.pi * df.index / 24)
        df['business_cycle_8y'] = np.sin(2 * np.pi * df.index / 32)
        
        # Quarterly cycle within year
        df['quarterly_cycle'] = np.sin(2 * np.pi * (df.index % 4) / 4)
        
        # Time since start (linear time trend)
        df['time_trend'] = df.index / len(df)  # Normalized time trend
        
        # Recession indicators (simplified)
        if 'unemployment_rate' in df.columns and 'real_gdp' in df.columns:
            # High unemployment + declining GDP often indicates recession
            unemp_threshold = df['unemployment_rate'].quantile(0.75)
            gdp_declining = df['real_gdp'].pct_change() < -0.01  # 1% quarterly decline
            df['recession_indicator'] = ((df['unemployment_rate'] > unemp_threshold) & gdp_declining).astype(int)
        
        logger.info("✅ Created cyclical features")
        return df
    
    def _create_cross_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create cross-lag features - how one indicator's past affects another's future."""
        df = data.copy()
        
        # Key relationships with economic intuition
        cross_relationships = [
            ('fed_funds_rate', 'unemployment_rate', 2),  # Fed policy affects unemployment with 2Q lag
            ('unemployment_rate', 'vix', 1),             # Rising unemployment leads to market fear
            ('vix', 'NASDAQ100_Close', 1),               # Market fear affects prices next quarter
            ('fed_funds_rate', 'treasury_10y', 1),       # Fed policy affects yield curve
            ('real_gdp', 'unemployment_rate', 1),        # GDP growth affects employment next quarter
        ]
        
        for indicator1, indicator2, lag in cross_relationships:
            if indicator1 in df.columns and indicator2 in df.columns:
                # Cross-correlation with lag
                lag_col = f'{indicator1}_lag{lag}q'
                if lag_col not in df.columns:
                    df[lag_col] = df[indicator1].shift(lag).fillna(df[indicator1].mean())
                
                # Interaction between lagged indicator1 and current indicator2
                df[f'{indicator1}_lag{lag}q_x_{indicator2}'] = df[lag_col] * df[indicator2]
                
                # Direction agreement (both increasing/decreasing)
                ind1_direction = (df[indicator1].diff() > 0).astype(int)
                ind2_direction = (df[indicator2].diff() > 0).astype(int)
                df[f'{indicator1}_{indicator2}_direction_agreement'] = (ind1_direction == ind2_direction).astype(int)
        
        logger.info("✅ Created cross-lag features")
        return df
    
    def _create_stationarity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create stationarity and change-based features."""
        df = data.copy()
        
        key_indicators = ['vix', 'unemployment_rate', 'fed_funds_rate', 'treasury_10y', 'real_gdp']
        
        for indicator in key_indicators:
            if indicator in df.columns:
                # First difference (change from previous quarter)
                df[f'{indicator}_diff_1q'] = df[indicator].diff(1).fillna(0)
                df[f'{indicator}_diff_2q'] = df[indicator].diff(2).fillna(0)
                
                # Second difference (acceleration)
                df[f'{indicator}_diff2_1q'] = df[f'{indicator}_diff_1q'].diff(1).fillna(0)
                
                # Percentage change
                df[f'{indicator}_pct_change_1q'] = df[indicator].pct_change(1).fillna(0)
                df[f'{indicator}_pct_change_4q'] = df[indicator].pct_change(4).fillna(0)  # Year-over-year
                
                # Z-score (standardized values)
                rolling_mean = df[indicator].expanding().mean()
                rolling_std = df[indicator].expanding().std()
                df[f'{indicator}_zscore'] = ((df[indicator] - rolling_mean) / (rolling_std + 1e-8)).fillna(0)
                
                # Regime change detection (significant deviations)
                df[f'{indicator}_regime_change'] = (np.abs(df[f'{indicator}_zscore']) > 2).astype(int)
        
        logger.info("✅ Created stationarity features")
        return df
    
    def _create_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for model training."""
        df = data.copy()
        
        if 'NASDAQ100_Return' in df.columns and not df['NASDAQ100_Return'].isna().all():
            # Create binary target: 1 = Bullish (positive return), 0 = Bearish (negative return)
            df['Market_Direction'] = (df['NASDAQ100_Return'] > 0).astype(int)
            df['Target'] = df['Market_Direction']
            
            # Handle any remaining NaN values in target
            if df['Target'].isna().any():
                # Forward fill missing targets
                df['Target'] = df['Target'].fillna(method='ffill')
                # If still NaN (first rows), use mode
                if df['Target'].isna().any():
                    mode_value = df['Target'].mode()
                    if len(mode_value) > 0:
                        df['Target'] = df['Target'].fillna(mode_value[0])
                    else:
                        # Last resort: random 0/1
                        df['Target'] = df['Target'].fillna(np.random.choice([0, 1]))
            
            logger.info(f"✅ Created target variable with {df['Target'].sum()}/{len(df)} bullish quarters")
        else:
            logger.error("❌ No valid NASDAQ return data found, cannot create target variable")
            logger.error("This should not happen as synthetic data should have been created")
            raise ValueError("Failed to create target variable - no market return data available")
        
        return df
    
    def _create_synthetic_target(self, quarterly_df: pd.DataFrame) -> pd.DataFrame:
        """Create a synthetic target variable based on macro indicators when NASDAQ data is unavailable."""
        try:
            df = quarterly_df.copy()
            
            logger.info("🎯 Creating synthetic target based on macro indicators...")
            
            # Create a composite market health score
            market_score = 0
            score_components = 0
            
            # VIX component (lower VIX = better market conditions)
            if 'vix' in df.columns:
                vix_norm = (df['vix'] - df['vix'].min()) / (df['vix'].max() - df['vix'].min())
                market_score += (1 - vix_norm)  # Invert VIX (lower is better)
                score_components += 1
                
            # Unemployment component (lower unemployment = better market)
            if 'unemployment_rate' in df.columns:
                unemp_norm = (df['unemployment_rate'] - df['unemployment_rate'].min()) / (df['unemployment_rate'].max() - df['unemployment_rate'].min())
                market_score += (1 - unemp_norm)  # Invert unemployment
                score_components += 1
                
            # Fed funds rate component (moderate rates are best)
            if 'fed_funds_rate' in df.columns:
                fed_median = df['fed_funds_rate'].median()
                fed_deviation = np.abs(df['fed_funds_rate'] - fed_median)
                fed_norm = fed_deviation / fed_deviation.max()
                market_score += (1 - fed_norm)  # Closer to median is better
                score_components += 1
                
            # Treasury 10Y component (lower yields often mean flight to safety)
            if 'treasury_10y' in df.columns:
                treasury_norm = (df['treasury_10y'] - df['treasury_10y'].min()) / (df['treasury_10y'].max() - df['treasury_10y'].min())
                market_score += (1 - treasury_norm)  # Lower yields = more cautious = lower market
                score_components += 1
            
            if score_components > 0:
                market_score = market_score / score_components
                
                # Create synthetic return based on market score changes
                market_score_change = market_score.pct_change()
                df['NASDAQ100_Return'] = market_score_change
                
                # Add some realistic market data
                df['NASDAQ100_Close'] = 100 * (1 + df['NASDAQ100_Return'].fillna(0)).cumprod()
                df['NASDAQ100_Volatility'] = df['NASDAQ100_Return'].rolling(4).std()
                
                logger.info("✅ Created synthetic NASDAQ data based on macro indicators")
            else:
                # Fallback: create simple random walk target
                logger.warning("⚠️ No suitable macro indicators for synthetic target, using random walk")
                df['NASDAQ100_Return'] = np.random.normal(0.02, 0.15, len(df))  # 2% avg quarterly return, 15% volatility
                df['NASDAQ100_Close'] = 100 * (1 + df['NASDAQ100_Return']).cumprod()
                df['NASDAQ100_Volatility'] = df['NASDAQ100_Return'].rolling(4).std()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Synthetic target creation failed: {e}")
            raise
    
    def transform_for_prediction(self, latest_data: pd.DataFrame) -> np.ndarray:
        """
        Transform latest data for prediction using the same features as training.
        
        Args:
            latest_data: Latest macro data
            
        Returns:
            Feature array ready for prediction
        """
        try:
            # Apply same feature engineering
            featured_data = self.create_quarterly_features(latest_data)
            
            # Check if we have any data after feature engineering
            if featured_data.empty:
                raise ValueError("No quarterly data available after feature engineering")
            
            # Get features in same order as training
            if not self.feature_columns:
                raise ValueError("No feature columns defined. Run create_quarterly_features first.")
            
            # Check which features are available
            available_features = [col for col in self.feature_columns if col in featured_data.columns]
            missing_features = [col for col in self.feature_columns if col not in featured_data.columns]
            
            if missing_features:
                logger.warning(f"⚠️ Missing features for prediction: {missing_features[:5]}...")
            
            if not available_features:
                raise ValueError("No training features available in current data")
            
            # Get the latest quarter with available features
            feature_data = featured_data[available_features].iloc[-1:].copy()
            
            # Fill missing features with zeros (or median from training data if available)
            for missing_col in missing_features:
                feature_data[missing_col] = 0.0
            
            # Reorder to match training feature order
            feature_data = feature_data[self.feature_columns]
            
            logger.info(f"🎯 Transformed data for prediction: {feature_data.shape} ({len(available_features)}/{len(self.feature_columns)} features available)")
            return feature_data.values
            
        except Exception as e:
            logger.error(f"❌ Prediction transformation failed: {e}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns.copy()
    
    def get_latest_quarter_features(self) -> Optional[Dict[str, float]]:
        """Get the latest quarter features as a dictionary."""
        if self.quarterly_data is None or self.quarterly_data.empty:
            return None
        
        try:
            latest_row = self.quarterly_data.iloc[-1]
            feature_dict = {col: latest_row[col] for col in self.feature_columns if col in latest_row.index}
            return feature_dict
        except Exception as e:
            logger.error(f"❌ Error getting latest features: {e}")
            return None
    
    def _create_compatible_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create exactly the 9 features that the realistic model expects.
        This ensures compatibility between training and prediction.
        """
        try:
            logger.info("🎯 Creating compatible features for realistic model...")
            
            # Check what data we actually have
            has_vix = 'vix' in data.columns
            has_unemployment = 'unemployment_rate' in data.columns
            
            if not has_vix and not has_unemployment:
                logger.warning("⚠️ No key indicators available. Creating minimal feature set with defaults.")
            
            # The 9 specific features the realistic model expects
            compatible_features = [
                'vix_lag_2q',
                'unemployment_rate_lag_1q', 
                'unemployment_rate_lag_2q',
                'vix_trend_4q',
                'vix_ma_8q',
                'unemployment_rate_trend_4q',
                'unemployment_rate_yoy_change',
                'business_cycle',
                'time_trend'
            ]
            
            # Create each feature exactly as the realistic model expects
            for feature in compatible_features:
                if feature == 'vix_lag_2q':
                    data[feature] = data['vix'].shift(2) if has_vix else 20.0  # Default VIX value
                elif feature == 'unemployment_rate_lag_1q':
                    data[feature] = data['unemployment_rate'].shift(1) if has_unemployment else 4.0  # Default unemployment
                elif feature == 'unemployment_rate_lag_2q':
                    data[feature] = data['unemployment_rate'].shift(2) if has_unemployment else 4.0  # Default unemployment
                elif feature == 'vix_trend_4q':
                    if has_vix:
                        data[feature] = data['vix'].rolling(4, min_periods=1).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0.0, raw=False
                        ).fillna(0.0)
                    else:
                        data[feature] = 0.0  # Default trend
                elif feature == 'vix_ma_8q':
                    data[feature] = data['vix'].rolling(8, min_periods=1).mean() if has_vix else 20.0  # Default VIX MA
                elif feature == 'unemployment_rate_trend_4q':
                    if has_unemployment:
                        data[feature] = data['unemployment_rate'].rolling(4, min_periods=1).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0.0, raw=False
                        ).fillna(0.0)
                    else:
                        data[feature] = 0.0  # Default trend
                elif feature == 'unemployment_rate_yoy_change':
                    data[feature] = data['unemployment_rate'].diff(4) if has_unemployment else 0.0
                elif feature == 'business_cycle':
                    n_quarters = len(data)
                    data[feature] = np.sin(2 * np.pi * np.arange(n_quarters) / 24)
                elif feature == 'time_trend':
                    n_quarters = len(data)
                    data[feature] = np.arange(n_quarters) / max(1, n_quarters - 1)  # Avoid division by zero
                else:
                    logger.warning(f"⚠️ Unknown compatible feature: {feature}")
                    data[feature] = 0.0
            
            # Fill any remaining NaN values with defaults
            for feature in compatible_features:
                if feature in data.columns:
                    data[feature] = data[feature].fillna(0.0)
            
            # Update feature columns to only include compatible features
            self.feature_columns = compatible_features
            
            logger.info(f"✅ Created {len(compatible_features)} compatible features")
            logger.info(f"🔍 Compatible features: {compatible_features}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to create compatible features: {e}")
            raise 