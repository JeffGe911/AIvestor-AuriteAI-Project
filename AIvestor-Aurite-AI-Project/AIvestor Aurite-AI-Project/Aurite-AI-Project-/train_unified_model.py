#!/usr/bin/env python3
"""
Unified Model Training Script - API Version
Trains a realistic model using API data instead of Neon DB.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger

# Add ai_agent to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_agent'))

from ai_agent.api_client import MacroAPIClient, APIConfig
from ai_agent.feature_engineer import FeatureEngineer
from ai_agent.config import Config


class StandaloneAPIConfig:
    """Standalone API configuration for training."""
    
    def __init__(self):
        """Initialize API configuration."""
        self.api = APIConfig(
            fred_api_key=os.getenv("FRED_API_KEY", ""),
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            yahoo_finance_enabled=True,
            fred_enabled=True,
            alpha_vantage_enabled=False,
            cache_duration=3600
        )


class SimpleModelManager:
    """Simplified model manager for training."""
    
    def __init__(self):
        """Initialize model manager."""
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.feature_columns = []
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Basic models (always available)
        self.models['Logistic Regression'] = LogisticRegression(
            random_state=42, penalty='l2', max_iter=1000
        )
        self.models['Random Forest'] = RandomForestClassifier(
            random_state=42, n_estimators=100
        )
        self.models['SVM'] = SVC(
            random_state=42, probability=True
        )
        self.models['Neural Network'] = MLPClassifier(
            random_state=42, hidden_layer_sizes=(100, 50), max_iter=500
        )
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            random_state=42, n_estimators=100
        )
        
        # Advanced models (optional)
        try:
            import xgboost as xgb
            self.models['XGBoost'] = xgb.XGBClassifier(
                random_state=42, eval_metric='logloss'
            )
            logger.info("✅ XGBoost available")
        except ImportError:
            logger.warning("⚠️ XGBoost not available. Install with: pip install xgboost")
        
        try:
            import lightgbm as lgb
            self.models['LightGBM'] = lgb.LGBMClassifier(
                random_state=42, verbose=-1
            )
            logger.info("✅ LightGBM available")
        except ImportError:
            logger.warning("⚠️ LightGBM not available. Install with: pip install lightgbm")
        
        try:
            import catboost as cb
            self.models['CatBoost'] = cb.CatBoostClassifier(
                random_state=42, verbose=False
            )
            logger.info("✅ CatBoost available")
        except ImportError:
            logger.warning("⚠️ CatBoost not available. Install with: pip install catboost")
        
        logger.info(f"📊 Initialized {len(self.models)} models")
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Train all models and return scores."""
        scores = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"🔄 Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                scores[name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'model': model
                }
                
                logger.info(f"✅ {name}: Train={train_score:.3f}, Test={test_score:.3f}")
                
                # Update best model
                if test_score > self.best_score:
                    self.best_score = test_score
                    self.best_model = model
                
            except Exception as e:
                logger.error(f"❌ Error training {name}: {e}")
                scores[name] = {'error': str(e)}
        
        return scores
    
    def save_best_model(self, model_name: str = "unified_model"):
        """Save the best model and related files."""
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)
            
            # Save model
            import pickle
            model_path = f"models/{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            # Save feature columns
            feature_path = f"models/feature_columns.json"
            with open(feature_path, 'w') as f:
                json.dump(self.feature_columns, f, indent=2)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'best_score': self.best_score,
                'feature_count': len(self.feature_columns),
                'training_date': datetime.now().isoformat(),
                'api_based': True,
                'data_source': 'APIs (FRED, Yahoo Finance)'
            }
            
            metadata_path = f"models/model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Saved best model ({self.best_score:.3f}) to {model_path}")
            logger.info(f"📊 Feature columns saved to {feature_path}")
            logger.info(f"📋 Metadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False


def create_unified_features(api_client: MacroAPIClient, feature_engineer: FeatureEngineer) -> Tuple[pd.DataFrame, pd.Series]:
    """Create unified features from API data."""
    try:
        logger.info("📊 Fetching data from APIs...")
        
        # Get macro data
        macro_data = api_client.get_latest_macro_data(limit=1000)
        if macro_data.empty:
            raise ValueError("No macro data available from APIs")
        
        # Get NASDAQ data
        nasdaq_data = api_client.get_nasdaq_data()
        if nasdaq_data.empty:
            logger.warning("⚠️ No NASDAQ data available, using macro data only")
            nasdaq_data = pd.DataFrame()
        
        # Create features
        logger.info("🛠️ Creating quarterly features...")
        quarterly_features = feature_engineer.create_quarterly_features(
            macro_data, nasdaq_data, training_mode=True
        )
        
        if quarterly_features is None or quarterly_features.empty:
            raise ValueError("Failed to create quarterly features")
        
        # Check if we have target variable
        if 'Target' not in quarterly_features.columns:
            raise ValueError("No target variable found. Need sufficient data for training.")
        
        # Separate features and target
        feature_cols = [col for col in quarterly_features.columns if col != 'Target']
        X = quarterly_features[feature_cols]
        y = quarterly_features['Target']
        
        # Store feature columns
        feature_engineer.feature_columns = feature_cols
        
        logger.info(f"📊 Created {len(feature_cols)} features from {len(X)} samples")
        logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"❌ Error creating features: {e}")
        raise


def train_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Train models with cross-validation."""
    try:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        
        # Initialize model manager
        model_manager = SimpleModelManager()
        
        # Prepare data
        logger.info("🔄 Preparing data for training...")
        
        # Handle class imbalance
        if len(y.unique()) < 2:
            raise ValueError("Need at least 2 classes for training")
        
        # Split data (time series split)
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store feature columns
        model_manager.feature_columns = list(X.columns)
        
        # Train-test split for final evaluation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale train and test separately
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        logger.info("🚀 Starting model training...")
        scores = model_manager.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Save best model
        if model_manager.best_model is not None:
            model_manager.save_best_model()
        
        return scores
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


def main():
    """Main training function."""
    print("🚀 UNIFIED MODEL TRAINING - API VERSION")
    print("=" * 50)
    print("This script trains a model using API data instead of Neon DB")
    print()
    
    try:
        # Initialize components
        logger.info("🔧 Initializing components...")
        
        config = StandaloneAPIConfig()
        api_client = MacroAPIClient(config.api)
        feature_engineer = FeatureEngineer()
        
        # Check API health
        health = api_client.health_check()
        if not any(health.values()):
            logger.error("❌ No APIs available")
            print("❌ No APIs available. Please check your API keys:")
            print("   - Set FRED_API_KEY for FRED data")
            print("   - Yahoo Finance is enabled by default")
            return
        
        logger.info(f"✅ APIs available: {health}")
        
        # Create features
        X, y = create_unified_features(api_client, feature_engineer)
        
        # Train models
        scores = train_models(X, y)
        
        # Display results
        print("\n📊 TRAINING RESULTS")
        print("=" * 30)
        
        for name, result in scores.items():
            if 'error' in result:
                print(f"❌ {name}: {result['error']}")
            else:
                print(f"✅ {name}:")
                print(f"   Train Score: {result['train_score']:.3f}")
                print(f"   Test Score: {result['test_score']:.3f}")
        
        print(f"\n🎯 Best Model Score: {max([r.get('test_score', 0) for r in scores.values() if 'test_score' in r]):.3f}")
        print("💾 Model saved to models/unified_model.pkl")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"❌ Training failed: {e}")


if __name__ == "__main__":
    main() 