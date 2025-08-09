"""
Model management for the AI Macro Analysis Agent.
Handles loading, saving, and using trained machine learning models.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from loguru import logger

from .config import ModelConfig


class ModelManager:
    """Manager for machine learning models and predictions."""
    
    def __init__(self, config: ModelConfig):
        """Initialize model manager with configuration."""
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_metadata = {}
        
        # Create models directory if it doesn't exist
        Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, 
                   model: Any, 
                   scaler: StandardScaler, 
                   feature_columns: List[str],
                   model_name: str = "Enhanced Time Series Model",
                   accuracy: float = 0.0,
                   features_used: int = 0) -> bool:
        """
        Save trained model, scaler, and metadata.
        
        Args:
            model: Trained scikit-learn model
            scaler: Fitted StandardScaler
            feature_columns: List of feature column names
            model_name: Name of the model
            accuracy: Model accuracy on test set
            features_used: Number of features used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save model
            with open(self.config.model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save scaler
            with open(self.config.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save feature columns
            with open(self.config.feature_columns_path, 'w') as f:
                json.dump(feature_columns, f)
            
            # Analyze feature composition for enhanced metadata
            feature_analysis = self._analyze_feature_composition(feature_columns)
            
            # Save enhanced metadata
            metadata = {
                'model_name': model_name,
                'accuracy': accuracy,
                'features_used': features_used,
                'feature_count': len(feature_columns),
                'created_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'confidence_threshold': self.config.confidence_threshold,
                'time_series_enhanced': True,
                'feature_composition': feature_analysis
            }
            
            metadata_path = Path(self.config.model_path).parent / 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Model saved: {model_name} (Accuracy: {accuracy:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def _analyze_feature_composition(self, feature_columns: List[str]) -> Dict[str, int]:
        """
        Analyze the composition of time series features.
        
        Args:
            feature_columns: List of feature column names
            
        Returns:
            Dictionary with feature category counts
        """
        composition = {
            'base_features': 0,
            'lag_features': 0,
            'autoregressive_features': 0,
            'trend_features': 0,
            'cyclical_features': 0,
            'cross_lag_features': 0,
            'stationarity_features': 0,
            'interaction_features': 0
        }
        
        for feature in feature_columns:
            # Categorize each feature
            if any(keyword in feature for keyword in ['_lag_', '_lag1_', '_lag2_', '_lag3_']):
                composition['lag_features'] += 1
            elif any(keyword in feature for keyword in ['_ar1', '_mean_reversion', '_persistence']):
                composition['autoregressive_features'] += 1
            elif any(keyword in feature for keyword in ['_trend_', '_slope_', '_acceleration', '_ma_8q', '_ma_12q']):
                composition['trend_features'] += 1
            elif any(keyword in feature for keyword in ['cycle', 'recession', 'time_trend']):
                composition['cyclical_features'] += 1
            elif any(keyword in feature for keyword in ['_x_', '_agreement']):
                composition['cross_lag_features'] += 1
            elif any(keyword in feature for keyword in ['_diff_', '_pct_', '_zscore', '_regime']):
                composition['stationarity_features'] += 1
            elif any(keyword in feature for keyword in ['_interaction', '_spread', '_ratio']):
                composition['interaction_features'] += 1
            else:
                composition['base_features'] += 1
        
        return composition
    
    def load_model(self) -> bool:
        """
        Load saved model, scaler, and metadata.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model files exist
            if not all([
                Path(self.config.model_path).exists(),
                Path(self.config.scaler_path).exists(),
                Path(self.config.feature_columns_path).exists()
            ]):
                logger.error("❌ Model files not found. Train a model first.")
                return False
            
            # Load model
            with open(self.config.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(self.config.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature columns
            with open(self.config.feature_columns_path, 'r') as f:
                self.feature_columns = json.load(f)
            
            # Load metadata if exists
            metadata_path = Path(self.config.model_path).parent / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            
            logger.info(f"✅ Model loaded: {self.model_metadata.get('model_name', 'Unknown')}")
            logger.info(f"📊 Features: {len(self.feature_columns)}, Type: {type(self.model).__name__}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction using loaded model.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                raise ValueError("No model loaded. Call load_model() first.")
            
            # Check if we have the right features
            expected_features = set(self.feature_columns)
            provided_features = set(features.keys())
            
            missing_features = expected_features - provided_features
            extra_features = provided_features - expected_features
            
            if missing_features:
                logger.warning(f"⚠️ Missing features: {list(missing_features)}")
                # Fill missing features with 0
                for feature in missing_features:
                    features[feature] = 0.0
            
            if extra_features:
                logger.warning(f"⚠️ Extra features provided: {list(extra_features)}")
                # Remove extra features
                features = {k: v for k, v in features.items() if k in expected_features}
            
            # Prepare feature array
            feature_array = self._prepare_features(features)
            
            # Scale features
            feature_scaled = self.scaler.transform([feature_array])
            
            # Make prediction
            prediction = self.model.predict(feature_scaled)[0]
            probabilities = self.model.predict_proba(feature_scaled)[0]
            
            # Get confidence (max probability)
            confidence = max(probabilities)
            
            # Determine direction
            direction = "bullish" if prediction == 1 else "bearish"
            
            # Check if confidence meets threshold
            meets_threshold = confidence >= self.config.confidence_threshold
            
            result = {
                'prediction': direction,
                'confidence': confidence,
                'probabilities': {
                    'bearish': probabilities[0],
                    'bullish': probabilities[1]
                },
                'meets_threshold': meets_threshold,
                'threshold': self.config.confidence_threshold,
                'model_name': self.model_metadata.get('model_name', 'Unknown'),
                'features_used': len(self.feature_columns),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🎯 Prediction: {direction} ({confidence:.1%} confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare feature array from feature dictionary.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Feature array in correct order
        """
        try:
            feature_array = []
            missing_features = []
            
            for feature_name in self.feature_columns:
                if feature_name in features:
                    value = features[feature_name]
                    
                    # Handle pandas Series
                    if hasattr(value, 'iloc'):
                        value = value.iloc[-1] if len(value) > 0 else 0.0
                    
                    # Convert to float and handle NaN/infinite values
                    try:
                        value = float(value)
                        if np.isnan(value) or np.isinf(value):
                            value = 0.0  # Use 0 as default for missing values
                    except (ValueError, TypeError):
                        value = 0.0  # Use 0 as default for invalid values
                    
                    feature_array.append(value)
                else:
                    missing_features.append(feature_name)
                    feature_array.append(0.0)  # Default value for missing features
            
            if missing_features:
                logger.warning(f"⚠️ Missing features (using defaults): {missing_features[:5]}...")
            
            return np.array(feature_array)
            
        except Exception as e:
            logger.error(f"❌ Feature preparation failed: {e}")
            raise
    
    def get_feature_importance(self, top_n: int = 15) -> Optional[List[Tuple[str, float]]]:
        """
        Get feature importance from the loaded model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        try:
            if self.model is None:
                logger.warning("⚠️ No model loaded")
                return None
            
            # Handle different model types
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'estimators_'):  # Ensemble models
                # Try to get from the first estimator
                if hasattr(self.model.estimators_[0], 'feature_importances_'):
                    importances = self.model.estimators_[0].feature_importances_
                else:
                    logger.warning("⚠️ Feature importance not available for this model type")
                    return None
            else:
                logger.warning("⚠️ Feature importance not available for this model type")
                return None
            
            # Create list of (feature, importance) tuples
            feature_importance = list(zip(self.feature_columns, importances))
            
            # Sort by importance (descending)
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return feature_importance[:top_n]
            
        except Exception as e:
            logger.error(f"❌ Error getting feature importance: {e}")
            return None
    
    def validate_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate feature dictionary against expected features.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Validation results
        """
        try:
            if not self.feature_columns:
                return {'valid': False, 'error': 'No feature columns loaded'}
            
            expected_features = set(self.feature_columns)
            provided_features = set(features.keys())
            
            missing_features = expected_features - provided_features
            extra_features = provided_features - expected_features
            
            # Check for invalid values
            invalid_values = []
            for feature, value in features.items():
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    invalid_values.append(feature)
            
            validation = {
                'valid': len(missing_features) == 0 and len(invalid_values) == 0,
                'expected_count': len(expected_features),
                'provided_count': len(provided_features),
                'missing_features': list(missing_features),
                'extra_features': list(extra_features),
                'invalid_values': invalid_values,
                'coverage': len(provided_features & expected_features) / len(expected_features)
            }
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Feature validation failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'loaded': self.model is not None,
            'model_type': type(self.model).__name__ if self.model else None,
            'scaler_fitted': self.scaler is not None,
            'feature_count': len(self.feature_columns),
            'model_path': self.config.model_path,
            'metadata': self.model_metadata
        }
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def create_prediction_summary(self, prediction_result: Dict[str, Any], features: Dict[str, float]) -> str:
        """
        Create a human-readable summary of the prediction.
        
        Args:
            prediction_result: Result from predict() method
            features: Feature values used for prediction
            
        Returns:
            Human-readable prediction summary
        """
        try:
            direction = prediction_result['prediction'].upper()
            confidence = prediction_result['confidence']
            model_name = prediction_result.get('model_name', 'Unknown')
            
            # Get important features for this prediction
            top_features = self.get_feature_importance(5)
            
            summary = f"""
NASDAQ 100 QUARTERLY PREDICTION
{'=' * 40}

🎯 PREDICTION: {direction}
📊 CONFIDENCE: {confidence:.1%}
🤖 MODEL: {model_name}
📅 TIMESTAMP: {prediction_result['prediction_timestamp']}

📈 PROBABILITY BREAKDOWN:
   • Bullish: {prediction_result['probabilities']['bullish']:.1%}
   • Bearish: {prediction_result['probabilities']['bearish']:.1%}

🔍 TOP INFLUENTIAL FEATURES:"""
            
            if top_features:
                for i, (feature, importance) in enumerate(top_features, 1):
                    value = features.get(feature, 0.0)
                    summary += f"\n   {i}. {feature}: {value:.3f} (importance: {importance:.3f})"
            
            summary += f"""

⚖️ CONFIDENCE ASSESSMENT:
   • Threshold: {prediction_result['threshold']:.1%}
   • Meets Threshold: {'✅ YES' if prediction_result['meets_threshold'] else '❌ NO'}
   
📝 NOTE: This prediction is based on macro economic indicators and technical analysis.
         Always consider additional factors and risk management in investment decisions.
            """
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"❌ Error creating summary: {e}")
            return f"Error creating prediction summary: {e}" 