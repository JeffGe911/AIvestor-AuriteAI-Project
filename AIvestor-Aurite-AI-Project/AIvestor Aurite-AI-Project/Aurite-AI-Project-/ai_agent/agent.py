"""
Main AI Macro Analysis Agent.
Orchestrates all components to provide quarterly NASDAQ 100 predictions.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from .config import Config
from .api_client import MacroAPIClient, APIConfig
from .feature_engineer import FeatureEngineer
from .model_manager import ModelManager
from .openai_client import OpenAIClient


class MacroAnalysisAgent:
    """AI Agent for quarterly macro analysis and NASDAQ 100 predictions."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI agent with all components.
        
        Args:
            config_path: Optional path to configuration file
        """
        try:
            # Load configuration
            self.config = Config()
            
            # Initialize components
            self.api_client = MacroAPIClient(self.config.api)
            self.feature_engineer = FeatureEngineer()
            self.model_manager = ModelManager(self.config.model)
            self.openai_client = OpenAIClient(self.config.openai)
            
            # Agent state
            self.is_initialized = False
            self.last_prediction = None
            self.last_features = None
            
            logger.info(f"🤖 {self.config.agent.name} initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise
    
    def initialize(self) -> bool:
        """
        Initialize the agent by loading models and validating connections.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🚀 Initializing MacroAnalysisAgent...")
            
            # Load trained model first
            if not self.model_manager.load_model():
                logger.error("❌ Failed to load trained model")
                return False
            
            # Perform health checks after model is loaded
            health = self.health_check()
            
            # Be more lenient - only require model to be loaded
            if not health['model_manager']:
                logger.error(f"❌ Model not loaded: {health}")
                return False
            
            # Warn about API issues but don't fail
            if not health['api_client']:
                logger.warning("⚠️ API client not available - some features may not work")
            
            self.is_initialized = True
            logger.info("✅ Agent initialization complete!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            return False
    
    def predict_next_quarter(self, 
                           include_openai_analysis: bool = True,
                           market_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate prediction for the next quarter.
        
        Args:
            include_openai_analysis: Whether to include OpenAI-generated analysis
            market_context: Optional additional market context
            
        Returns:
            Complete prediction results with analysis
        """
        try:
            if not self.is_initialized:
                raise ValueError("Agent not initialized. Call initialize() first.")
            
            logger.info("🎯 Generating next quarter prediction...")
            
            # Step 1: Retrieve latest macro data from APIs
            macro_data = self.api_client.get_latest_macro_data()
            if macro_data is None or macro_data.empty:
                raise ValueError("No macro data available from APIs")
            
            # Step 2: Get NASDAQ data
            nasdaq_data = self.api_client.get_nasdaq_data()
            if nasdaq_data is None or nasdaq_data.empty:
                logger.warning("⚠️ No NASDAQ data available, using macro data only")
                nasdaq_data = pd.DataFrame()
            
            # Step 3: Create features
            quarterly_features = self.feature_engineer.create_quarterly_features(
                macro_data, nasdaq_data, training_mode=False
            )
            
            if quarterly_features is None or quarterly_features.empty:
                raise ValueError("Failed to create quarterly features")
            
            # Step 4: Make prediction
            prediction_result = self.model_manager.predict(quarterly_features)
            
            if prediction_result is None:
                raise ValueError("Failed to generate prediction")
            
            # Step 5: Get target quarter
            target_quarter = self._get_next_quarter(datetime.now())
            
            # Step 6: Save prediction
            self.api_client.save_prediction(
                prediction=prediction_result['prediction'],
                confidence=prediction_result['confidence'],
                features_used=prediction_result.get('features_used', {}),
                model_name=prediction_result.get('model_name', 'Unified Model'),
                target_quarter=target_quarter
            )
            
            # Step 7: Generate OpenAI analysis if requested
            openai_analysis = ""
            if include_openai_analysis and self.openai_client.is_available():
                try:
                    openai_analysis = self._generate_openai_analysis(
                        prediction_result, macro_data, market_context
                    )
                except Exception as e:
                    logger.warning(f"⚠️ OpenAI analysis failed: {e}")
            
            # Step 8: Compile results
            result = {
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'target_quarter': target_quarter,
                'features': quarterly_features.to_dict('records')[-1] if not quarterly_features.empty else {},
                'feature_importance': prediction_result.get('feature_importance', {}),
                'data_summary': self._create_data_summary(macro_data, nasdaq_data),
                'openai_analysis': openai_analysis,
                'agent_info': {
                    'name': self.config.agent.name,
                    'model_used': prediction_result.get('model_name', 'Unified Model'),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Update agent state
            self.last_prediction = result
            self.last_features = quarterly_features
            
            logger.info(f"✅ Prediction complete: {prediction_result['prediction'].upper()} ({prediction_result['confidence']:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def get_prediction_summary(self, include_technical: bool = True) -> str:
        """
        Get a summary of the latest prediction.
        
        Args:
            include_technical: Whether to include technical details
            
        Returns:
            Formatted prediction summary
        """
        if not self.last_prediction:
            return "❌ No prediction available. Run predict_next_quarter() first."
        
        prediction = self.last_prediction['prediction']
        confidence = self.last_prediction['confidence']
        target_quarter = self.last_prediction['target_quarter']
        
        summary = f"""
🎯 **QUARTERLY PREDICTION SUMMARY**

📊 **Prediction**: {prediction.upper()}
🎯 **Confidence**: {confidence:.1%}
📅 **Target Quarter**: {target_quarter}
🤖 **Model**: {self.last_prediction['agent_info']['model_used']}
⏰ **Generated**: {self.last_prediction['agent_info']['timestamp']}
"""
        
        if include_technical and self.last_features is not None:
            # Get top features by importance
            feature_importance = self.last_prediction.get('feature_importance', {})
            if feature_importance:
                top_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:5]
                
                summary += "\n🔍 **Top Features**:\n"
                for feature, importance in top_features:
                    summary += f"  • {feature}: {importance:.3f}\n"
        
        if self.last_prediction.get('openai_analysis'):
            summary += f"\n🤖 **AI Analysis**:\n{self.last_prediction['openai_analysis']}"
        
        return summary
    
    def get_market_analysis(self) -> str:
        """
        Get a comprehensive market analysis.
        
        Returns:
            Detailed market analysis
        """
        try:
            # Get latest macro data
            macro_data = self.api_client.get_latest_macro_data(limit=50)
            
            if macro_data.empty:
                return "❌ No macro data available for analysis."
            
            # Get latest values for key indicators
            latest = macro_data.iloc[-1] if not macro_data.empty else {}
            
            analysis = f"""
📈 **MARKET ANALYSIS**

🔍 **Key Indicators**:
"""
            
            # Add key indicators if available
            indicators = {
                'fed_funds_rate': 'Federal Funds Rate',
                'treasury_10y': '10-Year Treasury',
                'unemployment_rate': 'Unemployment Rate',
                'real_gdp': 'Real GDP',
                'cpi': 'Consumer Price Index',
                'vix': 'VIX Volatility'
            }
            
            for key, name in indicators.items():
                if key in latest and pd.notna(latest[key]):
                    value = latest[key]
                    analysis += f"  • {name}: {value:.2f}\n"
            
            # Add prediction if available
            if self.last_prediction:
                analysis += f"\n🎯 **Latest Prediction**: {self.last_prediction['prediction'].upper()} ({self.last_prediction['confidence']:.1%})"
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Market analysis failed: {e}")
            return f"❌ Market analysis failed: {e}"
    
    def retrain_model(self, save_model: bool = True) -> bool:
        """
        Retrain the model with latest data.
        
        Args:
            save_model: Whether to save the retrained model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🔄 Retraining model with latest data...")
            
            # Get latest data
            macro_data = self.api_client.get_latest_macro_data(limit=1000)
            nasdaq_data = self.api_client.get_nasdaq_data()
            
            if macro_data.empty:
                logger.error("❌ No macro data available for retraining")
                return False
            
            # Create features for training
            quarterly_features = self.feature_engineer.create_quarterly_features(
                macro_data, nasdaq_data, training_mode=True
            )
            
            if quarterly_features is None or quarterly_features.empty:
                logger.error("❌ Failed to create features for retraining")
                return False
            
            # Retrain model
            success = self.model_manager.retrain_model(quarterly_features, save_model)
            
            if success:
                logger.info("✅ Model retraining completed successfully")
            else:
                logger.error("❌ Model retraining failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary with health status for each component
        """
        health_status = {
            'api_client': False,
            'model_manager': False,
            'feature_engineer': False,
            'openai_client': False
        }
        
        try:
            # Check API client - be lenient, at least one API should work
            api_health = self.api_client.health_check()
            health_status['api_client'] = any(api_health.values())
            
            # Check model manager
            health_status['model_manager'] = self.model_manager.is_model_loaded()
            
            # Check feature engineer (always available)
            health_status['feature_engineer'] = True
            
            # Check OpenAI client
            health_status['openai_client'] = self.openai_client.is_available()
            
            logger.info(f"🔍 Health Check: {health_status}")
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return health_status
    
    def _get_next_quarter(self, current_date: datetime) -> str:
        """Get the next quarter label."""
        year = current_date.year
        current_quarter = (current_date.month - 1) // 3 + 1
        
        if current_quarter == 4:
            next_quarter = 1
            next_year = year + 1
        else:
            next_quarter = current_quarter + 1
            next_year = year
        
        return f"{next_year}Q{next_quarter}"
    
    def _get_recent_predictions(self, limit: int = 5) -> List[Dict]:
        """Get recent predictions from API client."""
        try:
            history_df = self.api_client.get_prediction_history(limit)
            if history_df.empty:
                return []
            
            return history_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"❌ Error getting recent predictions: {e}")
            return []
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status.
        
        Returns:
            Dictionary with agent status information
        """
        try:
            health = self.health_check()
            
            status = {
                'name': self.config.agent.name,
                'initialized': self.is_initialized,
                'health': health,
                'last_prediction': self.last_prediction is not None,
                'model_loaded': self.model_manager.is_model_loaded(),
                'api_available': any(health.values()),
                'openai_available': self.openai_client.is_available(),
                'timestamp': datetime.now().isoformat()
            }
            
            if self.last_prediction:
                status['last_prediction_details'] = {
                    'prediction': self.last_prediction['prediction'],
                    'confidence': self.last_prediction['confidence'],
                    'target_quarter': self.last_prediction['target_quarter']
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting agent status: {e}")
            return {'error': str(e)}
    
    def _generate_openai_analysis(self, 
                                 prediction_result: Dict, 
                                 macro_data: pd.DataFrame,
                                 market_context: Optional[str] = None) -> str:
        """Generate OpenAI analysis for the prediction."""
        try:
            # Prepare context for OpenAI
            context = f"""
Prediction: {prediction_result['prediction'].upper()}
Confidence: {prediction_result['confidence']:.1%}
Target Quarter: {self._get_next_quarter(datetime.now())}

Latest Macro Indicators:
"""
            
            if not macro_data.empty:
                latest = macro_data.iloc[-1]
                for col in macro_data.columns:
                    if col != 'date' and pd.notna(latest[col]):
                        context += f"- {col}: {latest[col]:.2f}\n"
            
            if market_context:
                context += f"\nAdditional Context: {market_context}"
            
            # Generate analysis
            prompt = f"""
Based on the following prediction and market data, provide a brief analysis:

{context}

Please provide a 2-3 sentence analysis of this prediction and the current market conditions.
"""
            
            return self.openai_client.generate_analysis(prompt)
            
        except Exception as e:
            logger.error(f"❌ OpenAI analysis generation failed: {e}")
            return ""
    
    def _create_data_summary(self, macro_data: pd.DataFrame, nasdaq_data: pd.DataFrame) -> Dict[str, Any]:
        """Create a summary of the data used for prediction."""
        try:
            summary = {
                'macro_records': len(macro_data) if not macro_data.empty else 0,
                'nasdaq_records': len(nasdaq_data) if not nasdaq_data.empty else 0,
                'macro_indicators': list(macro_data.columns) if not macro_data.empty else [],
                'data_range': {}
            }
            
            if not macro_data.empty:
                summary['data_range']['macro_start'] = macro_data['date'].min().isoformat() if 'date' in macro_data.columns else None
                summary['data_range']['macro_end'] = macro_data['date'].max().isoformat() if 'date' in macro_data.columns else None
            
            if not nasdaq_data.empty:
                summary['data_range']['nasdaq_start'] = nasdaq_data['date'].min().isoformat() if 'date' in nasdaq_data.columns else None
                summary['data_range']['nasdaq_end'] = nasdaq_data['date'].max().isoformat() if 'date' in nasdaq_data.columns else None
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error creating data summary: {e}")
            return {}
    
    def close(self) -> None:
        """Clean up agent resources."""
        try:
            if hasattr(self, 'api_client'):
                self.api_client.close()
            
            if hasattr(self, 'model_manager'):
                self.model_manager.close()
            
            if hasattr(self, 'openai_client'):
                self.openai_client.close()
            
            logger.info("🔌 Agent resources cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}") 