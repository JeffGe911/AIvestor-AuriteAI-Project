"""
OpenAI integration for the AI Macro Analysis Agent.
Generates natural language analysis and reports.
"""

from openai import OpenAI
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .config import OpenAIConfig


class OpenAIClient:
    """Client for OpenAI API integration."""
    
    def __init__(self, config: OpenAIConfig):
        """Initialize OpenAI client with configuration."""
        self.config = config
        self.client = None
        
        # Only initialize if API key is provided
        if config.api_key and config.api_key.strip():
            try:
                self.client = OpenAI(api_key=config.api_key)
                # Test API key validity
                models = self.client.models.list()
                logger.info(f"✅ OpenAI API connected successfully (Model: {config.model})")
            except Exception as e:
                logger.error(f"❌ OpenAI API connection failed: {e}")
                logger.warning("⚠️ Continuing without OpenAI - will use fallback reports")
                self.client = None
        else:
            logger.info("⚠️ No OpenAI API key provided - using fallback reports only")
    
    def generate_prediction_report(self, 
                                 prediction_result: Dict[str, Any], 
                                 features: Dict[str, float],
                                 market_context: Optional[str] = None) -> str:
        """
        Generate a comprehensive prediction report using OpenAI.
        
        Args:
            prediction_result: Prediction results from ModelManager
            features: Feature values used for prediction
            market_context: Optional additional market context
            
        Returns:
            Natural language prediction report
        """
        try:
            # Check if OpenAI client is available
            if self.client is None:
                return self._create_fallback_report(prediction_result, features)
            
            # Prepare the prompt
            prompt = self._create_prediction_prompt(prediction_result, features, market_context)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            report = response.choices[0].message.content.strip()
            
            logger.info("📝 Generated OpenAI prediction report")
            return report
            
        except Exception as e:
            logger.error(f"❌ OpenAI report generation failed: {e}")
            # Return fallback report
            return self._create_fallback_report(prediction_result, features)
    
    def generate_market_summary(self, 
                              macro_data: Dict[str, float],
                              historical_predictions: Optional[list] = None) -> str:
        """
        Generate a market summary based on current macro conditions.
        
        Args:
            macro_data: Current macro economic indicators
            historical_predictions: Optional list of recent predictions
            
        Returns:
            Natural language market summary
        """
        try:
            # Check if OpenAI client is available
            if self.client is None:
                return self._create_fallback_summary(macro_data)
                
            prompt = self._create_market_summary_prompt(macro_data, historical_predictions)
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self._get_market_analyst_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            summary = response.choices[0].message.content.strip()
            
            logger.info("📊 Generated market summary")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Market summary generation failed: {e}")
            return self._create_fallback_summary(macro_data)
    
    def explain_prediction_factors(self, 
                                 prediction_result: Dict[str, Any],
                                 top_features: list,
                                 feature_values: Dict[str, float]) -> str:
        """
        Generate an explanation of the key factors driving the prediction.
        
        Args:
            prediction_result: Prediction results
            top_features: List of (feature_name, importance) tuples
            feature_values: Current feature values
            
        Returns:
            Natural language explanation of prediction factors
        """
        try:
            # Check if OpenAI client is available
            if self.client is None:
                return self._create_fallback_explanation(top_features, feature_values)
                
            prompt = self._create_explanation_prompt(prediction_result, top_features, feature_values)
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self._get_explanation_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.5  # Lower temperature for factual explanations
            )
            
            explanation = response.choices[0].message.content.strip()
            
            logger.info("🔍 Generated prediction explanation")
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Explanation generation failed: {e}")
            return self._create_fallback_explanation(top_features, feature_values)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for prediction reports."""
        return """
You are an expert financial analyst specializing in macro economic analysis and NASDAQ 100 predictions. 
Your role is to interpret machine learning model predictions and provide clear, actionable insights.

Guidelines:
- Be professional and objective
- Explain complex concepts in accessible terms
- Provide context for predictions
- Include relevant market risks and considerations
- Use specific data points to support your analysis
- Maintain a balanced perspective on both bullish and bearish scenarios
- Format your response clearly with headers and bullet points
        """.strip()
    
    def _get_market_analyst_prompt(self) -> str:
        """Get the system prompt for market analysis."""
        return """
You are a senior market analyst providing macro economic commentary. 
Focus on current economic conditions and their implications for equity markets.

Guidelines:
- Synthesize macro data into coherent market narrative
- Identify key trends and inflection points
- Discuss implications for different market sectors
- Be concise but comprehensive
- Use professional financial terminology
- Provide forward-looking insights based on data
        """.strip()
    
    def _get_explanation_prompt(self) -> str:
        """Get the system prompt for prediction explanations."""
        return """
You are a quantitative analyst explaining machine learning model predictions to investment professionals.
Focus on translating technical features into economic insights.

Guidelines:
- Explain how each feature influences the prediction
- Provide economic context for technical indicators
- Use clear cause-and-effect reasoning
- Be specific about feature values and their significance
- Maintain objectivity and acknowledge uncertainties
        """.strip()
    
    def _create_prediction_prompt(self, 
                                prediction_result: Dict[str, Any], 
                                features: Dict[str, float],
                                market_context: Optional[str]) -> str:
        """Create prompt for prediction report generation."""
        
        direction = prediction_result['prediction'].upper()
        confidence = prediction_result['confidence']
        model_name = prediction_result.get('model_name', 'ML Model')
        
        # Extract key macro indicators
        key_indicators = {}
        for key in ['vix', 'unemployment_rate', 'fed_funds_rate', 'treasury_10y', 'real_gdp']:
            if key in features:
                key_indicators[key] = features[key]
        
        prompt = f"""
Please provide a comprehensive quarterly prediction report for NASDAQ 100 based on the following analysis:

PREDICTION RESULTS:
- Direction: {direction}
- Confidence: {confidence:.1%}
- Model: {model_name}
- Timestamp: {prediction_result.get('prediction_timestamp', 'N/A')}

KEY MACRO INDICATORS:
{self._format_indicators(key_indicators)}

PROBABILITY BREAKDOWN:
- Bullish: {prediction_result['probabilities']['bullish']:.1%}
- Bearish: {prediction_result['probabilities']['bearish']:.1%}

{f"ADDITIONAL CONTEXT: {market_context}" if market_context else ""}

Please provide:
1. Executive Summary of the prediction
2. Key factors driving this forecast
3. Risk assessment and scenarios to watch
4. Actionable insights for investors
5. Timeline and key events to monitor

Format as a professional investment report.
        """.strip()
        
        return prompt
    
    def _create_market_summary_prompt(self, 
                                    macro_data: Dict[str, float],
                                    historical_predictions: Optional[list]) -> str:
        """Create prompt for market summary generation."""
        
        prompt = f"""
Provide a concise market analysis based on current macro economic conditions:

CURRENT MACRO INDICATORS:
{self._format_indicators(macro_data)}

{f"RECENT PREDICTION HISTORY: {historical_predictions}" if historical_predictions else ""}

Please analyze:
1. Current macro economic environment
2. Key trends and inflection points
3. Market implications by sector
4. Risk factors to monitor
5. Overall market sentiment assessment

Keep the analysis concise (300-400 words) and focused on actionable insights.
        """.strip()
        
        return prompt
    
    def _create_explanation_prompt(self, 
                                 prediction_result: Dict[str, Any],
                                 top_features: list,
                                 feature_values: Dict[str, float]) -> str:
        """Create prompt for prediction factor explanation."""
        
        direction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        
        features_text = ""
        for i, (feature_name, importance) in enumerate(top_features[:5], 1):
            value = feature_values.get(feature_name, 0.0)
            features_text += f"{i}. {feature_name}: {value:.3f} (importance: {importance:.3f})\n"
        
        prompt = f"""
Explain why the model predicts {direction.upper()} direction with {confidence:.1%} confidence.

TOP INFLUENTIAL FEATURES:
{features_text}

Please explain:
1. How each top feature contributes to the {direction} prediction
2. Economic interpretation of these technical indicators
3. Why these factors are particularly relevant for NASDAQ 100
4. What these feature values suggest about market conditions
5. Potential risks or limitations of this analysis

Be specific about the feature values and their economic significance.
        """.strip()
        
        return prompt
    
    def _format_indicators(self, indicators: Dict[str, float]) -> str:
        """Format macro indicators for prompt inclusion."""
        formatted = ""
        for key, value in indicators.items():
            name = key.replace('_', ' ').title()
            if 'rate' in key.lower() or 'gdp' in key.lower():
                formatted += f"- {name}: {value:.2f}%\n"
            else:
                formatted += f"- {name}: {value:.2f}\n"
        return formatted.strip()
    
    def _create_fallback_report(self, 
                              prediction_result: Dict[str, Any], 
                              features: Dict[str, float]) -> str:
        """Create fallback report when OpenAI is unavailable."""
        
        direction = prediction_result['prediction'].upper()
        confidence = prediction_result['confidence']
        
        return f"""
NASDAQ 100 QUARTERLY PREDICTION REPORT
{'=' * 45}

🎯 PREDICTION: {direction}
📊 CONFIDENCE: {confidence:.1%}
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

EXECUTIVE SUMMARY:
Based on current macro economic indicators, our model predicts a {direction.lower()} 
outlook for NASDAQ 100 in the next quarter with {confidence:.1%} confidence.

KEY FACTORS:
- Current VIX level: {features.get('vix', 'N/A')}
- Fed Funds Rate: {features.get('fed_funds_rate', 'N/A')}%
- Unemployment Rate: {features.get('unemployment_rate', 'N/A')}%
- 10Y Treasury Yield: {features.get('treasury_10y', 'N/A')}%

RISK ASSESSMENT:
{"High confidence prediction - strong signal from macro indicators" if confidence > 0.8 else "Moderate confidence - mixed signals from macro environment"}

NOTE: This is a simplified report. Full analysis requires OpenAI integration.
        """
    
    def _create_fallback_summary(self, macro_data: Dict[str, float]) -> str:
        """Create fallback market summary."""
        return f"""
MARKET SUMMARY - {datetime.now().strftime('%Y-%m-%d')}

Current macro conditions show:
- VIX: {macro_data.get('vix', 'N/A')}
- Federal Funds Rate: {macro_data.get('fed_funds_rate', 'N/A')}%
- Unemployment: {macro_data.get('unemployment_rate', 'N/A')}%
- 10Y Treasury: {macro_data.get('treasury_10y', 'N/A')}%

Detailed analysis requires OpenAI API connection.
        """
    
    def _create_fallback_explanation(self, 
                                   top_features: list, 
                                   feature_values: Dict[str, float]) -> str:
        """Create fallback explanation."""
        explanation = "TOP PREDICTION FACTORS:\n"
        for i, (feature, importance) in enumerate(top_features[:5], 1):
            value = feature_values.get(feature, 0.0)
            explanation += f"{i}. {feature}: {value:.3f} (importance: {importance:.3f})\n"
        
        explanation += "\nDetailed factor analysis requires OpenAI API connection."
        return explanation 