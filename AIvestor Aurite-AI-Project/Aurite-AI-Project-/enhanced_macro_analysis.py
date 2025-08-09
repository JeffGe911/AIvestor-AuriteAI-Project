"""
Enhanced Macro Analysis using Advanced Time Series Features
Demonstrates how to use the created features for detailed macro analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add ai_agent to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_agent'))

from ai_agent.api_client import MacroAPIClient, APIConfig
from ai_agent.feature_engineer import FeatureEngineer
from ai_agent.agent import MacroAnalysisAgent


class EnhancedMacroAnalyzer:
    """Enhanced macro analysis using advanced time series features."""
    
    def __init__(self):
        """Initialize the enhanced macro analyzer."""
        # Configure API client with environment variables
        config = APIConfig()
        config.fred_api_key = os.getenv('FRED_API_KEY', '')
        config.fred_enabled = bool(config.fred_api_key)
        
        self.api_client = MacroAPIClient(config)
        self.feature_engineer = FeatureEngineer()
        self.agent = MacroAnalysisAgent()
        
    def analyze_macro_environment(self) -> Dict:
        """Comprehensive macro environment analysis using enhanced features."""
        try:
            logger.info("🔍 Starting enhanced macro analysis...")
            
            # Get data
            macro_data = self.api_client.get_latest_macro_data(limit=1000)
            nasdaq_data = self.api_client.get_nasdaq_data()
            
            if macro_data.empty:
                raise ValueError("No macro data available")
            
            # Create enhanced features
            quarterly_features = self.feature_engineer.create_quarterly_features(
                macro_data, nasdaq_data, training_mode=False
            )
            
            if quarterly_features is None or quarterly_features.empty:
                raise ValueError("Failed to create quarterly features")
            
            # Analyze different aspects
            analysis = {
                'trend_analysis': self._analyze_trends(quarterly_features),
                'volatility_analysis': self._analyze_volatility(quarterly_features),
                'cyclical_analysis': self._analyze_cyclical_patterns(quarterly_features),
                'regime_analysis': self._analyze_regime_changes(quarterly_features),
                'correlation_analysis': self._analyze_correlations(quarterly_features),
                'feature_importance': self._analyze_feature_importance(quarterly_features),
                'prediction_analysis': self._get_prediction_analysis()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed: {e}")
            return {}
    
    def _analyze_trends(self, features: pd.DataFrame) -> Dict:
        """Analyze trend patterns in macro indicators."""
        trend_analysis = {}
        
        # Identify trend features
        trend_features = [col for col in features.columns if 'trend' in col.lower() or 'slope' in col.lower()]
        
        for feature in trend_features:
            if feature in features.columns:
                latest_value = features[feature].iloc[-1]
                avg_value = features[feature].mean()
                
                trend_analysis[feature] = {
                    'current_value': latest_value,
                    'average_value': avg_value,
                    'trend_direction': 'increasing' if latest_value > avg_value else 'decreasing',
                    'trend_strength': abs(latest_value - avg_value) / (features[feature].std() + 1e-8)
                }
        
        return trend_analysis
    
    def _analyze_volatility(self, features: pd.DataFrame) -> Dict:
        """Analyze volatility patterns."""
        volatility_analysis = {}
        
        # VIX-related features
        vix_features = [col for col in features.columns if 'vix' in col.lower()]
        
        for feature in vix_features:
            if feature in features.columns:
                current_vix = features[feature].iloc[-1]
                vix_history = features[feature].dropna()
                
                volatility_analysis[feature] = {
                    'current_level': current_vix,
                    'percentile_25': vix_history.quantile(0.25),
                    'percentile_75': vix_history.quantile(0.75),
                    'volatility_regime': 'high' if current_vix > vix_history.quantile(0.75) else 
                                       'low' if current_vix < vix_history.quantile(0.25) else 'normal'
                }
        
        return volatility_analysis
    
    def _analyze_cyclical_patterns(self, features: pd.DataFrame) -> Dict:
        """Analyze cyclical patterns in the economy."""
        cyclical_analysis = {}
        
        # Cyclical features
        cyclical_features = [col for col in features.columns if 'cycle' in col.lower() or 'recession' in col.lower()]
        
        for feature in cyclical_features:
            if feature in features.columns:
                current_cycle = features[feature].iloc[-1]
                cycle_history = features[feature].dropna()
                
                cyclical_analysis[feature] = {
                    'current_phase': current_cycle,
                    'cycle_strength': abs(current_cycle),
                    'phase_description': self._describe_cycle_phase(current_cycle, feature)
                }
        
        return cyclical_analysis
    
    def _analyze_regime_changes(self, features: pd.DataFrame) -> Dict:
        """Analyze regime changes in the economy."""
        regime_analysis = {}
        
        # Regime features
        regime_features = [col for col in features.columns if 'regime' in col.lower() or 'zscore' in col.lower()]
        
        for feature in regime_features:
            if feature in features.columns:
                current_regime = features[feature].iloc[-1]
                regime_history = features[feature].dropna()
                
                regime_analysis[feature] = {
                    'current_regime': current_regime,
                    'regime_percentile': (regime_history < current_regime).mean(),
                    'regime_description': self._describe_regime(current_regime, feature)
                }
        
        return regime_analysis
    
    def _analyze_correlations(self, features: pd.DataFrame) -> Dict:
        """Analyze correlations between key indicators."""
        correlation_analysis = {}
        
        # Key macro indicators
        key_indicators = ['fed_funds_rate', 'treasury_10y', 'unemployment_rate', 'vix']
        available_indicators = [col for col in key_indicators if col in features.columns]
        
        if len(available_indicators) >= 2:
            # Calculate correlations
            corr_matrix = features[available_indicators].corr()
            
            # Find strongest correlations
            correlations = []
            for i in range(len(available_indicators)):
                for j in range(i+1, len(available_indicators)):
                    corr_value = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_value):
                        correlations.append({
                            'pair': f"{available_indicators[i]} vs {available_indicators[j]}",
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.7 else 
                                       'moderate' if abs(corr_value) > 0.4 else 'weak'
                        })
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            correlation_analysis['top_correlations'] = correlations[:5]
        
        return correlation_analysis
    
    def _analyze_feature_importance(self, features: pd.DataFrame) -> Dict:
        """Analyze feature importance for predictions."""
        try:
            # Get prediction to analyze feature importance
            if not self.agent.is_initialized:
                self.agent.initialize()
            
            result = self.agent.predict_next_quarter(include_openai_analysis=False)
            
            if result and 'feature_importance' in result:
                importance = result['feature_importance']
                
                # Categorize features
                categorized = {
                    'lag_features': [],
                    'trend_features': [],
                    'volatility_features': [],
                    'cyclical_features': [],
                    'regime_features': []
                }
                
                for feature, importance_value in importance.items():
                    if 'lag' in feature.lower():
                        categorized['lag_features'].append((feature, importance_value))
                    elif 'trend' in feature.lower() or 'slope' in feature.lower():
                        categorized['trend_features'].append((feature, importance_value))
                    elif 'vix' in feature.lower():
                        categorized['volatility_features'].append((feature, importance_value))
                    elif 'cycle' in feature.lower():
                        categorized['cyclical_features'].append((feature, importance_value))
                    elif 'regime' in feature.lower() or 'zscore' in feature.lower():
                        categorized['regime_features'].append((feature, importance_value))
                
                # Sort each category by importance
                for category in categorized:
                    categorized[category].sort(key=lambda x: abs(x[1]), reverse=True)
                
                return categorized
            
        except Exception as e:
            logger.error(f"❌ Feature importance analysis failed: {e}")
        
        return {}
    
    def _get_prediction_analysis(self) -> Dict:
        """Get prediction analysis with enhanced features."""
        try:
            if not self.agent.is_initialized:
                self.agent.initialize()
            
            result = self.agent.predict_next_quarter(include_openai_analysis=True)
            
            if result:
                return {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'target_quarter': result['target_quarter'],
                    'model_used': result['agent_info']['model_used'],
                    'openai_analysis': result.get('openai_analysis', ''),
                    'data_summary': result.get('data_summary', {})
                }
        
        except Exception as e:
            logger.error(f"❌ Prediction analysis failed: {e}")
        
        return {}
    
    def _describe_cycle_phase(self, value: float, feature: str) -> str:
        """Describe the current cycle phase."""
        if 'recession' in feature.lower():
            return 'Recession' if value > 0.5 else 'Expansion'
        elif 'cycle' in feature.lower():
            if value > 0.7:
                return 'Peak'
            elif value < -0.7:
                return 'Trough'
            elif value > 0:
                return 'Expansion'
            else:
                return 'Contraction'
        return 'Unknown'
    
    def _describe_regime(self, value: float, feature: str) -> str:
        """Describe the current regime."""
        if abs(value) > 2:
            return 'Extreme'
        elif abs(value) > 1:
            return 'High'
        elif abs(value) > 0.5:
            return 'Moderate'
        else:
            return 'Normal'
    
    def generate_analysis_report(self) -> str:
        """Generate a comprehensive analysis report."""
        analysis = self.analyze_macro_environment()
        
        if not analysis:
            return "❌ Analysis failed. Check logs for details."
        
        report = """
📊 ENHANCED MACRO ANALYSIS REPORT
==================================

🎯 PREDICTION SUMMARY
"""
        
        if 'prediction_analysis' in analysis:
            pred = analysis['prediction_analysis']
            if pred:
                report += f"""
Direction: {pred.get('prediction', 'N/A').upper()}
Confidence: {pred.get('confidence', 0):.1%}
Target Quarter: {pred.get('target_quarter', 'N/A')}
Model: {pred.get('model_used', 'N/A')}
"""
        
        report += "\n📈 TREND ANALYSIS\n"
        if 'trend_analysis' in analysis:
            for feature, trend_data in analysis['trend_analysis'].items():
                report += f"""
{feature}:
  Current: {trend_data['current_value']:.3f}
  Average: {trend_data['average_value']:.3f}
  Direction: {trend_data['trend_direction']}
  Strength: {trend_data['trend_strength']:.2f}
"""
        
        report += "\n🌊 VOLATILITY ANALYSIS\n"
        if 'volatility_analysis' in analysis:
            for feature, vol_data in analysis['volatility_analysis'].items():
                report += f"""
{feature}:
  Current Level: {vol_data['current_level']:.3f}
  Regime: {vol_data['volatility_regime']}
  Percentile Range: {vol_data['percentile_25']:.3f} - {vol_data['percentile_75']:.3f}
"""
        
        report += "\n🔄 CYCLICAL ANALYSIS\n"
        if 'cyclical_analysis' in analysis:
            for feature, cycle_data in analysis['cyclical_analysis'].items():
                report += f"""
{feature}:
  Current Phase: {cycle_data['current_phase']:.3f}
  Phase Description: {cycle_data['phase_description']}
  Cycle Strength: {cycle_data['cycle_strength']:.2f}
"""
        
        report += "\n🔗 CORRELATION ANALYSIS\n"
        if 'correlation_analysis' in analysis and 'top_correlations' in analysis['correlation_analysis']:
            for corr in analysis['correlation_analysis']['top_correlations']:
                report += f"""
{corr['pair']}:
  Correlation: {corr['correlation']:.3f}
  Strength: {corr['strength']}
"""
        
        report += "\n🎯 FEATURE IMPORTANCE\n"
        if 'feature_importance' in analysis:
            for category, features in analysis['feature_importance'].items():
                if features:
                    report += f"\n{category.replace('_', ' ').title()}:\n"
                    for feature, importance in features[:3]:  # Top 3
                        report += f"  • {feature}: {importance:.3f}\n"
        
        if 'prediction_analysis' in analysis and analysis['prediction_analysis'].get('openai_analysis'):
            report += f"\n🤖 AI ANALYSIS\n{analysis['prediction_analysis']['openai_analysis']}"
        
        return report

    def export_macro_signals_json(self, filepath: str = None) -> Dict[str, Any]:
        """
        Export macro analysis as structured JSON for asset analysis consumption.
        
        Args:
            filepath: Optional path to save JSON file
            
        Returns:
            Dictionary with structured macro signals
        """
        try:
            logger.info("📄 Generating macro signals JSON export...")
            
            # Get comprehensive analysis
            analysis = self.analyze_macro_environment()
            
            if not analysis:
                raise ValueError("No analysis data available")
            
            # Extract prediction data
            prediction_data = analysis.get('prediction_analysis', {})
            
            # Create structured output for asset analysis consumption
            macro_signals = {
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_type": "enhanced_macro_analysis",
                "macro_prediction": {
                    "direction": prediction_data.get('prediction', 'neutral').lower(),
                    "confidence": prediction_data.get('confidence', 0.5),
                    "probability": prediction_data.get('confidence', 0.5),
                    "target_quarter": prediction_data.get('target_quarter', 'N/A'),
                    "model_used": prediction_data.get('model_used', 'Enhanced Macro Model')
                },
                "economic_environment": {
                    "volatility_regime": self._extract_volatility_regime(analysis),
                    "trend_direction": self._extract_trend_direction(analysis),
                    "cycle_phase": self._extract_cycle_phase(analysis),
                    "market_stress": self._extract_market_stress(analysis)
                },
                "asset_class_signals": {
                    "stocks": {
                        "recommendation": "overweight" if prediction_data.get('prediction', '').lower() == 'bullish' else "underweight",
                        "confidence": prediction_data.get('confidence', 0.5),
                        "reasoning": f"Macro model predicts {prediction_data.get('prediction', 'neutral')} environment"
                    },
                    "bonds": {
                        "recommendation": "neutral",  # Can be enhanced with bond-specific logic
                        "confidence": 0.6,
                        "reasoning": "Bond allocation based on interest rate environment"
                    },
                    "gold": {
                        "recommendation": "neutral",  # Can be enhanced with gold-specific logic  
                        "confidence": 0.6,
                        "reasoning": "Gold allocation based on inflation and volatility"
                    }
                },
                "feature_insights": {
                    "top_features": self._extract_top_features(analysis),
                    "trend_analysis": analysis.get('trend_analysis', {}),
                    "volatility_analysis": analysis.get('volatility_analysis', {}),
                    "correlation_insights": analysis.get('correlation_analysis', {})
                },
                "risk_factors": self._extract_risk_factors(analysis),
                "data_quality": {
                    "indicators_available": len(analysis.get('trend_analysis', {})),
                    "analysis_completeness": self._assess_completeness(analysis)
                }
            }
            
            # Save to file if path provided
            if filepath:
                import json
                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w') as f:
                    json.dump(macro_signals, f, indent=2)
                logger.info(f"📄 Macro signals saved to: {filepath}")
            
            logger.info("✅ Macro signals JSON export completed")
            return macro_signals
            
        except Exception as e:
            logger.error(f"❌ JSON export failed: {e}")
            return {}
    
    def _extract_volatility_regime(self, analysis: Dict) -> str:
        """Extract overall volatility regime from analysis."""
        vol_analysis = analysis.get('volatility_analysis', {})
        if vol_analysis:
            # Look for VIX-based regime
            for feature, data in vol_analysis.items():
                if 'vix' in feature.lower():
                    return data.get('volatility_regime', 'normal')
        return 'normal'
    
    def _extract_trend_direction(self, analysis: Dict) -> str:
        """Extract overall trend direction from analysis."""
        trend_analysis = analysis.get('trend_analysis', {})
        if trend_analysis:
            # Count increasing vs decreasing trends
            increasing = sum(1 for data in trend_analysis.values() 
                           if data.get('trend_direction') == 'increasing')
            decreasing = sum(1 for data in trend_analysis.values() 
                           if data.get('trend_direction') == 'decreasing')
            
            if increasing > decreasing:
                return 'upward'
            elif decreasing > increasing:
                return 'downward'
        return 'sideways'
    
    def _extract_cycle_phase(self, analysis: Dict) -> str:
        """Extract economic cycle phase from analysis."""
        cyclical_analysis = analysis.get('cyclical_analysis', {})
        if cyclical_analysis:
            # Look for recession indicators
            for feature, data in cyclical_analysis.items():
                phase_desc = data.get('phase_description', '').lower()
                if 'recession' in phase_desc:
                    return 'contraction'
                elif 'expansion' in phase_desc:
                    return 'expansion'
        return 'uncertain'
    
    def _extract_market_stress(self, analysis: Dict) -> str:
        """Extract market stress level from analysis."""
        vol_analysis = analysis.get('volatility_analysis', {})
        if vol_analysis:
            for feature, data in vol_analysis.items():
                if 'vix' in feature.lower():
                    regime = data.get('volatility_regime', 'normal')
                    if regime == 'high':
                        return 'elevated'
                    elif regime == 'low':
                        return 'low'
        return 'normal'
    
    def _extract_top_features(self, analysis: Dict) -> List[Dict]:
        """Extract top contributing features."""
        feature_importance = analysis.get('feature_importance', {})
        top_features = []
        
        for category, features in feature_importance.items():
            if features:
                for feature, importance in features[:2]:  # Top 2 per category
                    top_features.append({
                        'feature': feature,
                        'importance': importance,
                        'category': category
                    })
        
        # Sort by importance and return top 10
        top_features.sort(key=lambda x: abs(x['importance']), reverse=True)
        return top_features[:10]
    
    def _extract_risk_factors(self, analysis: Dict) -> List[str]:
        """Extract key risk factors from analysis."""
        risks = []
        
        # Check volatility regime
        if self._extract_volatility_regime(analysis) == 'high':
            risks.append("Elevated market volatility")
        
        # Check trend consistency
        if self._extract_trend_direction(analysis) == 'downward':
            risks.append("Negative trend momentum")
        
        # Check cycle phase
        if self._extract_cycle_phase(analysis) == 'contraction':
            risks.append("Economic contraction signals")
        
        # Add generic risks if none found
        if not risks:
            risks = ["Normal market conditions", "Standard macro uncertainties"]
        
        return risks
    
    def _assess_completeness(self, analysis: Dict) -> float:
        """Assess the completeness of the analysis."""
        expected_sections = ['prediction_analysis', 'trend_analysis', 'volatility_analysis', 
                           'cyclical_analysis', 'correlation_analysis', 'feature_importance']
        
        available_sections = sum(1 for section in expected_sections if section in analysis and analysis[section])
        return available_sections / len(expected_sections)


def main():
    """Main function to run enhanced macro analysis."""
    print("🔍 ENHANCED MACRO ANALYSIS")
    print("=" * 40)
    print("Using advanced time series features for comprehensive analysis")
    print()
    
    try:
        analyzer = EnhancedMacroAnalyzer()
        
        print("📊 Generating comprehensive analysis...")
        report = analyzer.generate_analysis_report()
        
        print(report)
        
        # Export macro signals as JSON for asset analysis consumption
        print("\n📄 Exporting macro signals as JSON...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure analysis_outputs directory exists
        output_dir = "analysis_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to analysis_outputs folder
        json_filename = f"macro_signals_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        macro_signals = analyzer.export_macro_signals_json(json_filepath)
        
        if macro_signals:
            print(f"✅ Macro signals exported to: {json_filepath}")
            print("\n🎯 KEY MACRO SIGNALS FOR ASSET ANALYSIS:")
            print(f"   • Direction: {macro_signals['macro_prediction']['direction'].upper()}")
            print(f"   • Confidence: {macro_signals['macro_prediction']['confidence']:.1%}")
            print(f"   • Volatility Regime: {macro_signals['economic_environment']['volatility_regime']}")
            print(f"   • Trend Direction: {macro_signals['economic_environment']['trend_direction']}")
            print(f"   • Asset Class Signals:")
            for asset_class, signal in macro_signals['asset_class_signals'].items():
                print(f"     - {asset_class.title()}: {signal['recommendation']} ({signal['confidence']:.1%})")
        else:
            print("⚠️ JSON export failed, but analysis completed")
        
        print("\n✅ Analysis complete!")
        print("💡 This analysis uses advanced time series features including:")
        print("   • Lag features for temporal relationships")
        print("   • Trend features for long-term patterns")
        print("   • Volatility features for market stress")
        print("   • Cyclical features for business cycles")
        print("   • Regime features for structural changes")
        print("\n📄 JSON output can be consumed by asset analysis agents:")
        print("   • Stock analysis agent")
        print("   • Bond analysis agent") 
        print("   • Gold analysis agent")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Enhanced analysis failed: {e}")


if __name__ == "__main__":
    main() 