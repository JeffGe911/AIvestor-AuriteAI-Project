# master_investment_workflow.py
"""
Master Investment Workflow - Aurite AI Project
Orchestrates Agent 1 → Agent 2 → Agent 3 for complete investment advisory

Author: Aurite AI Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import asdict

# Import our agents
from enhanced_aivestor_agent1 import agent1_for_workflow
from stock_analysis_agent import StockAnalysisAgent, AgentConfig, LLMConfig
from gold_analysis_agent import GoldAnalysisAgent
from etf_analysis_agent import BondAnalysisAgent
from enhanced_macro_analysis import EnhancedMacroAnalyzer
from portfolio_agent import run as portfolio_agent_run, Profile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MasterWorkflow')

class AuriteInvestmentWorkflow:
    """
    Master workflow orchestrator for the complete investment advisory system
    """
    
    def __init__(self):
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'agent1_profile': None,
            'agent2_analysis': {},
            'agent3_portfolio': None,
            'final_recommendation': None
        }
        
        # Ensure output directory exists
        os.makedirs("analysis_outputs", exist_ok=True)
        
    async def run_complete_workflow(self) -> Dict[str, Any]:
        """
        Execute the complete 3-agent investment workflow
        """
        print("🚀 AURITE AI INVESTMENT ADVISOR")
        print("=" * 60)
        print("Complete Investment Analysis & Portfolio Construction")
        print("=" * 60)
        
        try:
            # AGENT 1: User Profiling
            print("\n🤖 STEP 1: User Preference Analysis")
            print("-" * 40)
            user_profile = self._run_agent1()
            self.results['agent1_profile'] = user_profile
            
            # AGENT 2: Market & Asset Analysis  
            print("\n📊 STEP 2: Market & Asset Analysis")
            print("-" * 40)
            analysis_results = await self._run_agent2(user_profile)
            self.results['agent2_analysis'] = analysis_results
            
            # AGENT 3: Portfolio Construction
            print("\n🎯 STEP 3: Portfolio Construction")
            print("-" * 40)
            portfolio = self._run_agent3(user_profile, analysis_results)
            self.results['agent3_portfolio'] = portfolio
            
            # Generate Final Recommendation
            final_recommendation = self._generate_final_output()
            self.results['final_recommendation'] = final_recommendation
            
            # Save complete results
            self._save_complete_results()
            
            print("\n✅ WORKFLOW COMPLETE!")
            print("=" * 60)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {"error": str(e), "session_id": self.session_id}
    
    def _run_agent1(self) -> Dict[str, Any]:
        """
        Run Agent 1: User Preference Collection
        """
        print("Collecting your investment preferences...")
        
        # Run the enhanced aivestor agent
        user_profile = agent1_for_workflow()
        
        # Save user profile to JSON file immediately
        self._save_user_profile(user_profile)
        
        print(f"✅ User profile created: {user_profile['profile_id']}")
        return user_profile
    
    async def _run_agent2(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Agent 2: Comprehensive Market Analysis
        """
        print("Running comprehensive market analysis...")
        
        analysis_results = {}
        
        # 1. Macro Analysis
        print("  📈 Analyzing macro economic conditions...")
        try:
            macro_analyzer = EnhancedMacroAnalyzer()
            macro_analysis = macro_analyzer.generate_analysis_report()
            macro_signals = macro_analyzer.export_macro_signals_json()
            analysis_results['macro'] = {
                'report': macro_analysis,
                'signals': macro_signals
            }
            
            # Save macro analysis to JSON file
            self._save_macro_analysis(analysis_results['macro'])
            
            print("  ✅ Macro analysis complete")
        except Exception as e:
            print(f"  ⚠️ Macro analysis failed: {e}")
            analysis_results['macro'] = {"error": str(e)}
        
        # 2. Stock Analysis - Use pre-computed NASDAQ-100 analysis
        print("  📊 Loading pre-computed NASDAQ-100 analysis...")
        try:
            # Load existing NASDAQ-100 analysis
            stock_analysis = self._load_precomputed_stock_analysis()
            
            if stock_analysis:
                # Filter based on user preferences
                filtered_analysis = self._filter_stocks_for_user(stock_analysis, user_profile)
                analysis_results['stocks'] = filtered_analysis
                
                # Save the filtered stock analysis to JSON file for portfolio agent
                self._save_filtered_stock_analysis(filtered_analysis)
                
                print(f"  ✅ Loaded and filtered {len(filtered_analysis.get('detailed_results', filtered_analysis.get('predictions', [])))} stocks from NASDAQ-100 analysis")
            else:
                # Fallback to live analysis if pre-computed not available
                print("  ⚠️ Pre-computed analysis not found, running live analysis...")
                api_key = os.getenv("OPENAI_API_KEY")
                config = AgentConfig(llm_config=LLMConfig(api_key=api_key))
                stock_agent = StockAnalysisAgent(config)
                
                symbols = self._select_stocks_for_user(user_profile)
                stock_analysis = await stock_agent.analyze_stocks(
                    symbols=symbols, 
                    macro_context=analysis_results.get('macro', {}).get('signals', {})
                )
                analysis_results['stocks'] = stock_analysis
                print("  ✅ Stock analysis complete")
        except Exception as e:
            print(f"  ⚠️ Stock analysis failed: {e}")
            analysis_results['stocks'] = {"error": str(e)}
        
        # 3. Bond Analysis
        print("  🏛️ Analyzing bond market...")
        try:
            bond_agent = BondAnalysisAgent()
            bond_symbols = self._select_bonds_for_user(user_profile)
            bond_analysis = await bond_agent.analyze_bonds(symbols=bond_symbols)
            analysis_results['bonds'] = bond_analysis
            
            # Save bond analysis to JSON file
            self._save_bond_analysis(bond_analysis)
            
            print("  ✅ Bond analysis complete")
        except Exception as e:
            print(f"  ⚠️ Bond analysis failed: {e}")
            analysis_results['bonds'] = {"error": str(e)}
        
        # 4. Gold Analysis
        print("  🏆 Analyzing precious metals...")
        try:
            gold_agent = GoldAnalysisAgent()
            gold_symbols = ["GLD", "IAU", "GC=F", "GDX", "SGOL"]
            gold_analysis = await gold_agent.analyze_gold(symbols=gold_symbols)
            analysis_results['gold'] = gold_analysis
            
            # Save gold analysis to JSON file
            self._save_gold_analysis(gold_analysis)
            
            print("  ✅ Gold analysis complete")
        except Exception as e:
            print(f"  ⚠️ Gold analysis failed: {e}")
            analysis_results['gold'] = {"error": str(e)}
        
        return analysis_results
    
    def _save_user_profile(self, user_profile: Dict[str, Any]):
        """Save user profile to JSON file with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_profile_{timestamp}.json"
            filepath = os.path.join("analysis_outputs", filename)
            
            with open(filepath, 'w') as f:
                json.dump(user_profile, f, indent=2, default=str)
            
            print(f"    💾 User profile saved to: {filename}")
            logger.info(f"User profile saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save user profile: {e}")
    
    def _save_macro_analysis(self, macro_analysis: Dict[str, Any]):
        """Save macro analysis to JSON file with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"macro_analysis_{timestamp}.json"
            filepath = os.path.join("analysis_outputs", filename)
            
            with open(filepath, 'w') as f:
                json.dump(macro_analysis, f, indent=2, default=str)
            
            print(f"    💾 Macro analysis saved to: {filename}")
            logger.info(f"Macro analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save macro analysis: {e}")
    
    def _save_bond_analysis(self, bond_analysis: Dict[str, Any]):
        """Save bond analysis to JSON file with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bond_analysis_{timestamp}.json"
            filepath = os.path.join("analysis_outputs", filename)
            
            with open(filepath, 'w') as f:
                json.dump(bond_analysis, f, indent=2, default=str)
            
            print(f"    💾 Bond analysis saved to: {filename}")
            logger.info(f"Bond analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save bond analysis: {e}")
    
    def _save_filtered_stock_analysis(self, filtered_analysis: Dict[str, Any]):
        """Save filtered stock analysis to JSON file with timestamp for portfolio agent"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_analysis_filtered_{timestamp}.json"
            filepath = os.path.join("analysis_outputs", filename)
            
            # Ensure the filtered analysis has the right format for loading
            if filtered_analysis.get('format_type') == 'prediction':
                # Convert prediction format to the expected structure
                formatted_analysis = {
                    "prediction_date": filtered_analysis.get('timestamp', datetime.now().strftime("%Y-%m-%d")),
                    "total_analyzed": filtered_analysis.get('total_requested', 0),
                    "successful_predictions": filtered_analysis.get('successful', 0),
                    "horizons": {
                        "next_quarter": filtered_analysis.get('predictions', [])
                    },
                    "filter_criteria": filtered_analysis.get('filter_criteria', {}),
                    "signals_summary": filtered_analysis.get('signals_summary', []),
                    "format_type": "prediction"
                }
            else:
                # Keep original format for detailed results
                formatted_analysis = filtered_analysis
            
            with open(filepath, 'w') as f:
                json.dump(formatted_analysis, f, indent=2, default=str)
            
            print(f"    💾 Filtered stock analysis saved to: {filename}")
            logger.info(f"Filtered stock analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save filtered stock analysis: {e}")
    
    def _save_gold_analysis(self, gold_analysis: Dict[str, Any]):
        """Save gold analysis to JSON file with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gold_analysis_{timestamp}.json"
            filepath = os.path.join("analysis_outputs", filename)
            
            with open(filepath, 'w') as f:
                json.dump(gold_analysis, f, indent=2, default=str)
            
            print(f"    💾 Gold analysis saved to: {filename}")
            logger.info(f"Gold analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save gold analysis: {e}")
    
    def _load_analysis_outputs(self) -> Dict[str, Any]:
        """
        Load the most recent analysis outputs from JSON files in analysis_outputs directory
        """
        print("Loading analysis outputs from JSON files...")
        
        analysis_data = {}
        output_dir = Path("analysis_outputs")
        
        if not output_dir.exists():
            logger.warning("analysis_outputs directory not found")
            return analysis_data
        
        # Load user profile
        try:
            user_files = list(output_dir.glob("user_profile_*.json"))
            if user_files:
                latest_user_file = max(user_files, key=lambda f: f.stat().st_mtime)
                with open(latest_user_file, 'r') as f:
                    analysis_data['user_profile'] = json.load(f)
                print(f"  ✅ Loaded user profile from {latest_user_file.name}")
            else:
                logger.warning("No user profile file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load user profile: {e}")
        
        # Load stock analysis
        try:
            stock_files = list(output_dir.glob("stock_analysis_*.json"))
            if stock_files:
                latest_stock_file = max(stock_files, key=lambda f: f.stat().st_mtime)
                with open(latest_stock_file, 'r') as f:
                    analysis_data['stock_analysis'] = json.load(f)
                print(f"  ✅ Loaded stock analysis from {latest_stock_file.name}")
            else:
                logger.warning("No stock analysis file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load stock analysis: {e}")
        
        # Load bond analysis
        try:
            bond_files = list(output_dir.glob("bond_analysis_*.json"))
            if bond_files:
                latest_bond_file = max(bond_files, key=lambda f: f.stat().st_mtime)
                with open(latest_bond_file, 'r') as f:
                    analysis_data['bond_analysis'] = json.load(f)
                print(f"  ✅ Loaded bond analysis from {latest_bond_file.name}")
            else:
                logger.warning("No bond analysis file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load bond analysis: {e}")
        
        # Load gold analysis
        try:
            gold_files = list(output_dir.glob("gold_analysis_*.json"))
            if gold_files:
                latest_gold_file = max(gold_files, key=lambda f: f.stat().st_mtime)
                with open(latest_gold_file, 'r') as f:
                    analysis_data['gold_analysis'] = json.load(f)
                print(f"  ✅ Loaded gold analysis from {latest_gold_file.name}")
            else:
                logger.warning("No gold analysis file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load gold analysis: {e}")
        
        # Load macro analysis
        try:
            macro_files = list(output_dir.glob("macro_analysis_*.json"))
            if macro_files:
                latest_macro_file = max(macro_files, key=lambda f: f.stat().st_mtime)
                with open(latest_macro_file, 'r') as f:
                    analysis_data['macro_analysis'] = json.load(f)
                print(f"  ✅ Loaded macro analysis from {latest_macro_file.name}")
            else:
                logger.warning("No macro analysis file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load macro analysis: {e}")
        
        return analysis_data
    
    def _run_agent3(self, user_profile: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Agent 3: Professional Portfolio Construction using fresh analysis results
        """
        print("Constructing optimal portfolio using professional portfolio agent...")
        
        # Always use the fresh analysis results from the current run
        print("\n  � Using fresh analysis results from current session:")
        user_data = user_profile
        stock_data = analysis_results.get('stocks', {})
        bond_data = analysis_results.get('bonds', {})
        gold_data = analysis_results.get('gold', {})
        macro_data = analysis_results.get('macro', {})
        
        print(f"    - User Profile: ✅ Fresh (current session)")
        print(f"    - Stock Analysis: ✅ Fresh ({len(stock_data.get('detailed_results', stock_data.get('horizons', {}).get('next_quarter', [])))} stocks)")
        print(f"    - Bond Analysis: ✅ Fresh")
        print(f"    - Gold Analysis: ✅ Fresh") 
        print(f"    - Macro Analysis: ✅ Fresh")
        
        try:
            print("\n  🔨 Running professional portfolio construction...")
            
            # Call the new portfolio agent
            print("\n  🔨 Running enhanced portfolio analysis...")
            
            # Create temporary profile JSON for the portfolio agent
            temp_profile_path = f"analysis_outputs/temp_profile_{self.session_id}.json"
            with open(temp_profile_path, 'w') as f:
                json.dump(user_data, f, indent=2, default=str)
            
            # Generate output paths
            portfolio_md_path = f"analysis_outputs/portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            portfolio_json_path = f"analysis_outputs/portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Run the new portfolio agent
            result = portfolio_agent_run(
                profile_path=temp_profile_path,
                analysis_dir="analysis_outputs",
                out_md=portfolio_md_path,
                out_json=portfolio_json_path,
                plots_dir=None,
                overrides={}
            )
            
            # Load the generated portfolio JSON
            with open(portfolio_json_path, 'r') as f:
                portfolio_dict = json.load(f)
            
            # Clean up temp file
            os.remove(temp_profile_path)
            
            print("  ✅ Enhanced portfolio analysis complete")
            print(f"  📄 Report saved to: {os.path.basename(portfolio_md_path)}")
            print(f"  � Data saved to: {os.path.basename(portfolio_json_path)}")
            
            return portfolio_dict
            
        except Exception as e:
            logger.error(f"Professional portfolio construction failed: {e}")
            print(f"  ❌ Portfolio construction failed: {e}")
            
            # Fall back to basic portfolio construction
            print("  🔄 Falling back to basic portfolio construction...")
            return self._basic_portfolio_fallback(user_data, stock_data, bond_data, gold_data)
    
    def _basic_portfolio_fallback(self, user_profile: Dict, stock_data: Dict, bond_data: Dict, gold_data: Dict) -> Dict[str, Any]:
        """
        Basic portfolio construction fallback if professional agent fails
        """
        
        risk_level = user_profile.get('risk_level', 'moderate')
        investment_amount = user_profile.get('investment_amount', 10000)
        
        # Basic portfolio allocation based on risk level
        if risk_level == 'aggressive':
            allocation = {'stocks': 0.70, 'bonds': 0.20, 'gold': 0.10}
        elif risk_level == 'conservative':
            allocation = {'stocks': 0.40, 'bonds': 0.50, 'gold': 0.10}
        else:  # moderate
            allocation = {'stocks': 0.60, 'bonds': 0.30, 'gold': 0.10}
        
        # Extract top recommendations from analysis
        top_stocks = self._extract_top_picks(stock_data, 'stocks')
        top_bonds = self._extract_top_picks(bond_data, 'bonds')
        top_gold = self._extract_top_picks(gold_data, 'gold')
        
        # Create basic analysis results structure for compatibility
        analysis_results = {
            'stocks': stock_data,
            'bonds': bond_data,
            'gold': gold_data
        }
        
        portfolio = {
            'allocation': allocation,
            'recommendations': {
                'stocks': top_stocks,
                'bonds': top_bonds,
                'gold': top_gold
            },
            'total_investment': investment_amount,
            'expected_return': self._calculate_expected_return(allocation, analysis_results),
            'risk_level': risk_level,
            'reasoning': self._generate_portfolio_reasoning(user_profile, analysis_results, allocation)
        }
        
        print("  ✅ Portfolio construction complete")
        return portfolio
    
    def _select_stocks_for_user(self, user_profile: Dict[str, Any]) -> List[str]:
        """Select appropriate stocks based on user profile"""
        risk_level = user_profile.get('risk_level', 'moderate')
        preferred_sectors = user_profile.get('preferred_sectors', [])
        
        # Base stock selection
        if risk_level == 'aggressive':
            base_stocks = ['NVDA', 'TSLA', 'AMD', 'SHOP', 'SQ']
        elif risk_level == 'conservative':
            base_stocks = ['AAPL', 'MSFT', 'JNJ', 'PG', 'V']
        else:
            base_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Add sector-specific stocks
        sector_stocks = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV'],
            'Financial': ['JPM', 'BAC', 'V', 'MA']
        }
        
        for sector in preferred_sectors:
            if sector in sector_stocks:
                base_stocks.extend(sector_stocks[sector])
        
        return list(set(base_stocks))  # Remove duplicates
    
    def _select_bonds_for_user(self, user_profile: Dict[str, Any]) -> List[str]:
        """Select appropriate bonds based on user profile"""
        risk_level = user_profile.get('risk_level', 'moderate')
        
        if risk_level == 'aggressive':
            return ['HYG', 'JNK', 'EMB']  # High yield, emerging markets
        elif risk_level == 'conservative':
            return ['SHY', 'IEF', 'BND', 'AGG']  # Safe government and aggregate bonds
        else:
            return ['IEF', 'LQD', 'HYG', 'TIP']  # Balanced mix
    
    def _load_precomputed_stock_analysis(self) -> Dict[str, Any]:
        """
        Load pre-computed NASDAQ-100 analysis from JSON file
        Returns the analysis data if found, None otherwise
        """
        try:
            output_dir = Path("analysis_outputs")
            
            # First, look for filtered stock analysis files (these take priority)
            filtered_files = list(output_dir.glob("stock_analysis_filtered_*.json"))
            
            # Then look for regular stock analysis files
            stock_files = list(output_dir.glob("stock_analysis_*.json"))
            
            # Exclude filtered files from stock_files to avoid duplicates
            stock_files = [f for f in stock_files if 'filtered' not in f.name]
            
            # Combine and prioritize filtered files
            all_files = filtered_files + stock_files
            
            if not all_files:
                print("    ⚠️ No pre-computed stock analysis files found")
                return None
            
            # Get the most recent file
            latest_file = max(all_files, key=lambda f: f.stat().st_mtime)
            
            print(f"    📊 Loading pre-computed analysis: {latest_file.name}")
            
            with open(latest_file, 'r') as f:
                stock_analysis = json.load(f)
            
            # Validate the structure - support both old and new formats
            if 'detailed_results' in stock_analysis:
                # Old format with detailed_results
                print(f"    ✅ Loaded detailed analysis for {len(stock_analysis['detailed_results'])} stocks")
                stock_analysis['format_type'] = 'detailed'
            elif 'horizons' in stock_analysis and 'next_quarter' in stock_analysis['horizons']:
                # New prediction format
                predictions = stock_analysis['horizons']['next_quarter']
                print(f"    ✅ Loaded prediction analysis for {len(predictions)} stocks")
                stock_analysis['format_type'] = 'prediction'
            else:
                print("    ⚠️ Invalid stock analysis format - missing 'detailed_results' or 'horizons'")
                return None
            
            return stock_analysis
            
        except Exception as e:
            print(f"    ❌ Failed to load pre-computed stock analysis: {e}")
            return None
    
    def _filter_stocks_for_user(self, stock_analysis: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter the comprehensive NASDAQ-100 analysis based on user preferences
        Supports both detailed and prediction formats
        """
        try:
            risk_level = user_profile.get('risk_level', 'moderate')
            preferred_sectors = user_profile.get('preferred_sectors', [])
            avoid_sectors = user_profile.get('avoid_sectors', [])
            investment_amount = user_profile.get('investment_amount', 10000)
            
            print(f"    🎯 Filtering stocks for {risk_level} risk level")
            if preferred_sectors:
                print(f"    📈 Preferred sectors: {', '.join(preferred_sectors)}")
            if avoid_sectors:
                print(f"    🚫 Avoiding sectors: {', '.join(avoid_sectors)}")
            
            format_type = stock_analysis.get('format_type', 'detailed')
            
            if format_type == 'prediction':
                return self._filter_prediction_format(stock_analysis, user_profile, risk_level, preferred_sectors, avoid_sectors)
            else:
                return self._filter_detailed_format(stock_analysis, user_profile, risk_level, preferred_sectors, avoid_sectors)
                
        except Exception as e:
            print(f"    ❌ Failed to filter stocks: {e}")
            # Return original analysis if filtering fails
            return stock_analysis
    
    def _filter_prediction_format(self, stock_analysis: Dict[str, Any], user_profile: Dict[str, Any], 
                                 risk_level: str, preferred_sectors: List[str], avoid_sectors: List[str]) -> Dict[str, Any]:
        """Filter the new prediction format"""
        try:
            predictions = stock_analysis['horizons']['next_quarter']
            
            # Filter by signal type based on risk level
            if risk_level == 'aggressive':
                target_sentiments = ['Buy', 'Strong Buy']
                min_expected_return = 6.0
                max_stocks = 15
            elif risk_level == 'conservative':
                target_sentiments = ['Buy', 'Hold']
                min_expected_return = 5.0
                max_stocks = 8
            else:  # moderate
                target_sentiments = ['Buy', 'Hold']
                min_expected_return = 5.5
                max_stocks = 12
            
            # Filter predictions
            filtered_predictions = []
            for prediction in predictions:
                sentiment = prediction.get('sentiment', 'Hold')
                expected_return = prediction.get('expected_return', 0)
                ticker = prediction.get('symbol', '')
                summary = prediction.get('summary', '')
                
                # Filter by sentiment and expected return
                if sentiment not in target_sentiments or expected_return < min_expected_return:
                    continue
                
                # Extract sector from summary (format: "Technology stock with...")
                sector = 'Unknown'
                if ' stock with' in summary:
                    sector = summary.split(' stock with')[0].strip()
                
                # Apply sector filtering
                if avoid_sectors and sector in avoid_sectors:
                    print(f"    🚫 Excluding {ticker} ({sector} - avoided sector)")
                    continue
                
                if preferred_sectors and sector not in preferred_sectors:
                    continue
                
                # Add sector info to prediction
                prediction['sector'] = sector
                filtered_predictions.append(prediction)
            
            # Sort by expected return (descending) and confidence
            filtered_predictions.sort(key=lambda x: (
                x.get('expected_return', 0),
                x.get('confidence', 0)
            ), reverse=True)
            
            # Limit to max_stocks
            top_predictions = filtered_predictions[:max_stocks]
            top_tickers = [pred['symbol'] for pred in top_predictions]
            
            print(f"    📊 Selected {len(top_tickers)} stocks: {', '.join(top_tickers[:5])}{'...' if len(top_tickers) > 5 else ''}")
            
            # Create filtered analysis in a format compatible with the workflow
            sentiment_distribution = {}
            signals_summary = []
            
            for pred in top_predictions:
                sentiment = pred.get('sentiment', 'Hold')
                sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
                
                # Convert to signals_summary format for compatibility
                signals_summary.append({
                    'ticker': pred['symbol'],
                    'signal': sentiment,
                    'score': pred.get('expected_return', 5.0),
                    'confidence': 'high' if pred.get('confidence', 0) >= 0.8 else 'medium',
                    'reason': f"Expected return: {pred.get('expected_return', 0)}%",
                    'sector': pred.get('sector', 'Unknown')
                })
            
            # Create filtered analysis structure
            filtered_analysis = {
                'timestamp': stock_analysis.get('prediction_date'),
                'total_requested': len(top_predictions),
                'successful': len(top_predictions),
                'failed': 0,
                'success_rate': 1.0,
                'signal_distribution': sentiment_distribution,
                'signals_summary': signals_summary,
                'predictions': top_predictions,  # Keep original predictions
                'filter_criteria': {
                    'risk_level': risk_level,
                    'target_sentiments': target_sentiments,
                    'min_expected_return': min_expected_return,
                    'max_stocks': max_stocks,
                    'preferred_sectors': preferred_sectors,
                    'avoid_sectors': avoid_sectors
                },
                'format_type': 'prediction'
            }
            
            return filtered_analysis
            
        except Exception as e:
            print(f"    ❌ Failed to filter prediction format: {e}")
            return stock_analysis
    
    def _filter_detailed_format(self, stock_analysis: Dict[str, Any], user_profile: Dict[str, Any],
                               risk_level: str, preferred_sectors: List[str], avoid_sectors: List[str]) -> Dict[str, Any]:
        """Filter the original detailed format (for backward compatibility)"""
        try:
            individual_analyses = stock_analysis.get('detailed_results', {})
            signals_summary = stock_analysis.get('signals_summary', [])
            
            # Filter by signal type based on risk level
            if risk_level == 'aggressive':
                target_signals = ['Buy', 'Strong Buy']
                max_stocks = 15
            elif risk_level == 'conservative':
                target_signals = ['Hold', 'Buy']
                max_stocks = 8
            else:  # moderate
                target_signals = ['Hold', 'Buy']
                max_stocks = 12
            
            # Filter signals by target signals and apply sector filtering
            filtered_signals = []
            for signal in signals_summary:
                if signal.get('signal') not in target_signals or 'ticker' not in signal:
                    continue
                
                ticker = signal['ticker']
                
                # Get sector information from individual analysis
                ticker_analysis = individual_analyses.get(ticker, {})
                company_info = ticker_analysis.get('company_info', {})
                sector = company_info.get('sector', 'Unknown')
                
                # Apply sector filtering
                if avoid_sectors and sector in avoid_sectors:
                    print(f"    🚫 Excluding {ticker} ({sector} - avoided sector)")
                    continue
                
                if preferred_sectors and sector not in preferred_sectors:
                    continue
                
                filtered_signals.append(signal)
            
            # Sort by score (descending) and confidence
            filtered_signals.sort(key=lambda x: (
                x.get('score', 0),
                1 if x.get('confidence') == 'high' else 0
            ), reverse=True)
            
            # Limit to max_stocks
            top_signals = filtered_signals[:max_stocks]
            top_tickers = [signal['ticker'] for signal in top_signals]
            
            print(f"    📊 Selected {len(top_tickers)} stocks: {', '.join(top_tickers[:5])}{'...' if len(top_tickers) > 5 else ''}")
            
            # Create filtered analysis with only selected stocks
            filtered_individual_analyses = {
                ticker: individual_analyses[ticker] 
                for ticker in top_tickers 
                if ticker in individual_analyses
            }
            
            # Create filtered analysis structure
            filtered_analysis = {
                'timestamp': stock_analysis.get('timestamp'),
                'total_requested': len(top_tickers),
                'successful': len(filtered_individual_analyses),
                'failed': len(top_tickers) - len(filtered_individual_analyses),
                'success_rate': len(filtered_individual_analyses) / len(top_tickers) if top_tickers else 0,
                'signal_distribution': self._calculate_signal_distribution(top_signals),
                'signals_summary': top_signals,
                'detailed_results': filtered_individual_analyses,
                'filter_criteria': {
                    'risk_level': risk_level,
                    'target_signals': target_signals,
                    'max_stocks': max_stocks,
                    'preferred_sectors': preferred_sectors,
                    'avoid_sectors': avoid_sectors
                },
                'format_type': 'detailed'
            }
            
            return filtered_analysis
            
        except Exception as e:
            print(f"    ❌ Failed to filter detailed format: {e}")
            return stock_analysis
            
        except Exception as e:
            print(f"    ❌ Failed to filter stocks: {e}")
            # Return original analysis if filtering fails
            return stock_analysis
    
    def _extract_top_picks(self, analysis_data: Dict, asset_type: str) -> List[str]:
        """Extract top picks from analysis data for portfolio construction"""
        try:
            if asset_type == 'stocks':
                # Handle both prediction and detailed formats
                if analysis_data.get('format_type') == 'prediction':
                    predictions = analysis_data.get('predictions', [])
                    return [pred.get('symbol', '') for pred in predictions[:8]]
                else:
                    signals = analysis_data.get('signals_summary', [])
                    return [signal.get('ticker', '') for signal in signals[:8]]
            
            elif asset_type == 'bonds':
                # Extract bond recommendations
                if 'recommendations' in analysis_data:
                    recs = analysis_data['recommendations']
                    if isinstance(recs, list):
                        return [rec.get('symbol', '') for rec in recs[:5]]
                    elif isinstance(recs, dict):
                        return list(recs.keys())[:5]
                # Fallback to default bonds
                return ['BND', 'AGG', 'IEF', 'TIP', 'LQD']
            
            elif asset_type == 'gold':
                # Extract gold recommendations
                if 'recommendations' in analysis_data:
                    recs = analysis_data['recommendations']
                    if isinstance(recs, list):
                        return [rec.get('symbol', '') for rec in recs[:3]]
                    elif isinstance(recs, dict):
                        return list(recs.keys())[:3]
                # Fallback to default gold
                return ['GLD', 'IAU', 'GDX']
            
            return []
        except Exception as e:
            logger.error(f"Failed to extract top picks for {asset_type}: {e}")
            # Return sensible defaults
            if asset_type == 'stocks':
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
            elif asset_type == 'bonds':
                return ['BND', 'AGG', 'IEF']
            elif asset_type == 'gold':
                return ['GLD', 'IAU']
            return []

    def _calculate_signal_distribution(self, signals: List[Dict]) -> Dict[str, int]:
        """Calculate signal distribution for filtered stocks"""
        distribution = {}
        for signal in signals:
            signal_type = signal.get('signal', 'Unknown')
            distribution[signal_type] = distribution.get(signal_type, 0) + 1
        return distribution
    
    def _calculate_expected_return(self, allocation: Dict, analysis_results: Dict) -> float:
        """Calculate portfolio expected return"""
        # Simplified calculation
        base_returns = {'stocks': 0.08, 'bonds': 0.04, 'gold': 0.06}
        
        expected_return = sum(
            allocation[asset] * base_returns[asset] 
            for asset in allocation
        )
        
        return round(expected_return, 3)
    
    def _generate_portfolio_reasoning(self, user_profile: Dict, analysis_results: Dict, allocation: Dict) -> str:
        """Generate reasoning for portfolio construction"""
        risk_level = user_profile.get('risk_level', 'moderate')
        investment_goal = user_profile.get('investment_goal', 'wealth building')
        
        reasoning = f"""
Portfolio constructed for {risk_level} risk investor with {investment_goal} goal.

Allocation Reasoning:
- Stocks ({allocation['stocks']*100:.0f}%): Selected for growth potential based on current market analysis
- Bonds ({allocation['bonds']*100:.0f}%): Provides stability and income generation
- Gold ({allocation['gold']*100:.0f}%): Hedge against inflation and market volatility

Market Conditions Considered:
- Current macro economic environment
- Individual asset analysis and rankings
- Risk-adjusted return optimization
"""
        return reasoning.strip()
    
    def _generate_final_output(self) -> Dict[str, Any]:
        """Generate the final comprehensive recommendation"""
        portfolio = self.results['agent3_portfolio']
        user_profile = self.results['agent1_profile']
        
        final_output = {
            'session_id': self.session_id,
            'user_profile_summary': {
                'investment_amount': user_profile.get('investment_amount'),
                'risk_level': user_profile.get('risk_level'),
                'time_horizon': user_profile.get('time_horizon'),
                'investment_goal': user_profile.get('investment_goal')
            },
            'portfolio_allocation': portfolio.get('allocation'),
            'specific_recommendations': portfolio.get('recommendations'),
            'expected_annual_return': f"{portfolio.get('expected_return', 0)*100:.1f}%",
            'reasoning': portfolio.get('reasoning'),
            'implementation_plan': {
                'immediate_actions': [
                    "Review and approve the recommended allocation",
                    "Open investment accounts if needed",
                    "Execute trades according to allocation"
                ],
                'monitoring': [
                    "Rebalance quarterly or when allocation drifts >5%",
                    "Review annually or when life circumstances change"
                ]
            },
            'risk_warnings': [
                "Past performance does not guarantee future results",
                "All investments carry risk of loss",
                "Consider consulting with a financial advisor"
            ]
        }
        
        return final_output
    
    def _save_complete_results(self):
        """Save complete workflow results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_investment_recommendation_{timestamp}.json"
        filepath = os.path.join("analysis_outputs", filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"💾 Complete results saved to: {filepath}")
            logger.info(f"Complete workflow results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    async def run_portfolio_only_workflow(self) -> Dict[str, Any]:
        """
        Run only Agent 3 (portfolio construction) using existing JSON outputs
        This enables modular testing and re-running portfolio construction
        """
        print("🎯 AURITE AI PORTFOLIO CONSTRUCTION")
        print("=" * 60)
        print("Using existing analysis outputs for portfolio construction")
        print("=" * 60)
        
        try:
            # Load all saved analysis outputs
            print("\n📁 Loading saved analysis outputs...")
            saved_analysis = self._load_analysis_outputs()
            
            if not saved_analysis:
                print("❌ No analysis outputs found. Please run the complete workflow first.")
                return {"error": "No analysis outputs found"}
            
            # Check which data sources we have
            required_keys = ['user_profile', 'stock_analysis', 'bond_analysis', 'gold_analysis']
            missing_keys = [key for key in required_keys if key not in saved_analysis]
            
            if missing_keys:
                print(f"⚠️ Missing analysis data: {', '.join(missing_keys)}")
                print("Portfolio construction will proceed with available data.")
            
            # Run portfolio construction
            print("\n🎯 Running Portfolio Construction (Agent 3)")
            print("-" * 40)
            
            user_data = saved_analysis.get('user_profile', {})
            stock_data = saved_analysis.get('stock_analysis', {})
            bond_data = saved_analysis.get('bond_analysis', {})
            gold_data = saved_analysis.get('gold_analysis', {})
            macro_data = saved_analysis.get('macro_analysis', {})
            
            # Use professional portfolio agent
            portfolio_recommendation = await self.portfolio_agent.construct_portfolio(
                user_profile=user_data,
                stock_analysis=stock_data,
                bond_analysis=bond_data,
                gold_analysis=gold_data,
                market_conditions=macro_data
            )
            
            # Convert to dict and save
            if hasattr(portfolio_recommendation, '__dict__'):
                portfolio_dict = asdict(portfolio_recommendation) if hasattr(portfolio_recommendation, '__dataclass_fields__') else vars(portfolio_recommendation)
            else:
                portfolio_dict = portfolio_recommendation
            
            # Save the portfolio recommendation
            portfolio_filename = f"portfolio_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            portfolio_path = f"analysis_outputs/{portfolio_filename}"
            
            with open(portfolio_path, 'w') as f:
                json.dump(portfolio_dict, f, indent=2, default=str)
            
            print(f"✅ Portfolio saved to: {portfolio_filename}")
            print("\n✅ PORTFOLIO CONSTRUCTION COMPLETE!")
            print("=" * 60)
            
            return {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'portfolio': portfolio_dict,
                'data_sources': list(saved_analysis.keys()),
                'output_file': portfolio_filename
            }
            
        except Exception as e:
            logger.error(f"Portfolio-only workflow failed: {e}")
            return {"error": str(e), "session_id": self.session_id}

# ===================== MAIN EXECUTION =====================

async def main():
    """
    Main function to run the Aurite Investment Workflow
    """
    import sys
    
    workflow = AuriteInvestmentWorkflow()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--portfolio-only":
        # Run only portfolio construction using existing JSON files
        result = await workflow.run_portfolio_only_workflow()
    else:
        # Run complete workflow
        result = await workflow.run_complete_workflow()
    
    if 'error' in result:
        print(f"\n❌ Workflow failed: {result['error']}")
        return 1
    else:
        print(f"\n🎉 Workflow completed successfully!")
        print(f"Session ID: {result['session_id']}")
        return 0

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
