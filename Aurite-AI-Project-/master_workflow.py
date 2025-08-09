"""
Master Workflow - Aurite AI Multi-Agent Financial System
Orchestrates all agents for complete portfolio recommendations

Author: Aurite AI Project Team
Version: 1.0.0
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MasterWorkflow')

class AuriteMultiAgentWorkflow:
    """
    Master workflow that orchestrates all agents:
    1. Agent 1: User Profile Collection
    2. Agent 2a: Macro Economic Analysis  
    3. Agent 2b: Stock Analysis
    4. Agent 2c: Gold Analysis
    5. Agent 2d: ETF/Bond Analysis
    6. Agent 3: Portfolio Construction
    """
    
    def __init__(self):
        self.workflow_id = f"AURITE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = "analysis_outputs"
        self.ensure_output_directory()
        logger.info(f"Master Workflow initialized: {self.workflow_id}")
    
    def ensure_output_directory(self):
        """Ensure analysis_outputs directory exists"""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory ready: {self.output_dir}")
    
    async def run_complete_workflow(self, 
                                   user_input: Optional[Dict[str, Any]] = None,
                                   skip_user_input: bool = False) -> Dict[str, Any]:
        """
        Run the complete multi-agent workflow
        
        Args:
            user_input: Optional user preferences to skip interactive input
            skip_user_input: If True, uses default demo profile
        
        Returns:
            Complete portfolio recommendation
        """
        
        logger.info("="*80)
        logger.info("🚀 AURITE MULTI-AGENT WORKFLOW STARTING")
        logger.info("="*80)
        
        try:
            # Step 1: Collect User Profile (Agent 1)
            logger.info("📊 Step 1: Collecting User Profile...")
            user_profile = await self._run_agent_1(user_input, skip_user_input)
            logger.info(f"✅ User profile collected: {user_profile.get('profile_id', 'N/A')}")
            
            # Step 2: Run Market Analysis Agents (Agent 2a-2d)
            logger.info("📈 Step 2: Running Market Analysis Agents...")
            
            # Run all analysis agents in parallel for efficiency
            macro_analysis = await self._run_macro_analysis()
            stock_analysis = await self._run_stock_analysis()
            gold_analysis = await self._run_gold_analysis()
            etf_analysis = await self._run_etf_analysis()
            
            logger.info("✅ All market analysis agents completed")
            
            # Step 3: Portfolio Construction (Agent 3)
            logger.info("🎯 Step 3: Constructing Optimal Portfolio...")
            portfolio_recommendation = await self._run_portfolio_agent(
                user_profile=user_profile,
                stock_analysis=stock_analysis,
                bond_analysis=etf_analysis,  # Using ETF analysis for bonds
                gold_analysis=gold_analysis
            )
            
            # Step 4: Generate Master Report
            logger.info("📋 Step 4: Generating Master Report...")
            master_report = self._generate_master_report(
                workflow_id=self.workflow_id,
                user_profile=user_profile,
                macro_analysis=macro_analysis,
                stock_analysis=stock_analysis,
                gold_analysis=gold_analysis,
                etf_analysis=etf_analysis,
                portfolio_recommendation=portfolio_recommendation
            )
            
            # Step 5: Save Results
            self._save_master_report(master_report)
            
            logger.info("="*80)
            logger.info("✅ AURITE WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
            return master_report
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {str(e)}")
            return {
                'status': 'error',
                'workflow_id': self.workflow_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_agent_1(self, user_input: Optional[Dict], skip_user_input: bool) -> Dict[str, Any]:
        """Run Agent 1 - User Profile Collection"""
        
        if skip_user_input:
            # Use demo profile for presentations
            return {
                'profile_id': f'DEMO_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'investment_amount': 250000,
                'risk_tolerance': 'moderate',
                'time_horizon': 10,
                'investment_goals': ['retirement', 'growth'],
                'preferred_sectors': ['Technology', 'Healthcare'],
                'avoid_sectors': [],
                'monthly_contribution': 2000,
                'tax_sensitive': True,
                'esg_preference': False,
                'liquidity_needs': 'low',
                'experience_level': 'intermediate'
            }
        
        if user_input:
            return user_input
        
        # Try to run the actual Agent 1
        try:
            # Import and run enhanced_aivestor_agent1
            import enhanced_aivestor_agent1
            
            # Check if the agent has a main function
            if hasattr(enhanced_aivestor_agent1, 'collect_user_profile'):
                profile = enhanced_aivestor_agent1.collect_user_profile()
            elif hasattr(enhanced_aivestor_agent1, 'main'):
                profile = enhanced_aivestor_agent1.main()
            else:
                # Fallback to demo profile
                logger.warning("Agent 1 interface not found, using demo profile")
                return await self._run_agent_1(None, True)
            
            return profile
            
        except ImportError:
            logger.warning("Agent 1 not found, using demo profile")
            return await self._run_agent_1(None, True)
        except Exception as e:
            logger.error(f"Agent 1 failed: {e}, using demo profile")
            return await self._run_agent_1(None, True)
    
    async def _run_macro_analysis(self) -> Dict[str, Any]:
        """Run enhanced macro analysis"""
        try:
            import enhanced_macro_analysis
            
            # Check if it has an async main function
            if hasattr(enhanced_macro_analysis, 'main'):
                result = enhanced_macro_analysis.main()
                if hasattr(result, '__await__'):  # Check if it's a coroutine
                    return await result
                return result
            else:
                logger.warning("Macro analysis main function not found")
                return self._get_fallback_macro_analysis()
                
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            return self._get_fallback_macro_analysis()
    
    async def _run_stock_analysis(self) -> Dict[str, Any]:
        """Run stock analysis agent"""
        try:
            import stock_analysis_agent
            
            if hasattr(stock_analysis_agent, 'main'):
                result = stock_analysis_agent.main()
                if hasattr(result, '__await__'):
                    return await result
                return result
            else:
                logger.warning("Stock analysis main function not found")
                return self._get_fallback_stock_analysis()
                
        except Exception as e:
            logger.error(f"Stock analysis failed: {e}")
            return self._get_fallback_stock_analysis()
    
    async def _run_gold_analysis(self) -> Dict[str, Any]:
        """Run gold analysis agent"""
        try:
            import gold_analysis_agent
            
            if hasattr(gold_analysis_agent, 'main'):
                result = gold_analysis_agent.main()
                if hasattr(result, '__await__'):
                    return await result
                return result
            else:
                logger.warning("Gold analysis main function not found")
                return self._get_fallback_gold_analysis()
                
        except Exception as e:
            logger.error(f"Gold analysis failed: {e}")
            return self._get_fallback_gold_analysis()
    
    async def _run_etf_analysis(self) -> Dict[str, Any]:
        """Run ETF analysis agent"""
        try:
            import etf_analysis_agent
            
            if hasattr(etf_analysis_agent, 'main'):
                result = etf_analysis_agent.main()
                if hasattr(result, '__await__'):
                    return await result
                return result
            else:
                logger.warning("ETF analysis main function not found")
                return self._get_fallback_etf_analysis()
                
        except Exception as e:
            logger.error(f"ETF analysis failed: {e}")
            return self._get_fallback_etf_analysis()
    
    async def _run_portfolio_agent(self, user_profile, stock_analysis, bond_analysis, gold_analysis) -> Dict[str, Any]:
        """Run Agent 3 - Portfolio Construction"""
        try:
            from portfolio_agent import portfolio_agent_for_workflow
            
            result = portfolio_agent_for_workflow(
                user_profile=user_profile,
                stock_analysis=stock_analysis,
                bond_analysis=bond_analysis,
                gold_analysis=gold_analysis
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio agent failed: {e}")
            raise e
    
    def _generate_master_report(self, **kwargs) -> Dict[str, Any]:
        """Generate comprehensive master report"""
        
        return {
            'aurite_workflow': {
                'workflow_id': kwargs['workflow_id'],
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            },
            
            'executive_summary': {
                'investment_amount': kwargs['user_profile'].get('investment_amount', 0),
                'risk_profile': kwargs['user_profile'].get('risk_tolerance', 'moderate'),
                'time_horizon': f"{kwargs['user_profile'].get('time_horizon', 5)} years",
                'portfolio_sharpe_ratio': kwargs['portfolio_recommendation']['risk_analytics']['sharpe_ratio'],
                'expected_return': kwargs['portfolio_recommendation']['executive_summary']['expected_annual_return'],
                'total_positions': kwargs['portfolio_recommendation']['executive_summary']['total_positions']
            },
            
            'user_profile': kwargs['user_profile'],
            
            'market_analysis': {
                'macro_conditions': kwargs['macro_analysis'],
                'stock_opportunities': kwargs['stock_analysis'],
                'gold_outlook': kwargs['gold_analysis'],
                'bond_etf_analysis': kwargs['etf_analysis']
            },
            
            'portfolio_recommendation': kwargs['portfolio_recommendation'],
            
            'implementation_summary': {
                'total_steps': len(kwargs['portfolio_recommendation']['implementation']['step_by_step_plan']),
                'execution_timeframe': kwargs['portfolio_recommendation']['implementation']['execution_timeframe'],
                'estimated_costs': kwargs['portfolio_recommendation']['implementation']['estimated_transaction_cost'],
                'priority_actions': kwargs['portfolio_recommendation']['implementation']['step_by_step_plan'][:3]
            },
            
            'monitoring_dashboard': {
                'rebalancing_frequency': kwargs['portfolio_recommendation']['rebalancing']['frequency'],
                'alert_count': len(kwargs['portfolio_recommendation']['monitoring']['alerts']),
                'next_review': kwargs['portfolio_recommendation']['monitoring']['next_review']
            },
            
            'ai_insights_summary': kwargs['portfolio_recommendation']['ai_insights']
        }
    
    def _save_master_report(self, report: Dict[str, Any]):
        """Save the master report"""
        filename = f"master_portfolio_report_{self.workflow_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"✅ Master report saved: {filepath}")
            print(f"\n🎯 FINAL REPORT SAVED: {filepath}")
            
            # Also save a human-readable summary
            self._save_human_readable_summary(report)
            
        except Exception as e:
            logger.error(f"❌ Failed to save master report: {e}")
    
    def _save_human_readable_summary(self, report: Dict[str, Any]):
        """Save a human-readable summary"""
        summary_filename = f"portfolio_summary_{self.workflow_id}.txt"
        summary_filepath = os.path.join(self.output_dir, summary_filename)
        
        try:
            with open(summary_filepath, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("AURITE AI - PROFESSIONAL PORTFOLIO RECOMMENDATION\n")
                f.write("=" * 80 + "\n\n")
                
                # Executive Summary
                exec_summary = report['executive_summary']
                f.write("EXECUTIVE SUMMARY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Investment Amount: ${exec_summary['investment_amount']:,.0f}\n")
                f.write(f"Risk Profile: {exec_summary['risk_profile'].title()}\n")
                f.write(f"Time Horizon: {exec_summary['time_horizon']}\n")
                f.write(f"Expected Return: {exec_summary['expected_return']}\n")
                f.write(f"Sharpe Ratio: {exec_summary['portfolio_sharpe_ratio']}\n")
                f.write(f"Total Positions: {exec_summary['total_positions']}\n\n")
                
                # Portfolio Allocation
                f.write("PORTFOLIO ALLOCATION:\n")
                f.write("-" * 20 + "\n")
                for position in report['portfolio_recommendation']['asset_allocation']['detailed_positions'][:10]:
                    f.write(f"{position['ticker']:>6} | {position['asset_class']:>8} | {position['weight']:>7} | {position['dollar_amount']}\n")
                f.write("\n")
                
                # Implementation
                f.write("IMPLEMENTATION PLAN:\n")
                f.write("-" * 20 + "\n")
                for step in report['implementation_summary']['priority_actions']:
                    f.write(f"• {step}\n")
                f.write("\n")
                
                # AI Insights
                f.write("AI INSIGHTS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"{report['ai_insights_summary']}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("Generated by Aurite AI Multi-Agent System\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"✅ Human-readable summary saved: {summary_filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save summary: {e}")
    
    # Fallback analysis methods for demo purposes
    def _get_fallback_macro_analysis(self):
        """Fallback macro analysis for demo"""
        return {
            'horizons': {
                'next_year': {
                    'gdp_growth': 0.025,
                    'inflation': 0.032,
                    'interest_rates': 0.045,
                    'unemployment': 0.038,
                    'market_sentiment': 'cautiously_optimistic'
                }
            },
            'status': 'fallback_demo_data'
        }
    
    def _get_fallback_stock_analysis(self):
        """Fallback stock analysis for demo"""
        return {
            'horizons': {
                'next_year': [
                    {'ticker': 'AAPL', 'expected_return': 0.12, 'volatility': 0.22, 'confidence': 0.85, 
                     'sentiment': 'bullish', 'summary': 'Strong fundamentals', 'sector': 'Technology'},
                    {'ticker': 'MSFT', 'expected_return': 0.11, 'volatility': 0.20, 'confidence': 0.87,
                     'sentiment': 'bullish', 'summary': 'Cloud growth', 'sector': 'Technology'},
                    {'ticker': 'NVDA', 'expected_return': 0.15, 'volatility': 0.28, 'confidence': 0.82,
                     'sentiment': 'bullish', 'summary': 'AI leadership', 'sector': 'Technology'},
                    {'ticker': 'JPM', 'expected_return': 0.09, 'volatility': 0.18, 'confidence': 0.80,
                     'sentiment': 'neutral', 'summary': 'Banking leader', 'sector': 'Financial'},
                    {'ticker': 'JNJ', 'expected_return': 0.07, 'volatility': 0.14, 'confidence': 0.85,
                     'sentiment': 'neutral', 'summary': 'Defensive play', 'sector': 'Healthcare'}
                ]
            },
            'status': 'fallback_demo_data'
        }
    
    def _get_fallback_gold_analysis(self):
        """Fallback gold analysis for demo"""
        return {
            'horizons': {
                'next_year': [
                    {'ticker': 'GLD', 'expected_return': 0.05, 'volatility': 0.15, 'confidence': 0.70,
                     'sentiment': 'neutral', 'summary': 'Inflation hedge'},
                    {'ticker': 'IAU', 'expected_return': 0.048, 'volatility': 0.14, 'confidence': 0.68,
                     'sentiment': 'neutral', 'summary': 'Low-cost gold'}
                ]
            },
            'status': 'fallback_demo_data'
        }
    
    def _get_fallback_etf_analysis(self):
        """Fallback ETF analysis for demo"""
        return {
            'horizons': {
                'next_year': [
                    {'ticker': 'AGG', 'expected_return': 0.04, 'volatility': 0.05, 'confidence': 0.80,
                     'sentiment': 'neutral', 'summary': 'Bond market', 'bond_type': 'Aggregate'},
                    {'ticker': 'TLT', 'expected_return': 0.045, 'volatility': 0.08, 'confidence': 0.75,
                     'sentiment': 'bullish', 'summary': 'Long treasuries', 'bond_type': 'Treasury'},
                    {'ticker': 'HYG', 'expected_return': 0.065, 'volatility': 0.12, 'confidence': 0.70,
                     'sentiment': 'neutral', 'summary': 'High yield', 'bond_type': 'High Yield'}
                ]
            },
            'status': 'fallback_demo_data'
        }

# ===================== Main Execution Functions =====================

async def run_demo_workflow():
    """Run a complete demo workflow for presentation"""
    print("\n" + "🚀" * 30)
    print("AURITE AI - COMPLETE MULTI-AGENT WORKFLOW DEMO")
    print("🚀" * 30)
    
    workflow = AuriteMultiAgentWorkflow()
    
    # Run with demo data
    result = await workflow.run_complete_workflow(skip_user_input=True)
    
    if result.get('status') != 'error':
        print("\n✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved in: {workflow.output_dir}")
        print(f"🆔 Workflow ID: {workflow.workflow_id}")
        
        # Quick summary
        exec_summary = result['executive_summary']
        print(f"\n📊 PORTFOLIO SUMMARY:")
        print(f"   💰 Investment: ${exec_summary['investment_amount']:,.0f}")
        print(f"   📈 Expected Return: {exec_summary['expected_return']}")
        print(f"   📊 Sharpe Ratio: {exec_summary['portfolio_sharpe_ratio']}")
        print(f"   🎯 Positions: {exec_summary['total_positions']}")
        
    else:
        print(f"\n❌ WORKFLOW FAILED: {result.get('error', 'Unknown error')}")
    
    return result

async def run_interactive_workflow():
    """Run interactive workflow with user input"""
    print("\n" + "🎯" * 30)
    print("AURITE AI - INTERACTIVE PORTFOLIO BUILDER")
    print("🎯" * 30)
    
    workflow = AuriteMultiAgentWorkflow()
    
    # Run with user interaction
    result = await workflow.run_complete_workflow(skip_user_input=False)
    
    return result

def main():
    """Main entry point"""
    import asyncio
    
    print("Select mode:")
    print("1. Demo Workflow (Quick demonstration)")
    print("2. Interactive Workflow (Full user input)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            result = asyncio.run(run_demo_workflow())
        elif choice == "2":
            result = asyncio.run(run_interactive_workflow())
        elif choice == "3":
            print("Goodbye!")
            return
        else:
            print("Invalid choice. Running demo workflow...")
            result = asyncio.run(run_demo_workflow())
            
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
