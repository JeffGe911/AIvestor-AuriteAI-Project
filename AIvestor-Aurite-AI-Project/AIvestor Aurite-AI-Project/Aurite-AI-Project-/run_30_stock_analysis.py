#!/usr/bin/env python3
"""
Run stock analysis on 30 NASDAQ-100 stocks and save results to analysis_outputs
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from stock_analysis_agent import StockAnalysisAgent, AgentConfig, LLMConfig

# Select 30 diverse NASDAQ-100 stocks from different sectors
test_tickers = [
    # Technology
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'ADBE', 'CRM', 'INTC', 'AMD', 'CSCO',
    # Consumer Cyclical
    'AMZN', 'TSLA', 'NFLX', 'SBUX', 'BKNG', 'MAR',
    # Communication Services
    'GOOG', 'CMCSA', 'T',
    # Consumer Defensive
    'COST', 'PEP', 'KO', 'WBA',
    # Healthcare
    'JNJ', 'PFE', 'ABBV', 'UNH',
    # Industrials
    'HON', 'LMT',
    # Semiconductors
    'AVGO', 'QCOM'
]

async def run_30_stock_analysis():
    """Run analysis on 30 NASDAQ-100 stocks"""
    
    print("🚀 Running Stock Analysis on 30 NASDAQ-100 Stocks")
    print(f"📊 Selected stocks: {', '.join(test_tickers)}")
    print("-" * 80)
    
    # Ensure output directory exists
    os.makedirs("analysis_outputs", exist_ok=True)
    
    # Create agent config with the fixed scoring logic
    llm_config = LLMConfig()
    llm_config.model = "gpt-4o-mini"
    llm_config.temperature = 0.3
    llm_config.max_tokens = 4000
    
    config = AgentConfig()
    config.llm_config = llm_config
    
    # Initialize agent
    agent = StockAnalysisAgent(config)
    
    print("🤖 Initializing Stock Analysis Agent...")
    
    # Run analysis
    print("📈 Starting analysis (this may take several minutes)...")
    results = await agent.analyze_stocks(test_tickers)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"analysis_outputs/stock_analysis_30stocks_{timestamp}.json")
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Display summary
    if results:
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"- Total stocks analyzed: {results.get('successful', 0)}")
        print(f"- Success rate: {results.get('success_rate', 0):.1%}")
        
        # Show signal distribution
        signal_dist = results.get('signal_distribution', {})
        if signal_dist:
            print(f"- Signal distribution:")
            for signal, count in signal_dist.items():
                print(f"  • {signal}: {count} stocks")
        
        # Show score distribution from detailed results
        detailed_results = results.get('detailed_results', {})
        if detailed_results:
            scores = []
            for ticker, data in detailed_results.items():
                score = data.get('signal', {}).get('score', 5.0)
                scores.append(score)
            
            score_dist = {}
            for score in scores:
                score_dist[score] = score_dist.get(score, 0) + 1
            
            print(f"- Score distribution:")
            for score, count in sorted(score_dist.items()):
                print(f"  • Score {score}: {count} stocks")
        
        # Show some examples
        signals_summary = results.get('signals_summary', [])
        if signals_summary:
            print(f"\n🎯 Sample Results:")
            for signal in signals_summary[:10]:  # Show first 10
                ticker = signal.get('ticker', 'N/A')
                signal_type = signal.get('signal', 'N/A')
                score = signal.get('score', 'N/A')
                confidence = signal.get('confidence', 'N/A')
                print(f"  {ticker}: {signal_type} (Score: {score}, Confidence: {confidence})")
    
    return output_file

if __name__ == "__main__":
    asyncio.run(run_30_stock_analysis())
