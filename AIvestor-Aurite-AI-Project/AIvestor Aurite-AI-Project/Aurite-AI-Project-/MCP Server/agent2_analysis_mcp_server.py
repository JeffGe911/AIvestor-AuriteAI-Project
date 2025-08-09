# agent2_analysis_mcp_server.py
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import ai_agent modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from mcp.server import FastMCP
    import yfinance as yf
    import openai
    import pandas as pd
    MCP_AVAILABLE = True
except ImportError as e:
    logger.error(f"⚠️  MCP dependencies not available: {e}")
    MCP_AVAILABLE = False
    sys.exit(1)

# Import enhanced macro analysis
try:
    from enhanced_macro_analysis import EnhancedMacroAnalyzer
    ENHANCED_MACRO_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  Enhanced macro analysis not available")
    ENHANCED_MACRO_AVAILABLE = False

# Import macro analysis components
from ai_agent.agent import MacroAnalysisAgent
from ai_agent.api_client import MacroAPIClient, APIConfig
from ai_agent.feature_engineer import FeatureEngineer
from ai_agent.model_manager import ModelManager
from ai_agent.config import ModelConfig

# Import analysis agents with unified error handling
ANALYSIS_AGENTS = {}

try:
    from etf_analysis_agent import BondAnalysisAgent, AgentConfig, LLMConfig, BondPrediction, MarketConditions
    ANALYSIS_AGENTS['bond'] = {
        'class': BondAnalysisAgent,
        'config_class': AgentConfig,
        'llm_config_class': LLMConfig,
        'name': 'Bond Analysis Agent',
        'system_prompt': "You are an expert bond and fixed income analyst specializing in comprehensive bond analysis.",
        'symbols': ["TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "TIPS"]
    }
except ImportError:
    logger.warning("⚠️  Bond analysis agent not available")

try:
    from gold_analysis_agent import GoldAnalysisAgent, AgentConfig as GoldAgentConfig, LLMConfig as GoldLLMConfig, GoldPrediction, GoldMarketConditions
    ANALYSIS_AGENTS['gold'] = {
        'class': GoldAnalysisAgent,
        'config_class': GoldAgentConfig,
        'llm_config_class': GoldLLMConfig,
        'name': 'Gold Analysis Agent',
        'system_prompt': "You are an expert gold and precious metals analyst specializing in comprehensive gold analysis.",
        'symbols': ["GLD", "IAU", "GC=F", "GDX", "SGOL"]
    }
except ImportError:
    logger.warning("⚠️  Gold analysis agent not available")

try:
    from stock_analysis_agent import StockAnalysisAgent, AgentConfig as StockAgentConfig, LLMConfig as StockLLMConfig, StockPrediction, StockMarketConditions, get_nasdaq100_tickers
    # Get NASDAQ 100 tickers for comprehensive stock analysis
    nasdaq100_symbols = get_nasdaq100_tickers()
    logger.info(f"🚀 Loaded {len(nasdaq100_symbols)} NASDAQ-100 stocks for analysis")
    
    ANALYSIS_AGENTS['stock'] = {
        'class': StockAnalysisAgent,
        'config_class': StockAgentConfig,
        'llm_config_class': StockLLMConfig,
        'name': 'Stock Analysis Agent (NASDAQ-100)',
        'system_prompt': "You are an expert stock market analyst specializing in comprehensive NASDAQ-100 stock analysis.",
        'symbols': nasdaq100_symbols
    }
except ImportError:
    logger.warning("⚠️  Stock analysis agent not available")

warnings.filterwarnings('ignore')

# Create the MCP server instance
mcp = FastMCP("Agent 2 Portfolio Analysis Assistant")

# Initialize global components
macro_agent = None
api_client = None
model_manager = None
agents = {}
openai_client = None
output_dir = Path("analysis_outputs")

# Analysis configuration
ANALYSIS_CONFIG = {
    'horizon': 'next_quarter',  # Q3 focus
    'top_picks': 5,            # Top 5 picks per asset class
    'total_target': 15         # 15 total assets (5 per class)
}

async def initialize_components():
    """Initialize all analysis components."""
    global macro_agent, api_client, model_manager, agents, openai_client
    
    try:
        logger.info("🚀 Initializing Agent 2 MCP Server components...")
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Initialize macro components
        api_client = MacroAPIClient(APIConfig())
        model_manager = ModelManager(ModelConfig())
        macro_agent = MacroAnalysisAgent()
        
        # Initialize OpenAI client
        openai_client = openai.OpenAI()
        
        # Initialize analysis agents
        for agent_type, agent_info in ANALYSIS_AGENTS.items():
            try:
                llm_config = agent_info['llm_config_class']()
                config = agent_info['config_class'](
                    name=agent_info['name'],
                    system_prompt=agent_info['system_prompt'],
                    llm_config=llm_config
                )
                
                agent = agent_info['class'](config=config)
                agents[agent_type] = agent
                logger.info(f"✅ {agent_info['name']} initialized")
                
            except Exception as e:
                logger.warning(f"⚠️  {agent_info['name']} initialization failed: {e}")
        
        logger.info(f"✅ Initialized {len(agents)} analysis agents")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        return False

async def load_enhanced_macro_context() -> Optional[Dict]:
    """Load enhanced macro analysis JSON file as context."""
    try:
        # Look for the most recent enhanced macro analysis file
        macro_files = list(Path('.').glob('macro_signals_*.json'))
        
        if not macro_files:
            logger.info("No enhanced macro analysis files found")
            return None
        
        # Get the most recent file
        latest_file = max(macro_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            macro_context = json.load(f)
        
        logger.info(f"✅ Loaded enhanced macro context from: {latest_file}")
        return macro_context
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load enhanced macro context: {e}")
        return None

async def generate_enhanced_macro_signals() -> Optional[Dict]:
    """Generate fresh enhanced macro analysis signals."""
    if not ENHANCED_MACRO_AVAILABLE:
        return None
        
    try:
        logger.info("🔍 Generating enhanced macro analysis...")
        analyzer = EnhancedMacroAnalyzer()
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"macro_signals_{timestamp}.json"
        
        # Export macro signals
        macro_signals = analyzer.export_macro_signals_json(filepath)
        
        if macro_signals:
            logger.info(f"✅ Enhanced macro signals generated: {filepath}")
            return macro_signals
        else:
            logger.warning("⚠️ Enhanced macro analysis failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Enhanced macro analysis error: {e}")
        return None

async def get_macro_context() -> Optional[Dict]:
    """Get macro context for analysis - prioritize enhanced analysis."""
    # First try to load existing enhanced macro analysis
    enhanced_context = await load_enhanced_macro_context()
    
    if enhanced_context:
        return enhanced_context
    
    # If no enhanced analysis available, generate fresh one
    fresh_context = await generate_enhanced_macro_signals()
    
    if fresh_context:
        return fresh_context
    
    # Fallback to basic macro prediction
    try:
        if macro_agent:
            prediction = macro_agent.predict_next_quarter()
            return {
                "analysis_type": "basic_macro_prediction",
                "macro_prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.warning(f"Warning: Could not get macro context: {e}")
    
    return None

@mcp.tool()
async def analyze_market_with_prompt(user_prompt: str, analysis_depth: str = "comprehensive") -> Dict[str, Any]:
    """
    Perform complete market analysis with macro-integrated asset analysis.
    
    Args:
        user_prompt: User's investment query or context
        analysis_depth: Level of analysis ("basic", "comprehensive", "detailed")
        
    Returns:
        Complete analysis results with macro prediction and asset rankings
    """
    logger.info(f"[Agent 2] Received user prompt: {user_prompt}")
    
    try:
        # Step 1: Get macro context
        macro_context = await get_macro_context()
        
        # Step 2: Collect market data
        market_data = await collect_market_data()
        
        # Step 3: Perform parallel asset analysis
        analysis_tasks = []
        for agent_type, agent in agents.items():
            task = perform_asset_analysis(agent_type, agent, user_prompt, macro_context)
            analysis_tasks.append(task)
        
        # Execute all analyses in parallel
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Step 4: Save results and build response
        saved_files = await save_all_analyses(analysis_results, macro_context)
        
        return {
            "status": "success",
            "user_prompt": user_prompt,
            "analysis_timestamp": datetime.now().isoformat(),
            "macro_context": macro_context,
            "analysis_results": {
                agent_type: result for agent_type, result in 
                zip(agents.keys(), analysis_results) 
                if not isinstance(result, Exception)
            },
            "output_files": saved_files,
            "total_assets_analyzed": sum(
                len(result.get('analysis', {}).get('horizons', {}).get('next_quarter', [])) 
                for result in analysis_results 
                if isinstance(result, dict) and 'analysis' in result
            )
        }
        
    except Exception as e:
        logger.error(f"❌ Market analysis failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
async def get_macro_prediction() -> Dict[str, Any]:
    """
    Get macro economic prediction for the next quarter.
    
    Returns:
        Macro prediction with confidence scores and economic outlook
    """
    logger.info("[Agent 2] Getting macro prediction...")
    
    try:
        if not macro_agent:
            return {"error": "Macro agent not initialized"}
        
        prediction = macro_agent.predict_next_quarter()
        
        return {
            "status": "success",
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Macro prediction failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
async def get_analysis(analysis_type: str, symbols: Optional[List[str]] = None, 
                      use_macro_context: bool = True) -> Dict[str, Any]:
    """
    Get specific asset analysis (stock, bond, or gold).
    
    Args:
        analysis_type: Type of analysis ("stock", "bond", "gold")
        symbols: Optional list of symbols to analyze
        use_macro_context: Whether to include macro context
        
    Returns:
        Asset analysis results with rankings and recommendations
    """
    logger.info(f"[Agent 2] Getting {analysis_type} analysis...")
    
    try:
        if analysis_type not in agents:
            return {
                "status": "error",
                "error": f"Analysis type '{analysis_type}' not available"
            }
        
        agent = agents[analysis_type]
        macro_context = await get_macro_context() if use_macro_context else None
        
        # Get default symbols if not provided
        if not symbols:
            symbols = ANALYSIS_AGENTS[analysis_type]['symbols']
        
        # Perform analysis
        if analysis_type == 'stock':
            analysis = await agent.analyze_stocks(symbols, macro_context=macro_context)
        elif analysis_type == 'bond':
            analysis = await agent.analyze_bonds(symbols, macro_context=macro_context)
        elif analysis_type == 'gold':
            analysis = await agent.analyze_gold(symbols, macro_context=macro_context)
        
        return {
            "status": "success",
            "analysis_type": analysis_type,
            "analysis": analysis,
            "macro_context_used": macro_context is not None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ {analysis_type} analysis failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """
    Perform comprehensive health check of all system components.
    
    Returns:
        Health status of macro agent, analysis agents, and dependencies
    """
    logger.info("[Agent 2] Performing health check...")
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "macro_agent": macro_agent is not None,
        "api_client": api_client is not None,
        "model_manager": model_manager is not None,
        "openai_client": openai_client is not None,
        "analysis_agents": {
            agent_type: agent is not None 
            for agent_type, agent in agents.items()
        },
        "enhanced_macro_available": ENHANCED_MACRO_AVAILABLE,
        "total_agents": len(agents)
    }
    
    # Test macro agent if available
    if macro_agent:
        try:
            health = macro_agent.health_check()
            health_status["macro_agent_status"] = health.get("status") == "healthy"
        except:
            health_status["macro_agent_status"] = False
    
    # Test API client if available
    if api_client:
        try:
            health = api_client.health_check()
            health_status["api_client_status"] = any(health.values())
        except:
            health_status["api_client_status"] = False
    
    # Test model manager if available
    if model_manager:
        health_status["model_loaded"] = model_manager.is_model_loaded()
    
    return health_status

async def collect_market_data() -> Dict[str, Any]:
    """Collect current market data for analysis."""
    try:
        market_data = {
            "macro_indicators": {},
            "market_indices": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Collect basic market indicators
        if api_client:
            try:
                macro_data = api_client.get_latest_macro_data(limit=10)
                if not macro_data.empty:
                    market_data["macro_indicators"] = macro_data.tail(1).to_dict('records')[0]
            except Exception as e:
                logger.warning(f"Could not collect macro data: {e}")
        
        return market_data
        
    except Exception as e:
        logger.error(f"Error collecting market data: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

async def perform_asset_analysis(agent_type: str, agent: Any, user_prompt: str, 
                                macro_context: Optional[Dict]) -> Dict[str, Any]:
    """Perform analysis for a specific asset type."""
    try:
        logger.info(f"[Agent 2] Performing {agent_type} analysis...")
        
        symbols = ANALYSIS_AGENTS[agent_type]['symbols']
        horizon = ANALYSIS_CONFIG['horizon']
        
        if agent_type == 'stock':
            analysis = await agent.analyze_stocks(symbols, macro_context=macro_context)
            # Get top picks with consistent configuration
            top_picks = agent.get_top_stock_picks(
                analysis, 
                horizon=horizon, 
                top_n=ANALYSIS_CONFIG['top_picks']
            )
        elif agent_type == 'bond':
            analysis = await agent.analyze_bonds(symbols, macro_context=macro_context)
            # Get top picks with consistent configuration
            top_picks = agent.get_top_bond_picks(
                analysis, 
                horizon=horizon, 
                top_n=ANALYSIS_CONFIG['top_picks']
            )
        elif agent_type == 'gold':
            analysis = await agent.analyze_gold(symbols, macro_context=macro_context)
            # Get top picks with consistent configuration
            top_picks = agent.get_top_gold_picks(
                analysis, 
                horizon=horizon, 
                top_n=ANALYSIS_CONFIG['top_picks']
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Add top picks to analysis results
        if 'horizons' not in analysis:
            analysis['horizons'] = {}
        analysis['horizons'][f'{horizon}_top_picks'] = top_picks
        
        return {
            "agent_type": agent_type,
            "analysis": analysis,
            "top_picks_count": len(top_picks),
            "horizon": horizon,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ {agent_type} analysis failed: {e}")
        return {
            "agent_type": agent_type,
            "error": str(e),
            "status": "error"
        }

async def save_all_analyses(analysis_results: List[Any], macro_context: Optional[Dict]) -> Dict[str, str]:
    """Save all analysis results to files."""
    saved_files = {}
    
    try:
        # Save macro context if available
        if macro_context:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            macro_file = output_dir / f"macro_analysis_{timestamp}.json"
            with open(macro_file, 'w') as f:
                json.dump(macro_context, f, indent=2)
            saved_files["macro_analysis_json"] = str(macro_file)
        
        # Save analysis results
        for result in analysis_results:
            if isinstance(result, dict) and result.get("status") == "success":
                agent_type = result["agent_type"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = output_dir / f"{agent_type}_analysis_{timestamp}.json"
                
                with open(file_path, 'w') as f:
                    json.dump(result["analysis"], f, indent=2)
                
                saved_files[f"{agent_type}_analysis_json"] = str(file_path)
                logger.info(f"[Agent 2] {agent_type} analysis saved to: {file_path}")
        
        return saved_files
        
    except Exception as e:
        logger.error(f"❌ Error saving analyses: {e}")
        return {}

# Initialize components when the server starts
async def startup():
    """Initialize all components on server startup."""
    success = await initialize_components()
    if not success:
        logger.error("❌ Failed to initialize components")
        sys.exit(1)
    logger.info("✅ Agent 2 MCP Server initialized successfully")

# Allow the script to be run directly
if __name__ == "__main__":
    # Run initialization
    asyncio.run(startup())
    
    # Start the MCP server
    logger.info("🚀 Starting Agent 2 Portfolio Analysis MCP Server...")
    mcp.run()