# 📁 Project Structure

```
my_first_aurite_project/
├── 📋 README.md                              # Complete setup and usage guide
├── ⚙️ .env                                   # Environment variables (API keys)
├── 📝 .env.example                           # Template for environment setup
├── 🔧 setup.sh                               # Quick setup script (macOS/Linux)
├── 🔧 setup.bat                              # Quick setup script (Windows)
├── 📊 PROJECT_STRUCTURE.md                   # This file
├── 🐍 .venv/                                 # Python virtual environment
│
├── 🚀 AURITE-AI-PROJECT-/                    # Main application directory
│   ├── 📦 requirements.txt                   # Python dependencies
│   │
│   ├── 🎯 MAIN WORKFLOWS
│   ├── master_investment_workflow.py         # 🎯 Complete end-to-end workflow
│   ├── run_30_stock_analysis.py             # 📈 Stock analysis only
│   ├── enhanced_macro_analysis.py           # 🌍 Macro economic analysis
│   ├── gold_analysis_agent.py               # 🏆 Precious metals analysis
│   │
│   ├── 🧠 AI AGENTS & CORE LOGIC
│   ├── ai_agent/                            # Core AI agent modules
│   │   ├── __init__.py
│   │   ├── agent.py                         # Main macro analysis agent
│   │   ├── api_client.py                    # 🔗 FRED API & data fetching
│   │   ├── config.py                        # Configuration management
│   │   ├── feature_engineer.py             # 🛠️ ML feature engineering
│   │   ├── model_manager.py                # 🤖 ML model management
│   │   └── openai_client.py                # 🤖 OpenAI integration
│   │
│   ├── 📊 ANALYSIS AGENTS
│   ├── stock_analysis_agent.py              # Individual stock analysis
│   ├── etf_analysis_agent.py                # ETF analysis
│   ├── train_unified_model.py               # ML model training
│   │
│   ├── 🧪 TESTING & EXAMPLES
│   ├── example_usage.py                     # Usage examples
│   ├── test_agent2_integration.py          # Integration tests
│   ├── test_enhanced_macro_integration.py  # Macro analysis tests
│   ├── test_simple.py                      # Basic functionality tests
│   │
│   ├── 🤖 MACHINE LEARNING MODELS
│   ├── models/                             # Pre-trained ML models
│   │   ├── enhanced_nasdaq_model.pkl      # NASDAQ prediction model
│   │   ├── feature_columns.json           # Model feature definitions
│   │   ├── feature_scaler.pkl             # Data normalization
│   │   └── model_metadata.json            # Model configuration
│   │
│   ├── 🖥️ MCP SERVER (Optional)
│   ├── MCP Server/                         # Model Context Protocol server
│   │   └── agent2_analysis_mcp_server.py  # MCP integration
│   │
│   └── 📁 OUTPUT DIRECTORY
│       └── analysis_outputs/               # 📈 All generated reports
│           ├── portfolio_report_*.md       # 📋 Investment reports
│           ├── stock_analysis_*.json      # 📊 Stock analysis data
│           ├── macro_analysis_*.json      # 🌍 Economic analysis
│           ├── user_profile_*.json        # 👤 User preferences
│           └── complete_investment_*.json # 🎯 Full recommendations
│
└── 📁 AURITE CONFIGURATION
    └── config/                             # System configuration
        ├── agents/agents.json              # Agent configurations
        ├── llms/llms.json                 # LLM model settings
        ├── workflows/                      # Workflow definitions
        ├── custom_workflows/              # Custom user workflows
        └── mcp_servers/                   # MCP server configs
```

## 🔍 Key File Descriptions

### 🎯 Main Entry Points
- **`master_investment_workflow.py`** - Run this for complete investment analysis
- **`run_30_stock_analysis.py`** - Stock analysis on 30 NASDAQ-100 stocks  
- **`enhanced_macro_analysis.py`** - Economic analysis using FRED data

### 🧠 Core AI Components
- **`ai_agent/api_client.py`** - Handles FRED API, Yahoo Finance data fetching
- **`ai_agent/feature_engineer.py`** - Creates 191 ML features from economic data
- **`ai_agent/model_manager.py`** - Manages ML models for predictions
- **`ai_agent/agent.py`** - Main orchestrator for macro analysis

### 📊 Analysis Outputs
All analysis results are saved to `analysis_outputs/`:
- **Portfolio Reports** (`.md`) - Professional investment recommendations
- **Analysis Data** (`.json`) - Raw analysis results for further processing
- **User Profiles** (`.json`) - Risk tolerance and preferences

### ⚙️ Configuration
- **`.env`** - API keys and environment variables
- **`config/`** - System-wide configuration files
- **`requirements.txt`** - Python package dependencies

## 🚀 Quick Start Paths

### For New Users
1. Run `./setup.sh` (macOS/Linux) or `setup.bat` (Windows)
2. Edit `.env` with your API keys
3. Run `python master_investment_workflow.py`

### For Developers
1. Explore `ai_agent/` for core logic
2. Check `test_*.py` files for examples
3. Modify `config/` for customization

### For Researchers
1. Look at `models/` for ML models
2. Examine `feature_engineer.py` for feature creation
3. Use `enhanced_macro_analysis.py` for economic research

## 📈 Data Flow

```
User Input → User Preferences Agent
     ↓
FRED API → Macro Analysis → ML Features → Predictions
     ↓
Yahoo Finance → Stock Analysis → Scoring → Rankings
     ↓
Portfolio Optimizer → Asset Allocation → Reports
```

## 🔧 Customization Points

- **User Preferences**: Modify risk tolerance, sector exclusions
- **ML Models**: Retrain with `train_unified_model.py`
- **Data Sources**: Add new APIs in `api_client.py`
- **Portfolio Logic**: Customize allocation in portfolio agent
- **Reporting**: Modify report templates

---

**Need help?** Check the main README.md for detailed setup instructions!
