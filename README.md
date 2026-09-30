# 🚀 Aurite AI Investment Advisor-AIvestor
Created based on Aurite AI, by Yuqi Ge, Po-Yuan Chen, Chi-Fang Cheng, Shengyun Gao, Lixuan Wei

A five-person USC team project with Aurite AI exploring a multi-agent investment-research workflow: user preferences, market analysis, portfolio construction, and structured reporting.

**My contribution — Yuqi (Jeff) Ge:** User Preference and Portfolio Optimization agents, connecting user goals and constraints to the portfolio workflow. Start with the [user preference agent](Aurite-AI-Project-/enhanced_aivestor_agent1.py), [portfolio agent](Aurite-AI-Project-/portfolio_agent.py), and [workflow orchestrator](Aurite-AI-Project-/master_investment_workflow.py). Other analysis components are team contributions.

**Scope:** An educational research prototype. Forecasts, confidence values, and allocation scenarios are illustrative outputs; they do not establish validated predictive accuracy or realized investment returns.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## 🌟 Features

- **🤖 AI-Powered Analysis**: Advanced machine learning models for stock predictions and macro analysis
- **📊 Real-Time Data**: Integration with FRED API, Yahoo Finance, and other financial data sources
- **🎯 Personalized Portfolios**: Custom portfolio construction based on user preferences and risk tolerance
- **📈 Multi-Asset Coverage**: Stocks, bonds, precious metals, and alternatives analysis
- **🔄 Automated Workflows**: End-to-end investment analysis pipeline
- **📋 Professional Reporting**: Comprehensive markdown and JSON reports
- **⚙️ Sector Filtering**: Respect user preferences for sector inclusion/exclusion

## 🏗️ System Architecture

```
Aurite AI Investment Advisor
├── User Preference Analysis (Risk, Goals, Constraints)
├── Market & Economic Analysis
│   ├── Macro Analysis (FRED API + ML Models)
│   ├── Stock Analysis (NASDAQ-100 + Custom Scoring)
│   ├── Bond Analysis (Multiple Bond Types)
│   └── Gold/Alternatives Analysis
├── Portfolio Construction & Optimization
└── Professional Reporting (MD + JSON)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- Internet connection for API access

### 1. Clone the Repository

```bash
git clone https://github.com/JeffGe911/AIvestor-AuriteAI-Project.git
cd AIvestor-AuriteAI-Project
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
cd Aurite-AI-Project-
pip install -r requirements.txt
```

### 4. Configure Environment Variables

From the `Aurite-AI-Project-` working directory, create a local `.env` file:

```bash
# Copy the example and edit with your API keys
cp ../.env.example .env
```

Edit `.env` file:
```env
# Required: OpenAI API Key for LLM integration
OPENAI_API_KEY=sk-your-openai-api-key-here

# Required: FRED API Key for economic data (free from https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY=your-fred-api-key-here

# Optional: Additional API keys for enhanced data
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
QUANDL_API_KEY=your-quandl-key
```

### 5. Test the Installation

```bash
# Test FRED API integration
python -c "
from ai_agent.api_client import MacroAPIClient, APIConfig
from dotenv import load_dotenv
import os

load_dotenv()
config = APIConfig()
config.fred_api_key = os.getenv('FRED_API_KEY', '')
print(f'FRED API Key configured: {bool(config.fred_api_key)}')

client = MacroAPIClient(config)
health = client.health_check()
print(f'API Health: {health}')
"
```

## 📖 Usage Guide

### Option 1: Complete Investment Workflow (Recommended)
Run stock analysis first. The portfolio workflow reads the latest precomputed stock-analysis output in `analysis_outputs/`; check that the latest file is the input you intend to use.

Run the full end-to-end investment analysis:

```bash
python run_30_stock_analysis.py
python master_investment_workflow.py
```

This will:
1. Collect your investment preferences interactively
2. Analyze macro economic conditions using real FRED data
3. Perform stock analysis on NASDAQ-100 stocks
4. Analyze bonds and precious metals
5. Construct optimized portfolio
6. Generate professional reports

After execution, inspect the generated reports in `analysis_outputs/`.

### Option 2: Individual Analysis Components

#### Stock Analysis Only
```bash
python run_30_stock_analysis.py
```

#### Macro Analysis Only
```bash
python enhanced_macro_analysis.py
```

#### Gold Analysis Only
```bash
python gold_analysis_agent.py
```


## 📊 Output Files

All analysis results are saved to the `analysis_outputs/` directory:

### Generated Reports
- `portfolio_report_YYYYMMDD_HHMMSS.md` - Professional investment report
- `complete_investment_recommendation_YYYYMMDD_HHMMSS.json` - Complete analysis data

### Individual Analysis Files
- `user_profile_YYYYMMDD_HHMMSS.json` - User preferences and risk profile
- `macro_analysis_YYYYMMDD_HHMMSS.json` - Economic analysis with FRED data
- `stock_analysis_30stocks_YYYYMMDD_HHMMSS.json` - Stock analysis results
- `bond_analysis_YYYYMMDD_HHMMSS.json` - Bond market analysis
- `gold_analysis_YYYYMMDD_HHMMSS.json` - Precious metals analysis

## ⚙️ Configuration

### User Preferences
The system supports various user preferences:
- **Risk Tolerance**: Conservative, Moderate, Aggressive
- **Investment Goals**: Retirement, Wealth, Income, Capital Preservation
- **Time Horizon**: 1-30+ years
- **Sector Preferences**: Include/exclude specific sectors
- **ESG Preferences**: Environmental, Social, Governance considerations
- **Liquidity Needs**: Short-term access requirements

### API Configuration
Edit `ai_agent/config.py` or use environment variables:

```python
# Example configuration
config = APIConfig()
config.fred_api_key = "your-key"
config.fred_enabled = True
config.yahoo_finance_enabled = True
config.cache_duration = 3600  # 1 hour cache
config.max_retries = 3
```

## 🧠 Machine Learning Models

The system includes several pre-trained models:

### Macro Economic Model
- **Location**: `models/enhanced_nasdaq_model.pkl`
- **Features**: 191 engineered features from economic indicators
- **Target**: NASDAQ-100 quarterly performance prediction
- **Confidence**: 91.5% for bullish Q4 2025 prediction

### Stock Scoring Model
- **Method**: Multi-factor scoring combining technical and fundamental analysis
- **Factors**: P/E ratios, market cap, sector rotation, momentum
- **Output**: Buy/Hold/Sell signals with confidence scores

## 🔧 API Integration Details

### FRED (Federal Reserve Economic Data)
- **Purpose**: Real-time US economic indicators
- **Data**: Fed funds rate, unemployment, inflation, GDP, VIX, money supply
- **Update Frequency**: Daily/Monthly depending on indicator
- **Free Tier**: 1000 requests/day

### Yahoo Finance
- **Purpose**: Stock prices, company fundamentals
- **Data**: OHLCV data, financial statements, market cap
- **Update Frequency**: Real-time during market hours
- **Free Tier**: No API key required

### OpenAI API
- **Purpose**: Natural language analysis and report generation
- **Models**: GPT-4, GPT-3.5-turbo
- **Usage**: Investment reasoning, risk analysis, market commentary

## 🐛 Troubleshooting

### Common Issues

#### FRED API Not Working
```bash
# Check API key configuration
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('FRED key configured:', bool(os.getenv('FRED_API_KEY')))"

# Test API directly
curl "https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key=YOUR_KEY&limit=1&file_type=json"
```

#### Module Import Errors
```bash
# Ensure you're in the correct directory and virtual environment is activated
cd Aurite-AI-Project-
python -c "import sys; print(sys.path)"
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Error Logs
Check the console output for detailed error messages. The system uses `loguru` for comprehensive logging.

## 📈 Evaluation status

The repository includes [example portfolio reports](Aurite-AI-Project-/analysis_outputs/) and an [integration summary](Aurite-AI-Project-/INTEGRATION_SUCCESS_SUMMARY.md). They demonstrate workflow outputs and component integration, rather than validated investment performance.

- **Processing coverage is not predictive accuracy.** The stock agent computes `success_rate` as generated predictions divided by input symbols. Its error path can append a neutral fallback prediction, so that ratio may include fallback results. See [the implementation](Aurite-AI-Project-/stock_analysis_agent.py).
- **Confidence is not measured accuracy.** A model's confidence for a forecast is distinct from out-of-sample accuracy or a calibrated probability.
- **Returns and allocation scenarios are illustrative.** Generated scenario values are not realized investment returns.
- **Further evaluation:** document time-based holdouts, benchmark comparisons, fallback frequency, data freshness, and measured API availability before presenting accuracy or reliability claims.

## 🔐 Security & Privacy

- **API Keys**: Stored in `.env` file (never commit to version control)
- **Data**: No personal financial data stored permanently
- **Caching**: Economic data cached for 1 hour to reduce API calls
- **Output**: All reports saved locally only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# A repository-level tests/ directory is not included in this snapshot.
# Inspect sample outputs and integration notes; a reproducible test suite remains future work.

# Format code
black .
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Federal Reserve Economic Data (FRED)** for economic indicators
- **Yahoo Finance** for stock market data
- **OpenAI** for natural language processing
- **Scikit-learn** for machine learning models
- **Pandas/Numpy** for data processing

## 📞 Support

For questions, issues, or feature requests:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](https://github.com/JeffGe911/AIvestor-AuriteAI-Project/issues)
3. Create a new issue with detailed information

## 🗺️ Roadmap

### Upcoming Features
- [ ] Real-time portfolio monitoring
- [ ] Options and derivatives analysis
- [ ] International market expansion
- [ ] Advanced backtesting framework
- [ ] Web-based dashboard
- [ ] Mobile app integration

---

**⚠️ Disclaimer**: This system is for educational and research purposes only. All output should be considered illustrative and not investment advice. Always consult with qualified financial professionals before making investment decisions.

**Built with ❤️ by the Aurite AI Team**
