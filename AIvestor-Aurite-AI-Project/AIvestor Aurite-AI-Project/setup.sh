#!/bin/bash

# Aurite AI Investment Advisor - Quick Setup Script
# This script helps you set up the complete investment analysis system

echo "🚀 Aurite AI Investment Advisor - Setup Script"
echo "=============================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
if [[ $? -eq 0 ]]; then
    echo "✅ Found: $python_version"
else
    echo "❌ Python not found. Please install Python 3.9 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "🏗️  Creating virtual environment..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv || python -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Please manually activate: source .venv/bin/activate"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
cd AURITE-AI-PROJECT-
pip install -r requirements.txt
if [[ $? -eq 0 ]]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Set up environment file
echo ""
echo "⚙️  Setting up environment configuration..."
cd ..
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "📝 Please edit .env file with your API keys:"
    echo "   - OPENAI_API_KEY (required)"
    echo "   - FRED_API_KEY (required, free from https://fred.stlouisfed.org/docs/api/api_key.html)"
else
    echo "✅ .env file already exists"
fi

# Test installation
echo ""
echo "🧪 Testing installation..."
cd AURITE-AI-PROJECT-
python -c "
import sys
print(f'✅ Python version: {sys.version}')

try:
    import pandas, numpy, sklearn, yfinance, requests
    print('✅ Core dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY', '')
    fred_key = os.getenv('FRED_API_KEY', '')
    
    if openai_key and openai_key != 'sk-your-openai-api-key-here':
        print('✅ OpenAI API key configured')
    else:
        print('⚠️  OpenAI API key not configured')
    
    if fred_key and fred_key != 'your-fred-api-key-here':
        print('✅ FRED API key configured')
    else:
        print('⚠️  FRED API key not configured')
        
except Exception as e:
    print(f'⚠️  Environment configuration issue: {e}')
"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys (especially FRED_API_KEY)"
echo "2. Run the complete workflow:"
echo "   cd AURITE-AI-PROJECT-"
echo "   python master_investment_workflow.py"
echo ""
echo "3. Or try individual components:"
echo "   python run_30_stock_analysis.py"
echo "   python enhanced_macro_analysis.py"
echo ""
echo "📚 See README.md for detailed usage instructions"
echo ""
echo "🆘 Need help? Check the troubleshooting section in README.md"
