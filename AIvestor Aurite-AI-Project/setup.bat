@echo off
REM Aurite AI Investment Advisor - Quick Setup Script for Windows
REM This script helps you set up the complete investment analysis system

echo 🚀 Aurite AI Investment Advisor - Setup Script (Windows)
echo =======================================================

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.9 or higher.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo ✅ Found: %%i

REM Create virtual environment
echo.
echo 🏗️  Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo.
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✅ Virtual environment activated

REM Install dependencies
echo.
echo 📦 Installing dependencies...
cd AURITE-AI-PROJECT-
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully

REM Set up environment file
echo.
echo ⚙️  Setting up environment configuration...
cd ..
if not exist ".env" (
    copy .env.example .env
    echo ✅ Created .env file from template
    echo 📝 Please edit .env file with your API keys:
    echo    - OPENAI_API_KEY (required)
    echo    - FRED_API_KEY (required, free from https://fred.stlouisfed.org/docs/api/api_key.html)
) else (
    echo ✅ .env file already exists
)

echo.
echo 🎉 Setup Complete!
echo ==================
echo.
echo Next steps:
echo 1. Edit .env file with your API keys (especially FRED_API_KEY)
echo 2. Run the complete workflow:
echo    cd AURITE-AI-PROJECT-
echo    python master_investment_workflow.py
echo.
echo 3. Or try individual components:
echo    python run_30_stock_analysis.py
echo    python enhanced_macro_analysis.py
echo.
echo 📚 See README.md for detailed usage instructions
echo.
echo 🆘 Need help? Check the troubleshooting section in README.md
pause
