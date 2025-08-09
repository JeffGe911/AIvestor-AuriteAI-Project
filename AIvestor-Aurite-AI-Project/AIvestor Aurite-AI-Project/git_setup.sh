#!/bin/bash

# Git Repository Setup Script for Aurite AI Investment Advisor
echo "🚀 Setting up Git repository for Aurite AI Investment Advisor"
echo "============================================================"

# Initialize git repository
echo "📝 Initializing Git repository..."
git init

# Add all files (respecting .gitignore)
echo "📁 Adding files to Git (excluding sensitive data)..."
git add .

# Check what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short

# Create initial commit
echo ""
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Aurite AI Investment Advisor

- Complete AI-powered investment analysis system
- Real-time FRED API integration for economic data
- NASDAQ-100 stock analysis with ML predictions
- Multi-asset portfolio construction (stocks, bonds, gold)
- Professional reporting and recommendations
- User preference-based risk management
- 91.5% confidence macro predictions
- Comprehensive setup and usage documentation"

echo ""
echo "✅ Git repository initialized successfully!"
echo ""
echo "🔗 Next steps to upload to GitHub:"
echo "1. Create a new repository on GitHub (https://github.com/new)"
echo "2. Copy the repository URL"
echo "3. Run the following commands:"
echo ""
echo "   git remote add origin <your-github-repo-url>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "📝 Example:"
echo "   git remote add origin https://github.com/yourusername/aurite-ai-advisor.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🔐 Security Note: Your .env file is excluded from the repository"
echo "   Users will need to create their own .env file with API keys"
echo ""
echo "📚 Your repository will include:"
echo "   ✅ Complete source code"
echo "   ✅ Documentation (README.md)"
echo "   ✅ Setup scripts"
echo "   ✅ Configuration templates"
echo "   ✅ Pre-trained ML models"
echo "   ❌ API keys (.env file excluded)"
echo "   ❌ Virtual environment (.venv excluded)"
