#!/bin/bash

echo "🤖 Setting up Anthropic API for AI-powered processing..."

# Check if ANTHROPIC_API_KEY is already set
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✅ ANTHROPIC_API_KEY is already set"
    echo "Current key: ${ANTHROPIC_API_KEY:0:10}..."
else
    echo "❌ ANTHROPIC_API_KEY is not set"
    echo ""
    echo "To enable AI-powered processing:"
    echo "1. Get your API key from: https://console.anthropic.com/"
    echo "2. Set the environment variable:"
    echo "   export ANTHROPIC_API_KEY='your_api_key_here'"
    echo ""
    echo "3. Or add it to your shell profile (~/.zshrc or ~/.bashrc):"
    echo "   echo 'export ANTHROPIC_API_KEY=\"your_api_key_here\"' >> ~/.zshrc"
    echo "   source ~/.zshrc"
    echo ""
    echo "4. Then restart the web app:"
    echo "   python3 ai_kubernetes_web_app.py"
fi

echo ""
echo "🔧 Current status:"
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✅ AI-powered processing will be enabled"
else
    echo "⚠️  Regex-only processing (fallback mode)"
fi
