#!/bin/bash

echo "🎵 BABEL-BEATS Local Testing Suite"
echo "=================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if basic requirements are installed
echo "📦 Checking dependencies..."
if ! python -c "import numpy" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing basic requirements..."
    pip install numpy scipy librosa soundfile
fi

# Run tests
echo ""
echo "🧪 Running local tests..."
echo ""

# Test 1: Basic echo algorithm
echo "1️⃣ Testing echo algorithm..."
python local_echo_server.py test

echo ""
echo "2️⃣ Running component tests..."
python local_test.py

echo ""
echo "✅ Local testing complete!"
echo ""
echo "📖 Next steps:"
echo "   - To run the full server: python main.py"
echo "   - To test echo server: python local_echo_server.py"
echo "   - To install all dependencies: pip install -r requirements.txt"
echo ""