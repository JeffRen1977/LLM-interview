#!/bin/bash

# Setup script for Transformer implementation environment
# This script creates a virtual environment and installs PyTorch

echo "🚀 Setting up Transformer implementation environment..."

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.7 or higher."
    exit 1
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for compatibility)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install NumPy (usually comes with PyTorch, but let's be explicit)
echo "📊 Installing NumPy..."
pip install numpy

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source transformer_env/bin/activate"
echo ""
echo "To run the transformer implementation:"
echo "  python transformer_implementation.py"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"



