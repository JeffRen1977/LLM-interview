#!/bin/bash

# Setup script for LLM Interview Preparation environment
# This script creates a virtual environment and installs all dependencies

echo "🚀 Setting up LLM Interview Preparation environment..."

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

# Install additional dependencies for all implementations
echo "📈 Installing additional dependencies..."
pip install pandas matplotlib seaborn

# Install evaluation libraries (for evaluation framework)
echo "🔍 Installing evaluation libraries..."
pip install evaluate bert-score rouge-score sacrebleu

# Install text processing libraries
echo "📚 Installing text processing libraries..."
pip install nltk spacy

# Install transformers and ML libraries
echo "🤗 Installing transformers and ML libraries..."
pip install transformers

# Install visualization and development libraries
echo "📊 Installing visualization and development libraries..."
pip install plotly jupyter pytest

# Verify installation
echo "✅ Verifying installation..."
echo "Core libraries:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'  Pandas: {pandas.__version__}')"

echo "Evaluation libraries:"
python -c "import evaluate; print('  evaluate: OK')" 2>/dev/null || echo "  evaluate: Not available"
python -c "import bert_score; print('  bert_score: OK')" 2>/dev/null || echo "  bert_score: Not available"
python -c "import rouge_score; print('  rouge_score: OK')" 2>/dev/null || echo "  rouge_score: Not available"

echo "ML and text processing:"
python -c "import transformers; print('  transformers: OK')" 2>/dev/null || echo "  transformers: Not available"
python -c "import nltk; print('  nltk: OK')" 2>/dev/null || echo "  nltk: Not available"
python -c "import spacy; print('  spacy: OK')" 2>/dev/null || echo "  spacy: Not available"

echo "Visualization:"
python -c "import matplotlib; print('  matplotlib: OK')" 2>/dev/null || echo "  matplotlib: Not available"
python -c "import seaborn; print('  seaborn: OK')" 2>/dev/null || echo "  seaborn: Not available"
python -c "import plotly; print('  plotly: OK')" 2>/dev/null || echo "  plotly: Not available"

echo ""
echo "🎉 Complete LLM Interview Preparation environment setup complete!"
echo ""
echo "All dependencies installed for:"
echo "  ✅ Transformer attention mechanism"
echo "  ✅ Memory-efficient training algorithms"
echo "  ✅ Model evaluation framework"
echo ""
echo "To activate the environment in the future, run:"
echo "  source env/bin/activate"
echo ""
echo "Available implementations:"
echo "  python transformer_implementation.py          # Transformer attention mechanism"
echo "  python openAI_memory_efficient_training.py    # Memory-efficient training"
echo "  python openAI_evaluation_framework.py         # Model evaluation framework"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"



