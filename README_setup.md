# LLM Interview Preparation Setup

This directory contains complete implementations for LLM interview preparation, including:
- Transformer attention mechanism (PyTorch and NumPy)
- Memory-efficient training algorithms
- Model evaluation framework

## Quick Setup

### Option 1: Automated Setup (Recommended)

**For macOS/Linux:**
```bash
./setup_environment.sh
```

**For Windows:**
```cmd
setup_environment.bat
```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv transformer_env
   ```

2. **Activate virtual environment:**
   
   **macOS/Linux:**
   ```bash
   source transformer_env/bin/activate
   ```
   
   **Windows:**
   ```cmd
   transformer_env\Scripts\activate.bat
   ```

3. **Install dependencies (if needed):**
   ```bash
   pip install -r requirements.txt
   ```
   Note: The automated setup script installs all dependencies automatically.

## Running the Code

Once the environment is set up and activated, you can run any of the implementations:

```bash
# Transformer attention mechanism
python transformer_implementation.py

# Memory-efficient training algorithms
python openAI_memory_efficient_training.py

# Model evaluation framework
python openAI_evaluation_framework.py
```

Each implementation includes:
- ✅ Complete working code with demos
- ✅ Educational examples and analysis
- ✅ Performance benchmarking
- ✅ Comprehensive documentation

## Files Included

### Core Implementations
- `transformer_implementation.py` - Transformer attention mechanism
- `openAI_memory_efficient_training.py` - Memory-efficient training algorithms
- `openAI_evaluation_framework.py` - Model evaluation framework

### Documentation
- `transformer.md` - Transformer theory and implementation
- `openAI_memory_efficient_training.md` - Memory optimization techniques
- `openAI_evaluation_framework.md` - Evaluation framework theory

### Setup Scripts
- `setup_environment.sh` - Complete automated setup for macOS/Linux
- `setup_environment.bat` - Complete automated setup for Windows
- `requirements.txt` - All Python dependencies
- `README_setup.md` - This setup guide

## What You'll Learn

The implementations demonstrate:

### Transformer Attention Mechanism
- **Scaled Dot-Product Attention** - The core attention mechanism
- **Multi-Head Attention** - Parallel attention heads
- **Both PyTorch and NumPy** - Framework vs. pure math implementations
- **Complexity Analysis** - Time and space complexity understanding

### Memory-Efficient Training
- **Mixed Precision Training** - FP16/FP32 optimization
- **Gradient Checkpointing** - Memory vs. computation trade-offs
- **Model Sharding** - ZeRO and FSDP techniques
- **Performance Benchmarking** - Real-world memory usage analysis

### Model Evaluation Framework
- **Automated Metrics** - BLEU, ROUGE, BERTScore, Perplexity
- **Human Evaluation** - A/B testing, Likert ratings, red teaming
- **Online Evaluation** - Implicit signals, user feedback analysis
- **Comprehensive Reporting** - Multi-dimensional assessment

## Requirements

- Python 3.7 or higher
- 2GB+ RAM (for the demos)
- No GPU required (CPU version of PyTorch)

## Troubleshooting

**If PyTorch installation fails:**
- Try the CPU-only version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Check your Python version: `python --version`

**If virtual environment creation fails:**
- Make sure you have `venv` module: `python -m venv --help`
- Try using `virtualenv` instead: `pip install virtualenv && virtualenv transformer_env`

## Deactivating the Environment

When you're done:
```bash
deactivate
```



