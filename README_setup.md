# Transformer Implementation Setup

This directory contains a complete implementation of the Transformer's attention mechanism with both PyTorch and NumPy versions.

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

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

Once the environment is set up and activated:

```bash
python transformer_implementation.py
```

This will run:
- ✅ PyTorch implementation demo
- ✅ NumPy implementation demo  
- ✅ Attention mechanism analysis
- ✅ Complexity analysis

## Files Included

- `transformer_implementation.py` - Complete implementation with demos
- `transformer.md` - Detailed explanation and theory
- `setup_environment.sh` - Automated setup for macOS/Linux
- `setup_environment.bat` - Automated setup for Windows
- `requirements.txt` - Python dependencies
- `README_setup.md` - This setup guide

## What You'll Learn

The implementation demonstrates:
- **Scaled Dot-Product Attention** - The core attention mechanism
- **Multi-Head Attention** - Parallel attention heads
- **Both PyTorch and NumPy** - Framework vs. pure math implementations
- **Complexity Analysis** - Time and space complexity understanding
- **Shape Verification** - Understanding tensor dimensions

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



