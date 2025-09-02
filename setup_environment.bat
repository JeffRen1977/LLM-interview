@echo off
REM Setup script for Transformer implementation environment (Windows)
REM This script creates a virtual environment and installs PyTorch

echo 🚀 Setting up Transformer implementation environment...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

echo ✅ Found Python:
python --version

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv transformer_env

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call transformer_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch (CPU version for compatibility)
echo 🔥 Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install NumPy (usually comes with PyTorch, but let's be explicit)
echo 📊 Installing NumPy...
pip install numpy

REM Verify installation
echo ✅ Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo.
echo 🎉 Environment setup complete!
echo.
echo To activate the environment in the future, run:
echo   transformer_env\Scripts\activate.bat
echo.
echo To run the transformer implementation:
echo   python transformer_implementation.py
echo.
echo To deactivate the environment:
echo   deactivate
echo.
pause



