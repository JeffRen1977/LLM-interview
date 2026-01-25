#!/bin/bash
# PyTorch 安装修复脚本
# 用于修复 PyTorch 动态库缺失问题

echo "=========================================="
echo "PyTorch 安装修复脚本"
echo "=========================================="

# 激活虚拟环境
source env/bin/activate

echo ""
echo "步骤 1: 检查当前 PyTorch 安装"
echo "----------------------------------------"
pip list | grep -i torch

echo ""
echo "步骤 2: 卸载现有的 PyTorch 相关包"
echo "----------------------------------------"
pip uninstall -y torch torchvision torchaudio

echo ""
echo "步骤 3: 清理 pip 缓存"
echo "----------------------------------------"
pip cache purge

echo ""
echo "步骤 4: 重新安装 PyTorch (CPU 版本)"
echo "----------------------------------------"
echo "正在安装 PyTorch 2.5.1 (稳定版本)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "步骤 5: 验证安装"
echo "----------------------------------------"
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'安装路径: {torch.__file__}'); print('✅ PyTorch 安装成功!')"

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "如果还有问题，可以尝试："
echo "1. 使用 GPU 版本（如果有 GPU）:"
echo "   pip install torch torchvision torchaudio"
echo ""
echo "2. 使用特定版本:"
echo "   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2"
echo ""
