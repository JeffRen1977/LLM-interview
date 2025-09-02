import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint # 导入梯度检查点
from torch.cuda.amp import GradScaler, autocast # 导入混合精度工具