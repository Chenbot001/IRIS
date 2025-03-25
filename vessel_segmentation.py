# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from pytorch_grad_cam import GradCAM
import numpy as np
from typing import List, Tuple, Optional

# 深度可分离卷积模块
# 相比传统卷积，深度可分离卷积将卷积操作分为两步：
# 1. 深度卷积(depthwise)：对每个输入通道单独进行空间卷积
# 2. 逐点卷积(pointwise)：使用1x1卷积调整通道数
# 这种设计可以显著减少参数量和计算量，同时保持模型性能
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 自适应感受野模块
# 该模块通过多分支结构和不同大小的卷积核来捕获多尺度特征
# 包含4个并行分支，每个分支使用不同大小的卷积核(1x1, 3x3, 5x5, 7x7)
# 最后通过特征融合得到更丰富的特征表示
class AdaptiveReceptiveFieldBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels // 4, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.Conv2d(channels // 4, channels // 4, 5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.Conv2d(channels // 4, channels // 4, 7, padding=3)
        )
        self.fusion = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(out)

# 编码器模块
# 包含两个深度可分离卷积层，一个自适应感受野模块，以及归一化和激活函数
# 用于特征提取和下采样过程中的特征变换
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.adaptive_block = AdaptiveReceptiveFieldBlock(out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_block(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

# 解码器模块
# 包含上采样层和编码器模块
# 用于特征恢复和上采样过程中的特征变换
# 通过skip连接融合编码器对应层的特征，以保留更多细节信息
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = EncoderBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# 血管分割网络主体架构
# 基于U-Net结构，包含编码器路径、桥接层和解码器路径
# 特点：
# 1. 使用深度可分离卷积减少参数量
# 2. 引入自适应感受野模块增强特征提取能力
# 3. 采用三输出头设计，分别预测血管核心区域、边界区域和背景
class VascularUNet(nn.Module):
    def __init__(self, in_channels: int = 3, init_features: int = 32):
        super().__init__()
        
        # Encoder pathway
        self.encoder1 = EncoderBlock(in_channels, init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = EncoderBlock(init_features, init_features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = EncoderBlock(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = EncoderBlock(init_features * 4, init_features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge
        self.bridge = EncoderBlock(init_features * 8, init_features * 16)

        # Decoder pathway
        self.decoder4 = DecoderBlock(init_features * 16, init_features * 8)
        self.decoder3 = DecoderBlock(init_features * 8, init_features * 4)
        self.decoder2 = DecoderBlock(init_features * 4, init_features * 2)
        self.decoder1 = DecoderBlock(init_features * 2, init_features)

        # Triple output heads
        self.vessel_core = nn.Conv2d(init_features, 1, kernel_size=1)
        self.vessel_boundary = nn.Conv2d(init_features, 1, kernel_size=1)
        self.background = nn.Conv2d(init_features, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bridge
        bridge = self.bridge(self.pool4(enc4))

        # Decoder
        dec4 = self.decoder4(bridge, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        # Output heads
        core = torch.sigmoid(self.vessel_core(dec1))
        boundary = torch.sigmoid(self.vessel_boundary(dec1))
        bg = torch.sigmoid(self.background(dec1))

        return core, boundary, bg