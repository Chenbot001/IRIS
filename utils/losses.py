import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# 本模块实现了多种用于图像分割任务的损失函数，包括Dice损失、对比损失、拓扑损失、混合损失和SSIM损失

class DiceLoss(nn.Module):
    """Dice损失函数
    用于评估预测分割图和目标分割图之间的相似度
    Dice系数 = 2|X∩Y|/(|X|+|Y|)，其中X和Y分别为预测和目标图像
    损失值 = 1 - Dice系数
    """
    def __init__(self, smooth: float = 1e-5):
        """初始化Dice损失函数
        Args:
            smooth: 平滑项，防止分母为0，默认为1e-5
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class ContrastiveLoss(nn.Module):
    """对比损失函数
    通过特征空间中的样本对比学习来提高模型的表示能力
    使得相似样本的特征更接近，不相似样本的特征更远离
    """
    def __init__(self, temperature: float = 0.1):
        """初始化对比损失函数
        Args:
            temperature: 温度参数，控制特征分布的scale，默认为0.1
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        contrast_features = F.normalize(features)
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_features, contrast_features.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()

        return loss

class TopologicalLoss(nn.Module):
    """拓扑损失函数
    通过计算预测图和目标图的梯度差异来保持分割结果的拓扑结构
    特别适用于保持血管等细长结构的连续性
    """
    def __init__(self):
        super().__init__()

    def compute_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算图像在x和y方向的梯度
        Args:
            x: 输入张量，形状为[B,C,H,W]
        Returns:
            dx, dy: x方向和y方向的梯度
        """
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx, pred_dy = self.compute_gradient(pred)
        target_dx, target_dy = self.compute_gradient(target)

        grad_diff_x = torch.abs(pred_dx - target_dx)
        grad_diff_y = torch.abs(pred_dy - target_dy)

        return torch.mean(grad_diff_x) + torch.mean(grad_diff_y)

class HybridLoss(nn.Module):
    """混合损失函数
    结合Dice损失、对比损失和拓扑损失的优势
    在保证分割准确性的同时，维持图像的拓扑结构并提高特征表示能力
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.contrastive = ContrastiveLoss()
        self.topological = TopologicalLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        topo_loss = self.topological(pred, target)

        if features is not None:
            contrast_loss = self.contrastive(features, target)
            return 0.4 * dice_loss + 0.3 * contrast_loss + 0.3 * topo_loss
        else:
            return 0.7 * dice_loss + 0.3 * topo_loss

class SSIMLoss(nn.Module):
    """结构相似性(SSIM)损失函数
    通过比较图像的亮度、对比度和结构信息来评估图像相似度
    特别适用于保持图像的结构信息和视觉质量
    """
    def __init__(self, window_size: int = 11):
        """初始化SSIM损失函数
        Args:
            window_size: 高斯窗口大小，用于局部统计计算，默认为11
        """
        super().__init__()
        self.window_size = window_size
        self.register_buffer('window',
            self._create_window(window_size))

    def _create_window(self, window_size: int) -> torch.Tensor:
        """创建二维高斯窗口
        Args:
            window_size: 窗口大小
        Returns:
            window: 2D高斯窗口，形状为[1,1,window_size,window_size]
        """
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/2.0)
                             for x in range(window_size)])
        gauss = gauss/gauss.sum()
        window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()