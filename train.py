import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import logging
from datetime import datetime

from vessel_segmentation import VascularUNet
from utils.losses import HybridLoss, SSIMLoss, TopologicalLoss
from utils.augmentation import (
    ElasticDeformation,
    MaterialAwareAugmentation,
    MultiSpectralAugmentation,
    AnisotropicDiffusion,
    SyntheticShadowAugmentation
)

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

class SingleImageDataset(Dataset):
    """单图像自监督数据集类
    
    用于加载和预处理单张未标注图像，通过强数据增强生成多个视角。
    
    Args:
        image_path: 图像路径
        img_size: 图像大小
        num_views: 每次返回的视角数量
    """
    def __init__(self, image_path: str, img_size: int = 512, num_views: int = 4):
        self.image_path = Path(image_path)
        self.img_size = img_size
        self.num_views = num_views
        
        # 基础图像变换
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        # 加载原始图像
        self.original_image = Image.open(self.image_path).convert('RGB')
        self.original_image = self.transform(self.original_image)
        
        # 数据增强策略
        self.augmentations = [
            ElasticDeformation(sigma=20, alpha=40),  # 增强变形强度
            MaterialAwareAugmentation(),
            MultiSpectralAugmentation(intensity_range=(0.7, 1.3)),  # 扩大强度范围
            AnisotropicDiffusion(kappa=40, iterations=4),  # 增加迭代次数
            SyntheticShadowAugmentation(shadow_intensity=0.4, max_shadows=4)  # 增加阴影数量
        ]
    
    def __len__(self) -> int:
        return 1000  # 设置一个足够大的数字以支持长期训练
    
    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:
        # 生成两组不同强度的增强视角
        views = []
        for _ in range(self.num_views):
            view = self.original_image.clone()
            # 应用强数据增强
            for aug in self.augmentations:
                if np.random.random() > 0.3:  # 提高增强概率
                    view = aug(view)
            views.append(view)
        
        # 返回不同视角的图像对
        return torch.stack(views[:self.num_views//2]), torch.stack(views[self.num_views//2:])

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: str) -> None:
    """训练模型
    
    实现基于单图像的自监督训练循环，使用多视角一致性和结构保持损失。
    使用混合精度训练提高效率。
    
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备
        save_dir: 模型保存目录
    """
    scaler = GradScaler()  # 混合精度训练的梯度缩放器
    best_loss = float('inf')
    
    # 初始化损失函数
    ssim_loss = SSIMLoss().to(device)
    topo_loss = TopologicalLoss().to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')
        
        for views1, views2 in train_bar:
            views1, views2 = views1.to(device), views2.to(device)
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                # 获取两组视角的预测结果
                outputs1 = model(views1)
                outputs2 = model(views2)
                
                # 计算多视角一致性损失
                consistency_loss = 0.0
                for v1, v2 in zip(outputs1, outputs2):
                    consistency_loss += ssim_loss(v1, v2)
                
                # 计算结构保持损失
                structure_loss = 0.0
                for out in outputs1 + outputs2:
                    structure_loss += topo_loss(out, out)  # 使用自身作为目标
                
                # 组合损失
                loss = 0.7 * consistency_loss + 0.3 * structure_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cons_loss': f'{consistency_loss.item():.4f}',
                'struct_loss': f'{structure_loss.item():.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 记录训练信息
        logging.info(f'Epoch {epoch+1}/{num_epochs} - '
                    f'Train Loss: {avg_train_loss:.4f}')
        
        # 保存最佳模型
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss
            }, os.path.join(save_dir, 'best_model.pth'))
            logging.info(f'Saved best model with train loss: {avg_train_loss:.4f}')

def main():
    # 设置训练参数
    image_path = 'data/model.jpg'  # 单张未标注图像路径
    save_dir = 'checkpoints'  # 模型保存目录
    batch_size = 4  # 减小批次大小以适应多视角训练
    num_epochs = 200  # 增加训练轮数以确保充分学习
    learning_rate = 5e-5  # 降低学习率以提高稳定性
    img_size = 512
    num_views = 4  # 每次生成的视角数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据集和数据加载器
    train_dataset = SingleImageDataset(
        image_path=image_path,
        img_size=img_size,
        num_views=num_views
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型和优化器
    model = VascularUNet(in_channels=3, init_features=32).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 开始训练
    logging.info('Starting single-image self-supervised training...')
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_dir=save_dir
    )
    logging.info('Training completed!')

if __name__ == '__main__':
    main()