import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Tuple, Optional

# 本模块包含多种图像增强方法，用于数据增强和图像处理
# 主要包括弹性形变、材质感知增强、多光谱增强、各向异性扩散和合成阴影等技术

class ElasticDeformation:
    """弹性形变增强类
    
    通过生成随机位移场对图像进行非线性变形，模拟组织的弹性形变效果。
    适用于医学图像增强，可以模拟组织的自然变形。
    
    参数:
        sigma (float): 高斯滤波的标准差，控制变形的平滑程度
        alpha (float): 变形强度系数，控制最大变形幅度
    """
    def __init__(self, sigma: float = 15, alpha: float = 35):
        self.sigma = sigma  # 高斯滤波标准差
        self.alpha = alpha  # 变形强度系数

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """对输入图像应用弹性形变
        
        Args:
            image (torch.Tensor): 输入图像张量，形状为[C, H, W]
            
        Returns:
            torch.Tensor: 经过弹性形变后的图像张量
        """
        shape = image.shape
        # 生成随机位移场并进行高斯平滑
        dx = gaussian_filter((np.random.rand(*shape[1:]) * 2 - 1), self.sigma) * self.alpha  # x方向位移
        dy = gaussian_filter((np.random.rand(*shape[1:]) * 2 - 1), self.sigma) * self.alpha  # y方向位移

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        distorted = image.clone()
        for c in range(shape[0]):
            distorted[c] = torch.from_numpy(
                map_coordinates(image[c].numpy(), indices, order=1).reshape(shape[1:])
            )
        return distorted

class MaterialAwareAugmentation:
    """材质感知增强类
    
    通过生成随机纹理图案并与原图像结合，增强图像的材质特征。
    适用于需要突出材质细节的医学图像处理。
    """
    def __init__(self):
        self.texture_bank = self._create_texture_bank()  # 初始化纹理库

    def _create_texture_bank(self, size: int = 10):
        """创建随机纹理库
        
        Args:
            size (int): 纹理库中纹理的数量
            
        Returns:
            numpy.ndarray: 一维数组，包含随机生成的纹理
        """
        textures = []
        for _ in range(size):
            # 生成随机高斯噪声作为基础纹理
            texture = np.random.normal(0.5, 0.1, (64, 64))
            # 对纹理进行高斯平滑
            texture = gaussian_filter(texture, sigma=2)
            textures.append(texture.flatten())  # 将2D纹理展平为1D数组
        return np.concatenate(textures)  # 合并所有纹理为一个1D数组

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        texture = np.random.choice(self.texture_bank, size=64*64)
        texture = torch.from_numpy(texture.reshape(64, 64)).float()
        texture = F.interpolate(texture.unsqueeze(0).unsqueeze(0),
                              size=image.shape[1:],
                              mode='bilinear',
                              align_corners=False)[0, 0]
        return image * (0.8 + 0.4 * texture)

class MultiSpectralAugmentation:
    """多光谱增强类
    
    通过对不同通道应用不同的强度系数，模拟多光谱成像中的光谱变化。
    适用于多光谱医学图像的数据增强。
    
    参数:
        intensity_range (Tuple[float, float]): 强度调整的范围，默认为(0.8, 1.2)
    """
    def __init__(self, intensity_range: Tuple[float, float] = (0.8, 1.2)):
        self.intensity_range = intensity_range  # 强度调整范围

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        factors = torch.rand(image.shape[0]) * \
            (self.intensity_range[1] - self.intensity_range[0]) + \
            self.intensity_range[0]
        return image * factors.view(-1, 1, 1)

class AnisotropicDiffusion:
    """各向异性扩散类
    
    实现Perona-Malik各向异性扩散算法，用于图像去噪和边缘保持平滑。
    该方法可以在保持重要边缘特征的同时平滑图像。
    
    参数:
        kappa (float): 扩散系数，控制边缘检测的敏感度
        iterations (int): 迭代次数，控制平滑程度
    """
    def __init__(self, kappa: float = 50, iterations: int = 3):
        self.kappa = kappa  # 扩散系数
        self.iterations = iterations  # 迭代次数

    def _anisotropic_diffusion_step(self, image: np.ndarray) -> np.ndarray:
        """执行一次各向异性扩散步骤
        
        Args:
            image (np.ndarray): 输入图像数组
            
        Returns:
            np.ndarray: 经过一次扩散后的图像
        """
        # 计算四个方向的梯度
        gradN = np.roll(image, -1, axis=0) - image  # 北向梯度
        gradS = np.roll(image, 1, axis=0) - image   # 南向梯度
        gradE = np.roll(image, -1, axis=1) - image  # 东向梯度
        gradW = np.roll(image, 1, axis=1) - image   # 西向梯度

        cN = np.exp(-(gradN/self.kappa)**2)
        cS = np.exp(-(gradS/self.kappa)**2)
        cE = np.exp(-(gradE/self.kappa)**2)
        cW = np.exp(-(gradW/self.kappa)**2)

        return image + 0.25 * (cN*gradN + cS*gradS + cE*gradE + cW*gradW)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        result = image.clone()
        for c in range(image.shape[0]):
            img = image[c].numpy()
            for _ in range(self.iterations):
                img = self._anisotropic_diffusion_step(img)
            result[c] = torch.from_numpy(img)
        return result

class SyntheticShadowAugmentation:
    """合成阴影增强类
    
    通过生成随机的合成阴影效果增强图像。
    可以模拟手术场景中的光照变化和阴影效果。
    
    参数:
        shadow_intensity (float): 阴影的强度，值越小阴影越深
        max_shadows (int): 最大阴影数量
    """
    def __init__(self, shadow_intensity: float = 0.3, max_shadows: int = 3):
        self.shadow_intensity = shadow_intensity  # 阴影强度
        self.max_shadows = max_shadows  # 最大阴影数量

    def _create_shadow_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """创建阴影遮罩
        
        Args:
            shape (Tuple[int, int]): 输出遮罩的形状
            
        Returns:
            np.ndarray: 生成的阴影遮罩
        """
        mask = np.ones(shape)  # 初始化全白遮罩
        num_shadows = np.random.randint(1, self.max_shadows + 1)  # 随机确定阴影数量
        
        for _ in range(num_shadows):
            x1, x2 = np.random.randint(0, shape[1], 2)
            y1, y2 = np.random.randint(0, shape[0], 2)
            xpoints = np.array([x1, x2])
            ypoints = np.array([y1, y2])
            
            xnum = np.maximum(np.abs(x2-x1), 1)
            ynum = np.maximum(np.abs(y2-y1), 1)
            xlin = np.linspace(x1, x2, num=xnum)
            ylin = np.linspace(y1, y2, num=ynum)
            
            shadow_width = np.random.randint(20, 50)
            for x, y in zip(xlin, ylin):
                cv2.circle(mask, (int(x), int(y)), shadow_width, 
                          self.shadow_intensity, -1)
        
        return gaussian_filter(mask, sigma=10)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        shadow_mask = torch.from_numpy(
            self._create_shadow_mask(image.shape[1:])
        ).float()
        return image * shadow_mask