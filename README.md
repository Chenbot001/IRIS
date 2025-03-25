# 血管分割模型

基于自监督学习的单图像血管分割模型，使用多视角一致性和结构保持损失进行训练。

## 特点

- 支持单张未标注图像的自监督训练
- 采用强数据增强策略生成多个视角
- 结合多视角一致性损失和结构保持损失
- 使用混合精度训练提高效率

## 数据增强策略

- 弹性形变：模拟组织的自然变形
- 材质感知增强：增强图像的材质特征
- 多光谱增强：模拟不同光照条件
- 各向异性扩散：保持边缘的同时平滑图像
- 合成阴影：模拟手术场景中的光照变化

## 训练参数

- 图像大小：512x512
- 批次大小：4
- 训练轮数：200
- 学习率：5e-5
- 每批次视角数：4

## 使用方法

1. 准备单张未标注的血管图像，放置在 `data/unlabeled_image.jpg`

2. 运行训练脚本：
```bash
python train.py
```

3. 训练过程中的最佳模型将保存在 `checkpoints/best_model.pth`

## 训练日志

训练日志将保存在当前目录下，格式为 `training_YYYYMMDD_HHMMSS.log`