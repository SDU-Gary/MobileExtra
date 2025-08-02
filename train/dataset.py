"""
@file dataset.py
@brief NoiseBase预处理数据集

核心功能：
- 加载预处理后的NoiseBase 6通道训练数据
- 支持数据增强和批处理
- 提供PyTorch Dataset接口

数据格式：
- 输入: RGB(3) + Mask(1) + ResidualMV(2) = 6通道
- 输出: RGB(3) Ground Truth
- 支持多种patch大小

@author AI算法团队
@date 2025-08-02
@version 2.0 (简化版)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


class NoiseBaseDataset(Dataset):
    """
    NoiseBase预处理数据集
    
    功能：
    - 加载预处理后的6通道训练数据
    - 支持数据增强和patch提取
    - 提供训练/验证/测试分割
    """
    
    def __init__(self, 
                 data_root: str,
                 scene_name: str,
                 split: str = 'train',
                 patch_size: int = 64,
                 augmentation: bool = True):
        """
        初始化数据集
        
        Args:
            data_root: 预处理后的数据根目录
            scene_name: 场景名称 (如 'bistro1')
            split: 数据分割 ('train', 'val', 'test')
            patch_size: patch大小
            augmentation: 是否使用数据增强
        """
        super(NoiseBaseDataset, self).__init__()
        
        self.data_root = Path(data_root)
        self.scene_name = scene_name
        self.split = split
        self.patch_size = patch_size
        self.augmentation = augmentation and (split == 'train')
        
        # 加载数据文件列表
        self.data_files = self._load_data_files()
        
        # 数据增强变换
        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
            ])
        else:
            self.transform = None
        
        print(f"=== NoiseBaseDataset ({split}) ===")
        print(f"Scene: {scene_name}")
        print(f"Data samples: {len(self.data_files)}")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Augmentation: {self.augmentation}")
    
    def _load_data_files(self) -> List[Path]:
        """加载数据文件列表"""
        training_data_dir = self.data_root / self.scene_name / 'training_data'
        
        if not training_data_dir.exists():
            raise FileNotFoundError(f"训练数据目录不存在: {training_data_dir}")
        
        # 获取所有.npy文件
        all_files = list(training_data_dir.glob("*.npy"))
        all_files.sort()  # 确保顺序一致
        
        if len(all_files) == 0:
            raise ValueError(f"没有找到训练数据文件: {training_data_dir}")
        
        # 简单的数据分割
        total_files = len(all_files)
        
        if self.split == 'train':
            # 前80%作为训练
            end_idx = int(total_files * 0.8)
            selected_files = all_files[:end_idx]
        elif self.split == 'val':
            # 80%-95%作为验证
            start_idx = int(total_files * 0.8)
            end_idx = int(total_files * 0.95)
            selected_files = all_files[start_idx:end_idx]
        else:  # test
            # 最后5%作为测试
            start_idx = int(total_files * 0.95)
            selected_files = all_files[start_idx:]
        
        return selected_files
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据项
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: 包含input和target的字典
        """
        # 加载数据
        data_file = self.data_files[idx]
        sample_data = np.load(data_file)  # [6, H, W]
        
        # 分离输入和目标
        input_data = sample_data  # 6通道输入: RGB + Mask + ResidualMV
        target_data = sample_data[:3]  # 3通道目标: RGB (Ground Truth)
        
        # 转换为tensor
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target_data).float()
        
        # 随机裁剪到指定patch大小
        if input_tensor.shape[1] > self.patch_size or input_tensor.shape[2] > self.patch_size:
            input_tensor, target_tensor = self._random_crop(input_tensor, target_tensor)
        
        # 数据增强
        if self.transform is not None:
            # 将输入和目标拼接进行同步变换
            combined = torch.cat([input_tensor, target_tensor], dim=0)  # [9, H, W]
            combined = self.transform(combined)
            
            # 分离回输入和目标
            input_tensor = combined[:6]
            target_tensor = combined[6:9]
        
        # 数据归一化
        input_tensor = self._normalize_input(input_tensor)
        target_tensor = self._normalize_target(target_tensor)
        
        return {
            'input': input_tensor,      # [6, H, W]
            'target': target_tensor,    # [3, H, W]
            'filename': data_file.stem,
            'scene': self.scene_name
        }
    
    def _random_crop(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        随机裁剪
        
        Args:
            input_tensor: 输入张量 [6, H, W]
            target_tensor: 目标张量 [3, H, W]
            
        Returns:
            cropped_input, cropped_target: 裁剪后的张量
        """
        _, H, W = input_tensor.shape
        
        if H <= self.patch_size and W <= self.patch_size:
            return input_tensor, target_tensor
        
        # 随机选择裁剪位置
        top = random.randint(0, max(0, H - self.patch_size))
        left = random.randint(0, max(0, W - self.patch_size))
        
        # 裁剪
        cropped_input = input_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
        cropped_target = target_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
        
        return cropped_input, cropped_target
    
    def _normalize_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        归一化输入数据
        
        Args:
            input_tensor: 输入张量 [6, H, W]
            
        Returns:
            normalized_input: 归一化后的输入
        """
        normalized = input_tensor.clone()
        
        # RGB通道 (0-2): 已经在[0,1]范围内，保持不变
        # Mask通道 (3): 已经在[0,1]范围内，保持不变
        # ResidualMV通道 (4-5): 可能需要缩放
        
        mv_channels = normalized[4:6]
        mv_magnitude = torch.sqrt(mv_channels[0]**2 + mv_channels[1]**2)
        max_magnitude = torch.max(mv_magnitude)
        
        if max_magnitude > 0:
            # 将运动矢量缩放到合理范围
            scale_factor = min(1.0, 10.0 / max_magnitude)
            normalized[4:6] *= scale_factor
        
        return normalized
    
    def _normalize_target(self, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        归一化目标数据
        
        Args:
            target_tensor: 目标张量 [3, H, W]
            
        Returns:
            normalized_target: 归一化后的目标
        """
        # RGB目标已经在[0,1]范围内，直接返回
        return target_tensor


def create_noisebase_dataloader(data_root: str, 
                               scene_name: str,
                               split: str = 'train',
                               batch_size: int = 8,
                               patch_size: int = 64,
                               num_workers: int = 4,
                               shuffle: bool = None) -> DataLoader:
    """
    创建NoiseBase数据加载器
    
    Args:
        data_root: 数据根目录
        scene_name: 场景名称
        split: 数据分割
        batch_size: 批次大小
        patch_size: patch大小
        num_workers: 工作线程数
        shuffle: 是否打乱数据
        
    Returns:
        dataloader: 数据加载器
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    # 创建数据集
    dataset = NoiseBaseDataset(
        data_root=data_root,
        scene_name=scene_name,
        split=split,
        patch_size=patch_size,
        augmentation=(split == 'train')
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试NoiseBase数据集')
    parser.add_argument('--data-root', type=str, required=True, help='预处理后的数据根目录')
    parser.add_argument('--scene', type=str, default='bistro1', help='场景名称')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧪 测试NoiseBase数据集加载")
    print("="*60)
    
    try:
        # 创建数据加载器
        train_loader = create_noisebase_dataloader(
            data_root=args.data_root,
            scene_name=args.scene,
            split='train',
            batch_size=2,
            patch_size=64
        )
        
        print(f"✅ 训练数据加载器创建成功")
        print(f"   数据集大小: {len(train_loader.dataset)}")
        print(f"   批次数: {len(train_loader)}")
        
        # 测试数据加载
        for i, batch in enumerate(train_loader):
            print(f"\n📊 批次 {i+1}:")
            print(f"   输入形状: {batch['input'].shape}")
            print(f"   目标形状: {batch['target'].shape}")
            print(f"   输入数值范围: [{batch['input'].min():.3f}, {batch['input'].max():.3f}]")
            print(f"   目标数值范围: [{batch['target'].min():.3f}, {batch['target'].max():.3f}]")
            print(f"   文件名: {batch['filename']}")
            
            if i >= 2:  # 只测试前3个批次
                break
        
        print("\n✅ 数据集测试通过！")
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
    
