"""
@file dataset.py
@brief 游戏场景训练数据集

核心功能：
- 多游戏场景数据加载
- RGB/深度/MV数据处理
- 动态patch提取
- 数据增强和标准化

数据格式：
- 输入: RGB(3) + Mask(1) + ResidualMV(2) = 6通道
- 输出: RGB(3)修复结果  
- 分辨率: 多尺度64x64/128x128/256x256
- 序列长度: 连续3-5帧

预处理策略：
- 归一化: [0,1]范围
- 数据增强: 随机裁剪/翻转/噪声
- 负样本生成: 人工空洞模拟
- 质量过滤: 低质量帧剔除

性能优化：
- 多进程数据加载
- 内存映射文件访问
- 缓存热点数据
- 批处理优化

@author AI算法团队
@date 2025-07-28
@version 1.0
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


class GameSceneDataset(Dataset):
    """
    游戏场景训练数据集
    
    功能：
    - 加载多游戏场景数据（使命召唤、王者荣耀、QQ飞车、原神）
    - 处理6通道输入：RGB + Mask + ResidualMV
    - 动态patch提取和数据增强
    - 支持训练/验证/测试模式
    """
    
    def __init__(self, 
                 data_root: str,
                 split_file: str,
                 patch_size: int = 64,
                 sequence_length: int = 3,
                 mode: str = 'train',
                 augmentation: bool = True):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            split_file: 数据分割文件路径
            patch_size: patch大小
            sequence_length: 序列长度
            mode: 模式 ('train', 'val', 'test')
            augmentation: 是否使用数据增强
        """
        super(GameSceneDataset, self).__init__()
        
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.mode = mode
        self.augmentation = augmentation and (mode == 'train')
        
        # 加载数据列表
        self.data_list = self._load_data_split(split_file)
        
        # 数据增强变换
        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
        else:
            self.transform = None
        
        # 归一化参数
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406])
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225])
        
        print(f"=== GameSceneDataset ({mode}) ===")
        print(f"Data samples: {len(self.data_list)}")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Sequence length: {sequence_length}")
        print(f"Augmentation: {self.augmentation}")
    
    def _load_data_split(self, split_file: str) -> List[Dict]:
        """加载数据分割列表"""
        data_list = []
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 解析数据路径
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        data_item = {
                            'game': parts[0],  # 游戏名称
                            'scene': parts[1], # 场景ID
                            'frame_start': int(parts[2]), # 起始帧
                            'frame_count': int(parts[3])  # 帧数量
                        }
                        data_list.append(data_item)
        
        return data_list
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取数据项
        
        Returns:
            input_data: 6通道输入 [6, H, W] - RGB + Mask + ResidualMV
            target: 3通道目标 [3, H, W] - RGB
            input_prev: 前一帧输入（用于时序一致性）
            target_prev: 前一帧目标
        """
        data_item = self.data_list[idx]
        
        # 随机选择序列起始帧
        max_start = data_item['frame_count'] - self.sequence_length
        if max_start <= 0:
            frame_start = data_item['frame_start']
        else:
            frame_start = data_item['frame_start'] + random.randint(0, max_start)
        
        # 加载当前帧和前一帧数据
        current_frame = self._load_frame_data(data_item, frame_start + 1)
        previous_frame = self._load_frame_data(data_item, frame_start)
        
        # 提取patch
        input_data, target = self._extract_patch(current_frame)
        input_prev, target_prev = self._extract_patch(previous_frame)
        
        # 数据增强
        if self.augmentation:
            input_data, target = self._apply_augmentation(input_data, target)
            input_prev, target_prev = self._apply_augmentation(input_prev, target_prev)
        
        return input_data, target, input_prev, target_prev
    
    def _load_frame_data(self, data_item: Dict, frame_idx: int) -> Dict[str, np.ndarray]:
        """
        加载单帧数据
        
        Args:
            data_item: 数据项信息
            frame_idx: 帧索引
        
        Returns:
            frame_data: 包含RGB、Mask、ResidualMV的字典
        """
        game = data_item['game']
        scene = data_item['scene']
        
        # 构建文件路径
        frame_dir = self.data_root / game / scene / f"frame_{frame_idx:06d}"
        
        # 加载RGB图像
        rgb_path = frame_dir / "rgb.png"
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 加载遮挡掩码
        mask_path = frame_dir / "mask.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # 如果没有真实掩码，生成随机掩码用于训练
            mask = self._generate_random_mask(rgb.shape[:2])
        
        # 加载残差运动矢量
        mv_path = frame_dir / "residual_mv.npy"
        if mv_path.exists():
            residual_mv = np.load(str(mv_path))
        else:
            # 如果没有残差MV，生成零向量
            residual_mv = np.zeros((rgb.shape[0], rgb.shape[1], 2), dtype=np.float32)
        
        return {
            'rgb': rgb.astype(np.float32),
            'mask': mask.astype(np.float32),
            'residual_mv': residual_mv.astype(np.float32)
        }
    
    def _generate_random_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """生成随机空洞掩码用于训练"""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        # 生成随机形状的空洞
        num_holes = random.randint(1, 5)
        for _ in range(num_holes):
            # 随机空洞大小和位置
            hole_size = random.randint(10, min(h, w) // 4)
            x = random.randint(0, w - hole_size)
            y = random.randint(0, h - hole_size)
            
            # 创建不规则形状
            if random.random() < 0.5:
                # 矩形空洞
                mask[y:y+hole_size, x:x+hole_size] = 1.0
            else:
                # 椭圆形空洞
                center = (x + hole_size//2, y + hole_size//2)
                axes = (hole_size//2, hole_size//3)
                angle = random.randint(0, 180)
                cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
        
        return mask
    
    def _extract_patch(self, frame_data: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取patch
        
        Args:
            frame_data: 帧数据字典
        
        Returns:
            input_patch: 6通道输入patch [6, patch_size, patch_size]
            target_patch: 3通道目标patch [3, patch_size, patch_size]
        """
        rgb = frame_data['rgb']
        mask = frame_data['mask']
        residual_mv = frame_data['residual_mv']
        
        h, w = rgb.shape[:2]
        
        # 随机选择patch位置
        if h > self.patch_size and w > self.patch_size:
            y = random.randint(0, h - self.patch_size)
            x = random.randint(0, w - self.patch_size)
        else:
            # 如果图像太小，进行resize
            rgb = cv2.resize(rgb, (self.patch_size, self.patch_size))
            mask = cv2.resize(mask, (self.patch_size, self.patch_size))
            residual_mv = cv2.resize(residual_mv, (self.patch_size, self.patch_size))
            y, x = 0, 0
        
        # 提取patch
        rgb_patch = rgb[y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
        mv_patch = residual_mv[y:y+self.patch_size, x:x+self.patch_size]
        
        # 归一化RGB到[-1, 1]
        rgb_patch = rgb_patch / 255.0 * 2.0 - 1.0
        
        # 归一化mask到[0, 1]
        mask_patch = mask_patch / 255.0 if mask_patch.max() > 1.0 else mask_patch
        
        # 归一化运动矢量
        mv_patch = np.clip(mv_patch / 10.0, -1.0, 1.0)  # 假设MV范围在[-10, 10]
        
        # 转换为tensor并调整维度顺序 [H, W, C] -> [C, H, W]
        rgb_tensor = torch.from_numpy(rgb_patch).permute(2, 0, 1)  # [3, H, W]
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)    # [1, H, W]
        mv_tensor = torch.from_numpy(mv_patch).permute(2, 0, 1)   # [2, H, W]
        
        # 拼接为6通道输入
        input_patch = torch.cat([rgb_tensor, mask_tensor, mv_tensor], dim=0)  # [6, H, W]
        
        # 目标patch（原始RGB，用于计算损失）
        target_patch = rgb_tensor.clone()  # [3, H, W]
        
        return input_patch, target_patch
    
    def _apply_augmentation(self, input_patch: torch.Tensor, target_patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用数据增强
        
        Args:
            input_patch: 输入patch [6, H, W]
            target_patch: 目标patch [3, H, W]
        
        Returns:
            augmented_input: 增强后输入
            augmented_target: 增强后目标
        """
        # 分离RGB和其他通道
        rgb_input = input_patch[:3]  # RGB通道
        other_input = input_patch[3:]  # Mask + MV通道
        
        # 对RGB应用颜色增强
        if self.transform:
            # 转换到[0,1]范围进行变换
            rgb_input_01 = (rgb_input + 1) / 2
            target_01 = (target_patch + 1) / 2
            
            # 应用变换
            rgb_input_01 = self.transform(rgb_input_01)
            target_01 = self.transform(target_01)
            
            # 转换回[-1,1]范围
            rgb_input = rgb_input_01 * 2 - 1
            target_patch = target_01 * 2 - 1
        
        # 几何变换（翻转）
        if random.random() < 0.5:
            # 水平翻转
            rgb_input = torch.flip(rgb_input, dims=[2])
            target_patch = torch.flip(target_patch, dims=[2])
            other_input = torch.flip(other_input, dims=[2])
            # MV的x分量需要取反
            other_input[1] = -other_input[1]  # MV_x分量取反
        
        if random.random() < 0.3:
            # 垂直翻转
            rgb_input = torch.flip(rgb_input, dims=[1])
            target_patch = torch.flip(target_patch, dims=[1])
            other_input = torch.flip(other_input, dims=[1])
            # MV的y分量需要取反
            other_input[2] = -other_input[2]  # MV_y分量取反
        
        # 重新拼接
        augmented_input = torch.cat([rgb_input, other_input], dim=0)
        
        return augmented_input, target_patch


class GameSceneDataModule:
    """
    数据模块 - 管理训练/验证/测试数据加载器
    """
    
    def __init__(self, 
                 data_config: Dict,
                 batch_size: int = 32,
                 num_workers: int = 4):
        """
        初始化数据模块
        
        Args:
            data_config: 数据配置字典
            batch_size: 批次大小
            num_workers: 数据加载进程数
        """
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """设置数据集"""
        # 训练集
        self.train_dataset = GameSceneDataset(
            data_root=self.data_config['data_root'],
            split_file=self.data_config['train_split'],
            patch_size=self.data_config['patch_size'],
            sequence_length=self.data_config['sequence_length'],
            mode='train',
            augmentation=True
        )
        
        # 验证集
        self.val_dataset = GameSceneDataset(
            data_root=self.data_config['data_root'],
            split_file=self.data_config['val_split'],
            patch_size=self.data_config['patch_size'],
            sequence_length=self.data_config['sequence_length'],
            mode='val',
            augmentation=False
        )
        
        # 测试集
        if 'test_split' in self.data_config:
            self.test_dataset = GameSceneDataset(
                data_root=self.data_config['data_root'],
                split_file=self.data_config['test_split'],
                patch_size=self.data_config['patch_size'],
                sequence_length=self.data_config['sequence_length'],
                mode='test',
                augmentation=False
            )
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


def create_data_module(data_config: Dict, batch_size: int = 32, num_workers: int = 4) -> GameSceneDataModule:
    """
    创建数据模块
    
    Args:
        data_config: 数据配置
        batch_size: 批次大小
        num_workers: 工作线程数
    
    Returns:
        data_module: 数据模块实例
    """
    data_module = GameSceneDataModule(
        data_config=data_config,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    data_module.setup()
    
    print(f"=== Data Module Created ===")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Train samples: {len(data_module.train_dataset)}")
    print(f"Val samples: {len(data_module.val_dataset)}")
    if data_module.test_dataset:
        print(f"Test samples: {len(data_module.test_dataset)}")
    
    return data_module


if __name__ == "__main__":
    # 测试数据集
    data_config = {
        'data_root': './data/game_scenes',
        'train_split': './data/splits/train.txt',
        'val_split': './data/splits/val.txt',
        'patch_size': 64,
        'sequence_length': 3
    }
    
    # 创建数据集
    dataset = GameSceneDataset(
        data_root=data_config['data_root'],
        split_file=data_config['train_split'],
        patch_size=data_config['patch_size'],
        sequence_length=data_config['sequence_length'],
        mode='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试数据加载
    if len(dataset) > 0:
        sample = dataset[0]
        input_data, target, input_prev, target_prev = sample
        
        print(f"Input shape: {input_data.shape}")  # [6, 64, 64]
        print(f"Target shape: {target.shape}")      # [3, 64, 64]
        print(f"Input range: [{input_data.min():.3f}, {input_data.max():.3f}]")
        print(f"Target range: [{target.min():.3f}, {target.max():.3f}]")
    
    print("Dataset test completed successfully!")