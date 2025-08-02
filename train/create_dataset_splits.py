#!/usr/bin/env python3
"""
@file create_dataset_splits.py  
@brief 创建NoiseBase数据集分割文件

功能描述：
- 分析预处理后的NoiseBase数据
- 创建训练/验证/测试分割
- 生成适配dataset类的分割文件格式
- 考虑数据质量进行智能分割

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class DatasetSplitCreator:
    """数据集分割创建器"""
    
    def __init__(self, data_root: str, scene_name: str):
        """
        初始化分割创建器
        
        Args:
            data_root: 预处理数据根目录
            scene_name: 场景名称
        """
        self.data_root = Path(data_root)
        self.scene_name = scene_name
        self.scene_path = self.data_root / scene_name
        
        # 分割比例
        self.train_ratio = 0.7
        self.val_ratio = 0.2  
        self.test_ratio = 0.1
        
        # 质量阈值
        self.min_quality_score = 0.3
        self.max_hole_ratio = 0.4
        self.min_motion_diversity = 0.1
        
    def analyze_frame_quality(self, frame_idx: int) -> Dict:
        """
        分析单帧数据质量
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            quality_info: 质量信息字典
        """
        frame_name = f"frame_{frame_idx:04d}"
        
        # 加载训练数据
        training_data_path = self.scene_path / 'training_data' / f"{frame_name}.npy"
        
        if not training_data_path.exists():
            return {'valid': False, 'reason': 'missing_training_data'}
        
        try:
            training_data = np.load(training_data_path)  # [6, H, W]
            
            # 分离各通道
            rgb = training_data[:3]        # RGB通道
            mask = training_data[3:4]      # 掩码通道  
            residual_mv = training_data[4:6]  # 残差MV通道
            
            # 质量评估
            quality_info = {
                'valid': True,
                'frame_idx': frame_idx,
                'shape': training_data.shape
            }
            
            # 1. RGB质量检查
            rgb_std = np.std(rgb)
            rgb_range_valid = np.all((rgb >= -1.1) & (rgb <= 1.1))
            quality_info['rgb_std'] = float(rgb_std)
            quality_info['rgb_range_valid'] = bool(rgb_range_valid)
            
            # 2. 掩码质量检查
            hole_ratio = np.mean(mask)
            quality_info['hole_ratio'] = float(hole_ratio)
            quality_info['hole_valid'] = hole_ratio <= self.max_hole_ratio
            
            # 3. 运动矢量质量检查
            mv_magnitude = np.sqrt(residual_mv[0]**2 + residual_mv[1]**2)
            mv_avg_magnitude = np.mean(mv_magnitude)
            mv_std = np.std(mv_magnitude)
            quality_info['mv_magnitude'] = float(mv_avg_magnitude)
            quality_info['mv_diversity'] = float(mv_std)
            quality_info['mv_valid'] = mv_std >= self.min_motion_diversity
            
            # 4. 综合质量评分
            quality_factors = []
            
            # RGB对比度评分
            contrast_score = min(rgb_std / 0.3, 1.0) if rgb_range_valid else 0.0
            quality_factors.append(contrast_score)
            
            # 掩码评分（适中的空洞比例得分更高）
            if 0.05 <= hole_ratio <= 0.25:
                mask_score = 1.0
            else:
                mask_score = max(0.2, 1.0 - abs(hole_ratio - 0.15) * 3)
            quality_factors.append(mask_score)
            
            # 运动多样性评分
            diversity_score = min(mv_std / 1.0, 1.0)
            quality_factors.append(diversity_score)
            
            quality_score = np.mean(quality_factors)
            quality_info['quality_score'] = float(quality_score)
            quality_info['high_quality'] = quality_score >= self.min_quality_score
            
            return quality_info
            
        except Exception as e:
            return {'valid': False, 'reason': f'analysis_error: {e}'}
    
    def scan_available_frames(self) -> List[Dict]:
        """
        扫描所有可用帧并分析质量
        
        Returns:
            frame_infos: 帧信息列表
        """
        print(f"🔍 扫描场景 {self.scene_name} 的可用帧...")
        
        training_data_dir = self.scene_path / 'training_data'
        if not training_data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {training_data_dir}")
        
        # 获取所有训练数据文件
        training_files = sorted(training_data_dir.glob("frame_*.npy"))
        
        frame_infos = []
        for file_path in training_files:
            # 从文件名提取帧索引
            frame_name = file_path.stem
            frame_idx = int(frame_name.split('_')[1])
            
            # 分析帧质量
            quality_info = self.analyze_frame_quality(frame_idx)
            frame_infos.append(quality_info)
        
        print(f"✅ 扫描完成，找到 {len(frame_infos)} 个数据文件")
        
        # 统计质量信息
        valid_frames = [info for info in frame_infos if info.get('valid', False)]
        high_quality_frames = [info for info in valid_frames if info.get('high_quality', False)]
        
        print(f"📊 有效帧: {len(valid_frames)}")
        print(f"📊 高质量帧: {len(high_quality_frames)}")
        
        if len(valid_frames) > 0:
            avg_quality = np.mean([info['quality_score'] for info in valid_frames])
            avg_hole_ratio = np.mean([info['hole_ratio'] for info in valid_frames])
            avg_mv_magnitude = np.mean([info['mv_magnitude'] for info in valid_frames])
            
            print(f"📊 平均质量评分: {avg_quality:.3f}")
            print(f"📊 平均空洞比例: {avg_hole_ratio:.3f}") 
            print(f"📊 平均运动幅度: {avg_mv_magnitude:.3f}")
        
        return valid_frames
    
    def create_balanced_splits(self, frame_infos: List[Dict]) -> Dict[str, List[int]]:
        """
        创建均衡的数据分割
        
        Args:
            frame_infos: 帧信息列表
            
        Returns:
            splits: 分割字典
        """
        print(f"📋 创建数据集分割...")
        
        # 过滤有效帧
        valid_frames = [info for info in frame_infos if info.get('valid', False)]
        
        if len(valid_frames) == 0:
            raise ValueError("No valid frames found for splitting")
        
        # 按质量排序
        valid_frames.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # 计算分割大小
        total_frames = len(valid_frames)
        train_size = int(total_frames * self.train_ratio)
        val_size = int(total_frames * self.val_ratio)
        test_size = total_frames - train_size - val_size
        
        print(f"📊 分割计划: 训练{train_size}, 验证{val_size}, 测试{test_size}")
        
        # 策略：保证各个分割都有高中低质量的样本
        frame_indices = [info['frame_idx'] for info in valid_frames]
        
        # 交错分配以保证质量分布均匀
        train_indices = []
        val_indices = []
        test_indices = []
        
        for i, frame_idx in enumerate(frame_indices):
            if i % 10 < 7:  # 70%用于训练
                train_indices.append(frame_idx)
            elif i % 10 < 9:  # 20%用于验证
                val_indices.append(frame_idx)
            else:  # 10%用于测试
                test_indices.append(frame_idx)
        
        # 如果分割大小不符合预期，进行调整
        while len(train_indices) < train_size and (len(val_indices) > val_size or len(test_indices) > test_size):
            if len(val_indices) > val_size:
                train_indices.append(val_indices.pop())
            elif len(test_indices) > test_size:
                train_indices.append(test_indices.pop())
        
        splits = {
            'train': sorted(train_indices),
            'val': sorted(val_indices), 
            'test': sorted(test_indices)
        }
        
        print(f"✅ 实际分割: 训练{len(splits['train'])}, 验证{len(splits['val'])}, 测试{len(splits['test'])}")
        
        return splits
    
    def create_split_file(self, splits: Dict[str, List[int]], frame_infos: List[Dict]) -> str:
        """
        创建分割文件
        
        Args:
            splits: 分割字典
            frame_infos: 帧信息列表
            
        Returns:
            split_file_path: 分割文件路径
        """
        # 创建详细的分割信息
        split_data = {
            'dataset_info': {
                'scene_name': self.scene_name,
                'data_root': str(self.data_root),
                'total_frames': len(frame_infos),
                'creation_date': str(np.datetime64('today'))
            },
            'splits': splits,
            'statistics': {
                'train_samples': len(splits['train']),
                'val_samples': len(splits['val']),
                'test_samples': len(splits['test']),
                'ratios': {
                    'train': self.train_ratio,
                    'val': self.val_ratio,
                    'test': self.test_ratio
                }
            },
            'quality_thresholds': {
                'min_quality_score': self.min_quality_score,
                'max_hole_ratio': self.max_hole_ratio,
                'min_motion_diversity': self.min_motion_diversity
            }
        }
        
        # 添加每个分割的质量统计
        for split_name, indices in splits.items():
            split_frame_infos = [info for info in frame_infos if info['frame_idx'] in indices]
            
            if split_frame_infos:
                split_stats = { 
                    'avg_quality_score': float(np.mean([info['quality_score'] for info in split_frame_infos])),
                    'avg_hole_ratio': float(np.mean([info['hole_ratio'] for info in split_frame_infos])),
                    'avg_mv_magnitude': float(np.mean([info['mv_magnitude'] for info in split_frame_infos])),
                    'high_quality_count': sum(1 for info in split_frame_infos if info.get('high_quality', False))
                }
                split_data['statistics'][f'{split_name}_quality'] = split_stats
        
        # 保存分割文件
        split_file_path = self.data_root / f"{self.scene_name}_splits.json"
        
        with open(split_file_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 分割文件已保存: {split_file_path}")
        
        return str(split_file_path)
    
    def create_splits(self) -> str:
        """
        创建完整的数据集分割
        
        Returns:
            split_file_path: 分割文件路径
        """
        print("="*60)
        print(f"🎯 为场景 {self.scene_name} 创建数据集分割")
        print("="*60)
        
        # 1. 扫描可用帧
        frame_infos = self.scan_available_frames()
        
        if len(frame_infos) == 0:
            raise ValueError(f"No valid frames found in scene {self.scene_name}")
        
        # 2. 创建分割
        splits = self.create_balanced_splits(frame_infos)
        
        # 3. 生成分割文件
        split_file_path = self.create_split_file(splits, frame_infos)
        
        print("="*60)
        print("🎉 数据集分割创建完成!")
        print("="*60)
        
        return split_file_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Create NoiseBase Dataset Splits')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Preprocessed data root directory')
    parser.add_argument('--scene', type=str, default='bistro1',
                       help='Scene name (bistro1, kitchen)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio')
    
    args = parser.parse_args()
    
    # 验证比例
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("❌ 分割比例之和必须等于1.0")
        return 1
    
    try:
        # 创建分割器
        splitter = DatasetSplitCreator(args.data_root, args.scene)
        splitter.train_ratio = args.train_ratio
        splitter.val_ratio = args.val_ratio
        splitter.test_ratio = args.test_ratio
        
        # 创建分割
        split_file_path = splitter.create_splits()
        
        print(f"\n📖 使用方法:")
        print(f"python training/train_mobile_inpainting.py \\")
        print(f"  --data-root {args.data_root} \\")
        print(f"  --split-file {split_file_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 分割创建失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())