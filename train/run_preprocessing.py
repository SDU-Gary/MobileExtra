#!/usr/bin/env python3
"""
@file run_preprocessing.py
@brief NoiseBase数据预处理执行脚本

功能描述：
- 自动运行NoiseBase数据预处理
- 生成适配MobileInpaintingNetwork的6通道训练数据
- 创建数据集分割文件
- 验证处理结果

使用方法：
python run_preprocessing.py --scene bistro1 --frames 0 50

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "training"))

from noisebase_preprocessor import NoiseBasePreprocessor


def create_data_splits(output_dir: str, scene_name: str, total_frames: int) -> Dict[str, List[int]]:
    """
    创建数据集分割
    
    Args:
        output_dir: 输出目录
        scene_name: 场景名称
        total_frames: 总帧数
    
    Returns:
        splits: 分割字典
    """
    # 分割比例：训练80%，验证15%，测试5%
    train_ratio = 0.8
    val_ratio = 0.15
    test_ratio = 0.05
    
    # 计算分割点
    train_end = int(total_frames * train_ratio)
    val_end = int(total_frames * (train_ratio + val_ratio))
    
    # 创建分割列表（从帧1开始，因为需要前一帧作为参考）
    splits = {
        'train': list(range(1, train_end)),
        'val': list(range(train_end, val_end)),
        'test': list(range(val_end, total_frames))
    }
    
    # 保存分割文件
    split_file = Path(output_dir) / f"{scene_name}_splits.json"
    
    split_data = {
        'scene': scene_name,
        'total_frames': total_frames,
        'splits': splits,
        'statistics': {
            'train_samples': len(splits['train']),
            'val_samples': len(splits['val']),
            'test_samples': len(splits['test'])
        }
    }
    
    with open(split_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"Data splits saved to: {split_file}")
    print(f"Train: {len(splits['train'])} samples")
    print(f"Val: {len(splits['val'])} samples") 
    print(f"Test: {len(splits['test'])} samples")
    
    return splits


def validate_preprocessing_results(output_dir: str, scene_name: str, expected_frames: int):
    """
    验证预处理结果
    
    Args:
        output_dir: 输出目录
        scene_name: 场景名称
        expected_frames: 预期帧数
    """
    base_path = Path(output_dir) / scene_name
    
    # 检查目录结构
    required_dirs = ['rgb', 'warped', 'masks', 'residual_mv', 'training_data', 'visualization']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    # 检查文件数量
    training_data_dir = base_path / 'training_data'
    training_files = list(training_data_dir.glob("*.npy"))
    
    print(f"✅ Directory structure complete")
    print(f"✅ Generated {len(training_files)} training samples")
    
    # 检查样本文件大小和格式
    if training_files:
        import numpy as np
        sample_file = training_files[0]
        sample_data = np.load(sample_file)
        
        print(f"✅ Sample data shape: {sample_data.shape}")
        
        if sample_data.shape[0] == 6:
            print("✅ 6-channel format correct (RGB + Mask + ResidualMV)")
        else:
            print(f"❌ Expected 6 channels, got {sample_data.shape[0]}")
            return False
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run NoiseBase Preprocessing')
    parser.add_argument('--input-dir', type=str, default='./training',
                       help='NoiseBase data directory')
    parser.add_argument('--output-dir', type=str, default='./training/processed',
                       help='Output directory')
    parser.add_argument('--scene', type=str, default='bistro1',
                       help='Scene name (bistro1, kitchen)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Start frame index')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='End frame index (auto-detect if None)')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate preprocessing results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 NoiseBase数据预处理开始")
    print("="*80)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"场景名称: {args.scene}")
    print(f"起始帧: {args.start_frame}")
    print(f"结束帧: {args.end_frame if args.end_frame else 'auto-detect'}")
    print("="*80)
    
    try:
        # 1. 运行预处理
        print("\n📊 Step 1: 运行数据预处理...")
        preprocessor = NoiseBasePreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            scene_name=args.scene
        )
        
        # 自动检测帧数范围
        if args.end_frame is None:
            input_scene_dir = Path(args.input_dir) / args.scene
            frame_count = 0
            while (input_scene_dir / f"frame{frame_count:04d}.zip").exists():
                frame_count += 1
            args.end_frame = frame_count - 1
            print(f"检测到 {frame_count} 帧数据")
        
        # 执行预处理
        preprocessor.process_sequence(
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
        
        # 2. 创建数据分割
        if args.create_splits:
            print("\n📋 Step 2: 创建数据集分割...")
            total_frames = args.end_frame - args.start_frame + 1
            splits = create_data_splits(args.output_dir, args.scene, total_frames)
        
        # 3. 验证结果
        if args.validate:
            print("\n✅ Step 3: 验证预处理结果...")
            expected_frames = args.end_frame - args.start_frame
            success = validate_preprocessing_results(args.output_dir, args.scene, expected_frames)
            
            if success:
                print("\n🎉 预处理完成！所有验证通过")
            else:
                print("\n⚠️ 预处理完成，但验证发现问题")
        
        # 4. 输出使用指南
        print("\n" + "="*80)
        print("📖 使用指南")
        print("="*80)
        print("下一步可以执行以下操作：")
        print("")
        print("1. 训练网络：")
        print(f"   python training/train_mobile_inpainting.py \\")
        print(f"     --data-root {args.output_dir} \\")
        print(f"     --split-file {args.output_dir}/{args.scene}_splits.json")
        print("")
        print("2. 可视化预处理结果：")
        print(f"   查看 {args.output_dir}/{args.scene}/visualization/ 目录")
        print("")
        print("3. 检查训练数据：")
        print(f"   加载 {args.output_dir}/{args.scene}/training_data/*.npy 文件")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())