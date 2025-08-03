#!/usr/bin/env python3
"""
修正后的NoiseBase数据预处理脚本

基于学长detach.py脚本的分析，修正了数据加载部分：
- 正确处理zip+zarr格式的NoiseBase数据
- 实现RGBE颜色解压缩
- 处理多采样数据聚合
- 计算正确的几何信息

使用方法:
python run_preprocessing_corrected.py --data-root ./data --scene bistro1 --output ./processed
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from noisebase_data_loader import NoiseBaseDataLoader
from noisebase_preprocessor_corrected import NoiseBasePreprocessorCorrected


class NoiseBasePreprocessorWithRealData(NoiseBasePreprocessorCorrected):
    """
    使用真实NoiseBase数据的预处理器
    
    继承修正后的预处理器，替换数据加载部分
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 scene_name: str = "bistro1"):
        """
        初始化预处理器
        
        Args:
            input_dir: NoiseBase数据输入目录
            output_dir: 处理后数据输出目录  
            scene_name: 场景名称
        """
        super().__init__(input_dir, output_dir, scene_name)
        
        # 创建数据加载器
        self.data_loader = NoiseBaseDataLoader(input_dir)
        
        # 验证场景数据
        available_scenes = self.data_loader.list_available_scenes()
        if scene_name not in available_scenes:
            raise ValueError(f"场景 '{scene_name}' 不存在。可用场景: {available_scenes}")
        
        frame_count = self.data_loader.count_frames(scene_name)
        print(f"场景 '{scene_name}' 包含 {frame_count} 帧数据")
        
        # 验证数据完整性
        validation = self.data_loader.validate_data_integrity(scene_name, max_frames=3)
        if validation['valid_frames'] == 0:
            raise ValueError(f"场景 '{scene_name}' 没有有效的帧数据")
        
        print(f"数据验证通过: {validation['valid_frames']} 帧有效")
    
    def load_frame_data(self, frame_idx: int) -> Dict:
        """
        加载真实的NoiseBase帧数据
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            frame_data: 帧数据字典
        """
        # 使用真实的数据加载器
        frame_data = self.data_loader.load_frame_data(self.scene_name, frame_idx)
        
        if frame_data is None:
            return None
        
        # 确保数据格式符合预处理器要求
        processed_data = {}
        
        # 参考图像 (目标图像)
        if 'reference' in frame_data:
            processed_data['reference'] = frame_data['reference']
        elif 'color' in frame_data:
            # 如果没有reference，使用解压缩后的color作为参考
            processed_data['reference'] = frame_data['color']
        else:
            raise ValueError(f"帧 {frame_idx} 缺少参考图像数据")
        
        # 世界空间位置
        if 'position' in frame_data:
            processed_data['position'] = frame_data['position']
        else:
            raise ValueError(f"帧 {frame_idx} 缺少位置数据")
        
        # 运动矢量 (优先使用屏幕空间运动矢量)
        if 'screen_motion' in frame_data:
            processed_data['motion'] = frame_data['screen_motion']
        elif 'motion' in frame_data:
            # 如果只有世界空间运动，需要转换为屏幕空间
            if 'view_proj_mat' in frame_data:
                screen_motion = self.data_loader.compute_screen_motion_vectors(
                    frame_data['position'],
                    frame_data['motion'], 
                    frame_data['view_proj_mat']
                )
                processed_data['motion'] = screen_motion
            else:
                # 如果无法转换，直接使用前两个通道
                processed_data['motion'] = frame_data['motion'][:2]
        else:
            raise ValueError(f"帧 {frame_idx} 缺少运动数据")
        
        # 相机位置
        if 'camera_pos' in frame_data:
            processed_data['camera_pos'] = frame_data['camera_pos']
        else:
            # 如果没有相机位置，使用默认值
            processed_data['camera_pos'] = np.array([0, 0, 5], dtype=np.float32)
            print(f"⚠️ 帧 {frame_idx} 缺少相机位置，使用默认值")
        
        # 其他可选数据
        for key in ['normal', 'albedo', 'view_proj_mat', 'exposure']:
            if key in frame_data:
                processed_data[key] = frame_data[key]
        
        return processed_data
    
    def process_sequence(self, start_frame: int = 0, end_frame: int = None):
        """
        处理帧序列
        
        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引 (None表示处理到最后一帧)
        """
        # 确定处理范围
        total_frames = self.data_loader.count_frames(self.scene_name)
        
        if end_frame is None:
            end_frame = total_frames - 1
        
        end_frame = min(end_frame, total_frames - 1)
        
        if start_frame >= total_frames:
            raise ValueError(f"起始帧 {start_frame} 超出数据范围 (0-{total_frames-1})")
        
        print(f"🚀 开始处理帧序列: {start_frame} -> {end_frame}")
        print(f"总共需要处理 {end_frame - start_frame + 1} 帧")
        
        # 处理每一帧 (从第1帧开始，因为需要前一帧作为参考)
        success_count = 0
        error_count = 0
        
        for frame_idx in range(max(1, start_frame), end_frame + 1):
            print(f"\n📊 处理帧 {frame_idx}/{end_frame}")
            
            try:
                success = self.process_frame_pair(frame_idx)
                
                if success:
                    success_count += 1
                    print(f"✅ 帧 {frame_idx} 处理成功")
                else:
                    error_count += 1
                    print(f"❌ 帧 {frame_idx} 处理失败")
                    
            except Exception as e:
                error_count += 1
                print(f"❌ 帧 {frame_idx} 处理异常: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 序列处理完成!")
        print(f"   成功: {success_count} 帧")
        print(f"   失败: {error_count} 帧")
        print(f"   成功率: {success_count/(success_count+error_count)*100:.1f}%")
        
        return success_count, error_count


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
        sample_file = training_files[0]
        sample_data = np.load(sample_file)
        
        print(f"✅ Sample data shape: {sample_data.shape}")
        
        if sample_data.shape[0] == 6:
            print("✅ 6-channel format correct (RGB + OcclusionMask + ResidualMV)")
        else:
            print(f"❌ Expected 6 channels, got {sample_data.shape[0]}")
            return False
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run NoiseBase Preprocessing with Real Data')
    parser.add_argument('--data-root', type=str, required=True,
                       help='NoiseBase data root directory')
    parser.add_argument('--output-dir', type=str, default='./processed_data',
                       help='Output directory')
    parser.add_argument('--scene', type=str, default='bistro1',
                       help='Scene name (bistro1, kitchen, etc.)')
    parser.add_argument('--start-frame', type=int, default=1,
                       help='Start frame index')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='End frame index (auto-detect if None)')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate preprocessing results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 NoiseBase数据预处理开始 (使用真实数据)")
    print("="*80)
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"场景名称: {args.scene}")
    print(f"起始帧: {args.start_frame}")
    print(f"结束帧: {args.end_frame if args.end_frame else 'auto-detect'}")
    print("="*80)
    
    try:
        # 1. 创建预处理器
        print("\n📊 Step 1: 初始化预处理器...")
        preprocessor = NoiseBasePreprocessorWithRealData(
            input_dir=args.data_root,
            output_dir=args.output_dir,
            scene_name=args.scene
        )
        
        # 2. 自动检测帧数范围
        if args.end_frame is None:
            total_frames = preprocessor.data_loader.count_frames(args.scene)
            args.end_frame = total_frames - 1
            print(f"检测到 {total_frames} 帧数据，处理到第 {args.end_frame} 帧")
        
        # 3. 执行预处理
        print("\n🔄 Step 2: 执行数据预处理...")
        success_count, error_count = preprocessor.process_sequence(
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
        
        # 4. 创建数据分割
        if args.create_splits:
            print("\n📋 Step 3: 创建数据集分割...")
            total_processed = success_count
            splits = create_data_splits(args.output_dir, args.scene, total_processed)
        
        # 5. 验证结果
        if args.validate:
            print("\n✅ Step 4: 验证预处理结果...")
            validation_success = validate_preprocessing_results(
                args.output_dir, args.scene, success_count
            )
            
            if validation_success:
                print("\n🎉 预处理完成！所有验证通过")
            else:
                print("\n⚠️ 预处理完成，但验证发现问题")
        
        # 6. 输出使用指南
        print("\n" + "="*80)
        print("📖 使用指南")
        print("="*80)
        print("下一步可以执行以下操作：")
        print("")
        print("1. 检查处理结果：")
        print(f"   查看 {args.output_dir}/{args.scene}/visualization/ 目录")
        print("")
        print("2. 训练网络：")
        print(f"   python training/train_mobile_inpainting.py \\")
        print(f"     --data-root {args.output_dir} \\")
        print(f"     --split-file {args.output_dir}/{args.scene}_splits.json")
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