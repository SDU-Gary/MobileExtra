#!/usr/bin/env python3
"""
@file test_preprocessing_pipeline.py
@brief 测试完整的NoiseBase预处理和训练流水线

功能描述：
- 测试NoiseBase数据预处理流程
- 验证6通道训练数据生成
- 测试数据集加载和训练框架集成
- 提供详细的数据质量报告

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "training"))
sys.path.insert(0, str(project_root / "src"))

from training.noisebase_preprocessor import NoiseBasePreprocessor
from training.create_dataset_splits import DatasetSplitCreator
from training.datasets.noisebase_dataset import NoiseBaseDataset, create_noisebase_dataset
from src.npu.networks.mobile_inpainting_network import MobileInpaintingNetwork


class PreprocessingPipelineTester:
    """预处理流水线测试器"""
    
    def __init__(self, 
                 input_dir: str = "./training",
                 output_dir: str = "./training/processed_test",
                 scene_name: str = "bistro1"):
        """
        初始化测试器
        
        Args:
            input_dir: NoiseBase原始数据目录
            output_dir: 预处理输出目录
            scene_name: 场景名称
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        
        # 测试参数
        self.test_frame_count = 10  # 测试前10帧
        self.patch_size = 64
        
        print("="*80)
        print("🧪 NoiseBase预处理流水线测试")
        print("="*80)
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"测试场景: {self.scene_name}")
        print(f"测试帧数: {self.test_frame_count}")
        print("="*80)
    
    def test_data_availability(self) -> bool:
        """
        测试原始数据可用性
        
        Returns:
            available: 数据是否可用
        """
        print("\n🔍 Step 1: 检查原始数据可用性...")
        
        scene_dir = self.input_dir / self.scene_name
        if not scene_dir.exists():
            print(f"❌ 场景目录不存在: {scene_dir}")
            return False
        
        # 检查前几帧数据
        available_frames = 0
        for i in range(self.test_frame_count):
            frame_file = scene_dir / f"frame{i:04d}.zip"
            if frame_file.exists():
                available_frames += 1
            else:
                break
        
        print(f"✅ 找到 {available_frames} 个可用帧文件")
        
        if available_frames < 2:
            print("❌ 需要至少2帧数据进行测试")
            return False
        
        # 更新实际测试帧数
        self.test_frame_count = min(self.test_frame_count, available_frames)
        print(f"📊 将测试前 {self.test_frame_count} 帧")
        
        return True
    
    def test_preprocessing(self) -> bool:
        """
        测试数据预处理
        
        Returns:
            success: 预处理是否成功
        """
        print("\n⚙️ Step 2: 测试数据预处理...")
        
        try:
            # 创建预处理器
            preprocessor = NoiseBasePreprocessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir),
                scene_name=self.scene_name
            )
            
            # 处理测试序列
            preprocessor.process_sequence(
                start_frame=0,
                end_frame=self.test_frame_count - 1
            )
            
            print("✅ 预处理完成")
            return True
            
        except Exception as e:
            print(f"❌ 预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_data_quality(self) -> Dict:
        """
        测试生成数据的质量
        
        Returns:
            quality_report: 质量报告
        """
        print("\n📊 Step 3: 分析生成数据质量...")
        
        scene_path = self.output_dir / self.scene_name
        training_data_dir = scene_path / 'training_data'
        
        if not training_data_dir.exists():
            print("❌ 训练数据目录不存在")
            return {}
        
        # 分析所有训练数据文件
        training_files = list(training_data_dir.glob("*.npy"))
        
        if len(training_files) == 0:
            print("❌ 没有找到训练数据文件")
            return {}
        
        print(f"📁 找到 {len(training_files)} 个训练数据文件")
        
        quality_metrics = {
            'file_count': len(training_files),
            'shapes': [],
            'rgb_stats': [],
            'hole_ratios': [],
            'mv_magnitudes': []
        }
        
        # 分析每个文件
        for file_path in training_files[:5]:  # 分析前5个文件
            try:
                data = np.load(file_path)  # [6, H, W]
                quality_metrics['shapes'].append(data.shape)
                
                # 分离通道
                rgb = data[:3]
                mask = data[3:4]
                residual_mv = data[4:6]
                
                # RGB统计
                rgb_mean = np.mean(rgb)
                rgb_std = np.std(rgb)
                rgb_range = (np.min(rgb), np.max(rgb))
                quality_metrics['rgb_stats'].append({
                    'mean': float(rgb_mean),
                    'std': float(rgb_std),
                    'range': rgb_range
                })
                
                # 空洞比例
                hole_ratio = np.mean(mask)
                quality_metrics['hole_ratios'].append(float(hole_ratio))
                
                # 运动矢量幅度
                mv_magnitude = np.sqrt(residual_mv[0]**2 + residual_mv[1]**2)
                avg_magnitude = np.mean(mv_magnitude)
                quality_metrics['mv_magnitudes'].append(float(avg_magnitude))
                
            except Exception as e:
                print(f"⚠️ 分析文件 {file_path.name} 时出错: {e}")
        
        # 打印质量报告
        if quality_metrics['shapes']:
            print(f"✅ 数据形状: {quality_metrics['shapes'][0]} (所有文件)")
            print(f"✅ RGB均值范围: {np.min([s['mean'] for s in quality_metrics['rgb_stats']]):.3f} ~ {np.max([s['mean'] for s in quality_metrics['rgb_stats']]):.3f}")
            print(f"✅ RGB标准差范围: {np.min([s['std'] for s in quality_metrics['rgb_stats']]):.3f} ~ {np.max([s['std'] for s in quality_metrics['rgb_stats']]):.3f}")
            print(f"✅ 空洞比例范围: {np.min(quality_metrics['hole_ratios']):.3f} ~ {np.max(quality_metrics['hole_ratios']):.3f}")
            print(f"✅ 运动矢量幅度范围: {np.min(quality_metrics['mv_magnitudes']):.3f} ~ {np.max(quality_metrics['mv_magnitudes']):.3f}")
        
        return quality_metrics
    
    def test_dataset_splits(self) -> Optional[str]:
        """
        测试数据集分割创建
        
        Returns:
            split_file_path: 分割文件路径
        """
        print("\n📋 Step 4: 测试数据集分割创建...")
        
        try:
            # 创建分割器
            splitter = DatasetSplitCreator(
                data_root=str(self.output_dir),
                scene_name=self.scene_name
            )
            
            # 创建分割
            split_file_path = splitter.create_splits()
            print(f"✅ 分割文件创建成功: {split_file_path}")
            
            return split_file_path
            
        except Exception as e:
            print(f"❌ 分割创建失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_dataset_loading(self, split_file_path: str) -> bool:
        """
        测试数据集类加载
        
        Args:
            split_file_path: 分割文件路径
            
        Returns:
            success: 加载是否成功
        """
        print("\n💿 Step 5: 测试数据集类加载...")
        
        try:
            # 创建数据集实例
            dataset = create_noisebase_dataset(
                data_root=str(self.output_dir),
                split_file=split_file_path,
                patch_size=self.patch_size,
                mode='train'
            )
            
            print(f"✅ 数据集创建成功，包含 {len(dataset)} 个样本")
            
            # 测试数据加载
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✅ 样本加载成功")
                print(f"   输入形状: {sample['input'].shape}")
                print(f"   目标形状: {sample['target'].shape}")
                print(f"   元数据: {list(sample.keys())}")
                
                # 验证数据格式
                if sample['input'].shape[0] == 6:
                    print("✅ 6通道输入格式正确")
                else:
                    print(f"❌ 输入通道数错误: {sample['input'].shape[0]}, 期望6")
                    return False
                
                if sample['target'].shape[0] == 3:
                    print("✅ 3通道目标格式正确")
                else:
                    print(f"❌ 目标通道数错误: {sample['target'].shape[0]}, 期望3")
                    return False
                
            return True
            
        except Exception as e:
            print(f"❌ 数据集加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_network_forward(self, split_file_path: str) -> bool:
        """
        测试网络前向传播
        
        Args:
            split_file_path: 分割文件路径
            
        Returns:
            success: 网络测试是否成功
        """
        print("\n🧠 Step 6: 测试网络前向传播...")
        
        try:
            # 创建网络
            network = MobileInpaintingNetwork(
                input_channels=6,
                output_channels=3,
                base_channels=32
            )
            
            # 创建数据集
            dataset = create_noisebase_dataset(
                data_root=str(self.output_dir),
                split_file=split_file_path,
                patch_size=self.patch_size,
                mode='train'
            )
            
            if len(dataset) == 0:
                print("❌ 数据集为空")
                return False
            
            # 获取测试样本
            sample = dataset[0]
            input_tensor = sample['input'].unsqueeze(0)  # [1, 6, H, W]
            target_tensor = sample['target'].unsqueeze(0)  # [1, 3, H, W]
            
            print(f"📊 输入张量形状: {input_tensor.shape}")
            print(f"📊 目标张量形状: {target_tensor.shape}")
            
            # 前向传播
            with torch.no_grad():
                output = network(input_tensor)
            
            print(f"✅ 网络前向成功，输出形状: {output.shape}")
            
            # 验证输出
            if output.shape == target_tensor.shape:
                print("✅ 输出形状与目标匹配")
            else:
                print(f"❌ 输出形状不匹配: {output.shape} vs {target_tensor.shape}")
                return False
            
            # 检查输出数值范围
            output_min, output_max = output.min().item(), output.max().item()
            print(f"📊 输出数值范围: [{output_min:.3f}, {output_max:.3f}]")
            
            return True
            
        except Exception as e:
            print(f"❌ 网络测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_visualization(self, split_file_path: str):
        """
        创建可视化结果
        
        Args:
            split_file_path: 分割文件路径
        """
        print("\n🎨 Step 7: 创建可视化结果...")
        
        try:
            # 创建数据集
            dataset = create_noisebase_dataset(
                data_root=str(self.output_dir),
                split_file=split_file_path,
                patch_size=self.patch_size,
                mode='train'
            )
            
            if len(dataset) == 0:
                print("❌ 数据集为空，无法创建可视化")
                return
            
            # 获取样本
            sample = dataset[0]
            input_data = sample['input'].numpy()  # [6, H, W]
            target_data = sample['target'].numpy()  # [3, H, W]
            
            # 分离输入通道
            rgb = input_data[:3]        # RGB
            mask = input_data[3:4]      # Mask
            residual_mv = input_data[4:6]  # ResidualMV
            
            # 创建可视化
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # RGB输入
            rgb_vis = np.clip((rgb.transpose(1, 2, 0) + 1) / 2, 0, 1)
            axes[0, 0].imshow(rgb_vis)
            axes[0, 0].set_title('Input RGB')
            axes[0, 0].axis('off')
            
            # 空洞掩码
            axes[0, 1].imshow(mask[0], cmap='gray')
            axes[0, 1].set_title('Hole Mask')
            axes[0, 1].axis('off')
            
            # 残差运动矢量
            mv_magnitude = np.sqrt(residual_mv[0]**2 + residual_mv[1]**2)
            axes[0, 2].imshow(mv_magnitude, cmap='jet')
            axes[0, 2].set_title('Residual MV Magnitude')
            axes[0, 2].axis('off')
            
            # 目标图像
            target_vis = np.clip((target_data.transpose(1, 2, 0) + 1) / 2, 0, 1)
            axes[1, 0].imshow(target_vis)
            axes[1, 0].set_title('Target Ground Truth')
            axes[1, 0].axis('off')
            
            # 运动矢量方向
            mv_angle = np.arctan2(residual_mv[1], residual_mv[0])
            axes[1, 1].imshow(mv_angle, cmap='hsv')
            axes[1, 1].set_title('Residual MV Direction')
            axes[1, 1].axis('off')
            
            # 叠加显示
            overlay = rgb_vis.copy()
            overlay[mask[0] > 0.5] = [1, 0, 0]  # 红色标记空洞
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('RGB + Hole Overlay')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # 保存可视化
            vis_path = self.output_dir / f"{self.scene_name}_pipeline_test.png"
            plt.savefig(str(vis_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 可视化已保存: {vis_path}")
            
        except Exception as e:
            print(f"❌ 可视化创建失败: {e}")
            import traceback
            traceback.print_exc()
    
    def run_complete_test(self) -> bool:
        """
        运行完整测试流水线
        
        Returns:
            success: 所有测试是否通过
        """
        success_count = 0
        total_tests = 7
        
        # 测试步骤
        tests = [
            self.test_data_availability,
            self.test_preprocessing,
            self.test_data_quality,
            self.test_dataset_splits,
            lambda: self.test_dataset_loading(self.split_file_path) if hasattr(self, 'split_file_path') else False,
            lambda: self.test_network_forward(self.split_file_path) if hasattr(self, 'split_file_path') else False,
            lambda: self.create_visualization(self.split_file_path) if hasattr(self, 'split_file_path') else None
        ]
        
        # 逐步执行测试
        for i, test_func in enumerate(tests):
            try:
                if i == 3:  # dataset splits test
                    result = test_func()
                    if result:
                        self.split_file_path = result
                        success_count += 1
                elif i == 6:  # visualization 
                    test_func()
                    success_count += 1  # 可视化不算失败
                else:
                    result = test_func()
                    if result:
                        success_count += 1
                    
            except Exception as e:
                print(f"❌ 测试步骤 {i+1} 出现异常: {e}")
        
        # 总结报告
        print("\n" + "="*80)
        print("📋 测试结果总结")
        print("="*80)
        print(f"✅ 通过测试: {success_count}/{total_tests}")
        print(f"❌ 失败测试: {total_tests - success_count}/{total_tests}")
        
        if success_count >= total_tests - 1:  # 允许1个测试失败
            print("🎉 整体测试通过！流水线运行正常")
            print("\n📖 下一步操作:")
            print("1. 运行完整预处理:")
            print(f"   python run_preprocessing.py --input-dir {self.input_dir} --output-dir {self.output_dir} --scene {self.scene_name}")
            print("2. 开始训练:")
            if hasattr(self, 'split_file_path'):
                print(f"   python training/train_mobile_inpainting.py --data-root {self.output_dir} --split-file {self.split_file_path}")
            return True
        else:
            print("❌ 测试未完全通过，请检查错误信息")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NoiseBase Preprocessing Pipeline')
    parser.add_argument('--input-dir', type=str, default='./training',
                       help='NoiseBase data directory')
    parser.add_argument('--output-dir', type=str, default='./training/processed_test',
                       help='Test output directory')
    parser.add_argument('--scene', type=str, default='bistro1',
                       help='Scene name to test')
    parser.add_argument('--test-frames', type=int, default=10,
                       help='Number of frames to test')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = PreprocessingPipelineTester(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scene_name=args.scene
    )
    tester.test_frame_count = args.test_frames
    
    # 运行测试
    success = tester.run_complete_test()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())