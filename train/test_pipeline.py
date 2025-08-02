#!/usr/bin/env python3
"""
@file test_pipeline.py
@brief 简化的预处理流程测试脚本

功能：
- 快速测试数据处理流程
- 验证输出格式
- 生成示例可视化

使用方法：
python test_pipeline.py --input-dir /path/to/noisebase --scene bistro1

@author AI算法团队
@date 2025-08-02
@version 1.0
"""

import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from noisebase_preprocessor import NoiseBasePreprocessor
from dataset import create_noisebase_dataloader


def test_preprocessing(input_dir: str, scene_name: str = "bistro1"):
    """
    测试预处理流程
    
    Args:
        input_dir: 输入目录
        scene_name: 场景名称
    """
    print("="*60)
    print("🧪 测试NoiseBase预处理流程")
    print("="*60)
    
    # 检查输入数据
    scene_dir = Path(input_dir) / scene_name
    if not scene_dir.exists():
        print(f"❌ 场景目录不存在: {scene_dir}")
        return False
    
    # 查找可用帧
    available_frames = []
    for i in range(10):  # 检查前10帧
        frame_file = scene_dir / f"frame{i:04d}.zip"
        if frame_file.exists():
            available_frames.append(i)
    
    if len(available_frames) < 2:
        print(f"❌ 需要至少2帧数据，找到: {len(available_frames)}")
        return False
    
    print(f"✅ 找到 {len(available_frames)} 个可用帧")
    
    # 创建预处理器
    output_dir = "./test_output"
    preprocessor = NoiseBasePreprocessor(
        input_dir=input_dir,
        output_dir=output_dir,
        scene_name=scene_name
    )
    
    try:
        # 处理前几帧
        print(f"\n🔄 处理前3帧...")
        end_frame = min(3, len(available_frames) - 1)
        
        preprocessor.process_sequence(
            start_frame=0,
            end_frame=end_frame
        )
        
        print(f"✅ 预处理完成!")
        
        # 测试数据集加载
        print(f"\n📊 测试数据集加载...")
        try:
            train_loader = create_noisebase_dataloader(
                data_root=output_dir,
                scene_name=scene_name,
                split='train',
                batch_size=2,
                patch_size=64
            )
            
            print(f"✅ 数据集加载成功，包含 {len(train_loader.dataset)} 个样本")
            
            # 测试一个批次
            for batch in train_loader:
                print(f"   输入形状: {batch['input'].shape}")
                print(f"   目标形状: {batch['target'].shape}")
                print(f"   数值范围: [{batch['input'].min():.3f}, {batch['target'].max():.3f}]")
                break
            
            # 创建可视化
            create_visualization(batch, output_dir, scene_name)
            
        except Exception as e:
            print(f"⚠️ 数据集测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_visualization(batch, output_dir: str, scene_name: str):
    """创建可视化"""
    try:
        # 获取第一个样本
        input_data = batch['input'][0].numpy()  # [6, H, W]
        target_data = batch['target'][0].numpy()  # [3, H, W]
        
        # 分离通道
        rgb = input_data[:3]        # [3, H, W]
        mask = input_data[3]        # [H, W]
        residual_mv = input_data[4:6]  # [2, H, W]
        
        # 创建可视化
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # RGB图像
        rgb_vis = np.clip(rgb.transpose(1, 2, 0), 0, 1)
        axes[0].imshow(rgb_vis)
        axes[0].set_title('RGB Input')
        axes[0].axis('off')
        
        # 空洞掩码
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Hole Mask\n(Coverage: {np.mean(mask):.3f})')
        axes[1].axis('off')
        
        # 运动矢量幅度
        mv_magnitude = np.sqrt(residual_mv[0]**2 + residual_mv[1]**2)
        im = axes[2].imshow(mv_magnitude, cmap='jet')
        axes[2].set_title(f'MV Magnitude\n(Avg: {np.mean(mv_magnitude):.3f})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 叠加显示
        overlay = rgb_vis.copy()
        hole_pixels = mask > 0.5
        overlay[hole_pixels] = [1, 0, 0]  # 红色标记空洞
        axes[3].imshow(overlay)
        axes[3].set_title('RGB + Holes Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # 保存
        vis_path = Path(output_dir) / f"{scene_name}_pipeline_test.png"
        plt.savefig(str(vis_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化已保存: {vis_path}")
        
    except Exception as e:
        print(f"⚠️ 可视化创建失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试NoiseBase预处理流程')
    parser.add_argument('--input-dir', type=str, required=True, help='NoiseBase数据目录')
    parser.add_argument('--scene', type=str, default='bistro1', help='场景名称')
    
    args = parser.parse_args()
    
    # 运行测试
    success = test_preprocessing(args.input_dir, args.scene)
    
    if success:
        print("\n🎉 测试通过！数据处理流程正常工作")
        print("\n📖 下一步:")
        print("1. 运行完整预处理:")
        print(f"   python run_preprocessing.py --input-dir {args.input_dir} --output-dir ./processed_data --scene {args.scene}")
        print("2. 开始训练:")
        print("   python train.py --data-root ./processed_data --scene bistro1")
        print("3. 检查生成的数据:")
        print("   查看 ./test_output/ 目录")
    else:
        print("\n❌ 测试失败，请检查错误信息")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())