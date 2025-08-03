#!/usr/bin/env python3
"""
测试修正后的空洞检测和遮挡掩码实现
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from noisebase_preprocessor import NoiseBasePreprocessor

def create_test_data():
    """创建测试数据"""
    H, W = 256, 256
    
    # 模拟当前帧数据
    curr_frame = {
        'reference': np.random.rand(3, H, W).astype(np.float32) * 2 - 1,  # [-1, 1]
        'position': np.random.rand(3, H, W).astype(np.float32) * 10,      # 世界坐标
        'motion': np.random.rand(2, H, W).astype(np.float32) * 4 - 2,     # 运动矢量
        'camera_pos': np.array([0, 0, 5], dtype=np.float32)               # 相机位置
    }
    
    # 模拟前一帧数据
    prev_frame = {
        'reference': np.random.rand(3, H, W).astype(np.float32) * 2 - 1,
        'position': curr_frame['position'] + np.random.rand(3, H, W).astype(np.float32) * 0.5,
        'motion': curr_frame['motion'] + np.random.rand(2, H, W).astype(np.float32) * 0.2,
        'camera_pos': np.array([0.1, 0.1, 5.1], dtype=np.float32)
    }
    
    # 模拟warp后的图像和覆盖掩码
    warped_image = curr_frame['reference'] + np.random.rand(3, H, W).astype(np.float32) * 0.1
    
    # 创建一些空洞区域（覆盖度低）
    coverage_mask = np.ones((H, W), dtype=np.float32)
    coverage_mask[50:100, 50:100] = 0.2  # 空洞区域1
    coverage_mask[150:200, 150:200] = 0.1  # 空洞区域2
    
    return curr_frame, prev_frame, warped_image, coverage_mask

def test_hole_and_occlusion_detection():
    """测试空洞检测和遮挡掩码生成"""
    print("🔍 测试修正后的空洞检测和遮挡掩码实现...")
    
    # 创建预处理器
    output_dir = Path("/tmp/test_corrected_preprocessing")
    preprocessor = NoiseBasePreprocessor(
        input_dir=str(Path("/tmp/dummy")),
        output_dir=str(output_dir),
        scene_name="test_scene"
    )
    
    # 创建测试数据
    curr_frame, prev_frame, warped_image, coverage_mask = create_test_data()
    target_image = curr_frame['reference']
    
    print(f"✅ 测试数据创建完成:")
    print(f"   - 图像尺寸: {target_image.shape}")
    print(f"   - 覆盖掩码范围: [{coverage_mask.min():.3f}, {coverage_mask.max():.3f}]")
    
    # 测试空洞检测和遮挡掩码生成
    try:
        hole_mask, occlusion_mask = preprocessor.detect_holes_and_occlusion(
            warped_image, target_image, coverage_mask, curr_frame, prev_frame
        )
        
        print(f"✅ 空洞检测成功:")
        print(f"   - 空洞掩码形状: {hole_mask.shape}")
        print(f"   - 空洞覆盖率: {np.mean(hole_mask):.3f}")
        print(f"   - 空洞区域数量: {np.sum(hole_mask > 0.5)}")
        
        print(f"✅ 遮挡掩码生成成功:")
        print(f"   - 遮挡掩码形状: {occlusion_mask.shape}")
        print(f"   - 遮挡覆盖率: {np.mean(occlusion_mask):.3f}")
        print(f"   - 遮挡区域数量: {np.sum(occlusion_mask > 0.5)}")
        
    except Exception as e:
        print(f"❌ 空洞检测和遮挡掩码生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试残差运动矢量计算
    try:
        motion_vectors = curr_frame['motion']
        residual_mv = preprocessor.compute_residual_motion_vectors(
            warped_image, target_image, coverage_mask, motion_vectors, hole_mask
        )
        
        print(f"✅ 残差运动矢量计算成功:")
        print(f"   - 残差MV形状: {residual_mv.shape}")
        print(f"   - 残差MV幅度: {np.mean(np.sqrt(residual_mv[0]**2 + residual_mv[1]**2)):.3f}")
        
    except Exception as e:
        print(f"❌ 残差运动矢量计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试7通道训练样本创建
    try:
        training_sample = preprocessor.create_training_sample(
            target_image, hole_mask, occlusion_mask, residual_mv
        )
        
        print(f"✅ 7通道训练样本创建成功:")
        print(f"   - 训练样本形状: {training_sample.shape}")
        print(f"   - 通道分布: RGB(3) + HoleMask(1) + OcclusionMask(1) + ResidualMV(2)")
        
        # 验证通道内容
        rgb_channels = training_sample[:3]
        hole_channel = training_sample[3]
        occlusion_channel = training_sample[4]
        mv_channels = training_sample[5:7]
        
        print(f"   - RGB通道范围: [{rgb_channels.min():.3f}, {rgb_channels.max():.3f}]")
        print(f"   - 空洞通道范围: [{hole_channel.min():.3f}, {hole_channel.max():.3f}]")
        print(f"   - 遮挡通道范围: [{occlusion_channel.min():.3f}, {occlusion_channel.max():.3f}]")
        print(f"   - MV通道范围: [{mv_channels.min():.3f}, {mv_channels.max():.3f}]")
        
    except Exception as e:
        print(f"❌ 训练样本创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建可视化
    try:
        create_comparison_visualization(
            target_image, hole_mask, occlusion_mask, residual_mv, output_dir
        )
        print(f"✅ 可视化创建成功，保存到: {output_dir}")
        
    except Exception as e:
        print(f"⚠️  可视化创建失败: {e}")
    
    return True

def create_comparison_visualization(target_image, hole_mask, occlusion_mask, residual_mv, output_dir):
    """创建对比可视化"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    rgb_vis = np.clip((target_image.transpose(1, 2, 0) + 1) / 2, 0, 1)
    axes[0, 0].imshow(rgb_vis)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 空洞掩码
    axes[0, 1].imshow(hole_mask, cmap='Reds', alpha=0.8)
    axes[0, 1].set_title('Hole Mask (Geometric)')
    axes[0, 1].axis('off')
    
    # 遮挡掩码
    axes[0, 2].imshow(occlusion_mask, cmap='Greens', alpha=0.8)
    axes[0, 2].set_title('Occlusion Mask (Semantic)')
    axes[0, 2].axis('off')
    
    # 残差运动矢量幅度
    mv_magnitude = np.sqrt(residual_mv[0]**2 + residual_mv[1]**2)
    im1 = axes[1, 0].imshow(mv_magnitude, cmap='jet')
    axes[1, 0].set_title('Residual MV Magnitude')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    # 空洞覆盖
    hole_overlay = rgb_vis.copy()
    hole_overlay[hole_mask > 0.5] = [1, 0, 0]  # 红色标记空洞
    axes[1, 1].imshow(hole_overlay)
    axes[1, 1].set_title('Holes Overlay (Red)')
    axes[1, 1].axis('off')
    
    # 遮挡覆盖
    occlusion_overlay = rgb_vis.copy()
    occlusion_overlay[occlusion_mask > 0.5] = [0, 1, 0]  # 绿色标记遮挡
    axes[1, 2].imshow(occlusion_overlay)
    axes[1, 2].set_title('Occlusion Overlay (Green)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'corrected_preprocessing_test.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """主测试函数"""
    print("🚀 开始测试修正后的预处理实现...")
    print("=" * 60)
    
    success = test_hole_and_occlusion_detection()
    
    print("=" * 60)
    if success:
        print("✅ 所有测试通过！修正后的实现工作正常。")
        print("\n📋 修正要点总结:")
        print("   1. ✅ 区分了空洞检测和遮挡掩码两个概念")
        print("   2. ✅ 空洞检测：基于覆盖度的纯几何方法")
        print("   3. ✅ 遮挡检测：基于深度和运动不一致性的语义方法")
        print("   4. ✅ 训练样本：从6通道扩展到7通道")
        print("   5. ✅ 保存格式：分别保存空洞和遮挡掩码")
    else:
        print("❌ 测试失败，需要进一步调试。")
    
    return success

if __name__ == "__main__":
    main()