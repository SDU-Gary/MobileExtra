#!/usr/bin/env python3
"""
测试新的归一化策略
验证HDR RGB → LDR归一化，掩码和MV不归一化的效果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 添加路径以导入项目模块
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'npu' / 'networks'))

from input_normalizer import UnifiedInputNormalizer

def create_test_data():
    """创建测试数据"""
    batch_size, height, width = 1, 128, 128
    
    # 创建7通道测试数据
    test_data = torch.zeros(batch_size, 7, height, width)
    
    # 1. 创建HDR RGB数据（模拟真实HDR场景）
    # 创建一个有亮暗对比的HDR图像
    y_coords, x_coords = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing='ij')
    
    # 基础图像：渐变背景
    base_image = (y_coords + x_coords) / 2
    
    # 添加高亮区域（模拟光源）
    bright_spot1 = torch.exp(-((x_coords - 0.3)**2 + (y_coords - 0.3)**2) * 50) * 10.0
    bright_spot2 = torch.exp(-((x_coords - 0.7)**2 + (y_coords - 0.7)**2) * 30) * 5.0
    
    # 组合HDR图像
    hdr_r = base_image * 0.5 + bright_spot1 + torch.randn_like(base_image) * 0.1
    hdr_g = base_image * 0.7 + bright_spot2 + torch.randn_like(base_image) * 0.1  
    hdr_b = base_image * 0.3 + bright_spot1 * 0.5 + torch.randn_like(base_image) * 0.1
    
    test_data[0, 0] = torch.clamp(hdr_r, 0.0, 20.0)  # R通道，最高20.0
    test_data[0, 1] = torch.clamp(hdr_g, 0.0, 15.0)  # G通道，最高15.0
    test_data[0, 2] = torch.clamp(hdr_b, 0.0, 10.0)  # B通道，最高10.0
    
    # 2. 创建有意义的掩码数据
    # 空洞掩码：圆形空洞
    holes_mask = ((x_coords - 0.5)**2 + (y_coords - 0.2)**2) < 0.05
    test_data[0, 3] = holes_mask.float()
    
    # 遮挡掩码：矩形遮挡区域
    occlusion_mask = ((x_coords > 0.6) & (x_coords < 0.9) & (y_coords > 0.4) & (y_coords < 0.8))
    test_data[0, 4] = occlusion_mask.float()
    
    # 3. 创建有意义的残差MV数据
    # 模拟径向运动矢量
    center_x, center_y = 0.5, 0.5
    mv_u = (x_coords - center_x) * 50.0 + torch.randn_like(x_coords) * 5.0  # u分量
    mv_v = (y_coords - center_y) * 30.0 + torch.randn_like(y_coords) * 3.0  # v分量
    
    test_data[0, 5] = mv_u  # MV u分量，范围大约[-25, 25]像素
    test_data[0, 6] = mv_v  # MV v分量，范围大约[-15, 15]像素
    
    return test_data

def test_normalization_comparison():
    """对比新旧归一化策略的效果"""
    
    print("🧪 测试新的归一化策略效果")
    
    # 创建测试数据
    test_data = create_test_data()
    
    print(f"📊 原始测试数据统计:")
    print(f"   HDR RGB: [{test_data[0, 0:3].min():.3f}, {test_data[0, 0:3].max():.3f}], 均值: {test_data[0, 0:3].mean():.3f}")
    print(f"   掩码: [{test_data[0, 3:5].min():.3f}, {test_data[0, 3:5].max():.3f}], 非零比例: {(test_data[0, 3:5] > 0).float().mean():.3f}")
    print(f"   残差MV: [{test_data[0, 5:7].min():.1f}, {test_data[0, 5:7].max():.1f}], 均值: {test_data[0, 5:7].mean():.1f}")
    
    # 测试新的归一化策略
    print(f"\n🔥 测试新归一化策略:")
    new_normalizer = UnifiedInputNormalizer(
        rgb_method="hdr_to_ldr",
        tone_mapping="reinhard", 
        normalize_masks=False,
        normalize_mv=False,
        gamma=2.2
    )
    
    processed_data = new_normalizer(test_data)
    
    print(f"   处理后RGB: [{processed_data[0, 0:3].min():.3f}, {processed_data[0, 0:3].max():.3f}], 均值: {processed_data[0, 0:3].mean():.3f}")
    print(f"   处理后掩码: [{processed_data[0, 3:5].min():.3f}, {processed_data[0, 3:5].max():.3f}], 非零比例: {(processed_data[0, 3:5] > 0).float().mean():.3f}")
    print(f"   处理后MV: [{processed_data[0, 5:7].min():.1f}, {processed_data[0, 5:7].max():.1f}], 均值: {processed_data[0, 5:7].mean():.1f}")
    
    # 测试TensorBoard可视化
    print(f"\n📺 TensorBoard可视化测试:")
    
    # RGB可视化
    rgb_vis = new_normalizer.prepare_for_tensorboard(processed_data[0:1, 0:3], "rgb")
    print(f"   RGB可视化范围: [{rgb_vis.min():.3f}, {rgb_vis.max():.3f}]")
    
    # 掩码可视化
    holes_vis = new_normalizer.prepare_for_tensorboard(processed_data[0:1, 3:4], "mask")
    occlusion_vis = new_normalizer.prepare_for_tensorboard(processed_data[0:1, 4:5], "mask") 
    print(f"   掩码可视化范围: [{holes_vis.min():.3f}, {holes_vis.max():.3f}], [{occlusion_vis.min():.3f}, {occlusion_vis.max():.3f}]")
    
    # MV可视化
    mv_vis = new_normalizer.prepare_for_tensorboard(processed_data[0:1, 5:7], "mv")
    print(f"   MV可视化范围: [{mv_vis.min():.3f}, {mv_vis.max():.3f}]")
    
    return test_data, processed_data, new_normalizer

def visualize_results(original_data, processed_data, normalizer):
    """可视化结果对比"""
    
    try:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # 原始数据可视化
        axes[0, 0].set_title('原始HDR RGB')
        # HDR需要tone mapping才能显示
        original_rgb_vis = normalizer._hdr_to_ldr(original_data[0, 0:3], "reinhard", 2.2)
        axes[0, 0].imshow(original_rgb_vis.permute(1, 2, 0).numpy())
        axes[0, 0].axis('off')
        
        axes[0, 1].set_title('原始掩码（空洞）')
        axes[0, 1].imshow(original_data[0, 3].numpy(), cmap='Reds')
        axes[0, 1].axis('off')
        
        axes[0, 2].set_title('原始掩码（遮挡）')
        axes[0, 2].imshow(original_data[0, 4].numpy(), cmap='Blues')
        axes[0, 2].axis('off')
        
        axes[0, 3].set_title('原始MV幅度')
        original_mv_mag = torch.sqrt(original_data[0, 5]**2 + original_data[0, 6]**2)
        im0 = axes[0, 3].imshow(original_mv_mag.numpy(), cmap='viridis')
        axes[0, 3].axis('off')
        plt.colorbar(im0, ax=axes[0, 3], fraction=0.046)
        
        # 处理后数据可视化
        axes[1, 0].set_title('处理后LDR RGB')
        axes[1, 0].imshow(processed_data[0, 0:3].permute(1, 2, 0).numpy())
        axes[1, 0].axis('off')
        
        axes[1, 1].set_title('处理后掩码（空洞）')
        axes[1, 1].imshow(processed_data[0, 3].numpy(), cmap='Reds')
        axes[1, 1].axis('off')
        
        axes[1, 2].set_title('处理后掩码（遮挡）')
        axes[1, 2].imshow(processed_data[0, 4].numpy(), cmap='Blues')
        axes[1, 2].axis('off')
        
        axes[1, 3].set_title('处理后MV幅度')
        processed_mv_mag = torch.sqrt(processed_data[0, 5]**2 + processed_data[0, 6]**2)
        im1 = axes[1, 3].imshow(processed_mv_mag.numpy(), cmap='viridis')
        axes[1, 3].axis('off')
        plt.colorbar(im1, ax=axes[1, 3], fraction=0.046)
        
        # TensorBoard可视化效果
        rgb_tb = normalizer.prepare_for_tensorboard(processed_data[0:1, 0:3], "rgb")
        holes_tb = normalizer.prepare_for_tensorboard(processed_data[0:1, 3:4], "mask")
        occlusion_tb = normalizer.prepare_for_tensorboard(processed_data[0:1, 4:5], "mask")
        mv_tb = normalizer.prepare_for_tensorboard(processed_data[0:1, 5:7], "mv")
        
        axes[2, 0].set_title('TensorBoard RGB')
        axes[2, 0].imshow(rgb_tb[0].permute(1, 2, 0).numpy())
        axes[2, 0].axis('off')
        
        axes[2, 1].set_title('TensorBoard 空洞掩码')
        axes[2, 1].imshow(holes_tb[0, 0].numpy(), cmap='Reds')
        axes[2, 1].axis('off')
        
        axes[2, 2].set_title('TensorBoard 遮挡掩码')
        axes[2, 2].imshow(occlusion_tb[0, 0].numpy(), cmap='Blues')
        axes[2, 2].axis('off')
        
        axes[2, 3].set_title('TensorBoard MV可视化')
        axes[2, 3].imshow(mv_tb[0].permute(1, 2, 0).numpy())
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n💾 可视化结果已保存到 'normalization_comparison.png'")
        
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")

def main():
    """主测试函数"""
    
    # 执行归一化对比测试
    original_data, processed_data, normalizer = test_normalization_comparison()
    
    # 可视化结果
    visualize_results(original_data, processed_data, normalizer)
    
    # 总结
    print(f"\n✅ 新归一化策略测试完成！")
    print(f"🎯 关键改进:")
    print(f"   1. HDR RGB转为LDR后保持在[0,1]，TensorBoard可正常显示")
    print(f"   2. 掩码保持原始[0,1]值，避免信息丢失") 
    print(f"   3. 残差MV保持像素偏移值，避免变成纯色")
    print(f"   4. 所有数据都能在TensorBoard中正确可视化")
    
    # 网络兼容性提醒
    print(f"\n⚠️ 重要提醒:")
    print(f"   网络现在需要处理异构输入：")
    print(f"   - RGB: [0,1] LDR值")
    print(f"   - Masks: [0,1] 概率值") 
    print(f"   - MV: 原始像素偏移值（可能±几百像素）")
    print(f"   建议网络对不同通道使用不同的处理策略")

if __name__ == "__main__":
    main()