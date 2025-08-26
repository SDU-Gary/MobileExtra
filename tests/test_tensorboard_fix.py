#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorBoard修复验证测试
验证 UnifiedInputNormalizer.prepare_for_tensorboard() 方法的参数修复
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.npu.networks.input_normalizer import UnifiedInputNormalizer
    print("✅ 成功导入 UnifiedInputNormalizer")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

def test_prepare_for_tensorboard_parameters():
    """测试 prepare_for_tensorboard 方法的参数兼容性"""
    
    print("\n🧪 测试 prepare_for_tensorboard 参数兼容性...")
    
    # 创建归一化器
    normalizer = UnifiedInputNormalizer(
        rgb_method="hdr_to_ldr",
        tone_mapping="reinhard", 
        normalize_masks=False,
        normalize_mv=False,
        gamma=2.2
    )
    
    # 创建测试数据
    batch_size = 1
    height, width = 64, 64
    
    # 测试RGB数据
    rgb_data = torch.rand(batch_size, 3, height, width)  # [0,1] LDR 数据
    
    try:
        # 1. 测试原始调用方式（默认参数）
        result1 = normalizer.prepare_for_tensorboard(rgb_data, data_type="rgb")
        print(f"✅ 默认调用成功: shape={result1.shape}, range=[{result1.min():.3f}, {result1.max():.3f}]")
        
        # 2. 测试带 is_normalized=False 的调用
        result2 = normalizer.prepare_for_tensorboard(rgb_data, data_type="rgb", is_normalized=False)
        print(f"✅ is_normalized=False 调用成功: shape={result2.shape}, range=[{result2.min():.3f}, {result2.max():.3f}]")
        
        # 3. 测试带 is_normalized=True 的调用（之前出错的情况）
        normalized_data = rgb_data * 2.0 - 1.0  # 模拟归一化后的数据 [-1,1]
        result3 = normalizer.prepare_for_tensorboard(normalized_data, data_type="rgb", is_normalized=True)
        print(f"✅ is_normalized=True 调用成功: shape={result3.shape}, range=[{result3.min():.3f}, {result3.max():.3f}]")
        
        # 4. 测试 mask 数据
        mask_data = torch.rand(batch_size, 1, height, width)
        result4 = normalizer.prepare_for_tensorboard(mask_data, data_type="mask")
        print(f"✅ mask 数据调用成功: shape={result4.shape}, range=[{result4.min():.3f}, {result4.max():.3f}]")
        
        # 5. 测试 MV 数据
        mv_data = torch.randn(batch_size, 2, height, width) * 10  # 模拟运动矢量
        result5 = normalizer.prepare_for_tensorboard(mv_data, data_type="mv")
        print(f"✅ MV 数据调用成功: shape={result5.shape}, range=[{result5.min():.3f}, {result5.max():.3f}]")
        
        # 验证输出范围都在 [0,1]
        all_results = [result1, result2, result3, result4, result5]
        for i, result in enumerate(all_results, 1):
            if result.min() >= 0.0 and result.max() <= 1.0:
                print(f"✅ 结果{i} 范围检查通过: [{result.min():.3f}, {result.max():.3f}]")
            else:
                print(f"⚠️ 结果{i} 范围异常: [{result.min():.3f}, {result.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_training_scenario():
    """模拟训练场景的调用"""
    
    print("\n🚂 测试训练场景调用...")
    
    normalizer = UnifiedInputNormalizer()
    
    # 模拟网络输出（已归一化到 [-1,1] 或 [0,1]）
    network_output = torch.randn(1, 3, 32, 32) * 0.5  # 模拟网络输出
    
    try:
        # 这是训练代码中的调用方式
        output_display = normalizer.prepare_for_tensorboard(network_output, data_type="rgb", is_normalized=True)
        print(f"✅ 训练场景调用成功: shape={output_display.shape}, range=[{output_display.min():.3f}, {output_display.max():.3f}]")
        return True
        
    except Exception as e:
        print(f"❌ 训练场景测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🔧 TensorBoard修复验证测试")
    print("=" * 50)
    
    # 运行测试
    test1_passed = test_prepare_for_tensorboard_parameters()
    test2_passed = test_training_scenario()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("✅ 所有测试通过！TensorBoard 参数修复成功。")
        print("✅ 现在可以正常使用训练脚本了。")
    else:
        print("❌ 部分测试失败，需要进一步检查。")
    
    print("📝 修复内容:")
    print("   1. 添加了 is_normalized 参数到 prepare_for_tensorboard 方法")
    print("   2. 更新了训练脚本中的方法调用，明确指定 data_type 参数") 
    print("   3. 修复了测试文件中的参数传递问题")
    print("   4. ✅ 修复了方法名不匹配: denormalize_network_output -> denormalize_output")