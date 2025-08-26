#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速TensorBoard修复验证
专门测试 denormalize_output 方法调用问题
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_method_exists():
    """测试方法是否存在"""
    try:
        from src.npu.networks.input_normalizer import UnifiedInputNormalizer
        
        normalizer = UnifiedInputNormalizer()
        
        # 检查关键方法是否存在
        methods_to_check = [
            'denormalize_output',
            'prepare_for_tensorboard', 
            'hdr_to_ldr_for_display'
        ]
        
        for method_name in methods_to_check:
            if hasattr(normalizer, method_name):
                print(f"✅ 方法存在: {method_name}")
            else:
                print(f"❌ 方法缺失: {method_name}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_problematic_call():
    """测试之前出错的具体调用"""
    try:
        from src.npu.networks.input_normalizer import UnifiedInputNormalizer
        
        normalizer = UnifiedInputNormalizer()
        
        # 创建模拟的网络输出数据
        network_output = torch.randn(1, 3, 32, 32) * 0.5  # [-0.5, 0.5] 范围
        
        # 这是之前出错的调用
        result = normalizer.prepare_for_tensorboard(network_output, data_type="rgb", is_normalized=True)
        
        print(f"✅ 问题调用成功执行:")
        print(f"   输入范围: [{network_output.min():.3f}, {network_output.max():.3f}]")
        print(f"   输出范围: [{result.min():.3f}, {result.max():.3f}]")
        print(f"   输出形状: {result.shape}")
        
        # 验证输出在合理范围内
        if 0.0 <= result.min() and result.max() <= 1.0:
            print("✅ 输出范围验证通过 [0,1]")
            return True
        else:
            print(f"⚠️ 输出范围异常: [{result.min():.3f}, {result.max():.3f}]")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 快速TensorBoard修复验证")
    print("=" * 40)
    
    # 1. 检查方法存在
    methods_ok = test_method_exists()
    
    # 2. 测试具体的问题调用
    if methods_ok:
        call_ok = test_problematic_call()
    else:
        call_ok = False
    
    print("\n" + "=" * 40)
    if methods_ok and call_ok:
        print("✅ 修复成功！TensorBoard 调用问题已解决。")
        print("✅ 训练脚本现在应该能正常运行图像记录功能。")
    else:
        print("❌ 仍有问题，需要进一步检查。")
        
    print(f"\n🔧 修复详情:")
    print(f"   原错误: 'UnifiedInputNormalizer' object has no attribute 'denormalize_network_output'")
    print(f"   问题原因: 方法名不匹配")
    print(f"   修复方案: denormalize_network_output() → denormalize_output()")