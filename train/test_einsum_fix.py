#!/usr/bin/env python3
"""
测试einsum维度修复
"""

import numpy as np
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from projective import motion_vectors


def test_motion_vectors_dimensions():
    """测试运动矢量计算的维度处理"""
    print("🧪 测试运动矢量维度处理...")
    
    # 模拟数据
    H, W = 100, 100
    
    # 测试1: 3HWS格式 (原始格式)
    print("\n1. 测试3HWS格式 (原始):")
    w_position_4d = np.random.randn(3, H, W, 1)
    w_motion_4d = np.random.randn(3, H, W, 1)
    pv = np.random.randn(4, 4)
    prev_pv = np.random.randn(4, 4)
    
    try:
        result_4d = motion_vectors(w_position_4d, w_motion_4d, pv, prev_pv, H, W)
        print(f"   ✅ 成功: 输入{w_position_4d.shape} -> 输出{result_4d.shape}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    # 测试2: 3HW格式 (平均后格式) - 这个应该会失败
    print("\n2. 测试3HW格式 (平均后):")
    w_position_3d = np.random.randn(3, H, W)
    w_motion_3d = np.random.randn(3, H, W)
    
    try:
        result_3d = motion_vectors(w_position_3d, w_motion_3d, pv, prev_pv, H, W)
        print(f"   ✅ 成功: 输入{w_position_3d.shape} -> 输出{result_3d.shape}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    # 测试3: 手动添加维度
    print("\n3. 测试手动添加维度:")
    w_position_fixed = w_position_3d[..., np.newaxis]
    w_motion_fixed = w_motion_3d[..., np.newaxis]
    
    try:
        result_fixed = motion_vectors(w_position_fixed, w_motion_fixed, pv, prev_pv, H, W)
        print(f"   ✅ 成功: 输入{w_position_fixed.shape} -> 输出{result_fixed.shape}")
        
        # 移除样本维度
        if result_fixed.ndim == 4 and result_fixed.shape[3] == 1:
            result_final = result_fixed.squeeze(axis=3)
            print(f"   ✅ 最终: {result_final.shape}")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")


if __name__ == "__main__":
    test_motion_vectors_dimensions()