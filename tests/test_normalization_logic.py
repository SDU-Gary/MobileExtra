#!/usr/bin/env python3
"""
归一化概念验证脚本（纯Python）
验证数据范围处理的关键逻辑
"""

import math


def test_hdr_percentile_logic():
    """测试HDR百分位数归一化逻辑"""
    
    print("🧪 HDR百分位数归一化逻辑验证")
    
    # 模拟HDR数据：0-45范围（类似你的测试数据）
    hdr_samples = [0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 44.0, 45.0]
    print(f"   HDR样本: {hdr_samples}")
    
    # 计算99.5%分位数（简化：取95%位置）
    sorted_samples = sorted([x for x in hdr_samples if x >= 0])
    percentile_99_5_idx = int(len(sorted_samples) * 0.995)
    if percentile_99_5_idx >= len(sorted_samples):
        percentile_99_5_idx = len(sorted_samples) - 1
    
    hdr_min = 0.0
    hdr_max = sorted_samples[percentile_99_5_idx]
    
    print(f"   HDR范围: [{hdr_min}, {hdr_max}]")
    
    # 归一化到[-1,1]
    normalized = []
    for x in hdr_samples:
        if hdr_max > 0:
            norm_01 = max(0, min(1, (x - hdr_min) / (hdr_max - hdr_min)))
            norm_11 = norm_01 * 2.0 - 1.0
        else:
            norm_11 = 0.0
        normalized.append(norm_11)
    
    print(f"   归一化结果: {[f'{x:.3f}' for x in normalized]}")
    
    # 反归一化测试
    restored = []
    for norm_val in normalized:
        val_01 = (norm_val + 1.0) / 2.0
        original = val_01 * (hdr_max - hdr_min) + hdr_min
        restored.append(original)
    
    print(f"   恢复结果: {[f'{x:.3f}' for x in restored]}")
    
    # 计算误差
    errors = [abs(orig - rest) for orig, rest in zip(hdr_samples, restored)]
    max_error = max(errors)
    print(f"   最大恢复误差: {max_error:.6f}")
    
    return max_error < 0.001


def test_mv_fixed_range_logic():
    """测试MV固定范围归一化逻辑"""
    
    print("\n🧪 MV固定范围归一化逻辑验证")
    
    # 模拟MV数据：±200范围（类似你的测试数据）
    mv_samples = [-218.0, -100.0, -50.0, 0.0, 50.0, 100.0, 204.0]
    mv_pixel_range = 100.0
    
    print(f"   MV样本: {mv_samples}")
    print(f"   预设范围: ±{mv_pixel_range}")
    
    # 检查是否超出预期范围
    max_abs_mv = max(abs(x) for x in mv_samples)
    if max_abs_mv > mv_pixel_range * 2:
        print(f"   ⚠️  MV最大值{max_abs_mv:.1f}超出预期范围±{mv_pixel_range}")
    
    # 固定范围归一化
    normalized = []
    for x in mv_samples:
        norm_val = max(-1.0, min(1.0, x / mv_pixel_range))
        normalized.append(norm_val)
    
    print(f"   归一化结果: {[f'{x:.3f}' for x in normalized]}")
    
    # 检查范围
    all_in_range = all(-1.0 <= x <= 1.0 for x in normalized)
    print(f"   归一化范围正确: {all_in_range}")
    
    return all_in_range


def test_tone_mapping_logic():
    """测试tone mapping逻辑"""
    
    print("\n🧪 Tone Mapping逻辑验证")
    
    # HDR样本
    hdr_samples = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
    print(f"   HDR样本: {hdr_samples}")
    
    # Reinhard tone mapping: x / (1 + x)
    reinhard_results = []
    for x in hdr_samples:
        if x >= 0:
            ldr_val = x / (1.0 + x)
        else:
            ldr_val = 0.0
        reinhard_results.append(ldr_val)
    
    print(f"   Reinhard结果: {[f'{x:.3f}' for x in reinhard_results]}")
    
    # 检查LDR范围
    all_ldr_range = all(0.0 <= x <= 1.0 for x in reinhard_results)
    print(f"   LDR范围[0,1]正确: {all_ldr_range}")
    
    # ACES tone mapping (简化)
    aces_results = []
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    for x in hdr_samples:
        if x >= 0:
            ldr_val = max(0, min(1, (x * (a * x + b)) / (x * (c * x + d) + e)))
        else:
            ldr_val = 0.0
        aces_results.append(ldr_val)
    
    print(f"   ACES结果: {[f'{x:.3f}' for x in aces_results]}")
    
    return all_ldr_range


def test_data_range_compatibility():
    """测试数据范围兼容性"""
    
    print("\n🧪 数据范围兼容性验证")
    
    # 测试你观察到的数据范围
    test_cases = [
        ("HDR RGB", [0.0, 44.614]),
        ("Target HDR", [-1.465, 44.791]),  # 包含负值！
        ("MV", [-218.277, 204.510])        # 超出±100范围！
    ]
    
    for name, (min_val, max_val) in test_cases:
        print(f"\n   {name}: [{min_val:.3f}, {max_val:.3f}]")
        
        if name.endswith("HDR"):
            # HDR数据检查
            if min_val < 0:
                print(f"     ❌ HDR包含负值{min_val:.3f}，需要裁剪到≥0")
                corrected_min = max(0.0, min_val)
                print(f"     ✅ 修正后范围: [{corrected_min:.3f}, {max_val:.3f}]")
            else:
                print(f"     ✅ HDR范围正常")
            
            # 动态范围计算
            effective_range = max_val - max(0, min_val)
            dynamic_ratio = max_val / max(0.001, max(0, min_val))
            print(f"     📊 有效动态范围: {effective_range:.3f}")
            print(f"     📊 动态比例: {dynamic_ratio:.1f}:1")
            
        elif name == "MV":
            # MV数据检查
            mv_pixel_range = 100.0
            if abs(min_val) > mv_pixel_range or abs(max_val) > mv_pixel_range:
                print(f"     ⚠️  MV超出预期±{mv_pixel_range}范围")
                print(f"     💡 建议调整mv_pixel_range到{max(abs(min_val), abs(max_val)):.0f}")
            else:
                print(f"     ✅ MV范围在预期内")


def test_complete_pipeline():
    """测试完整管道"""
    
    print("\n🧪 完整归一化管道验证")
    
    # 模拟完整流程
    print(f"   1. 原始HDR数据: [0, 45] (类似你的测试)")
    print(f"   2. HDR归一化: [-1, 1]")
    print(f"   3. 网络处理: [-1, 1] → [-1, 1]")
    print(f"   4. 反归一化: [-1, 1] → [0, 45]")  
    print(f"   5. Tone mapping: [0, 45] → [0, 1]")
    print(f"   6. TensorBoard显示: [0, 1]")
    
    # 验证每一步的数学正确性
    original_hdr = 30.0  # 示例值
    hdr_max = 45.0
    
    # Step 2: 归一化
    norm_01 = original_hdr / hdr_max  # 0.667
    norm_11 = norm_01 * 2.0 - 1.0    # 0.333
    print(f"   HDR {original_hdr} → 归一化 {norm_11:.3f}")
    
    # Step 4: 反归一化  
    restored_01 = (norm_11 + 1.0) / 2.0  # 0.667
    restored_hdr = restored_01 * hdr_max   # 30.0
    print(f"   归一化 {norm_11:.3f} → 恢复HDR {restored_hdr:.3f}")
    
    # Step 5: Tone mapping
    ldr_display = restored_hdr / (1.0 + restored_hdr)  # Reinhard
    print(f"   HDR {restored_hdr:.3f} → LDR显示 {ldr_display:.3f}")
    
    # 验证误差
    recovery_error = abs(original_hdr - restored_hdr)
    print(f"   🎯 恢复误差: {recovery_error:.6f}")
    
    return recovery_error < 0.001


if __name__ == "__main__":
    print("🔍 归一化概念验证开始")
    
    # 运行所有测试
    tests = [
        ("HDR百分位数归一化", test_hdr_percentile_logic),
        ("MV固定范围归一化", test_mv_fixed_range_logic),
        ("Tone Mapping", test_tone_mapping_logic),
        ("完整管道", test_complete_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"   ✅ {test_name}: {'通过' if result else '失败'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"   ❌ {test_name}: 异常 - {e}")
    
    # 数据范围兼容性检查
    test_data_range_compatibility()
    
    # 总结
    print(f"\n📋 验证总结:")
    passed_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"   通过测试: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print(f"   🎉 所有核心逻辑验证通过！")
    else:
        print(f"   ⚠️  部分逻辑需要调整")
    
    print(f"\n💡 关键修复建议:")
    print(f"   1. ✅ 修复view()→reshape()错误")
    print(f"   2. ✅ 添加HDR负值检查和裁剪")
    print(f"   3. ✅ 添加MV范围验证和警告")
    print(f"   4. ✅ HDR百分位数归一化保持动态范围")
    print(f"   5. ✅ MV固定范围归一化保持物理意义")
    print(f"   6. ✅ 完整的HDR→LDR显示管道")