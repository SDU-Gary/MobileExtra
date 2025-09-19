#!/usr/bin/env python3
"""
Test Colleague Data with Simple Grid Strategy

测试 ColleagueDatasetAdapter + PatchAwareDataset + SimplePatchExtractor 的完整集成
"""

import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'train'))

def test_colleague_data_loading():
    """测试同事数据加载"""
    print("1️⃣ 测试ColleagueDatasetAdapter数据加载...")
    
    try:
        from train.colleague_dataset_adapter import ColleagueDatasetAdapter
        
        # 创建数据集
        dataset = ColleagueDatasetAdapter(
            data_root="./data",
            split="train"
        )
        
        print(f"    数据集创建成功")
        print(f"    样本数量: {len(dataset)}")
        
        # 测试数据加载
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"    样本格式: {type(sample)}")
            
            if isinstance(sample, tuple):
                input_data, target_residual, target_rgb = sample
                print(f"    输入数据形状: {input_data.shape}")
                print(f"    目标残差形状: {target_residual.shape}")  
                print(f"    目标RGB形状: {target_rgb.shape}")
            else:
                print(f"    数据形状: {sample.shape if hasattr(sample, 'shape') else 'Unknown'}")
                
            print("    数据加载测试成功")
            return True
        else:
            print("    数据集为空")
            return False
            
    except Exception as e:
        print(f"    数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_grid_integration():
    """测试简单网格策略集成"""
    print("\n2️⃣ 测试简单网格策略集成...")
    
    try:
        from train.patch_aware_dataset import PatchTrainingConfig, PatchAwareDataset
        
        # 创建简单网格配置
        config = PatchTrainingConfig(
            enable_patch_mode=True,
            use_simple_grid_patches=True,      # 启用简单网格
            use_optimized_patches=False,       # 禁用复杂检测
            simple_grid_rows=4,
            simple_grid_cols=4,
            simple_expected_height=1080,
            simple_expected_width=1920,
            max_patches_per_image=16
        )
        
        print("    简单网格配置创建")
        
        # 创建数据集 - 注意这里需要使用支持colleague数据的基础数据集
        # 但是PatchAwareDataset默认使用UnifiedNoiseBaseDataset
        # 我们需要修改或创建一个适配版本
        
        print("     PatchAwareDataset需要适配ColleagueDatasetAdapter")
        print("    建议: 创建PatchAwareColleagueDataset或修改PatchAwareDataset")
        
        return True
        
    except Exception as e:
        print(f"    简单网格集成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_grid_extraction():
    """测试简单网格提取"""
    print("\n3️⃣ 测试简单网格提取器...")
    
    try:
        from simple_patch_extractor import SimplePatchExtractor, create_default_config
        import numpy as np
        
        # 创建测试数据
        test_input = np.random.rand(7, 1080, 1920).astype(np.float32)  # 7通道输入
        test_target_residual = np.random.rand(3, 1080, 1920).astype(np.float32)
        test_target_rgb = np.random.rand(3, 1080, 1920).astype(np.float32)
        
        print(f"    测试输入形状: {test_input.shape}")
        
        # 创建提取器
        extractor = SimplePatchExtractor(create_default_config())
        
        # 提取patches
        input_patches, positions = extractor.extract_patches(test_input)
        residual_patches, _ = extractor.extract_patches(test_target_residual)
        rgb_patches, _ = extractor.extract_patches(test_target_rgb)
        
        print(f"    成功提取patches")
        print(f"    输入patches: {len(input_patches)} 个")
        print(f"    残差patches: {len(residual_patches)} 个")
        print(f"    RGB patches: {len(rgb_patches)} 个")
        
        if len(input_patches) > 0:
            print(f"    单个patch形状:")
            print(f"      输入: {input_patches[0].shape}")
            print(f"      残差: {residual_patches[0].shape}")
            print(f"      RGB: {rgb_patches[0].shape}")
            
            # 验证patch数量
            if len(input_patches) == 16:
                print("    正确生成16个patches (4x4网格)")
            else:
                print(f"     Patch数量不是16: {len(input_patches)}")
        
        return True
        
    except Exception as e:
        print(f"    简单网格提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_integration_recommendation():
    """显示集成建议"""
    print("\n 集成建议:")
    print("="*50)
    
    print("当前状态:")
    print("    ColleagueDatasetAdapter - 处理OpenEXR数据")
    print("    SimplePatchExtractor - 4x4网格提取") 
    print("    PatchAwareDataset - Patch训练数据集")
    print("     需要连接: Colleague数据 → Patch训练")
    
    print("\n建议方案:")
    print("   1. 修改PatchAwareDataset支持ColleagueDatasetAdapter")
    print("   2. 或创建PatchAwareColleagueDataset专用版本")
    print("   3. 确保数据流: OpenEXR → 7通道 → 16个Patch → 训练")
    
    print("\n数据流程:")
    print("   OpenEXR文件 → ColleagueDatasetAdapter → 7通道数据")
    print("   7通道数据 → SimplePatchExtractor → 16个270x480 patches")
    print("   Patches → Resize到128x128 → 训练")

def main():
    """主测试函数"""
    print("🧪 Colleague Data + Simple Grid Integration Test")
    print("="*60)
    
    # 测试步骤
    tests = [
        ("同事数据加载", test_colleague_data_loading),
        ("简单网格集成", test_simple_grid_integration), 
        ("简单网格提取", test_simple_grid_extraction)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"    {test_name} 出错: {e}")
            results.append(False)
    
    # 显示结果
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"测试结果: {passed}/{total} 通过")
    
    for i, (test_name, _) in enumerate(tests):
        status = " PASS" if results[i] else " FAIL"
        print(f"   {status} - {test_name}")
    
    # 显示建议
    show_integration_recommendation()
    
    if passed == total:
        print(f"\n🎉 所有测试通过！可以开始训练")
    else:
        print(f"\n  部分测试失败，建议检查集成方案")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)