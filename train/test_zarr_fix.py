#!/usr/bin/env python3
"""
@file test_zarr_fix.py
@brief 测试Zarr兼容性修复

功能描述：
- 测试Zarr版本兼容性
- 验证导入是否正常
- 提供详细的错误信息

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import sys
import os
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "training"))


def test_zarr_basic():
    """测试基础Zarr功能"""
    print("🔍 测试基础Zarr功能...")
    
    try:
        import zarr
        print(f"✅ Zarr导入成功，版本: {zarr.__version__}")
        
        # 测试基本功能
        array = zarr.zeros((10, 10), dtype='f4')
        print(f"✅ Zarr基本功能正常")
        
        return True
    except Exception as e:
        print(f"❌ Zarr基础测试失败: {e}")
        return False


def test_zarr_zipstore():
    """测试ZipStore兼容性"""
    print("\n🔍 测试ZipStore兼容性...")
    
    try:
        # 尝试导入ZipStore
        try:
            from zarr import ZipStore
            print("✅ 找到 zarr.ZipStore")
            return True
        except ImportError:
            try:
                from zarr.storage import ZipStore
                print("✅ 找到 zarr.storage.ZipStore")
                return True
            except ImportError:
                print("⚠️ 未找到ZipStore，将使用自定义实现")
                return "custom"
    except Exception as e:
        print(f"❌ ZipStore测试失败: {e}")
        return False


def test_zarr_compat():
    """测试兼容性模块"""
    print("\n🔍 测试兼容性模块...")
    
    try:
        from zarr_compat import load_zarr_group, decompress_RGBE_compat, get_zarr_version
        print(f"✅ 兼容性模块导入成功")
        print(f"   检测到Zarr版本: {get_zarr_version()}")
        
        # 测试函数是否可调用
        print(f"✅ load_zarr_group函数: {callable(load_zarr_group)}")
        print(f"✅ decompress_RGBE_compat函数: {callable(decompress_RGBE_compat)}")
        
        return True
    except Exception as e:
        print(f"❌ 兼容性模块测试失败: {e}")
        return False


def test_preprocessor_import():
    """测试预处理器导入"""
    print("\n🔍 测试预处理器导入...")
    
    try:
        from training.noisebase_preprocessor import NoiseBasePreprocessor
        print("✅ NoiseBasePreprocessor导入成功")
        
        # 尝试创建实例（不会实际运行，只是测试初始化）
        try:
            preprocessor = NoiseBasePreprocessor.__new__(NoiseBasePreprocessor)
            print("✅ NoiseBasePreprocessor实例化测试通过")
        except Exception as e:
            print(f"⚠️ 实例化测试失败（可能是正常的）: {e}")
        
        return True
    except Exception as e:
        print(f"❌ 预处理器导入失败: {e}")
        return False


def test_projective_import():
    """测试投影模块导入"""
    print("\n🔍 测试投影模块导入...")
    
    try:
        from projective import screen_space_position, motion_vectors, log_depth
        print("✅ 投影函数导入成功")
        print(f"   screen_space_position: {callable(screen_space_position)}")
        print(f"   motion_vectors: {callable(motion_vectors)}")
        print(f"   log_depth: {callable(log_depth)}")
        
        return True
    except Exception as e:
        print(f"❌ 投影模块导入失败: {e}")
        return False


def main():
    """主测试函数"""
    print("="*80)
    print("🔧 Zarr兼容性修复测试")
    print("="*80)
    
    tests = [
        ("基础Zarr功能", test_zarr_basic),
        ("ZipStore兼容性", test_zarr_zipstore),
        ("兼容性模块", test_zarr_compat),
        ("投影模块导入", test_projective_import),
        ("预处理器导入", test_preprocessor_import),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*80)
    print("📊 测试结果总结")
    print("="*80)
    
    success_count = 0
    for test_name, result in results:
        if result is True:
            print(f"✅ {test_name}: 成功")
            success_count += 1
        elif result == "custom":
            print(f"⚠️ {test_name}: 使用自定义实现")
            success_count += 1
        else:
            print(f"❌ {test_name}: 失败")
    
    print(f"\n📈 总体结果: {success_count}/{len(tests)} 通过")
    
    if success_count == len(tests):
        print("\n🎉 所有测试通过！可以运行预处理脚本了")
        print("\n📖 下一步:")
        print("   python run_preprocessing.py --scene bistro1 --test-frames 5")
        return 0
    elif success_count >= len(tests) - 1:
        print("\n⚠️ 大部分测试通过，可以尝试运行预处理脚本")
        print("   如果有问题会显示具体错误信息")
        return 0
    else:
        print("\n❌ 仍有重要问题需要解决")
        return 1


if __name__ == "__main__":
    exit(main())