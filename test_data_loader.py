#!/usr/bin/env python3
"""
测试NoiseBase数据加载器

用于验证数据加载器是否能正确读取NoiseBase数据
"""

import sys
from pathlib import Path

# 添加train目录到路径
sys.path.insert(0, str(Path(__file__).parent / "train"))

from noisebase_data_loader import NoiseBaseDataLoader


def test_data_loader():
    """测试数据加载器"""
    print("🚀 测试NoiseBase数据加载器...")
    
    # 提示用户输入数据路径
    print("\n请提供NoiseBase数据路径。")
    print("数据目录结构应该如下:")
    print("data/")
    print("├── bistro1/")
    print("│   ├── frame0000.zip")
    print("│   ├── frame0001.zip")
    print("│   └── ...")
    print("├── kitchen/")
    print("│   └── ...")
    print("")
    
    data_root = input("请输入NoiseBase数据根目录路径: ").strip()
    
    if not data_root:
        print("❌ 未指定数据路径")
        return False
    
    data_path = Path(data_root)
    if not data_path.exists():
        print(f"❌ 数据路径不存在: {data_path}")
        return False
    
    try:
        # 创建数据加载器
        loader = NoiseBaseDataLoader(str(data_path))
        
        # 列出可用场景
        scenes = loader.list_available_scenes()
        print(f"\n📋 发现场景: {scenes}")
        
        if not scenes:
            print("❌ 未找到任何场景数据")
            print("请检查数据目录结构是否正确")
            return False
        
        # 测试第一个场景
        test_scene = scenes[0]
        print(f"\n🎯 测试场景: {test_scene}")
        
        # 统计帧数
        frame_count = loader.count_frames(test_scene)
        print(f"📊 场景包含 {frame_count} 帧")
        
        if frame_count == 0:
            print("❌ 场景中没有找到帧数据")
            return False
        
        # 加载第一帧
        print(f"\n📂 加载第一帧数据...")
        frame_data = loader.load_frame_data(test_scene, 0)
        
        if frame_data is None:
            print("❌ 无法加载第一帧数据")
            return False
        
        print(f"✅ 成功加载帧数据!")
        print(f"   数据通道: {list(frame_data.keys())}")
        
        # 显示数据详情
        print(f"\n📊 数据详情:")
        for key, value in frame_data.items():
            if hasattr(value, 'shape'):
                print(f"   {key:15s}: shape={str(value.shape):15s} dtype={value.dtype}")
                if hasattr(value, 'min') and value.size > 0:
                    try:
                        min_val = float(value.min())
                        max_val = float(value.max())
                        print(f"   {'':<15s}  range=[{min_val:.3f}, {max_val:.3f}]")
                    except:
                        pass
            else:
                print(f"   {key:15s}: {value}")
        
        # 验证关键数据
        required_keys = ['reference', 'position']
        missing_keys = [key for key in required_keys if key not in frame_data]
        
        if missing_keys:
            print(f"\n⚠️ 缺少关键数据: {missing_keys}")
        else:
            print(f"\n✅ 包含所有关键数据")
        
        # 检查数据是否为花屏（随机数据）
        if 'reference' in frame_data:
            ref_data = frame_data['reference']
            if hasattr(ref_data, 'std'):
                std_val = float(ref_data.std())
                mean_val = float(ref_data.mean())
                print(f"\n🔍 参考图像统计:")
                print(f"   均值: {mean_val:.3f}")
                print(f"   标准差: {std_val:.3f}")
                
                # 简单的花屏检测
                if std_val > 0.5 and abs(mean_val) < 0.1:
                    print("⚠️ 数据可能是随机噪声（花屏）")
                else:
                    print("✅ 数据看起来正常")
        
        # 验证数据完整性
        print(f"\n🔍 验证数据完整性...")
        validation = loader.validate_data_integrity(test_scene, max_frames=3)
        
        if validation['valid_frames'] > 0:
            print(f"✅ 数据验证通过!")
            print(f"   有效帧数: {validation['valid_frames']}")
            print(f"   公共通道: {sorted(validation['common_channels'])}")
        else:
            print(f"❌ 数据验证失败")
            return False
        
        print(f"\n🎉 数据加载器测试成功!")
        print(f"\n📖 下一步可以运行:")
        print(f"python train/run_preprocessing_corrected.py \\")
        print(f"  --data-root {data_root} \\")
        print(f"  --scene {test_scene} \\")
        print(f"  --output-dir ./processed_data")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_data_loader()
    if not success:
        sys.exit(1)