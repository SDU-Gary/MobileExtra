#!/usr/bin/env python3
"""
统一NoiseBase预处理器快速开始脚本
自动检测环境、验证数据、运行预处理

作者：AI算法团队
日期：2025-08-03
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def check_environment():
    """检查环境和依赖"""
    print("🔍 检查环境和依赖...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 7):
        print(f"❌ Python版本过低: {python_version}, 需要Python 3.7+")
        return False
    
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查依赖库
    required_packages = {
        'numpy': 'numpy',
        'cv2': 'opencv-python', 
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm'
    }
    
    optional_packages = {
        'zarr': 'zarr',
        'numba': 'numba'
    }
    
    missing_required = []
    missing_optional = []
    
    # 检查必需依赖
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}: 已安装")
        except ImportError:
            missing_required.append(package_name)
            print(f"❌ {package_name}: 未安装")
    
    # 检查可选依赖
    for import_name, package_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}: 已安装")
        except ImportError:
            missing_optional.append(package_name)
            print(f"⚠️ {package_name}: 未安装（可选）")
    
    # 安装缺失的依赖
    if missing_required:
        print(f"\n❌ 缺少必需依赖: {', '.join(missing_required)}")
        if input("是否自动安装? (y/N): ").lower().startswith('y'):
            try:
                cmd = [sys.executable, '-m', 'pip', 'install'] + missing_required
                subprocess.run(cmd, check=True)
                print("✅ 必需依赖安装完成")
            except subprocess.CalledProcessError:
                print("❌ 自动安装失败，请手动执行:")
                print(f"pip install {' '.join(missing_required)}")
                return False
        else:
            return False
    
    if missing_optional:
        print(f"\n⚠️ 缺少可选依赖: {', '.join(missing_optional)}")
        print("建议安装以获得更好的性能和兼容性:")
        print(f"pip install {' '.join(missing_optional)}")
    
    return True


def detect_data_structure(data_root):
    """检测和验证数据结构"""
    print(f"\n📂 检测数据结构: {data_root}")
    
    data_path = Path(data_root)
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_path}")
        return None
    
    # 寻找场景目录
    scene_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not scene_dirs:
        print(f"❌ 未找到场景目录")
        return None
    
    print(f"✅ 发现 {len(scene_dirs)} 个场景目录:")
    
    scene_info = {}
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        
        # 统计帧文件
        frame_files = list(scene_dir.glob('frame*.zip'))
        if frame_files:
            frame_count = len(frame_files)
            first_frame = min(int(f.stem.replace('frame', '')) for f in frame_files)
            last_frame = max(int(f.stem.replace('frame', '')) for f in frame_files)
            
            print(f"   📁 {scene_name}: {frame_count} 帧 (frame{first_frame:04d} - frame{last_frame:04d})")
            
            scene_info[scene_name] = {
                'frame_count': frame_count,
                'first_frame': first_frame,
                'last_frame': last_frame,
                'frame_files': frame_files
            }
        else:
            print(f"   📁 {scene_name}: 无帧数据")
    
    if not scene_info:
        print(f"❌ 所有场景目录都没有帧数据")
        return None
    
    return scene_info


def quick_test(data_root, scene_name):
    """快速测试数据加载"""
    print(f"\n🧪 快速测试数据加载...")
    
    try:
        # 动态导入统一预处理器
        sys.path.insert(0, str(Path(__file__).parent))
        from unified_noisebase_preprocessor import UnifiedNoiseBasePreprocessor
        
        # 创建临时预处理器实例
        preprocessor = UnifiedNoiseBasePreprocessor(
            data_root=data_root,
            output_dir="/tmp/quick_test_output",
            scene_name=scene_name
        )
        
        # 尝试加载第一帧
        frame_data = preprocessor.load_frame_data(scene_name, 0)
        if frame_data is None:
            print(f"❌ 无法加载场景 {scene_name} 的第一帧")
            return False
        
        print(f"✅ 数据加载测试成功!")
        print(f"   场景: {scene_name}")
        
        for key, value in frame_data.items():
            if hasattr(value, 'shape'):
                print(f"   {key}: {value.shape}")
            else:
                print(f"   {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False


def generate_run_command(data_root, output_dir, scene_name, config):
    """生成运行命令"""
    script_path = Path(__file__).parent / "unified_noisebase_preprocessor.py"
    
    cmd_parts = [
        "python",
        str(script_path),
        f"--data-root {data_root}",
        f"--output {output_dir}",
        f"--scene {scene_name}"
    ]
    
    if config.get('max_frames'):
        cmd_parts.append(f"--max-frames {config['max_frames']}")
    
    if config.get('test_mode'):
        cmd_parts.append("--test-mode")
    
    if config.get('use_numba'):
        cmd_parts.append("--use-numba")
    
    if config.get('hole_threshold', 0.3) != 0.3:
        cmd_parts.append(f"--hole-threshold {config['hole_threshold']}")
    
    return " ".join(cmd_parts)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NoiseBase预处理器快速开始')
    parser.add_argument('--data-root', type=str, 
                       help='NoiseBase数据根目录')
    parser.add_argument('--output', type=str,
                       help='输出目录')
    parser.add_argument('--scene', type=str,
                       help='场景名称')
    parser.add_argument('--auto', action='store_true',
                       help='自动模式（跳过交互）')
    parser.add_argument('--test-only', action='store_true',
                       help='只运行测试，不进行实际处理')
    
    args = parser.parse_args()
    
    print("🚀 NoiseBase预处理器快速开始")
    print("=" * 50)
    
    # 1. 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请解决依赖问题后重试")
        sys.exit(1)
    
    # 2. 获取数据目录
    if args.data_root:
        data_root = args.data_root
    else:
        if args.auto:
            print("❌ 自动模式需要指定 --data-root")
            sys.exit(1)
        
        print(f"\n📂 请指定NoiseBase数据目录")
        print(f"数据目录应包含如下结构:")
        print(f"data/")
        print(f"├── bistro1/")
        print(f"│   ├── frame0000.zip")
        print(f"│   ├── frame0001.zip")
        print(f"│   └── ...")
        print(f"└── kitchen/")
        print(f"    └── ...")
        
        data_root = input("数据目录路径: ").strip()
        if not data_root:
            print("❌ 未指定数据目录")
            sys.exit(1)
    
    # 3. 检测数据结构
    scene_info = detect_data_structure(data_root)
    if not scene_info:
        print("❌ 数据结构检测失败")
        sys.exit(1)
    
    # 4. 选择场景
    if args.scene:
        scene_name = args.scene
        if scene_name not in scene_info:
            print(f"❌ 指定的场景 '{scene_name}' 不存在")
            print(f"可用场景: {list(scene_info.keys())}")
            sys.exit(1)
    else:
        if args.auto:
            # 自动选择第一个场景
            scene_name = list(scene_info.keys())[0]
        else:
            print(f"\n📋 选择处理场景:")
            scene_names = list(scene_info.keys())
            for i, name in enumerate(scene_names, 1):
                info = scene_info[name]
                print(f"   {i}. {name} ({info['frame_count']} 帧)")
            
            while True:
                try:
                    choice = input(f"请选择场景 (1-{len(scene_names)}): ").strip()
                    if not choice:
                        scene_name = scene_names[0]  # 默认第一个
                        break
                    idx = int(choice) - 1
                    if 0 <= idx < len(scene_names):
                        scene_name = scene_names[idx]
                        break
                    else:
                        print("❌ 无效选择")
                except ValueError:
                    print("❌ 请输入数字")
    
    print(f"\n✅ 选择场景: {scene_name}")
    print(f"   帧数: {scene_info[scene_name]['frame_count']}")
    
    # 5. 快速测试
    print(f"\n🧪 运行快速测试...")
    if not quick_test(data_root, scene_name):
        if not args.auto:
            if not input("数据加载测试失败，是否继续? (y/N): ").lower().startswith('y'):
                sys.exit(1)
        else:
            print("❌ 自动模式下数据加载测试失败")
            sys.exit(1)
    
    if args.test_only:
        print("\n✅ 测试完成，退出")
        sys.exit(0)
    
    # 6. 配置处理参数
    config = {}
    
    if args.auto:
        # 自动配置
        config = {
            'max_frames': 10,
            'test_mode': False,
            'use_numba': True,
            'hole_threshold': 0.3
        }
        output_dir = args.output or "./processed_unified"
    else:
        # 交互配置
        print(f"\n⚙️ 配置处理参数:")
        
        # 输出目录
        output_dir = args.output or input("输出目录 (默认: ./processed_unified): ").strip()
        if not output_dir:
            output_dir = "./processed_unified"
        
        # 处理模式
        print(f"\n处理模式:")
        print(f"1. 测试模式 (3帧)")
        print(f"2. 小批量 (10帧)")
        print(f"3. 中批量 (50帧)")
        print(f"4. 全部帧 ({scene_info[scene_name]['frame_count']}帧)")
        print(f"5. 自定义")
        
        mode_choice = input("选择模式 (默认: 1): ").strip() or "1"
        
        if mode_choice == "1":
            config['test_mode'] = True
            config['max_frames'] = 3
        elif mode_choice == "2":
            config['max_frames'] = 10
        elif mode_choice == "3":
            config['max_frames'] = 50
        elif mode_choice == "4":
            config['max_frames'] = None
        elif mode_choice == "5":
            try:
                max_frames = input("最大处理帧数: ").strip()
                config['max_frames'] = int(max_frames) if max_frames else None
            except ValueError:
                config['max_frames'] = 10
        else:
            config['test_mode'] = True
            config['max_frames'] = 3
        
        # 性能选项
        config['use_numba'] = input("启用Numba加速? (Y/n): ").strip().lower() != 'n'
        
        # 算法参数
        hole_threshold = input("空洞检测阈值 (默认: 0.3): ").strip()
        try:
            config['hole_threshold'] = float(hole_threshold) if hole_threshold else 0.3
        except ValueError:
            config['hole_threshold'] = 0.3
    
    # 7. 生成和显示命令
    run_command = generate_run_command(data_root, output_dir, scene_name, config)
    
    print(f"\n📋 处理配置:")
    print(f"   数据目录: {data_root}")
    print(f"   输出目录: {output_dir}")
    print(f"   场景: {scene_name}")
    print(f"   最大帧数: {config.get('max_frames', '全部')}")
    print(f"   测试模式: {config.get('test_mode', False)}")
    print(f"   Numba加速: {config.get('use_numba', True)}")
    print(f"   空洞阈值: {config.get('hole_threshold', 0.3)}")
    
    print(f"\n🚀 运行命令:")
    print(f"{run_command}")
    
    # 8. 确认运行
    if not args.auto:
        if not input("\n开始处理? (Y/n): ").strip().lower() != 'n':
            print("取消处理")
            sys.exit(0)
    
    # 9. 执行处理
    print(f"\n🔄 开始处理...")
    print("=" * 50)
    
    try:
        # 执行命令
        result = subprocess.run(run_command.split(), check=True)
        
        print("=" * 50)
        print("🎉 处理完成!")
        print(f"输出目录: {output_dir}")
        print(f"\n查看结果:")
        print(f"ls -la {output_dir}/")
        print(f"\n可视化结果:")
        print(f"ls -la {output_dir}/visualization/")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 处理失败: {e}")
        print(f"请检查错误信息并重试")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中止处理")
        sys.exit(1)


if __name__ == "__main__":
    main()