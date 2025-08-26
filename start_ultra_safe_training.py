#!/usr/bin/env python3
"""
超安全训练启动脚本 - 一键解决内存问题
"""

import os
import sys
import subprocess
import yaml
import time
from pathlib import Path


def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("超安全移动端帧插值训练系统")
    print("=" * 60)
    print("目标: 彻底解决内存卡死问题")
    print("策略: 多层级内存优化")
    print("特点: 实时内存监控 + 渐进式训练")
    print("=" * 60)


def check_environment():
    print("\n环境检查...")
    
    checks = []
    
    # 检查CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        checks.append(("CUDA", "OK" if cuda_available else "FAIL", 
                      f"GPU count: {torch.cuda.device_count()}" if cuda_available else "No CUDA support"))
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            checks.append(("GPU", "OK", f"{gpu_name} ({gpu_memory:.1f}GB)"))
    except ImportError:
        checks.append(("PyTorch", "FAIL", "未安装"))
    
    # 检查数据
    data_path = "./output_motion_fix/training_data"
    data_exists = os.path.exists(data_path)
    if data_exists:
        npy_files = len([f for f in os.listdir(data_path) if f.endswith('.npy')])
        checks.append(("训练数据", "OK", f"{npy_files} 个文件"))
    else:
        checks.append(("训练数据", "FAIL", "数据目录不存在"))
    
    # 检查配置
    config_path = "./configs/ultra_safe_training_config.yaml"
    config_exists = os.path.exists(config_path)
    checks.append(("配置文件", "OK" if config_exists else "FAIL", 
                  "ultra_safe_training_config.yaml"))
    
    # 检查网络文件 - 修复：使用当前主要网络
    network_path = "./src/npu/networks/residual_mv_guided_network.py"
    network_exists = os.path.exists(network_path)
    checks.append(("残差MV网络", "OK" if network_exists else "FAIL", 
                  "residual_mv_guided_network.py"))
    
    # 打印检查结果
    print("\nEnvironment check results:")
    for name, status, detail in checks:
        print(f"  {status} {name}: {detail}")
    
    # 检查是否可以继续
    critical_missing = [name for name, status, _ in checks if status == "FAIL" and name in ["CUDA", "训练数据", "配置文件"]]
    
    if critical_missing:
        print(f"\nCritical components missing: {', '.join(critical_missing)}")
        return False
    
    print("\nEnvironment check passed")
    return True


def estimate_memory_requirements():
    print("\nMemory requirement estimation...")
    
    try:
        with open("./configs/ultra_safe_training_config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 获取配置参数
        batch_size = config['training']['batch_size']
        resolution = config['datasets']['preprocessing']['target_resolution']
        encoder_channels = config['model']['encoder_channels']
        
        # 计算内存需求
        h, w = resolution
        input_size_mb = batch_size * 7 * h * w * 4 / 1024**2  # 输入数据
        target_size_mb = batch_size * 3 * h * w * 4 / 1024**2  # 目标数据
        
        # 模型参数估算
        max_channels = max(encoder_channels)
        param_count = max_channels * 1000  # 粗略估算
        param_size_mb = param_count * 4 / 1024**2
        
        # 激活值估算
        max_activation_mb = batch_size * max_channels * h * w * 4 / 1024**2
        
        total_mb = input_size_mb + target_size_mb + param_size_mb + max_activation_mb * 2
        
        print(f"Target resolution: {h}×{w}")
        print(f"Batch size: {batch_size}")
        print(f"Network structure: {encoder_channels}")
        print(f"Estimated memory requirement:")
        print(f"   Input data: {input_size_mb:.1f} MB")
        print(f"   Target data: {target_size_mb:.1f} MB") 
        print(f"   Model parameters: {param_size_mb:.1f} MB")
        print(f"   Activation values: {max_activation_mb:.1f} MB")
        print(f"   Total: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
        
        # 安全性评估
        if total_mb < 1024:  # < 1GB
            safety_level = "VERY_SAFE"
        elif total_mb < 2048:  # < 2GB
            safety_level = "SAFE"
        elif total_mb < 4096:  # < 4GB
            safety_level = "CAUTION"
        else:
            safety_level = "DANGEROUS"
        
        print(f"Safety level: {safety_level}")
        
        return total_mb < 4096  # 4GB以下认为安全
        
    except Exception as e:
        print(f"Warning: 内存评估失败: {e}")
        return True  # 默认认为安全


def run_memory_debug():
    print("\nRunning memory pre-check...")
    
    try:
        # 修复：使用内置的PyTorch内存检查，而不是不存在的debug_memory_usage.py
        import torch
        
        if torch.cuda.is_available():
            # GPU内存信息
            gpu_count = torch.cuda.device_count()
            print(f"Detected {gpu_count} GPUs")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                
                print(f"  GPU {i} ({props.name}):")
                print(f"    Total memory: {total_memory:.1f} GB")
                print(f"    Allocated: {allocated:.1f} GB")
                print(f"    Cached: {cached:.1f} GB")
                print(f"    Available: {total_memory - cached:.1f} GB")
            
            # 内存建议
            if total_memory < 4.0:
                print("Warning: GPU memory less than 4GB, recommend minimal configuration")
            elif total_memory < 8.0:
                print("GPU memory moderate, recommend conservative configuration")
            else:
                print("GPU memory sufficient, can use standard configuration")
                
        else:
            print("Warning: No CUDA support detected, will use CPU training")
            
        print("Memory debug completed")
        
    except ImportError:
        print("Error: PyTorch未安装，无法进行内存检查")
    except Exception as e:
        print(f"Warning: 内存调试失败: {e}")


def start_training():
    print("\nStarting ultra-safe training...")
    
    try:
        # 启动训练脚本
        training_script = "train/ultra_safe_train.py"
        
        if not os.path.exists(training_script):
            print(f"Error: Training script not found: {training_script}")
            return False
        
        print(f"Executing script: {training_script}")
        print("Note: If freezing still occurs, try these strategies:")
        print("   1. Reduce batch_size to smaller value")
        print("   2. Lower target_resolution")
        print("   3. Simplify encoder_channels")
        print("   4. Use FP16 precision")
        print("\n" + "="*50)
        
        # 执行训练
        subprocess.run([sys.executable, training_script])
        
        return True
        
    except KeyboardInterrupt:
        print("\nWarning: Training interrupted by user")
        return False
    except Exception as e:
        print(f"\nError: Training startup failed: {e}")
        return False


def show_progressive_strategies():
    print("\nProgressive training strategies:")
    print("If memory issues persist, gradually increase complexity in these stages:")
    
    strategies = [
        {
            "阶段": "第一阶段（最保守）",
            "分辨率": "135×240",
            "批次大小": "1",
            "网络": "[16, 32, 64]",
            "特点": "确保能够启动训练"
        },
        {
            "阶段": "第二阶段（逐步增加）", 
            "分辨率": "270×480",
            "批次大小": "1",
            "网络": "[32, 64, 128]",
            "特点": "当前配置"
        },
        {
            "阶段": "第三阶段（目标配置）",
            "分辨率": "540×960", 
            "批次大小": "1",
            "网络": "[64, 128, 256]",
            "特点": "提升质量"
        },
        {
            "阶段": "第四阶段（最终目标）",
            "分辨率": "1080×1920",
            "批次大小": "1", 
            "网络": "完整UNet+FFC",
            "特点": "完整分辨率"
        }
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['阶段']}:")
        print(f"  Resolution: {strategy['分辨率']}")
        print(f"  Batch size: {strategy['批次大小']}")
        print(f"  Network: {strategy['网络']}")
        print(f"  Features: {strategy['特点']}")


def show_emergency_solutions():
    print("\nEmergency: If ultra-safe configuration still freezes, try these emergency measures:")
    
    solutions = [
        "1. Modify config file ultra_safe_training_config.yaml:",
        "   target_resolution: [135, 240]  # Smaller resolution",
        "   encoder_channels: [16, 32, 64]  # Simpler network",
        "",
        "2. Use CPU training test:",
        "   Force CPU usage in ultra_safe_train.py",
        "   device = torch.device('cpu')",
        "",
        "3. Check system memory:",
        "   Close other programs to free memory",
        "   Monitor system memory usage",
        "",
        "4. Use progressive approach:",
        "   Start with minimal configuration and gradually increase complexity",
        "   Verify each stage runs properly",
        "",
        "5. Hardware upgrade considerations:",
        "   GPU memory ≥ 8GB (recommend 16GB)",
        "   System memory ≥ 16GB (recommend 32GB)"
    ]
    
    for solution in solutions:
        print(solution)


def main():
    print_banner()
    
    # 环境检查
    if not check_environment():
        print("\nError: Environment check failed, please resolve issues and retry")
        return
    
    # 内存评估
    if not estimate_memory_requirements():
        print("\nWarning: Memory requirements may exceed limits, recommend progressive strategy")
        show_progressive_strategies()
    
    # 选择操作
    print("\n" + "="*50)
    print("Select operation:")
    print("1. Start ultra-safe training directly")
    print("2. Run memory pre-check")
    print("3. View progressive strategies")
    print("4. View emergency solutions")
    print("5. Exit")
    
    while True:
        choice = input("\nPlease select (1-5): ").strip()
        
        if choice == "1":
            start_training()
            break
        elif choice == "2":
            run_memory_debug()
        elif choice == "3":
            show_progressive_strategies()
        elif choice == "4":
            show_emergency_solutions()
        elif choice == "5":
            print("Exit")
            break
        else:
            print("Error: Invalid selection, please enter 1-5")


if __name__ == "__main__":
    main()