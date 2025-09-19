#!/usr/bin/env python3
"""
Ultra Safe Colleague Training - 结合超安全内存管理和简单网格策略
专门用于ColleagueDatasetAdapter + SimplePatchExtractor的训练流程
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("超安全同事数据集训练系统 (Simple Grid + Ultra Safe)")
    print("=" * 60)
    print("特点: 超安全内存管理 + 简单4x4网格策略")
    print("数据: ColleagueDatasetAdapter (OpenEXR)")
    print("策略: 渐进式内存优化 + 固定Patch数")
    print("=" * 60)

def create_ultra_safe_colleague_config():
    """创建超安全同事训练配置"""
    config_path = "./configs/ultra_safe_colleague_config.yaml"
    
    config = {
        # 实验配置
        'experiment': {
            'name': 'ultra_safe_colleague_training',
            'version': 'v1.0',
            'description': '超安全同事数据集训练 - ColleagueDatasetAdapter + SimplePatchExtractor'
        },
        
        # 数据配置 - 使用ColleagueDatasetAdapter
        'data': {
            'data_root': './data',
            'dataset_type': 'colleague',
            'data_format': 'processed_bistro',
            'processed_bistro_path': './data/processed_bistro',
            'required_subdirs': ['warp_hole', 'ref', 'bistro'],
            'splits': {
                'train': 'train',
                'val': 'val',
                'test': 'test'
            }
        },
        
        # Patch训练配置 - 启用简单网格策略
        'patch': {
            'enable_patch_mode': True,
            'patch_size': 128,
            'patch_mode_probability': 1.0,
            
            # 简单网格策略 (关键配置)
            'use_simple_grid_patches': True,
            'use_optimized_patches': False,
            
            # 简单网格参数
            'simple_grid_rows': 4,
            'simple_grid_cols': 4,
            'simple_expected_height': 1080,
            'simple_expected_width': 1920,
            
            # 固定patch数量
            'min_patches_per_image': 16,
            'max_patches_per_image': 16,
            
            # 缓存配置（保守设置）
            'enable_patch_cache': False,  # 避免内存开销
            'cache_size': 100,
            
            # 数据增强（禁用）
            'patch_augmentation': False,
            'augmentation_probability': 0.0
        },
        
        # 网络配置 - 保守设置
        'network': {
            'type': 'PatchNetwork',
            'input_channels': 7,
            'output_channels': 3,
            'base_channels': 16,  # 减少通道数节省内存
            'learning_mode': 'residual'
        },
        
        # 训练配置 - 超安全设置
        'training': {
            'batch_size': 1,  # 最小批次大小
            'num_workers': 0,  # 避免多进程开销
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'max_epochs': 50,  # 测试用较少轮数
            
            # 优化器配置
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'scheduler_params': {
                'T_max': 50,
                'eta_min': 0.000001
            }
        },
        
        # 损失函数配置 - 轻量化
        'loss': {
            'type': 'ResidualInpaintingLoss',
            'weights': {
                'mse': 1.0,
                'l1': 0.3,
                'perceptual': 0.2,  # 降低感知损失权重
                'edge': 0.3,
                'boundary': 0.2
            },
            'residual_scale_factor': 2.0,
            'spatial_attention': True
        },
        
        # 验证和监控 - 简化设置
        'validation': {
            'frequency': 10,  # 减少验证频率
            'metrics': ['mse', 'psnr']  # 简化指标
        },
        
        'monitoring': {
            'tensorboard_log_dir': './logs/ultra_safe_colleague',
            'model_save_dir': './models/ultra_safe_colleague',
            'save_frequency': 20  # 减少保存频率
        },
        
        # 内存管理 - 关键配置
        'memory': {
            'enable_ultra_safe_mode': True,
            'max_gpu_memory_gb': 2,  # 严格限制GPU内存
            'garbage_collection_frequency': 5  # 频繁垃圾回收
        }
    }
    
    # 保存配置文件
    os.makedirs('./configs', exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, indent=2)
    
    print(f" 超安全同事训练配置已创建: {config_path}")
    return config_path

def check_colleague_data():
    """检查同事数据"""
    print("\n 检查同事数据...")
    
    processed_bistro_path = "./data/processed_bistro"
    
    if not os.path.exists(processed_bistro_path):
        print(f" 同事数据目录不存在: {processed_bistro_path}")
        return False
    
    # 检查关键子目录
    critical_dirs = ['warp_hole', 'ref']
    data_counts = {}
    
    for subdir in critical_dirs:
        subdir_path = os.path.join(processed_bistro_path, subdir)
        if os.path.exists(subdir_path):
            exr_files = len([f for f in os.listdir(subdir_path) if f.endswith('.exr')])
            data_counts[subdir] = exr_files
            print(f"    {subdir}: {exr_files} EXR文件")
        else:
            print(f"    {subdir}: 目录不存在")
            return False
    
    # 验证数据对数量一致
    if len(set(data_counts.values())) == 1 and list(data_counts.values())[0] > 0:
        total_pairs = list(data_counts.values())[0]
        print(f"    数据验证通过: {total_pairs} 对训练样本")
        return True
    else:
        print(f"    数据不一致: {data_counts}")
        return False

def check_dependencies():
    """检查依赖文件"""
    print("\n 检查依赖文件...")
    
    dependencies = [
        ("simple_patch_extractor.py", "简单网格提取器"),
        ("train/colleague_dataset_adapter.py", "同事数据适配器"),
        ("train/patch_aware_dataset.py", "Patch数据集"),
        ("train/ultra_safe_train.py", "超安全训练脚本"),
        ("train/residual_inpainting_loss.py", "残差损失函数")
    ]
    
    missing_files = []
    for file_path, description in dependencies:
        if os.path.exists(file_path):
            print(f"    {description}: {file_path}")
        else:
            print(f"    {description}: {file_path} (缺失)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  缺失关键文件: {missing_files}")
        return False
    
    return True

def estimate_memory_usage():
    """估算内存使用"""
    print("\n 内存使用估算...")
    
    # 基于配置估算
    batch_size = 1
    patch_size = 128
    patches_per_image = 16
    
    # 输入数据: batch_size * patches * channels * H * W * 4字节
    input_mb = batch_size * patches_per_image * 7 * patch_size * patch_size * 4 / 1024**2
    target_mb = batch_size * patches_per_image * 3 * patch_size * patch_size * 4 / 1024**2
    
    # 网络参数估算 (保守)
    base_channels = 16
    param_count = base_channels * 1000  # 粗略估算
    param_mb = param_count * 4 / 1024**2
    
    total_mb = input_mb + target_mb + param_mb
    
    print(f"    Patch配置: {patches_per_image}个{patch_size}x{patch_size}的patch")
    print(f"    输入数据: {input_mb:.1f} MB")
    print(f"    目标数据: {target_mb:.1f} MB")
    print(f"    网络参数: {param_mb:.1f} MB")
    print(f"    总计估算: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
    
    if total_mb < 500:  # < 0.5GB
        print("   🟢 内存使用: 非常安全")
        return True
    elif total_mb < 1000:  # < 1GB
        print("   🟡 内存使用: 相对安全")
        return True
    else:
        print("   🔴 内存使用: 可能存在风险")
        return False

def start_ultra_safe_colleague_training():
    """启动超安全同事训练"""
    print("\n 启动超安全同事训练...")
    
    try:
        # 创建专用的训练脚本
        training_script_path = create_colleague_ultra_safe_script()
        
        print(f" 使用训练脚本: {training_script_path}")
        print("⏳ 正在启动训练...")
        print("=" * 60)
        
        # 执行训练
        result = subprocess.run([sys.executable, training_script_path])
        
        if result.returncode == 0:
            print("\n 训练完成!")
            return True
        else:
            print(f"\n 训练失败，退出码: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户终止")
        return False
    except Exception as e:
        print(f"\n 启动训练失败: {e}")
        return False

def create_colleague_ultra_safe_script():
    """创建专用的同事超安全训练脚本"""
    script_path = "./train_ultra_safe_colleague.py"
    
    script_content = '''#!/usr/bin/env python3
"""
Ultra Safe Colleague Training Script
结合ColleagueDatasetAdapter和SimplePatchExtractor的超安全训练
"""

import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train"))

def main():
    """主函数 - 使用ColleagueDatasetAdapter的超安全训练"""
    
    try:
        # 导入超安全训练器
        from train.ultra_safe_train import UltraSafeTrainer
        
        # 使用专门为同事数据创建的配置
        config_path = "./configs/ultra_safe_colleague_config.yaml"
        
        if not os.path.exists(config_path):
            print(f" 配置文件不存在: {config_path}")
            print("请先运行 start_ultra_safe_colleague_training.py 创建配置")
            return 1
        
        print(" 启动超安全同事数据训练...")
        print(f" 配置文件: {config_path}")
        
        # 创建训练器
        trainer = UltraSafeTrainer(config_path)
        
        # 开始训练
        trainer.train()
        
        print(" 训练完成!")
        return 0
        
    except Exception as e:
        print(f" 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    return script_path

def show_usage_guide():
    """显示使用指南"""
    print("\n📖 超安全同事训练使用指南:")
    print("=" * 50)
    
    print("\n 特点:")
    print("   • 超安全内存管理 - 防止系统卡死")
    print("   • 简单4x4网格策略 - 100%稳定patch生成")
    print("   • ColleagueDatasetAdapter - 处理OpenEXR数据")
    print("   • 固定16个patch - 可预测的内存使用")
    
    print("\n 配置:")
    print("   • 批次大小: 1 (最小设置)")
    print("   • Patch大小: 128x128")
    print("   • 网络通道: 16 (轻量化)")
    print("   • GPU内存限制: 2GB")
    
    print("\n🛠️  监控:")
    print("   • TensorBoard日志: ./logs/ultra_safe_colleague")
    print("   • 模型保存: ./models/ultra_safe_colleague")
    print("   • 实时内存监控和自动清理")

def main():
    """主函数"""
    print_banner()
    
    # 创建超安全同事配置
    config_path = create_ultra_safe_colleague_config()
    
    # 检查数据
    if not check_colleague_data():
        print("\n 同事数据检查失败")
        print("💡 请确保 ./data/processed_bistro 目录存在并包含:")
        print("   - warp_hole/*.exr (带洞的warped图像)")
        print("   - ref/*.exr (参考目标图像)")
        print("   - bistro/correct/*.exr (洞洞掩码)")
        return 1
    
    # 检查依赖
    if not check_dependencies():
        print("\n 依赖检查失败，请确保所有必要文件存在")
        return 1
    
    # 内存估算
    memory_safe = estimate_memory_usage()
    if not memory_safe:
        print("\n⚠️  内存使用可能较高，但已启用超安全模式")
    
    # 显示使用指南
    show_usage_guide()
    
    # 询问是否启动训练
    print("\n" + "=" * 60)
    choice = input("是否启动超安全同事训练? (y/N): ").strip().lower()
    
    if choice in ['y', 'yes', '是']:
        success = start_ultra_safe_colleague_training()
        return 0 if success else 1
    else:
        print("👋 训练已取消")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)