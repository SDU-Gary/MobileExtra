#!/usr/bin/env python3
"""
Colleague Training Starter - 启动同事数据集训练 (with Simple Grid Strategy)

专门用于启动使用colleague_training_config.yaml的训练流程
已集成简单网格策略，确保训练稳定性和可预测性
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("同事数据集训练系统 (Simple Grid Strategy)")
    print("=" * 60)
    print("配置文件: colleague_training_config.yaml")
    print("Patch策略: 简单4x4网格 (16 patches)")
    print("训练模式: Residual Learning + Patch-based")
    print("数据集: ColleagueDatasetAdapter")
    print("=" * 60)

def validate_config():
    """验证配置文件"""
    config_path = "./configs/colleague_training_config.yaml"
    
    if not os.path.exists(config_path):
        print(f" 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(" 配置验证:")
        
        # Patch 采样策略配置（支持简单网格/重叠crop）
        patch_config = config.get('patch', {})
        simple_grid_enabled = bool(patch_config.get('use_simple_grid_patches', False))
        overlap_enabled = bool(patch_config.get('use_overlapping_crops', False))
        print(" Patch 采样:")
        if overlap_enabled:
            crop_sz = int(patch_config.get('crop_size', 256))
            stride = int(patch_config.get('crop_stride', 128))
            keep_frac = float(patch_config.get('keep_top_frac', 0.5))
            print(f"    重叠crop: 启用 ({crop_sz}x{crop_sz}, stride={stride}, keep_top_frac={keep_frac})")
        else:
            grid_rows = patch_config.get('simple_grid_rows', 4)
            grid_cols = patch_config.get('simple_grid_cols', 4)
            print(f"    简单网格: {'启用' if simple_grid_enabled else '禁用'}，网格={grid_rows}x{grid_cols}")
        
        # 检查其他关键配置
        network_config = config.get('network', {})
        training_config = config.get('training', {})
        
        print(f"    网络类型: {network_config.get('type', 'Unknown')}")
        print(f"    学习模式: {network_config.get('learning_mode', 'Unknown')}")
        print(f"    批次大小: {training_config.get('batch_size', 'Unknown')}")
        print(f"    最大轮数: {training_config.get('max_epochs', 'Unknown')}")
        print(f"    梯度累计: {training_config.get('accumulate_grad_batches', 1)}")

        # Tone-mapping / 归一化 / log 残差配置
        hdr_cfg = config.get('hdr_processing', {})
        print(" HDR显示/损失域:")
        print(f"    tone-mapping: {hdr_cfg.get('tone_mapping_for_display', 'reinhard')}")
        if hdr_cfg.get('tone_mapping_for_display', 'reinhard').lower() == 'mulaw':
            print(f"    mu: {hdr_cfg.get('mulaw_mu', 500.0)}")
        norm_cfg = config.get('normalization', {})
        print(f"    归一化模式: {norm_cfg.get('type', 'none')}")
        if str(norm_cfg.get('type','')).lower() == 'log':
            print(f"    log_epsilon: {norm_cfg.get('log_epsilon', 'NA')} / log_delta_abs_max: {norm_cfg.get('log_delta_abs_max', 0)} / log_delta_alpha: {norm_cfg.get('log_delta_alpha', 1.0)}")

        return True
        
    except Exception as e:
        print(f" 配置文件验证失败: {e}")
        return False

def check_dependencies():
    """检查依赖项"""
    print("\n 依赖检查:")
    
    dependencies = [
        ("simple_patch_extractor.py", "简单网格提取器"),
        ("train/patch_aware_dataset.py", "Patch数据集"),
        ("train/colleague_dataset_adapter.py", "NoiseBase数据适配器"),
        ("train/patch_training_framework.py", "Patch训练框架"),
        ("train/patch_tensorboard_logger.py", "可视化记录器"),
        ("src/npu/networks/patch/patch_network.py", "Patch网络"),
        ("train/residual_inpainting_loss.py", "残差损失函数"),
        ("train/residual_learning_helper.py", "残差学习助手")
    ]
    
    all_good = True
    for file_path, description in dependencies:
        if os.path.exists(file_path):
            print(f"    {description}: {file_path}")
        else:
            print(f"    {description}: {file_path} (缺失)")
            all_good = False
    
    return all_good

def check_data():
    """检查数据目录 - 针对NoiseBase数据格式"""
    print("\n 数据检查:")
    
    data_root = "./data"  # 来自colleague_training_config.yaml
    processed_bistro_path = "./data/processed_bistro"  # NoiseBase数据路径
    
    if os.path.exists(data_root):
        print(f"    数据根目录存在: {data_root}")
        
        # 检查processed_bistro目录
        if os.path.exists(processed_bistro_path):
            print(f"    NoiseBase数据目录存在: {processed_bistro_path}")
            
            # 检查必需的子目录和文件
            required_subdirs = {
                'warp_hole': 'Warped RGB with holes (输入数据)',
                'ref': 'Reference images (目标数据)', 
                'bistro': 'Bistro scene data (语义数据)',
                'warped': 'Warped RGB images (可选)',
                'normal': 'Normal maps (可选)',
                'pre': 'Previous frames (可选)'
            }
            
            data_files_found = 0
            for subdir, description in required_subdirs.items():
                subdir_path = os.path.join(processed_bistro_path, subdir)
                if os.path.exists(subdir_path):
                    # 计算EXR文件数量
                    exr_files = len([f for f in os.listdir(subdir_path) if f.endswith('.exr')])
                    data_files_found += exr_files
                    status = "" if exr_files > 0 else " "
                    print(f"   {status} {subdir}: {exr_files} EXR文件 ({description})")
                else:
                    print(f"    {subdir}: 目录不存在 ({description})")
            
            # 检查关键目录
            critical_dirs = ['warp_hole', 'ref']
            critical_missing = []
            for critical_dir in critical_dirs:
                critical_path = os.path.join(processed_bistro_path, critical_dir)
                if not os.path.exists(critical_path):
                    critical_missing.append(critical_dir)
                else:
                    exr_count = len([f for f in os.listdir(critical_path) if f.endswith('.exr')])
                    if exr_count == 0:
                        critical_missing.append(f"{critical_dir}(空)")
            
            if critical_missing:
                print(f"    关键数据缺失: {', '.join(critical_missing)}")
                print("   💡 关键目录: warp_hole(输入), ref(目标)")
                return False
            else:
                print(f"    数据概况: 总共 {data_files_found} 个EXR文件")
                print("    关键数据完整，可以开始训练")
                return True
                
        else:
            print(f"    NoiseBase数据目录不存在: {processed_bistro_path}")
            print("   💡 数据应该位于: ./data/processed_bistro/")
            print("   💡 包含子目录: warp_hole, ref, bistro, warped, normal, pre")
            return False
    else:
        print(f"    数据根目录不存在: {data_root}")
        print("   💡 请确保数据已正确放置")
        return False

def start_training():
    """启动训练"""
    print("\n 启动训练:")
    
    # 选择训练脚本
    training_scripts = [
        ("train/patch_training_framework.py", "Patch训练框架 (推荐)")
    ]
    
    print("选择训练脚本:")
    for i, (script, desc) in enumerate(training_scripts, 1):
        exists = "" if os.path.exists(script) else ""
        print(f"   {i}. {exists} {desc}")
    
    try:
        # 直接使用唯一的训练脚本
        selected_script = training_scripts[0][0]
        
        if not os.path.exists(selected_script):
            print(f" 训练脚本不存在: {selected_script}")
            return False
        
        print(f" 使用训练脚本: {selected_script}")
        print(f" 使用配置文件: ./configs/colleague_training_config.yaml")
        
        # 启动训练
        cmd = [
            sys.executable, selected_script,
            "--config", "./configs/colleague_training_config.yaml"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print("\n" + "="*60)
        print("训练启动中... (按 Ctrl+C 可终止)")
        print("="*60 + "\n")
        
        # 执行训练
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n 训练完成!")
        else:
            print(f"\n 训练失败，退出码: {result.returncode}")
        
        return result.returncode == 0
        
    except ValueError:
        print(" 无效输入")
        return False
    except KeyboardInterrupt:
        print("\n  训练被用户终止")
        return False
    except Exception as e:
        print(f" 启动训练失败: {e}")
        return False

def show_simple_grid_info():
    return

def main():
    """主函数"""
    print_banner()
    
    # 验证配置文件
    if not validate_config():
        print("\n 配置验证失败，请检查配置文件")
        return 1
    
    # 检查依赖
    if not check_dependencies():
        print("\n 依赖检查失败，请确保所有文件存在")
        return 1
    
    # 检查数据
    if not check_data():
        print("\n  数据检查失败，但可以继续训练（如果使用测试数据）")
    
    # 询问是否启动训练
    print("\n" + "="*60)
    choice = input("是否启动训练? (y/N): ").strip().lower()
    
    if choice in ['y', 'yes', '是']:
        success = start_training()
        return 0 if success else 1
    else:
        print("👋 训练已取消")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
