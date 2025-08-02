#!/usr/bin/env python3
"""
@file setup_environment.py
@brief 自动化环境设置脚本

功能描述：
- 检测系统环境和硬件配置
- 自动选择合适的PyTorch版本
- 验证关键依赖是否正确安装
- 提供环境诊断和修复建议

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class EnvironmentSetup:
    """环境设置工具类"""
    
    def __init__(self):
        """初始化环境设置工具"""
        self.system_info = self.detect_system_info()
        self.cuda_info = self.detect_cuda_info()
        self.python_info = self.detect_python_info()
        
        print("="*80)
        print("🔧 MobileExtra环境设置工具")
        print("="*80)
        print(f"系统: {self.system_info['platform']} {self.system_info['version']}")
        print(f"Python: {self.python_info['version']}")
        print(f"CUDA: {self.cuda_info['version'] if self.cuda_info['available'] else 'Not Available'}")
        print("="*80)
    
    def detect_system_info(self) -> Dict:
        """检测系统信息"""
        return {
            'platform': platform.system(),
            'version': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
    
    def detect_python_info(self) -> Dict:
        """检测Python信息"""
        return {
            'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'executable': sys.executable,
            'pip_available': self.check_command_available('pip')
        }
    
    def detect_cuda_info(self) -> Dict:
        """检测CUDA信息"""
        cuda_info = {
            'available': False,
            'version': None,
            'device_count': 0,
            'devices': []
        }
        
        try:
            # 检查nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                cuda_info['available'] = True
                
                # 解析GPU信息
                gpu_lines = result.stdout.strip().split('\n')
                for line in gpu_lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpu_name = parts[0].strip()
                            memory_mb = parts[1].strip()
                            cuda_info['devices'].append({
                                'name': gpu_name,
                                'memory_mb': int(memory_mb)
                            })
                
                cuda_info['device_count'] = len(cuda_info['devices'])
                
                # 检查CUDA版本
                nvcc_result = subprocess.run(['nvcc', '--version'], 
                                           capture_output=True, text=True, timeout=30)
                if nvcc_result.returncode == 0:
                    # 从nvcc输出解析版本
                    for line in nvcc_result.stdout.split('\n'):
                        if 'release' in line.lower():
                            import re
                            version_match = re.search(r'release (\d+\.\d+)', line)
                            if version_match:
                                cuda_info['version'] = version_match.group(1)
                                break
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return cuda_info
    
    def check_command_available(self, command: str) -> bool:
        """检查命令是否可用"""
        try:
            result = subprocess.run([command, '--version'], 
                                  capture_output=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def get_pytorch_install_command(self) -> Tuple[str, str]:
        """获取推荐的PyTorch安装命令"""
        base_packages = "torch torchvision torchaudio"
        
        if self.cuda_info['available']:
            # 根据检测到的CUDA版本选择
            cuda_version = self.cuda_info.get('version', '11.8')
            
            if cuda_version.startswith('12.'):
                index_url = "https://download.pytorch.org/whl/cu121"
                reason = f"检测到CUDA {cuda_version}，使用CUDA 12.1版本"
            elif cuda_version.startswith('11.'):
                index_url = "https://download.pytorch.org/whl/cu118"  
                reason = f"检测到CUDA {cuda_version}，使用CUDA 11.8版本"
            else:
                index_url = "https://download.pytorch.org/whl/cu118"
                reason = f"未能确定CUDA版本，使用默认CUDA 11.8版本"
        else:
            index_url = "https://download.pytorch.org/whl/cpu"
            reason = "未检测到CUDA，使用CPU版本"
        
        command = f"pip install {base_packages} --index-url {index_url}"
        
        return command, reason
    
    def create_conda_environment(self, env_name: str = "mobile_interpolation") -> bool:
        """创建conda环境"""
        print(f"\n🐍 创建conda环境: {env_name}")
        
        # 检查conda是否可用
        if not self.check_command_available('conda'):
            print("❌ conda未安装或不在PATH中")
            print("请先安装Anaconda或Miniconda: https://docs.conda.io/en/latest/miniconda.html")
            return False
        
        try:
            # 检查环境是否已存在
            result = subprocess.run(['conda', 'env', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            if env_name in result.stdout:
                print(f"⚠️ 环境 {env_name} 已存在")
                response = input("是否删除并重新创建? (y/N): ").strip().lower()
                if response == 'y':
                    print(f"🗑️ 删除现有环境...")
                    subprocess.run(['conda', 'env', 'remove', '-n', env_name], 
                                 timeout=120)
                else:
                    print("跳过环境创建")
                    return True
            
            # 使用environment.yml创建环境
            env_file = Path(__file__).parent / 'environment.yml'
            if env_file.exists():
                print(f"📄 使用配置文件创建环境: {env_file}")
                result = subprocess.run(['conda', 'env', 'create', '-f', str(env_file)], 
                                      timeout=1800)  # 30分钟超时
                return result.returncode == 0
            else:
                # 手动创建基础环境
                print("📄 手动创建基础环境...")
                result = subprocess.run(['conda', 'create', '-n', env_name, 'python=3.9', '-y'], 
                                      timeout=300)
                return result.returncode == 0
                
        except subprocess.TimeoutExpired:
            print("❌ 环境创建超时")
            return False
        except Exception as e:
            print(f"❌ 环境创建失败: {e}")
            return False
    
    def install_requirements(self, env_name: str = "mobile_interpolation") -> bool:
        """安装项目依赖"""
        print(f"\n📦 安装项目依赖...")
        
        # 检查requirements.txt是否存在
        req_file = Path(__file__).parent / 'requirements.txt'
        if not req_file.exists():
            print(f"❌ requirements.txt文件不存在: {req_file}")
            return False
        
        try:
            # 获取PyTorch安装命令
            pytorch_cmd, reason = self.get_pytorch_install_command()
            print(f"🔥 PyTorch安装策略: {reason}")
            print(f"   命令: {pytorch_cmd}")
            
            # 激活环境并安装PyTorch
            if self.system_info['platform'] == 'Windows':
                activate_cmd = f"conda activate {env_name} && {pytorch_cmd}"
                shell = True
            else:
                activate_cmd = f"source activate {env_name} && {pytorch_cmd}"
                shell = True
            
            print("⏳ 安装PyTorch (可能需要几分钟)...")
            result = subprocess.run(activate_cmd, shell=shell, timeout=1800)
            
            if result.returncode != 0:
                print("❌ PyTorch安装失败")
                return False
            
            # 安装其他依赖
            print("⏳ 安装其他依赖...")
            if self.system_info['platform'] == 'Windows':
                install_cmd = f"conda activate {env_name} && pip install -r {req_file}"
            else:
                install_cmd = f"source activate {env_name} && pip install -r {req_file}"
            
            result = subprocess.run(install_cmd, shell=shell, timeout=1800)
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("❌ 依赖安装超时")
            return False
        except Exception as e:
            print(f"❌ 依赖安装失败: {e}")
            return False
    
    def verify_installation(self, env_name: str = "mobile_interpolation") -> bool:
        """验证安装是否成功"""
        print(f"\n✅ 验证安装...")
        
        verification_script = '''
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"PyTorch import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except ImportError as e:
    print(f"NumPy import failed: {e}")
    sys.exit(1)

try:
    import zarr
    print(f"Zarr: {zarr.__version__}")
except ImportError as e:
    print(f"Zarr import failed: {e}")
    sys.exit(1)

try:
    import pytorch_lightning as pl
    print(f"PyTorch Lightning: {pl.__version__}")
except ImportError as e:
    print(f"PyTorch Lightning import failed: {e}")
    sys.exit(1)

print("✅ All key dependencies verified!")
'''
        
        try:
            # 在conda环境中运行验证脚本
            if self.system_info['platform'] == 'Windows':
                verify_cmd = f'conda activate {env_name} && python -c "{verification_script}"'
            else:
                verify_cmd = f'source activate {env_name} && python -c "{verification_script}"'
            
            result = subprocess.run(verify_cmd, shell=True, timeout=60, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout)
                return True
            else:
                print("❌ 验证失败:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 验证过程出错: {e}")
            return False
    
    def print_usage_instructions(self, env_name: str = "mobile_interpolation"):
        """打印使用说明"""
        print("\n" + "="*80)
        print("📖 环境设置完成！使用说明:")
        print("="*80)
        
        if self.system_info['platform'] == 'Windows':
            activate_cmd = f"conda activate {env_name}"
        else:
            activate_cmd = f"conda activate {env_name}"
        
        print(f"1. 激活环境:")
        print(f"   {activate_cmd}")
        print("")
        print("2. 测试预处理流水线:")
        print("   python test_preprocessing_pipeline.py --scene bistro1 --test-frames 5")
        print("")
        print("3. 运行完整预处理:")
        print("   python run_preprocessing.py --scene bistro1 --create-splits")
        print("")
        print("4. 开始训练:")
        print("   python training/train_mobile_inpainting.py --data-root ./training/processed --split-file ./training/processed/bistro1_splits.json")
        print("")
        print("5. 验证环境 (任何时候):")
        print(f"   {activate_cmd}")
        print("   python setup_environment.py --verify-only")
        print("="*80)
    
    def setup_complete_environment(self, env_name: str = "mobile_interpolation") -> bool:
        """完整环境设置流程"""
        print("🚀 开始完整环境设置...")
        
        success_steps = 0
        total_steps = 4
        
        # Step 1: 创建conda环境
        if self.create_conda_environment(env_name):
            success_steps += 1
            print("✅ Step 1/4: Conda环境创建成功")
        else:
            print("❌ Step 1/4: Conda环境创建失败")
            return False
        
        # Step 2: 安装依赖
        if self.install_requirements(env_name):
            success_steps += 1
            print("✅ Step 2/4: 依赖安装成功")
        else:
            print("❌ Step 2/4: 依赖安装失败")
            return False
        
        # Step 3: 验证安装
        if self.verify_installation(env_name):
            success_steps += 1
            print("✅ Step 3/4: 安装验证成功")
        else:
            print("❌ Step 3/4: 安装验证失败")
            return False
        
        # Step 4: 打印使用说明
        self.print_usage_instructions(env_name)
        success_steps += 1
        print("✅ Step 4/4: 使用说明已显示")
        
        print(f"\n🎉 环境设置完成! ({success_steps}/{total_steps} 步骤成功)")
        return success_steps == total_steps


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MobileExtra Environment Setup')
    parser.add_argument('--env-name', type=str, default='mobile_interpolation',
                       help='Conda environment name')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing installation')
    parser.add_argument('--pytorch-only', action='store_true',
                       help='Only install PyTorch')
    
    args = parser.parse_args()
    
    # 创建环境设置工具
    setup_tool = EnvironmentSetup()
    
    if args.verify_only:
        # 仅验证现有安装
        success = setup_tool.verify_installation(args.env_name)
        if success:
            setup_tool.print_usage_instructions(args.env_name)
        return 0 if success else 1
    
    elif args.pytorch_only:
        # 仅显示PyTorch安装命令
        pytorch_cmd, reason = setup_tool.get_pytorch_install_command()
        print(f"\n🔥 推荐的PyTorch安装命令:")
        print(f"理由: {reason}")
        print(f"命令: {pytorch_cmd}")
        return 0
    
    else:
        # 完整环境设置
        success = setup_tool.setup_complete_environment(args.env_name)
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())