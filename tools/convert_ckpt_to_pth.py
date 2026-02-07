#!/usr/bin/env python3
"""
Checkpoint to PTH Converter
将PyTorch Lightning checkpoint转换为纯净的state_dict .pth文件

用法:
    python tools/convert_ckpt_to_pth.py --ckpt models/colleague/last.ckpt --output models/patch_network.pth
    python tools/convert_ckpt_to_pth.py --ckpt models/colleague/last.ckpt --output models/ --auto-name
"""

import torch
import argparse
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import classes that may be in checkpoint (for unpickling)
try:
    from train.patch_aware_dataset import PatchTrainingConfig
except ImportError:
    # Dummy class for unpickling if import fails
    class PatchTrainingConfig:
        pass


def convert_checkpoint_to_pth(ckpt_path: str, output_path: str,
                              save_full_model: bool = False,
                              extract_discriminator: bool = False) -> dict:
    """
    转换checkpoint到pth格式

    Args:
        ckpt_path: Lightning checkpoint路径
        output_path: 输出pth文件路径
        save_full_model: 是否保存完整模型（包括optimizer等）
        extract_discriminator: 是否同时提取discriminator

    Returns:
        转换信息字典
    """
    print(f"[1/3] Loading checkpoint: {ckpt_path}")

    # 加载checkpoint
    # PyTorch 2.6+ requires weights_only=False for checkpoints with custom classes
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    print(f"[2/3] Extracting model state_dict...")

    # 提取信息
    info = {
        'checkpoint_path': ckpt_path,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'global_step': checkpoint.get('global_step', 'unknown'),
    }

    # 提取主网络state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

        # 过滤出patch_network的权重
        # Lightning保存格式: 'patch_network.encoder1.conv1.weight'
        patch_network_state = {}
        discriminator_state = {}

        for key, value in state_dict.items():
            if key.startswith('patch_network.'):
                # 移除'patch_network.'前缀
                new_key = key.replace('patch_network.', '')
                patch_network_state[new_key] = value
            elif key.startswith('discriminator.'):
                # 移除'discriminator.'前缀
                new_key = key.replace('discriminator.', '')
                discriminator_state[new_key] = value

        # Count actual parameters (not just keys)
        patch_param_count = sum(v.numel() for v in patch_network_state.values())
        disc_param_count = sum(v.numel() for v in discriminator_state.values())

        info['patch_network_params'] = patch_param_count
        info['discriminator_params'] = disc_param_count

        print(f"  - PatchNetwork parameters: {patch_param_count:,} ({len(patch_network_state)} layers)")
        print(f"  - Discriminator parameters: {disc_param_count:,} ({len(discriminator_state)} layers)")

    else:
        raise ValueError("Checkpoint does not contain 'state_dict' key")

    # 准备保存内容
    if save_full_model:
        # 保存完整模型（包括optimizer等）
        save_content = {
            'model_state_dict': patch_network_state,
            'epoch': info['epoch'],
            'global_step': info['global_step'],
        }
        if 'optimizer_states' in checkpoint:
            save_content['optimizer_state_dict'] = checkpoint['optimizer_states']
        if 'lr_schedulers' in checkpoint:
            save_content['scheduler_state_dict'] = checkpoint['lr_schedulers']
    else:
        # 仅保存纯净的state_dict
        save_content = patch_network_state

    print(f"[3/3] Saving to: {output_path}")

    # 保存主网络
    torch.save(save_content, output_path)
    output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    info['output_size_mb'] = output_size_mb

    print(f"  ✅ Saved: {output_path} ({output_size_mb:.2f} MB)")

    # 可选：保存discriminator
    if extract_discriminator and len(discriminator_state) > 0:
        disc_output_path = str(output_path).replace('.pth', '_discriminator.pth')
        torch.save(discriminator_state, disc_output_path)
        disc_size_mb = Path(disc_output_path).stat().st_size / (1024 * 1024)
        print(f"  ✅ Saved discriminator: {disc_output_path} ({disc_size_mb:.2f} MB)")
        info['discriminator_output'] = disc_output_path

    return info


def main():
    parser = argparse.ArgumentParser(description='Convert Lightning checkpoint to PTH format')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to input checkpoint file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output .pth file or directory')
    parser.add_argument('--auto-name', action='store_true',
                       help='Auto-generate output filename based on checkpoint info')
    parser.add_argument('--full-model', action='store_true',
                       help='Save full model including optimizer state')
    parser.add_argument('--extract-discriminator', action='store_true',
                       help='Also extract discriminator weights')

    args = parser.parse_args()

    # 验证输入路径
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"❌ Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # 确定输出路径
    if args.auto_name:
        # 自动命名
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 从checkpoint文件名提取信息
        ckpt_name = ckpt_path.stem  # 例如: 'patch-model-epoch=234-val_loss=11.63'
        output_name = f"{ckpt_name}.pth"
        output_path = output_dir / output_name
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 执行转换
    try:
        info = convert_checkpoint_to_pth(
            str(ckpt_path),
            str(output_path),
            save_full_model=args.full_model,
            extract_discriminator=args.extract_discriminator
        )

        # 打印摘要
        print("\n" + "="*60)
        print("✅ Conversion Complete!")
        print("="*60)
        print(f"Checkpoint: {info['checkpoint_path']}")
        print(f"Epoch: {info['epoch']}")
        print(f"Global Step: {info['global_step']}")
        print(f"Output: {output_path} ({info['output_size_mb']:.2f} MB)")
        print(f"Parameters: {info['patch_network_params']}")

        # 计算压缩比
        ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        compression_ratio = ckpt_size_mb / info['output_size_mb']
        print(f"Compression: {ckpt_size_mb:.2f} MB → {info['output_size_mb']:.2f} MB "
              f"({compression_ratio:.1f}x smaller)")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
