#!/usr/bin/env python3
"""
权重迁移工具: PatchNetwork V1 → V2

将现有checkpoint的权重迁移到PatchNetworkV2：
- 复用84%的权重（未改变的模块）
- 跳过encoder1和decoder4（结构改变，需重新训练）
- 生成ready-to-finetune的V2 checkpoint

用法:
    python tools/migrate_weights_v1_to_v2.py \\
        --v1-ckpt models/colleague/last.ckpt \\
        --output models/colleague/v2_init.ckpt
"""

import torch
import argparse
import sys
import os
from pathlib import Path
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.npu.networks.patch.patch_network import PatchNetwork as PatchNetworkV1
from src.npu.networks.patch.patch_network_v2 import PatchNetworkV2


def load_v1_checkpoint(ckpt_path: str, base_channels: int = 24):
    """加载V1 checkpoint"""
    print(f"\n[1/4] 加载V1 Checkpoint...")
    print(f"  路径: {ckpt_path}")

    # Import dependency class for unpickling
    try:
        train_dir = os.path.join(os.path.dirname(__file__), '..', 'train')
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)
        from patch_aware_dataset import PatchTrainingConfig
    except ImportError:
        class PatchTrainingConfig:
            pass

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract patch_network weights
    state_dict = checkpoint['state_dict']
    patch_network_state = {}

    for key, value in state_dict.items():
        if key.startswith('patch_network.'):
            new_key = key.replace('patch_network.', '')
            patch_network_state[new_key] = value

    print(f"  ✓ V1 state_dict keys: {len(patch_network_state)}")
    print(f"  ✓ Epoch: {checkpoint.get('epoch', 'unknown')}")

    return patch_network_state, checkpoint


def create_v2_model(base_channels: int = 24):
    """创建V2模型"""
    print(f"\n[2/4] 创建V2模型...")
    model_v2 = PatchNetworkV2(
        input_channels=7,
        output_channels=3,
        base_channels=base_channels,
        residual_scale_factor=1.0
    )
    print(f"  ✓ V2模型参数: {sum(p.numel() for p in model_v2.parameters()):,}")
    return model_v2


def migrate_weights(v1_state_dict, v2_model):
    """迁移权重: V1 → V2"""
    print(f"\n[3/4] 迁移权重...")

    v2_state_dict = v2_model.state_dict()

    # 需要跳过的模块（结构改变 + 新增层）
    skip_modules = [
        'encoder1',        # Gated → Depthwise Separable + Hardswish
        'decoder4',        # Gated → Depthwise Separable + Hardswish
        'input_padding',   # 新增: 7ch → 8ch (INT8对齐)
        'output_expand',   # 新增: 24ch → 8ch (INT8对齐)
        'output_conv',     # 修改: 24→3 变为 8→3 (shape不兼容)
    ]

    # 需要特殊处理的层（可能需要部分迁移）
    special_layers = [
        'input_proj',  # 输入通道从7→8可能影响第一个conv
    ]

    # 需要处理的移除组件（Phase 1已移除）
    removed_components = [
        'bottleneck.pos_encoding',
        'bottleneck.hierarchical_attention',
    ]

    # 统计
    total_keys = len(v2_state_dict)
    migrated_keys = 0
    skipped_keys = 0
    missing_keys = 0
    unexpected_keys = 0

    migrated_modules = set()
    skipped_modules = set()

    print(f"\n  正在迁移权重...")

    for v2_key in v2_state_dict.keys():
        # 检查是否应该跳过
        should_skip = False
        for skip_module in skip_modules:
            if v2_key.startswith(skip_module + '.'):
                should_skip = True
                skipped_modules.add(skip_module)
                break

        if should_skip:
            skipped_keys += 1
            continue

        # 尝试从V1加载
        if v2_key in v1_state_dict:
            # 检查shape是否匹配
            if v1_state_dict[v2_key].shape == v2_state_dict[v2_key].shape:
                v2_state_dict[v2_key] = v1_state_dict[v2_key]
                migrated_keys += 1

                # 记录迁移的模块
                module_name = v2_key.split('.')[0]
                migrated_modules.add(module_name)
            else:
                print(f"  ⚠️  Shape mismatch: {v2_key}")
                print(f"      V1: {v1_state_dict[v2_key].shape}")
                print(f"      V2: {v2_state_dict[v2_key].shape}")
                missing_keys += 1
        else:
            missing_keys += 1

    # 检查V1中有但V2不需要的key（removed components）
    for v1_key in v1_state_dict.keys():
        if v1_key not in v2_state_dict:
            # 检查是否是已知的移除组件
            is_removed_component = False
            for removed in removed_components:
                if v1_key.startswith(removed):
                    is_removed_component = True
                    break

            if not is_removed_component:
                # 检查是否在skip modules中
                is_in_skip = False
                for skip_module in skip_modules:
                    if v1_key.startswith(skip_module + '.'):
                        is_in_skip = True
                        break

                if not is_in_skip:
                    unexpected_keys += 1

    # 加载迁移后的权重
    result = v2_model.load_state_dict(v2_state_dict, strict=False)

    # 打印统计
    print(f"\n  迁移统计:")
    print(f"    V2总keys:       {total_keys}")
    print(f"    成功迁移:       {migrated_keys} ({migrated_keys/total_keys*100:.1f}%)")
    print(f"    跳过(新模块):   {skipped_keys} ({skipped_keys/total_keys*100:.1f}%)")
    print(f"    未找到:         {missing_keys}")
    print(f"    V1多余keys:     {unexpected_keys}")

    print(f"\n  迁移的模块:")
    for module in sorted(migrated_modules):
        print(f"    ✓ {module}")

    print(f"\n  跳过的模块(需重新训练):")
    for module in sorted(skipped_modules):
        print(f"    ⏸ {module}")

    if result.missing_keys:
        print(f"\n  Missing keys总数: {len(result.missing_keys)}")
        # 按模块分组
        missing_by_module = {}
        for key in result.missing_keys:
            module = key.split('.')[0]
            if module not in missing_by_module:
                missing_by_module[module] = []
            missing_by_module[module].append(key)

        for module, keys in sorted(missing_by_module.items()):
            print(f"    {module}: {len(keys)} keys")

    if result.unexpected_keys:
        print(f"\n  Unexpected keys总数: {len(result.unexpected_keys)}")
        # 分组显示
        unexpected_by_prefix = {}
        for key in result.unexpected_keys:
            prefix = '.'.join(key.split('.')[:2]) if '.' in key else key
            if prefix not in unexpected_by_prefix:
                unexpected_by_prefix[prefix] = 0
            unexpected_by_prefix[prefix] += 1

        for prefix, count in sorted(unexpected_by_prefix.items()):
            print(f"    {prefix}: {count} keys")

    return v2_model


def save_v2_checkpoint(v2_model, v1_checkpoint, output_path: str, base_channels: int):
    """保存V2 checkpoint"""
    print(f"\n[4/4] 保存V2 Checkpoint...")

    # 创建新的checkpoint，保留原始checkpoint的部分元数据
    new_checkpoint = {
        'state_dict': {f'patch_network.{k}': v for k, v in v2_model.state_dict().items()},
        'epoch': v1_checkpoint.get('epoch', 0),
        'global_step': v1_checkpoint.get('global_step', 0),
        'v2_migration': True,
        'v2_base_channels': base_channels,
        'v2_modified_modules': ['encoder1', 'decoder4', 'input_padding', 'output_expand', 'output_conv'],
        'v2_roi_strategies': [
            'Strategy #1: Hard-Swish Activation (ROI=266.4)',
            'Strategy #2: Input/Output Padding (ROI=30.0)',
            'Strategy #3: Phase 2 Training (ROI=19.1)'
        ],
        'v2_migration_note': 'Migrated from V1 with Top 3 ROI strategies. New modules randomly initialized: encoder1, decoder4, input_padding, output_expand, output_conv. Needs fine-tuning.',
    }

    # 保存optimizer和scheduler state如果存在（但fine-tune时可能需要重新初始化）
    if 'optimizer_states' in v1_checkpoint:
        new_checkpoint['optimizer_states'] = v1_checkpoint['optimizer_states']

    if 'lr_schedulers' in v1_checkpoint:
        new_checkpoint['lr_schedulers'] = v1_checkpoint['lr_schedulers']

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(new_checkpoint, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ 保存路径: {output_path}")
    print(f"  ✓ 文件大小: {file_size_mb:.1f} MB")

    print(f"\n✅ 权重迁移完成！")
    print(f"\n📋 V2优化策略:")
    print(f"  ✓ Strategy #1: Hard-Swish Activation (ROI=266.4)")
    print(f"  ✓ Strategy #2: Input/Output Padding (ROI=30.0)")
    print(f"  ✓ Strategy #3: Phase 2 Training (ROI=19.1)")
    print(f"\n📊 预期性能提升:")
    print(f"  • FP32推理: 721ms → ~470ms (1.54×提速)")
    print(f"  • INT8推理: 721ms → <100ms (7×+提速)")
    print(f"\n🔧 下一步:")
    print(f"  1. Fine-tune 5-10 epochs")
    print(f"     python tools/train.py --presets base,train_ultra_safe --set network.type=v2 training.max_epochs=10 training.resume=false")
    print(f"  2. 在验证集上测试性能和质量")
    print(f"  3. 使用profiling工具验证FP32性能提升")
    print(f"  4. 进行INT8量化并测试移动端性能")


def main():
    parser = argparse.ArgumentParser(description='迁移V1 checkpoint到V2')
    parser.add_argument('--v1-ckpt', type=str, required=True,
                       help='V1 checkpoint路径 (e.g., models/colleague/last.ckpt)')
    parser.add_argument('--output', type=str, required=True,
                       help='输出V2 checkpoint路径 (e.g., models/colleague/v2_init.ckpt)')
    parser.add_argument('--base-channels', type=int, default=24,
                       help='网络base_channels (default: 24)')

    args = parser.parse_args()

    print("="*70)
    print("权重迁移: PatchNetwork V1 → V2 (Top 3 ROI策略)")
    print("="*70)
    print("V2优化:")
    print("  • Hard-Swish Activation (ROI=266.4)")
    print("  • Input/Output Padding Alignment (ROI=30.0)")
    print("  • Phase 2 Depthwise Separable Conv (ROI=19.1)")
    print("="*70)

    # 检查V1 checkpoint
    v1_ckpt_path = Path(args.v1_ckpt)
    if not v1_ckpt_path.exists():
        print(f"❌ 错误: V1 checkpoint不存在: {v1_ckpt_path}")
        sys.exit(1)

    # 1. 加载V1 checkpoint
    v1_state_dict, v1_checkpoint = load_v1_checkpoint(str(v1_ckpt_path), args.base_channels)

    # 2. 创建V2模型
    v2_model = create_v2_model(args.base_channels)

    # 3. 迁移权重
    v2_model = migrate_weights(v1_state_dict, v2_model)

    # 4. 保存V2 checkpoint
    save_v2_checkpoint(v2_model, v1_checkpoint, args.output, args.base_channels)

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
