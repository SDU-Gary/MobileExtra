#!/usr/bin/env python3
"""
Unified training entry point.

- Supports loading a single config file or merging preset fragments.
- Provides key-value override via --set KEY=VALUE (dot notation).
- Calls run_patch_training() directly to avoid extra subprocess layers.

Usage examples:
  python tools/train.py --config configs/colleague_training_config.yaml
  python tools/train.py --presets base,model_patch_v2 --set training.batch_size=2
  python tools/train.py --presets base --dry-run  # show merged config only
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:
    print("[ERROR] PyYAML 未安装，请先运行: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def _import_training_entrypoint():
    # Import lazily so `--dry-run` works without heavy deps (torch, lightning)
    from train.patch_training_framework import run_patch_training  # type: ignore

    return run_patch_training

from tools.common.artifacts import (
    create_run_dirs,
    default_artifacts_dir,
    write_env_snapshot,
    write_git_snapshot,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config {path} should be a mapping")
        return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (mutates base)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def set_by_path(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split('.')
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]  # type: ignore[index]
    cur[parts[-1]] = value


def parse_scalar(val: str) -> Any:
    lowered = val.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    # presets (comma separated)
    if args.presets:
        preset_dir = ROOT / 'configs' / 'presets'
        for name in args.presets.split(','):
            name = name.strip()
            if not name:
                continue
            path = preset_dir / f"{name}.yaml"
            cfg = deep_merge(cfg, load_yaml(path))

    # single config file (if provided)
    if args.config:
        cfg = deep_merge(cfg, load_yaml(Path(args.config)))

    # inline overrides
    for kv in args.set or []:
        if '=' not in kv:
            raise ValueError(f"Override must be KEY=VALUE: {kv}")
        k, v = kv.split('=', 1)
        set_by_path(cfg, k, parse_scalar(v))

    if not cfg:
        raise ValueError("No configuration provided. Use --config or --presets.")
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified training launcher")
    parser.add_argument('--config', type=str, help='Path to full config yaml')
    parser.add_argument('--presets', type=str, help='Comma separated preset names under configs/presets')
    parser.add_argument('--set', dest='set', nargs='*', help='Override key-value pairs, e.g., training.batch_size=2')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--artifacts-dir', type=str, default=None,
                        help='Artifacts root (default: $MOBILEEXTRA_ARTIFACTS_DIR or ../mobileExtra_artifacts)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional human-readable run name (used in output directory name)')
    parser.add_argument('--dry-run', action='store_true', help='Print merged config and exit')
    parser.add_argument('--save-merged', type=str, help='Optional path to write merged config yaml')
    args = parser.parse_args()

    try:
        cfg = build_config(args)
    except Exception as e:
        print(f"[ERROR] Failed to build config: {e}")
        return 1

    if args.dry_run:
        print(yaml.dump(cfg, allow_unicode=True, sort_keys=False))
        if args.save_merged:
            Path(args.save_merged).write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
        return 0

    # Create a clean run directory (artifacts are kept OUTSIDE the source tree by default)
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else default_artifacts_dir(ROOT)
    run_dirs = create_run_dirs(artifacts_dir, run_name=args.run_name, prefix="train")

    # Inject --resume_from parameter into config if provided
    if args.resume_from:
        if 'training' not in cfg:
            cfg['training'] = {}
        cfg['training']['resume_from_ckpt'] = args.resume_from
        print(f"[INFO] Resume training from checkpoint: {args.resume_from}")

    # Default behavior: do NOT auto-resume unless explicitly requested.
    training_cfg = cfg.setdefault('training', {})
    if 'resume' not in training_cfg and not args.resume_from:
        training_cfg['resume'] = False

    # Route all outputs to run_dir
    monitoring_cfg = cfg.setdefault('monitoring', {})
    # Always override to keep artifacts out of the source tree.
    monitoring_cfg['model_save_dir'] = str(run_dirs.checkpoints_dir)
    monitoring_cfg['tensorboard_log_dir'] = str(run_dirs.tensorboard_dir)
    training_cfg['log_dir'] = str(run_dirs.logs_dir)
    print(f"[INFO] Run dir: {run_dirs.run_dir}")

    # Save the final effective config (after run_dir/resume injection)
    merged_yaml = yaml.dump(cfg, allow_unicode=True, sort_keys=False)
    (run_dirs.run_dir / 'config_merged.yaml').write_text(merged_yaml, encoding='utf-8')
    if args.save_merged:
        Path(args.save_merged).write_text(merged_yaml, encoding='utf-8')

    # Record minimal run metadata for reproducibility
    (run_dirs.run_dir / 'cmd.txt').write_text(' '.join(sys.argv) + '\n', encoding='utf-8')
    write_git_snapshot(run_dirs.run_dir, ROOT)
    write_env_snapshot(run_dirs.run_dir)

    # Run training directly
    run_patch_training = _import_training_entrypoint()
    success = run_patch_training(cfg)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
