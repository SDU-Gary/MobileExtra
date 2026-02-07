# Project Status & Entry Points

## Active entry points
- `tools/train.py` — unified launcher (supports `--config` or `--presets base,model_patch_v2`).
- `train/patch_training_framework.py` — core training logic (still callable directly with `--config`).

## Clean pipeline
- See `docs/PIPELINE.md` for the recommended training/distillation/benchmark/export workflow.

## Recommended presets
- `base` — mirrors legacy `colleague_training_config.yaml`.
- `model_patch_v2` — switches to PatchNetworkV2 + simple grid patches + safe batch size.
- `dataset_colleague` — dataset root/structure for processed Bistro data.
- `train_ultra_safe` — conservative batch/epoch settings for low-memory GPUs.

## Deprecated/legacy wrappers
- Legacy wrappers have been removed. Use `tools/train.py`.

## Notes
- PatchNetworkV2 is selectable via `network.type: PatchNetworkV2` in config/preset.
- Presets live under `configs/presets/` and are merged in order; later presets override earlier ones.
- Use `python tools/train.py --dry-run --presets ...` to inspect the merged config before launching.

## Artifacts policy
- Training outputs (checkpoints, TensorBoard logs, metadata) should be written to an **artifacts directory** outside the source tree.
- Default artifacts dir: `../mobileExtra_artifacts` (or override with `$MOBILEEXTRA_ARTIFACTS_DIR`).
