# Training & Distillation Pipeline (Clean, Reproducible)

This project uses a single recommended entry point for training:

- `tools/train.py`

All outputs (checkpoints, TensorBoard logs, metadata) are written to an **artifacts directory** outside the source tree by default.

## 1) Artifacts directory

Default:

- `../mobileExtra_artifacts` (sibling of repo root)

Override:

- Set env: `MOBILEEXTRA_ARTIFACTS_DIR=/path/to/artifacts`
- Or pass: `python tools/train.py --artifacts-dir /path/to/artifacts ...`

Each run gets its own folder under:

- `$ARTIFACTS_DIR/runs/<timestamp>_train_<run_name>/`

Saved metadata:

- `config_merged.yaml`
- `cmd.txt`
- `git_commit.txt`, `git_diff.patch`, `git_status.txt`
- `env.txt`

## 2) Normal training

Example (student S1, no auto-resume):

```bash
python tools/train.py --presets base,dataset_colleague,train_ultra_safe \
  --set network.type=student_s1 \
       training.resume=false \
       training.batch_size=2
```

Notes:

- `tools/train.py` defaults `training.resume=false` unless `--resume_from` is specified.
- Checkpoints and TensorBoard logs will be under the run directory.

## 3) Distillation training (teacher -> student)

Example (distill from V2 teacher):

```bash
python tools/train.py --presets base,dataset_colleague,train_ultra_safe \
  --set network.type=student_s1 \
       training.resume=false \
       distill.enable=true \
       distill.teacher_ckpt=models/colleague/v2_init.ckpt \
       distill.teacher_type=v2 \
       distill.lambda=0.5
```

Safety:

- If `distill.enable=true`, missing/invalid `distill.teacher_ckpt` will fail fast.

## 4) Benchmarking policy

Project policy:

- **Forward-only latency** is the unified benchmark window for all benchmarking tools.
- End-to-end pipeline timing (log normalize / exp restore, etc.) must be reported separately.

Recommended forward-only compare:

```bash
python tools/model_profile_compare.py --models v1 v2 student_s1 student_s2 extranet \
  --height 256 --width 256 --iters 50 --warmup 10 --device cuda
```

## 5) Export (ONNX/TRT)

Export scripts live under `tools/export/`.

- `tools/export/export_onnx_raw.py` : forward-only ONNX (benchmarking)
- `tools/export/export_onnx_fused.py`: fused pre/post ONNX (deployment)

Artifacts are written under:

- `$ARTIFACTS_DIR/exports/onnx/`

