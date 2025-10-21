# Repository Guidelines

## Project Structure & Module Organization
Runtime code lives in `src/`: `gpu/` hosts warp kernels, `npu/` contains repair networks, `renderer/` offers motion-vector helpers, and `system/` implements schedulers. Training loops, adapters, and losses sit under `train/`, while `configs/` tracks YAML presets for colleague, residual-MV, and ultra-safe modes. Keep raw or processed assets in `data/`, checkpoints in `models/`, experiment logs in `logs/`, and consult `docs/training_data_flow.md` when wiring new ingestion stages.

## System Architecture Overview
### System Entry & Initialization
`start_colleague_training.py` validates `configs/colleague_training_config.yaml`, checks dependencies and dataset layout,然后通过 `subprocess` 调用 `train/patch_training_framework.py --config ...`。核心训练逻辑集中在 `train/patch_training_framework.py`，包括配置解析、Lightning 模块构建与调度、数据加载器装配，以及训练器实例化。所有参数均来源于 YAML 配置体系，便于版本化和环境复现。

### Training Flow
数据链路为 `ColleagueDatasetAdapter → ColleaguePatchDataset → torch.utils.data.DataLoader`，先生成 7 通道帧与目标，再在 patch 包装器内执行简单网格或重叠 crop 切片。优化策略使用 AdamW 结合 CosineAnnealingLR，支持 step 级 warmup。训练循环由 `PatchFrameInterpolationTrainer.training_step` 驱动，涵盖前向推理、`PatchAwareLoss` 损失分解、Lightning 反向传播与梯度裁剪。

### Inpainting Core Components
训练阶段直接使用 `PatchNetwork` 进行补丁级残差学习；在推理侧可选 `PatchBasedInpainting` 将 `HoleDetector`、`PatchExtractor` 与 `PatchFusion` 组装成完整的自动补洞流水线。`PatchNetwork` 支持全局标准化、per-patch 百分位归一化以及 log 域残差学习模式，可通过 YAML 配置切换。

### Network Parameters
主干为 5 层 U-Net（通道 64→96→128→192→256），编码/解码块由 `PatchGatedConv2d` 和 `PatchGatedConvBlock` 提供门控卷积与边界增强，瓶颈嵌入轻量自注意力模块；解码阶段统一采用 `ConvTranspose2d` 可学习上采样（必要时再用 `F.interpolate` 对齐尺寸），激活函数以 `LeakyReLU(0.2)` 为主，同时通过可配置的残差缩放因子控制输出幅度。

### Data Preprocessing
输入由 `warp_hole/ref/correct` EXR 组成的 7 通道张量，`ColleagueDatasetAdapter` 支持线性 HDR 预处理、可选 sRGB→Linear 转换、曝光缩放与 tone-mapping。洞区掩码约束至 `[0,1]`，运动矢量保持像素位移尺度，残差标签通过 `ResidualLearningHelper` 直接在 HDR 域内计算。

### Persistence & Recovery
Lightning `ModelCheckpoint` 负责 top-k 与 `last.ckpt` 存储，`pl.Trainer.fit` 会自动检测 `last.ckpt` 实现断点续训。TensorBoard Logger 支持版本复用，确保实验迭代共用同一日志目录。

### Monitoring & Visualization
`PatchTensorBoardLogger` 输出 patch 网格对比、洞区覆盖、训练/验证指标、梯度统计及误差分位数，内置 HDR tone-mapping（含 mu-law、自适应曝光）以便准确复现高动态范围效果。

## Build, Test, and Development Commands
Provision the Conda environment before running code:
```bash
conda env create -f environment.yml
conda activate mobileExtra
```
Common entry points include `python start_ultra_safe_colleague_training.py --config configs/ultra_safe_training_config.yaml` for guarded training, `python start_colleague_training.py --profile lightning` for the standard patch loop, and `python tools/dry_check.py --scene bistro1` for HDR pipeline checks. Run `pytest tests -q` to execute regressions; add device flags (e.g., `CUDA_VISIBLE_DEVICES=0`) if hardware-specific debugging is required.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case for functions and modules, and PascalCase for Lightning modules or dataclasses. Maintain type hints on public APIs, keep docstrings focused on safety constraints (boundary checks, memory guards), and prefer `logging` over bare `print`. Preserve environment toggles such as `TRAINING_FRAMEWORK_IMPORT_WARN` so optional imports remain silent by default.

## Testing Guidelines
PyTest is the test runner; place new files in `tests/` named `test_<component>.py` and seed randomness via `torch.manual_seed`. Synthetic inputs should respect the documented 270×480 patch size to keep GPU memory predictable. For targeted debugging, run a single file (`pytest tests/test_residual_network.py -k residual`) and stash artifacts only inside `logs/`.

## Commit & Pull Request Guidelines
The published bundle omits Git history, so follow Conventional Commits (`feat:`, `fix:`, `refactor:`) with subjects under 72 characters and mention affected configs or scripts in the body. Group related changes per commit, note GPU/NPU or latency implications when touching critical paths, and include reproduction commands in the PR description. Link issues, attach metric deltas (SSIM, latency, stability), and call out required datasets so reviewers can stage them.

## Data & Configuration Tips
Do not commit proprietary footage; use sanitized samples under `data/demo/` and update ignores when introducing new asset paths. Add configuration variants as new YAML files instead of overwriting defaults, and host checkpoints over 50 MB externally, linking them in the PR summary with access guidance.

## Adversarial Training (Optional)
- GAN 训练通过 `gan.*` 配置启用，默认关闭；启用后 Lightning 进入手动优化流程。
- 判别器位于 `src/npu/networks/discriminator.py`（PatchGAN，RGB+mask 条件输入），损失定义在 `train/gan_losses.py`。
- `warmup_epochs` 控制仅重建阶段；之后按 `lambda_adv_start → lambda_adv_end` 线性增大对抗权重，判别器优化器参数独立配置。
- 相关日志（`train_adv_loss`, `train_loss_d`, `gan_lambda_adv` 等）会写入 TensorBoard，便于监控训练稳定性。

## Known Architecture Issues (Pending Fix)
- **训练/推理补丁尺寸不一致**：训练 pipeline 在重叠裁剪模式下产出 256×256 patch（见 `configs/colleague_training_config.yaml` 与 `train/patch_training_framework.py`），而推理时 `PatchExtractorConfig.patch_size` 仍为 128，导致网络面对的输入分布不一致，需要统一补丁尺寸。
- **边界掩码语义错配**：已在 `train/patch_training_framework.py` 中移除 `boundary_override`，训练/推理解耦统一改为使用 warped 输入的内部边界检测。
- **轻量级注意力尺度设置不合理**：当前尺度固定为 `[32, 16, 8]`（`src/npu/networks/patch/lightweight_attention.py`），当瓶颈特征仅 16×16 时会被迫插值放大，无法提供真实多尺度上下文，需根据实际尺寸动态调整。
