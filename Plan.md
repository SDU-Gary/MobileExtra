# HDR数据流重构计划（优化版）

## 概述

在与Gemini的讨论和现有代码审阅基础上，确认当前HDR数据流的根本矛盾：输入端采用`log1p`压缩和多处`[-1,1]`限制（`clamp`/`tanh`），而HDR学习应在物理线性的空间进行。这导致色彩偏移、高亮自发光区域异常和损失空间与目标空间不一致。

本计划将重构为：端到端在线性HDR空间运行的训练/推理流程；移除不必要的`log1p`与范围压缩；用一致的可视化色调映射辅助观测与感知损失。计划遵循“渐进式实施+充分验证”的原则，以降低风险。


## 核心问题诊断

- 非线性压缩：`log1p`破坏了HDR线性特性
- 范围限制：多处`clamp(-1,1)`丢失高动态信息
- 输出激活：`Tanh`将网络输出强制压缩至`[-1,1]`
- 损失空间：在受限/非线性空间计算，与物理规律不符

目标：构建端到端线性HDR工作流，保持数据动态范围，确保损失与网络学习空间一致。


## 详细修改方案

### 步骤0：准备性分析与架构整理（关键）

新增与强化的准备任务：
- 修复残差缩放因子不一致
  - `src/npu/networks/patch/patch_network.py`中的`residual_scale_factor=3.0`与目标1.0不符，需要统一至1.0
- 统一HDR管道（强调）
  - 统一`train/colleague_dataset_adapter.py`、`src/npu/networks/input_normalizer.py`、`train/patch_tensorboard_logger.py`中所有HDR处理与显示路径，确保“单一事实来源（SCALE_FACTOR/色调映射/色彩空间转换）”
  - 建议新增配置节`hdr_processing`集中管理：`scale_factor`、`enable_srgb_linear`、`vgg_input_mode`、`tone_mapping_for_display`
- 训练数据数值范围分析（用于SCALE_FACTOR选型）
  - 对`./data/processed_bistro/{ref, warp_hole}`采样统计像素分布（p90/p95/p99，通道分布、亮度分布、负值占比）
  - 产出：统计报告与推荐初值`SCALE_FACTOR`（建议使p95附近映射到~1.0），同时给出上限保护策略（如异常极端值的软裁剪/鲁棒损失）
- 稳定性预检
  - 采用“梯度裁剪（clip_grad_norm_）”而不是对网络输出的强行`clamp`；建议初值`max_norm=0.5~1.0`
  - 评估初期学习率（建议≤5e-5）、禁用AMP（float32）
- 可视化一致性（强调）
  - TensorBoard显示与VGG感知损失统一采用同一种色调映射（见步骤4与步骤5），以避免“看见的”和“VGG感知”的不一致

交付物：
- 数据分布统计报告（含p95/p99、通道/亮度直方图）
- 初始`SCALE_FACTOR`与上限策略建议
- 统一HDR配置草案（供后续阶段引用）


### 步骤1：数据预处理重构（`train/colleague_dataset_adapter.py`）

- 函数重命名：`_normalize_to_tanh_range()` → `_linear_hdr_preprocessing()`
- 线性HDR预处理（替代log1p/[-1,1]）：
  - RGB通道：
    - 可选（已确认Kornia可用）：sRGB→Linear（`kornia.color.rgb_to_linear_rgb`）；如源EXR已线性则跳过
    - 非负约束：`clamp(min=0.0)`
    - 线性缩放：`rgb_linear / SCALE_FACTOR`，`SCALE_FACTOR`来自步骤0统计结果（初值建议100.0，按统计报告调整）
  - 掩码通道：保持[0,1]
  - MV通道：保持原始像素位移（不缩放/不[-1,1]），仅做必要的鲁棒性处理（如异常值日志）
- 不再缩放至`[-1,1]`，返回值允许>1的线性HDR范围（经过线性缩放后的相对范围）

注意：此步骤完成后，训练与推理路径必须使用同一`SCALE_FACTOR`并通过配置传递，其他模块不得私自假设[-1,1]。


### 步骤2：残差学习逻辑（`train/residual_learning_helper.py`）

- `compute_residual_target()`：取消所有`clamp`与额外缩放，直接返回线性空间残差
- `reconstruct_from_residual()`：取消输出`clamp(-1,1)`，保持线性HDR范围；如需显示或导出，再走显示/导出路径进行色调映射
- 全局`SCALE_FACTOR`统一为1.0（残差不再额外缩放）


### 步骤3：网络输出层（`src/npu/networks/patch/patch_network.py`）

- 移除输出层`Tanh`，改为线性输出
- 将`residual_scale_factor`从3.0统一为1.0
- 稳定性：不再在网络输出端做`clamp`；改为训练层面的梯度裁剪（Lightning配置或优化器hook），并适度降低初始学习率；必要时考虑权重衰减与EMA


### 步骤4：损失计算（线性HDR + 感知用色调映射）

- 重建逻辑：在计算损失前，先从残差预测重建完整图像：`reconstructed = warped_rgb + residual_pred`
- 在“线性HDR空间”计算核心重建损失（L1/MSE/边界/边缘）
- VGG感知损失（关键改进）：
  - 不直接在HDR线性空间送VGG。改为对`reconstructed`与`target_rgb`施加“与显示一致的可逆色调映射”（如Reinhard + gamma 或 ACES + gamma），得到类似LDR的[0,1]范围，再喂入VGG（配合ImageNet标准化）
  - 初期感知权重建议显著降低（如0.01~0.05），避免干扰主损失；后续可调
- 去除所有`clamp(-1,1)`；必要的稳健性可通过鲁棒损失/权重调制实现


### 步骤5：可视化与反归一化（统一）

- 反归一化/显示路径：
  - 线性HDR →（可选 Linear→sRGB）→ 色调映射（Reinhard/ACES/Exposure）→ gamma → clamp到[0,1]用于显示
  - 与步骤4中VGG输入采用相同的色调映射策略，以保持“你看到的”和“VGG感知到的”一致
- 统一实现位置：
  - 建议新增一个小工具模块（如`src/npu/utils/hdr_vis.py`）或集中在`input_normalizer`中暴露“显示用转换”函数，被Dataset/TensorBoard/VGG调用，避免多处复制


### 步骤6：训练流程适配

- 配置参数更新（见“配置文件更新”）
- 学习率：建议降低到`5e-5`起步
- 梯度裁剪：在Lightning/训练循环中启用`gradient_clip_val`（如0.5）和/或`clip_grad_norm_`
- AMP：初期禁用混合精度；等数值稳定后再评估开启
- 验证项：记录线性HDR空间损失与VGG感知损失（tone-mapped）；TensorBoard可视化与VGG保持一致的tone mapping


### 步骤7：全面测试与验证

- 端到端训练小跑（少量epoch）验证数值稳定
- 统计损失曲线是否平稳、是否出现爆NaN；必要时进一步降低LR或提高权重衰减
- 可视化主观检查：色偏是否消除、自发光区域是否合理
- 性能基准：仅PC端（移动端量化暂缓）


## 预期影响与注意事项

- 正面：色彩准确性明显提升，高亮与自发光区域物理合理；损失与学习空间一致
- 挑战：损失量级与梯度分布变化；训练不稳定风险；需要重调LR/权重/裁剪
- 缓解：渐进实施、严格监控、先禁用AMP、采用梯度裁剪、降低VGG权重
- 暂不考虑移动端量化：本阶段专注PC端训练与效果达成；量化与INT8范围控制留后续专门任务


## 配置文件更新（建议）

新增/调整：

```yaml
hdr_processing:
  enable_linear_preprocessing: true
  scale_factor: <由步骤0统计确定>   # 初值如 100.0，后续根据统计调整
  enable_srgb_linear: true            # 已确认Kornia可用
  tone_mapping_for_display: reinhard  # display与VGG统一，选reinhard/aces/exposure
  gamma: 2.2

loss:
  weights:
    l1: 1.0
    mse: 1.0
    vgg_perceptual: 0.02   # 显著降低；基于tone-mapped输入
    edge: 0.1
    boundary: 0.1
  vgg_input_mode: tone_mapped  # 感知损失输入采用显示同款色调映射
  enable_hdr_loss: true        # 在线性HDR空间计算核心重建损失

training:
  learning_rate: 5e-5
  gradient_clip_val: 0.5
  enable_amp: false

network:
  remove_output_tanh: true
  residual_scale_factor: 1.0
```

注：
- `input_normalizer`若继续用于推理/工具化场景，需切换至“线性HDR→显示映射”的路径，不得再假设`[-1,1]`或`log1p`。
- Dataset、TensorBoard、VGG感知损失共享同一tone-mapping与gamma设置，避免感知与显示不一致。


## 成功标准

技术指标：
- 训练损失平稳下降，无NaN/Inf；线性HDR空间L1/MSE逐步改善
- 色偏消除，自发光/高亮区域物理合理
- 与旧流程相比PSNR/SSIM不退化，主观质量提升

验证清单：
- 步骤0统计报告产出并据此设定`SCALE_FACTOR`
- 统一配置落地：可视化/感知与显示色调映射一致
- 输出层无`tanh`，残差/重建全链路无`clamp(-1,1)`
- 训练侧启用梯度裁剪；AMP关闭


## 渐进式发布与回退方案

- 阶段化切换：先仅替换数据预处理+可视化（保持旧损失/网络输出），确认稳定后再移除`tanh`与`clamp`，最后切换感知损失输入为tone-mapped
- 配置开关：为`enable_linear_preprocessing`、`remove_output_tanh`、`vgg_input_mode`等提供一键回退
- 回退路径：保留旧版`Plan.backup.md`与旧配置，问题出现时快速切回


## 结论

在保持端到端线性HDR空间学习的前提下，统一“数据→网络→损失→可视化（含VGG感知输入）”的变换与配置，将显著减少色偏与过度压缩带来的信息丢失，并提升自发光/高亮区域的还原能力。本优化版计划强调：
- 先做数据分布统计确定`SCALE_FACTOR`，再实施代码改动
- 移除`tanh/clamp(-1,1)`等范围束缚，改用训练端梯度裁剪保证稳定
- 统一显示色调映射既用于可视化也用于VGG感知损失输入，保证观测与优化目标一致
- 移动端量化延后，专注PC端训练达成质量目标
