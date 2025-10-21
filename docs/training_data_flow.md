# 项目训练数据流与关键逻辑（讲解稿）

本文档用于向团队讲解项目完整的数据与训练链路：从数据读取→预处理→训练样本构建→网络前向与重建→损失计算→可视化记录；并展开“新增的数据获取逻辑（重叠裁剪 + 洞占比排序）”与“局部感知损失”的实现方式，便于排查与优化。

---

## 总览流程图（ASCII）

```
OpenEXR数据（processed_bistro）
  ├─ warp_hole/warped_hole-*.exr     → 输入RGB（带洞）
  ├─ ref/ref-*.exr                   → 目标RGB
  └─ bistro/correct/hole-*.exr       → 洞掩码（1=洞，0=非洞）

       │  数据读取（OpenEXR/OpenCV）
       ▼
ColleagueDatasetAdapter（train/colleague_dataset_adapter.py）
  - 组装输入7通道： [warped_rgb(3), holes(1), occlusion(1=0), residual_mv(2=0)]
  - 目标3通道：target_rgb
  - 线性HDR预处理（可选sRGB→Linear，scale_factor）

       │
       │（数据模式选择）
       ├─ PatchAwareDataset（train/patch_aware_dataset.py）
       │      └─ 简单网格/自适应patch（备用系）
       └─ ColleaguePatchDataset（train/patch_training_framework.py）
              └─ 新增“重叠crop索引”：按洞占比排序，选Top，min_hole_frac筛选

       │
       ▼
DataLoader (collate)
  - batch: {'patch_input':[N,7,h,w], 'patch_target_rgb':[N,3,h,w], 'patch_target_residual':[N,3,h,w]}

       │
       ▼
PatchFrameInterpolationTrainer.training_step（train/patch_training_framework.py）
  - 归一化：global | per_patch | log（从 YAML.normalization 读取）
  - 使用网络内部的边界检测（基于 warped RGB + 洞掩码）
  - 前向：
    * global/per_patch: 直接在归一化域预测残差 → 反归一化重建HDR
    * log: 预测 log-delta → exp 还原线性HDR
  - 得到线性HDR重建 patch_pred_full

       │
       ▼
PatchAwareLoss（train/patch_training_framework.py）
  - 多项损失（详见下文）：
    * 感知（全局/局部），边缘，边界，洞内像素，上下文保持，对数域监督（log模式），可选颜色统计
    * LDR域项统一经过 hdr_vis.tone_map（mulaw/gamma/exposure）计算
    * 掩码（洞/环）与面积补偿
  - 求和得到 total_loss

       │
       ▼
优化器更新（梯度累计后）
  - AdamW + CosineAnnealingLR + (可选 warmup)

       │
       ▼
可视化（train/patch_tensorboard_logger.py）
  - 按“训练批次计数”每300步记录一次（自定义TensorBoard）
  - 对比图：输入RGB|目标RGB|预测重建
    * 先对每面板单独 tone-map（mulaw/gamma/exposure）→ clamp → 再拼接
  - 保存PNG至 logs/.../visualization_images
```

---

## 数据读取与预处理

- 数据路径（processed_bistro）：
  - `warp_hole/warped_hole-*.exr` → 输入 `warped_rgb`（3通道）
  - `ref/ref-*.exr` → 目标 `target_rgb`（3通道）
  - `bistro/correct/hole-*.exr` → 洞掩码（1通道，洞=1）
- 适配器（`train/colleague_dataset_adapter.py`）：
  - 组装输入7通道：`[warped_rgb(3), holes(1), occlusion(1=0), residual_mv(2=0)]`
  - 线性HDR预处理（默认启用）：
    - RGB非负：clamp ≥ 0；可选 sRGB→Linear（默认 false）；`scale_factor` 缩放（当前 1.0）
  - 输出三元组：`(input_tensor[7,H,W], target_residual[3,H,W], target_rgb[3,H,W])`
    - `target_residual = target_rgb - warped_rgb`（线性HDR）

---

## 训练数据构建（新增获取逻辑）

- `ColleaguePatchDataset`（`train/patch_training_framework.py`）：
  - 重叠裁剪索引（Overlapping crops）：
    - 参数：`crop_size / crop_stride / keep_top_frac / min_hole_frac`
    - 对每张图在滑窗上生成候选 crop：
      - 取洞掩码子块 `cell`，洞占比 `frac = mean(cell)`（洞=1）
      - 丢弃 `frac < min_hole_frac` 的候选
      - 按 `frac` 排序保留 Top 比例 `keep_top_frac`
    - 训练时按全局索引直接裁剪 `patch_input / target_rgb / target_residual`
  - 简单网格（SimplePatchExtractor，备用）：固定4×4网格，16个patch

---

## 归一化与网络前向

- 归一化（YAML.normalization.type）：
  - `global`：`Xn=(warped−mu)/sigma`；网络预测 `residual_norm`；重建 `(Xn+residual_norm)*sigma+mu`
  - `per_patch`：`Xn=warped/b`（分位数尺度）；重建 `(warped/b + residual_norm)*b`
  - `log`：`log_img=log(warped+eps)` → 标准化为Xn；网络预测 `delta_log`；重建 `exp(log_img+delta_log)−eps`
- 训练与推理一致：由网络内部的 `_generate_boundary_mask` 根据 warped 输入估计边界权重。

---

## 损失函数组成（当前权重）

- 统一原则：凡在 LDR 域计算的项都走 `hdr_vis.tone_map`（mulaw + gamma + exposure；当前 `mu=300, gamma=1.0, exposure=0.8`），与显示一致。
- 掩码：洞区（掩码=1）+ ring（`maxpool−mask` × `hole_ring_scale`）

权重来自 `configs/colleague_training_config.yaml`：
- 全局感知（LDR）：`perceptual = 1.4`
- 局部感知（LDR，洞内网格）：`local_perceptual = 1.0`
- 边缘（LDR）：`edge = 6.0`
- 边界（LDR，边界权重图加权 L1）：`boundary = 8.0`
- 洞内像素（LDR，掩码鲁棒 L1）：`hole = 50.0`
- 上下文保持（LDR，非洞鲁棒 L1）：`ctx = 10.0`
- 对数域监督（log模式，仅洞/环）：`log_supervision_weight = 1.0`
- 面积补偿（逐样本，作用于 perceptual/edge/boundary）：当洞占比小于阈值时放大该分项，`scale=(1/hole_frac)^gamma` 限幅。

说明：
- `patch_l1`（全局像素 L1，不掩码）当前仅日志监控，不计入总损失。
- 局部感知：按网格选择洞占比≥阈值的子块，对每个子块分别做 VGG 感知并以子块内“洞面积”加权平均（面积越大权重越大）。

---

## 可视化记录（TensorBoard + PNG）

- 自定义记录（`train/patch_tensorboard_logger.py`）：
  - 已改为按“训练批次计数”触发（默认每 300 个 batch 可视化一次；YAML 可配置 `visualization.visualization_frequency`）。
  - 记录对比图（输入RGB|目标RGB|预测重建）：先逐面板 tone‑map → clamp → 再拼接；同时保存 PNG 到 `logs/.../visualization_images`。
- Lightning 默认标量仍按优化步（global_step）计数，可忽略或与自定义面板分开查看。

---

## 重点展开

### 新增的数据获取逻辑（重叠裁剪 + 洞占比排序）
- 目的：提升洞区样本密度，优先训练“有修复价值”的区域。
- 策略：
  1) 生成候选窗口（`crop_size/stride`）
  2) 计算候选“洞占比” = `mean(洞掩码子块)` （洞=1，非洞=0）
  3) 丢弃占比低于阈值 `min_hole_frac` 的候选
  4) 按占比排序，保留前 `keep_top_frac`
  5) 训练时按索引取 patch（同步裁剪 input/target/residual）

### 局部感知损失（local_perceptual）
- 启用：`enhanced.local_perceptual.enable=true` 或 `weights.local_perceptual>0`（当前两者皆满足；权重来自 weights=1.0）
- 过程：
  1) 对预测与目标（线性HDR重建）先 tone‑map 到 LDR
  2) 将 patch 划分为 `grid_rows×grid_cols` 网格（当前 4×4）
  3) 对每个网格子块，计算洞占比 `frac=mean(mask_cell)`；仅当 `frac≥hole_threshold_frac`（当前 0.05）时选为候选，并带 `padding`（8 像素）扩展区域
  4) 对每个子块分别计算 VGG 感知损失（失败时退化为 SSIM）
  5) 以“洞面积”加权平均：`num += area * lp_loss`，`den += area`，`local_perc = num/den`
  6) 乘以 `weights.local_perceptual=1.0` 累加到总损失

---

## 已知问题与对策（给讲解/答疑使用）

- 视觉“发亮发白”：主要源于 tone‑map 的 gamma 提亮中灰；已改为 `gamma=1.0` 并调低 `mu/exposure`（当前 `mu=300, exposure=0.8`）。
- `patch_hole_l1` 量级与历史不一致：当前在 LDR 域 + 掩码均值归一 + Huber（beta=0.2）上记录“未乘权重”的 per‑pixel 平均；自然在 0.02~0.06 的量级。若需与旧曲线可比，可额外记录“加权后值”或 HDR 版本的参考线。
- Windows 多进程 DataLoader 不稳定：已将 `training.num_workers=0`，避免 OpenEXR/Imath 多进程不稳定问题。
- 训练步 vs 可视化步：已从“优化步（受梯度累计影响）”改为“训练步（按 batch）”每 300 步可视化一次，直观对齐 937 batch/epoch。

---

## 参考文件与配置

- 数据与适配：`train/colleague_dataset_adapter.py`
- 新数据包装器（重叠裁剪索引）：`train/patch_training_framework.py`（`ColleaguePatchDataset`）
- 训练器与损失：`train/patch_training_framework.py`（`PatchFrameInterpolationTrainer`、`PatchAwareLoss`）
- Patch网络：`src/npu/networks/patch/patch_network.py`
- Tone-map：`src/npu/utils/hdr_vis.py`
- 可视化：`train/patch_tensorboard_logger.py`
- 配置：`configs/colleague_training_config.yaml`

---

## 附：当前关键配置（摘要）

- hdr_processing：`tone_mapping=mulaw`，`gamma=1.0`，`exposure=0.8`，`mulaw_mu=300`（训练与显示统一）
- normalization：`type=log`（启用 log‑supervision，weight=1.0）
- loss.weights：`perceptual=1.4`，`local_perceptual=1.0`，`edge=6.0`，`boundary=8.0`，`hole=50.0`，`ctx=10.0`
- visualization：按“训练步”每 300 步记录一次（PNG + TB 图像）
- DataLoader：`num_workers=0`（Windows 稳定性）

---

如需将内容导入到 PPT，可直接复制本 Markdown；若需 Mermaid/Visio 版结构图，请告知，我可提供可视化源码或图片版本的流程图。
