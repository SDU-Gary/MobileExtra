# Patch Inpainting 改进实验路线图（2025-10-20）

下面列出的策略按照「原理说明 → 最简代码修改 → 预期效果」组织，便于逐项 A/B 实验。原理部分尽量使用易懂的语言；代码段采用 `diff` 形式标注修改位置，并保留真实上下文，方便快速定位。

---

## 1. 训练 / 推理补丁尺度统一 + 多尺度采样

- **策略原理描述**  
  训练阶段的 `PatchAwareDataset` 目前默认输出 128×128 patch，但历史流程仍存在 256×256 重叠裁剪的提示，意味着训练与推理的输入分布并未完全一致。统一 patch 尺度可以让模型在训练时看到与推理相同的分布，避免因尺度差异导致的恢复伪影。进一步地，引入多尺度采样（小尺寸 + 原尺寸）能让网络在训练中就学会处理不同上下文范围的洞，提升稳定性。

- **最简代码修改**

  ```diff
  --- a/train/patch_aware_dataset.py:346
  +++ b/train/patch_aware_dataset.py:346
  @@
       patches_input = []
       patches_target_residual = []
       patches_target_rgb = []
       patches_metadata = []
  +    multi_scale_factors = getattr(self.config, "multi_scale_factors", [1.0])
  +    multi_scale_factors = [f for f in multi_scale_factors if f > 0]
   
  -    for patch_info in selected_patches:
  +    for patch_info in selected_patches:
          ...
  -        patch_input_tensor = torch.from_numpy(input_patch).float()
  -        ...
  -        if patch_input_tensor.shape[1] != 128 or patch_input_tensor.shape[2] != 128:
  -            patch_input_tensor = torch.nn.functional.interpolate(
  -                patch_input_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
  -            ).squeeze(0)
  -            ...
  +        base_input = torch.from_numpy(input_patch).float()
  +        base_residual = torch.from_numpy(residual_patch).float()
  +        base_rgb = torch.from_numpy(rgb_patch).float()
  +
  +        for scale in multi_scale_factors:
  +            scaled_size = (int(128 * scale), int(128 * scale))
  +            patch_input_tensor = base_input
  +            patch_target_residual_tensor = base_residual
  +            patch_target_rgb_tensor = base_rgb
  +
  +            if scaled_size != base_input.shape[1:]:
  +                patch_input_tensor = torch.nn.functional.interpolate(
  +                    base_input.unsqueeze(0), size=scaled_size, mode='bilinear', align_corners=False
  +                ).squeeze(0)
  +                patch_target_residual_tensor = torch.nn.functional.interpolate(
  +                    base_residual.unsqueeze(0), size=scaled_size, mode='bilinear', align_corners=False
  +                ).squeeze(0)
  +                patch_target_rgb_tensor = torch.nn.functional.interpolate(
  +                    base_rgb.unsqueeze(0), size=scaled_size, mode='bilinear', align_corners=False
  +                ).squeeze(0)
  +
  +            patches_input.append(patch_input_tensor)
  +            patches_target_residual.append(patch_target_residual_tensor)
  +            patches_target_rgb.append(patch_target_rgb_tensor)
  +            patches_metadata.append({**base_metadata, "scale": scale})
   ```

  > 备注：`base_metadata` 为现有循环里构造的 patch 元信息，可复用现有字段。

- **预计效果**  
  同一模型在训练即暴露于推理规模的洞区，减少分布漂移；多尺度采样可提升大型洞口与细节洞口的鲁棒性，使恢复结果更贴近参考图像。

---

## 2. 双阶段上下文适配（Dual Context Adapter）

- **策略原理描述**  
  参考 Patch-Adapter 的做法，先在低分辨率构建全局语义，再在原分辨率细化局部纹理。我们可以在 `PatchBasedInpainting` 中新增一个轻量的全局上下文编码器，将其输出通过 1×1 卷积融合到现有的 U-Net skip-connection。这相当于让补丁网络提前“看见”更大范围的上下文，提高语义一致性。

- **最简代码修改**

  ```diff
  --- a/src/npu/networks/patch_inpainting.py:196
  +++ b/src/npu/networks/patch_inpainting.py:196
  @@
       if self.config.enable_patch_mode:
           self.hole_detector = HoleDetector(self.config.hole_detector)
           self.patch_extractor = PatchExtractor(self.config.patch_extractor)
           self.patch_network = PatchNetwork(
               input_channels=input_channels,
               output_channels=output_channels,
               base_channels=self.config.patch_network_channels,
               residual_scale_factor=self.config.residual_scale_factor
           )
           self.patch_fusion = PatchFusion(self.config)
  +
  +      # NEW: 全局上下文编码器（低分辨率）
  +      self.global_context = nn.Sequential(
  +          nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
  +          nn.LeakyReLU(0.2, inplace=True),
  +          nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
  +          nn.LeakyReLU(0.2, inplace=True),
  +          nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)
  +      )
  +      self.context_proj = nn.Conv2d(64, self.patch_network.ch1, kernel_size=1)
  ```

  ```diff
  --- a/src/npu/networks/patch/patch_network.py:320
  +++ b/src/npu/networks/patch/patch_network.py:320
  @@
       u4 = self.up4(u3)
       if u4.shape[2:] != e1.shape[2:]:
           u4 = F.interpolate(u4, size=e1.shape[2:], mode='bilinear', align_corners=False)
  -    u4 = torch.cat([u4, e1], dim=1)
  +    if context_feat is not None:
  +        u4 = torch.cat([u4, e1, context_feat], dim=1)
  +    else:
  +        u4 = torch.cat([u4, e1], dim=1)
       u4 = self.up_conv4(u4, boundary_mask)
       u4 = self.decoder4(u4, boundary_mask)
   ```

  在前向函数 `_process_patches_in_batches` 中，将 `context_feat = self.context_proj(global_feat)` 作为额外参数传入即可。

- **预计效果**  
  低分辨率分支提供更完整的场景语义，有助于模型在大面积洞区恢复时避免错位或语义漂移，让输出更接近参考图像。

---

## 3. 区域分离注意力（Region-Wise Separated Attention）

- **策略原理描述**  
  SafePaint 提出在洞区与上下文分别建立注意力图，再融合结果，可以减少“洞区染色”或背景漂移。我们可以在瓶颈层引入一个专门的区域分离注意力模块，让网络在生成残差时分别关注填补区域与保留区域。

- **最简代码修改**

  ```diff
  --- a/src/npu/networks/patch/patch_network.py:205
  +++ b/src/npu/networks/patch/patch_network.py:205
  @@
        self.encoder5 = PatchGatedConvBlock(self.ch5, use_boundary_aware=False)
  -    self.bottleneck = LightweightSelfAttention(self.ch5, enable_position_encoding=True)
  +    self.bottleneck = RegionSeparatedAttention(self.ch5)
  ```

  ```diff
  --- a/src/npu/networks/patch/patch_network.py:45
  +++ b/src/npu/networks/patch/patch_network.py:45
    @@
    class PatchGatedConvBlock(nn.Module):
        ...
    
  +class RegionSeparatedAttention(nn.Module):
  +    def __init__(self, channels):
  +        super().__init__()
  +        self.fg_attn = LightweightSelfAttention(channels // 2, enable_position_encoding=True)
  +        self.bg_attn = LightweightSelfAttention(channels // 2, enable_position_encoding=True)
  +        self.mix = nn.Conv2d(channels, channels, kernel_size=1)
  +
  +    def forward(self, x, mask=None):
  +        if mask is None:
  +            return self.mix(x)
  +        fg_mask = mask
  +        bg_mask = 1.0 - mask
  +        fg_feat, bg_feat = torch.chunk(x, 2, dim=1)
  +        fg = self.fg_attn(fg_feat * fg_mask)
  +        bg = self.bg_attn(bg_feat * bg_mask)
  +        return self.mix(torch.cat([fg, bg], dim=1))
  ```

  在前向传播时，将洞区 mask 经过插值传入 `RegionSeparatedAttention`。

- **预计效果**  
  洞区与上下文分别注意力建模后，模型更容易保持背景颜色与纹理一致，同时聚焦于洞区结构重建，从而减少色差和边缘伪影。

---

## 4. 结构频带协同头（高 / 低频拆分）

- **策略原理描述**  
  Wavelet Transformer 等研究表明，高低频拆分有助于同时保持结构和纹理。可以在残差头输出前，使用离散小波（或可微分的 Haar 变换）将 `u4` 分解，分别处理低频（结构）和高频（细节），再合成。

- **最简代码修改**

  ```diff
  --- a/src/npu/networks/patch/patch_network.py:330
  +++ b/src/npu/networks/patch/patch_network.py:330
  @@
        u4 = self.decoder4(u4, boundary_mask)
  -      residual_prediction = self.output_conv(u4)
  +      low, high = self.wavelet_decompose(u4)
  +      low = self.low_head(low)
  +      high = self.high_head(high)
  +      residual_prediction = self.wavelet_reconstruct(low, high)
  ```

  ```diff
  --- a/src/npu/networks/patch/patch_network.py:120
  +++ b/src/npu/networks/patch/patch_network.py:120
  @@
        self.output_conv = nn.Conv2d(self.ch1, output_channels, 1, 1, 0)
        self.register_buffer('residual_scale_factor', torch.tensor(float(residual_scale_factor)))
        self.register_buffer('boundary_kernel', self._create_boundary_kernel())
  +      self.low_head = nn.Conv2d(self.ch1, output_channels, kernel_size=3, padding=1)
  +      self.high_head = nn.Conv2d(self.ch1, output_channels, kernel_size=3, padding=1)
  ```

  在 `PatchNetwork` 内补充 `wavelet_decompose`/`wavelet_reconstruct` 函数，可使用简单 Haar 滤波实现。

- **预计效果**  
  高频分支专注纹理细节，低频分支保障结构与亮度一致，可减少过度平滑或噪点，使补洞结果更接近参考图像。

---

## 5. 动态对数残差约束与分布对齐

- **策略原理描述**  
  目前 log 域残差限制通过固定 `log_delta_abs_max`，容易出现亮度溢出或过度裁剪。动态缩放（随 patch 动态范围调整）与分布约束（让预测残差的分布接近目标）可以提升 HDR 区域的亮度稳定性。

- **最简代码修改**

  ```diff
  --- a/train/patch_training_framework.py:1524
  +++ b/train/patch_training_framework.py:1524
  @@
  -                if self.log_delta_abs_max > 0.0:
  -                    delta_log = self.log_delta_alpha * torch.tanh(residual_pred_log) * self.log_delta_abs_max * scale
  -                else:
  -                    delta_log = (self.log_delta_scale * denom) * torch.tanh(residual_pred_log) * scale
  +                dynamic_cap = self.log_delta_abs_max
  +                if dynamic_cap <= 0:
  +                    dynamic_cap = self.log_delta_scale * denom.mean(dim=[1, 2, 3], keepdim=True)
  +                delta_log = self.log_delta_alpha * torch.tanh(residual_pred_log) * dynamic_cap * scale
  ```

  ```diff
  --- a/train/patch_training_framework.py:1589
  +++ b/train/patch_training_framework.py:1589
  @@
                if self.log_sup_type == 'huber':
                    per = F.smooth_l1_loss(delta_log, gt_delta_log, beta=self.log_sup_huber_delta, reduction='none')
                elif self.log_sup_type == 'charbonnier':
                    diff = delta_log - gt_delta_log
                    per = torch.sqrt(diff * diff + 1.0e-6)
                else:
                    per = torch.abs(delta_log - gt_delta_log)
  +              # NEW: KL 正则，约束预测/真实 log 残差分布
  +              pred_norm = torch.nn.functional.log_softmax(delta_log.view(delta_log.size(0), -1), dim=1)
  +              gt_norm = torch.nn.functional.softmax(gt_delta_log.view(gt_delta_log.size(0), -1), dim=1)
  +              kl_div = torch.nn.functional.kl_div(pred_norm, gt_norm, reduction='batchmean')
                num = (per * sup_mask).sum()
                den = sup_mask.sum() + 1e-6
                log_sup_loss = torch.nan_to_num(num / den, nan=0.0, posinf=1e6, neginf=0.0)
  +              log_sup_loss = log_sup_loss + self.log_sup_weight * kl_div
  ```

- **预计效果**  
  动态截断避免 log 残差过度压缩，KL 正则帮助预测分布向参考对齐，从而减少 HDR 区域的亮度异常，使补洞结果更稳定、贴近参考图像。

---

> **执行建议**：可以先从策略 1 着手，确认训练 / 推理尺度统一后再逐步开启其余策略。每个策略可单独配置开关，以便快速回滚。实验时建议通过 TensorBoard 的 `Validation/` 分组观察各子损失及 KL 正则项的变化，及时调整权重。  
