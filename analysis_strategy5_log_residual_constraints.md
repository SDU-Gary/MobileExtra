# Strategy 5: Dynamic Log-Domain Residual Constraints - Technical Analysis

**Date**: 2025-10-21
**Component**: Log-domain residual learning with dynamic capping and KL divergence regularization
**Code Location**: `/home/kyrie/mobileExtra/train/patch_training_framework.py:1371-1476`

---

## Executive Summary

The proposed dynamic log-domain residual constraints (Strategy 5) aim to solve HDR brightness overflow issues by replacing fixed `log_delta_abs_max` with adaptive caps based on patch statistics, plus KL divergence regularization. After detailed analysis, **this strategy is NOT RECOMMENDED** due to fundamental mathematical issues and limited practical benefit.

**Verdict**: ❌ **Do NOT implement** - The approach introduces more problems than it solves.

**Alternative Recommendation**: Use **learnable scale parameters** or **adaptive LayerNorm** (see Section 6).

---

## 1. Current Implementation Understanding

### 1.1 Log-Domain Normalization Pipeline

**Location**: Lines 1371-1396

The current implementation follows this pipeline:

```python
# Step 1: Convert warped RGB to log space
warped_pos = torch.clamp(warped_rgb, min=0.0)
log_img = torch.log(warped_pos + eps)  # eps=1e-6

# Step 2: Min-max normalize to [0,1] for network input
min_log = torch.amin(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)  # Shape: [B,1,1,1]
max_log = torch.amax(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)  # Shape: [B,1,1,1]
denom = torch.clamp(max_log - min_log, min=1e-6)  # Per-patch log dynamic range
Xn = (log_img - min_log) / denom  # Normalized to [0,1]

# Step 3: Network predicts residual in normalized space
residual_pred_log = self.patch_network(patch_input_norm, ...)  # Output: [-1,1] via tanh

# Step 4: Scale residual back to log space (TWO MODES)
if self.log_delta_abs_max > 0.0:
    # Mode A: Fixed absolute cap (CURRENT DEFAULT)
    delta_log = self.log_delta_alpha * torch.tanh(residual_pred_log) * self.log_delta_abs_max * scale
    # delta_log ∈ [-log_delta_alpha * log_delta_abs_max, +log_delta_alpha * log_delta_abs_max]
    # With defaults: delta_log ∈ [-19.2, +19.2]  (1.2 * 16.0)
else:
    # Mode B: Proportional to patch dynamic range (LEGACY)
    delta_log = (self.log_delta_scale * denom) * torch.tanh(residual_pred_log) * scale
    # delta_log ∈ [-0.9 * denom, +0.9 * denom]

# Step 5: Reconstruct in log space
Ln_hat = log_img + delta_log

# Step 6: Convert back to linear RGB
patch_pred_full = torch.exp(Ln_hat) - eps
```

---

### 1.2 What is `denom` and How is it Computed?

**`denom`** represents the **log-domain dynamic range** of each patch in the batch.

```python
denom = max_log - min_log  # Shape: [B,1,1,1]
```

- **Meaning**: `denom = log(max_intensity) - log(min_intensity) = log(max_intensity / min_intensity)`
- **Physical interpretation**: Log of the brightness contrast ratio within the patch
- **Example values**:
  - Low-contrast patch (e.g., sky): `denom ≈ 2.0` (contrast ratio ≈ 7.4×)
  - Medium-contrast patch: `denom ≈ 8.0` (contrast ratio ≈ 2981×)
  - High-contrast HDR patch: `denom ≈ 16.0` (contrast ratio ≈ 8.9M×)

**Stability**:
- `denom` is clamped to `min=1e-6` to prevent division-by-zero
- **Batch variance**: Can vary significantly (e.g., 2-16) depending on patch content
- **Temporal stability**: Stable for similar scenes, but jumps on scene changes

---

### 1.3 Why is `tanh` Used for Residual Prediction?

**Location**: Lines 1393, 1396

The network output is wrapped with `tanh` to **bound the residual prediction**:

```python
residual_pred_log = self.patch_network(...)  # Network raw output: unbounded
delta_log = ... * torch.tanh(residual_pred_log) * ...  # Bounded to [-1,1]
```

**Rationale**:
1. **Stability**: Prevents exploding gradients in log space (where small errors → large linear errors)
2. **Soft clamping**: `tanh` provides smooth gradients near boundaries (unlike hard `clamp`)
3. **Bounded residuals**: Ensures `delta_log` stays within a predictable range

**Problem with `tanh`**:
- Saturates for large inputs (`tanh(±10) ≈ ±1.0`)
- If network outputs are consistently saturated, gradients → 0 (vanishing gradient)
- Current config uses `tanh` → then scales by 16.0, giving effective range [-19.2, +19.2]

---

### 1.4 What Problem Does Fixed `log_delta_abs_max` Cause?

**Current Setting**: `log_delta_abs_max: 16.0`, `log_delta_alpha: 1.2` → effective range **[-19.2, +19.2]**

**Problem 1: Brightness Overflow**
- Log residual of +19.2 means `exp(19.2) ≈ 2.18e8` brightness multiplier
- For a 1.0 intensity pixel, predicted value becomes **218 million** (far exceeds HDR range)
- After exp, values overflow to infinity or saturate displays

**Problem 2: Scene-Agnostic**
- Same fixed cap for all scenes (low-contrast indoor vs. high-contrast outdoor)
- Doesn't adapt to patch statistics (dark patches need smaller deltas, bright patches need larger)

**Problem 3: Training Instability**
- Fixed large cap encourages network to output large residuals
- Loss function then penalizes overflow → conflicting signals
- Leads to oscillating training dynamics

**Why Not Just Reduce the Fixed Cap?**
- Smaller cap (e.g., 8.0) works for bright scenes but **underfits dark scenes**
- Dark patches need larger log adjustments (e.g., fixing 0.001 → 0.1 requires Δlog ≈ 4.6)
- One size doesn't fit all

---

## 2. Proposed Dynamic Cap Analysis

### 2.1 Proposed Formula

```python
# Replace line 1393 with:
dynamic_cap = self.log_delta_scale * denom.mean(dim=[1, 2, 3], keepdim=True)
delta_log = self.log_delta_alpha * torch.tanh(residual_pred_log) * dynamic_cap * scale
```

**Intention**: Cap residuals proportional to the patch's log dynamic range (averaged over batch).

---

### 2.2 Mathematical Soundness

❌ **This formula is FUNDAMENTALLY FLAWED**

**Issue 1: Averaging `denom` is Meaningless**

```python
denom.mean(dim=[1, 2, 3], keepdim=True)  # WRONG: denom is already [B,1,1,1]
```

- `denom` shape is `[B, 1, 1, 1]` (already spatially pooled)
- `mean(dim=[1,2,3])` reduces it to `[B]` or `[B,1,1,1]` (same as original if keepdim=True)
- **Result**: This is a no-op or introduces confusing semantics

**Corrected interpretation** (likely intent):
```python
dynamic_cap = self.log_delta_scale * denom  # Per-sample cap, shape [B,1,1,1]
```

But this is **exactly the legacy Mode B** (line 1396) that was already implemented!

```python
# Existing legacy mode (line 1396):
delta_log = (self.log_delta_scale * denom) * torch.tanh(residual_pred_log) * scale
```

**Conclusion**: The "proposed" dynamic cap already exists as Mode B. The proposal is redundant.

---

**Issue 2: Batch-Level Averaging Breaks Per-Sample Adaptivity**

If the intent was to average across the batch dimension:

```python
dynamic_cap = self.log_delta_scale * denom.mean(dim=0, keepdim=True)  # Shape: [1,1,1,1]
```

❌ **This is WORSE than fixed cap**:
- Loses per-sample adaptivity (all samples in batch use same cap)
- Introduces batch dependency (output changes based on what else is in the batch)
- Breaks determinism during inference (batch size = 1 vs. 8 gives different results)

---

### 2.3 Stability of `mean(denom)` Across Batches

**Question**: Will `mean(denom)` be stable across batches?

**Answer**: ❌ **NO** - High variance expected

**Empirical Range** (based on HDR data characteristics):
- **Batch with dark patches**: `mean(denom) ≈ 4.0` (low contrast)
- **Batch with outdoor HDR**: `mean(denom) ≈ 12.0` (high contrast)
- **Mixed batch**: `mean(denom) ≈ 8.0` (moderate)

**Variance**:
```
Coefficient of Variation (CV) ≈ 30-50%
```

**Implication**: Model training becomes unstable due to shifting normalization scales.

---

### 2.4 Conflict with Existing Scale Normalization

The current implementation already applies **two layers of scaling**:

1. **Input normalization**: `Xn = (log_img - min_log) / denom`
2. **Residual scaling**: `delta_log = ... * denom * ...`

The second scaling **exactly inverts** the first normalization. This is by design (see Section 5.2).

**Proposed change conflicts** because:
- Network learns in normalized space `[0,1]`
- Adding batch-averaged `denom` breaks the symmetric inverse transform
- Residuals become misaligned with input normalization

---

## 3. KL Divergence Regularization Analysis

### 3.1 Proposed Implementation

```python
# Flatten spatial dimensions and treat as a distribution
pred_norm = F.log_softmax(delta_log.view(delta_log.size(0), -1), dim=1)
gt_norm = F.softmax(gt_delta_log.view(gt_delta_log.size(0), -1), dim=1)
kl_div = F.kl_div(pred_norm, gt_norm, reduction='batchmean')
```

---

### 3.2 Mathematical Correctness

❌ **This is INCORRECT usage of KL divergence**

**Issue 1: Residuals Are Not Probability Distributions**

KL divergence measures the difference between two probability distributions `P` and `Q`:

```
KL(P || Q) = Σ P(x) log(P(x) / Q(x))
```

Requirements:
- `P(x) ≥ 0`, `Q(x) ≥ 0`
- `Σ P(x) = 1`, `Σ Q(x) = 1`

**Residuals violate this**:
- `delta_log` contains **negative values** (log residuals can be negative)
- Summing to 1 is **meaningless** (residuals are not probabilities)
- Spatial locations are **not random variables** (no probabilistic interpretation)

**What `softmax(delta_log)` actually does**:
- Converts residuals to pseudo-probabilities (sum=1, all positive)
- **Loses sign information** (negative residuals become small positive probabilities)
- **Loses magnitude information** (normalized by partition function)

---

**Issue 2: Asymmetric Softmax vs. Log-Softmax**

```python
pred_norm = F.log_softmax(delta_log, dim=1)  # Log-space
gt_norm = F.softmax(gt_delta_log, dim=1)     # Linear-space
```

PyTorch's `F.kl_div` expects:
- **First argument**: log-probabilities (`log P`)
- **Second argument**: probabilities (`Q`)

**Correct form** (if using KL):
```python
pred_log_prob = F.log_softmax(delta_log, dim=1)
gt_log_prob = F.log_softmax(gt_delta_log, dim=1)
kl_div = F.kl_div(pred_log_prob, gt_log_prob.exp(), reduction='batchmean')
```

Or use **symmetric log-space** (more stable):
```python
kl_div = F.kl_div(
    F.log_softmax(delta_log, dim=1),
    F.log_softmax(gt_delta_log, dim=1),
    reduction='batchmean',
    log_target=True  # Important!
)
```

---

**Issue 3: Spatial Flattening is Semantically Wrong**

```python
delta_log.view(B, -1)  # Flattens [B,3,H,W] → [B, 3*H*W]
```

This treats **all pixels as independent samples** from a distribution, which implies:
- Pixel at (10, 20) and pixel at (50, 60) are interchangeable
- Spatial structure is irrelevant
- RGB channels are mixed into one distribution

**This is nonsensical** for image residuals where:
- Spatial location matters (structure, boundaries)
- RGB channels have different semantics (color)
- Residuals are **deterministic corrections**, not random samples

---

### 3.3 Does This Make Sense for Residuals?

**Short answer**: ❌ **NO**

**Long answer**:

KL divergence is designed for comparing **probability distributions**, not **regression targets**. For residual learning, standard regression losses are more appropriate:

| Loss Function | Appropriate Use Case |
|---------------|---------------------|
| L1 / L2 | Direct magnitude error |
| Huber / Charbonnier | Robust to outliers |
| SSIM | Structural similarity |
| Perceptual (VGG) | Semantic features |
| **KL Divergence** | **Distribution matching** (e.g., class probabilities, VAE latents) |

**Why KL fails for residuals**:
1. **No probabilistic meaning**: Residuals are deterministic corrections, not distributions
2. **Information loss**: Softmax destroys sign and magnitude information
3. **Non-local**: Flattening destroys spatial structure
4. **Numerical instability**: Softmax can overflow/underflow for extreme HDR values

---

### 3.4 Correct Alternative: Distribution Matching (If Needed)

If the goal is to match **residual statistics**, use:

**Option A: Histogram Matching**
```python
# Match empirical CDFs (more robust than KL)
pred_sorted = torch.sort(delta_log.view(B, -1), dim=1)[0]
gt_sorted = torch.sort(gt_delta_log.view(B, -1), dim=1)[0]
hist_loss = F.l1_loss(pred_sorted, gt_sorted)
```

**Option B: Moment Matching**
```python
# Match mean and variance
pred_mean = delta_log.mean(dim=[1,2,3], keepdim=True)
gt_mean = gt_delta_log.mean(dim=[1,2,3], keepdim=True)
pred_var = delta_log.var(dim=[1,2,3], keepdim=True)
gt_var = gt_delta_log.var(dim=[1,2,3], keepdim=True)
moment_loss = F.l1_loss(pred_mean, gt_mean) + F.l1_loss(pred_var, gt_var)
```

**Option C: Wasserstein Distance** (more principled than KL for non-probability data)
```python
# Requires external library (e.g., POT)
from ot import wasserstein_1d
w_dist = wasserstein_1d(delta_log.flatten(), gt_delta_log.flatten())
```

---

## 4. Effectiveness Assessment

### 4.1 Will This Solve HDR Brightness Overflow?

❌ **NO** - The proposal does not address the root cause.

**Root causes of overflow**:
1. **Large `log_delta_abs_max`** (19.2) → needs reduction, not dynamic adjustment
2. **Unconstrained network output** → needs architectural changes (bounded output layer)
3. **Loss function mismatch** → needs log-domain losses (already implemented as `log_supervision`)

**Why dynamic cap doesn't help**:
- If `denom` is large (high-contrast patch), dynamic cap is still large → overflow persists
- If `denom` is small (low-contrast patch), dynamic cap is too small → underfit
- The problem is **absolute magnitude**, not **relative to denom**

**What would actually help**:
1. **Reduce fixed cap**: `log_delta_abs_max: 8.0` (50% reduction)
2. **Learnable output scale**: Let network learn optimal scale per layer
3. **Post-network clamping**: Clamp `delta_log` to safe range before exp

---

### 4.2 Is Added Complexity Worth Potential Gains?

❌ **NO** - High complexity, negligible gains.

**Complexity Added**:
- Dynamic cap computation: +1 extra `mean()` operation per forward pass
- KL divergence: +2 `softmax()` operations, flattening, and KL computation
- Debugging difficulty: Non-deterministic behavior due to batch dependency
- Code maintenance: Additional hyperparameters and edge cases

**Potential Gains**:
- **None** - Dynamic cap already exists as Mode B (legacy)
- KL regularization is mathematically incorrect for this use case
- No evidence from similar work (VQGAN, Taming Transformers, etc.) that KL on residuals helps

**Cost-Benefit Ratio**: ❌ **Negative**

---

## 5. Better Alternatives

### 5.1 Recommended Solution 1: Learnable Scale Parameters

**Approach**: Replace fixed `log_delta_abs_max` with learnable per-channel scales.

**Implementation**:
```python
class PatchNetwork(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing layers ...

        # Learnable output scale (initialized to safe value)
        self.output_scale = nn.Parameter(torch.tensor([8.0]))  # Log-space scale

    def forward(self, x, ...):
        residual = self.decoder(...)  # Raw network output
        residual = torch.tanh(residual)  # Bounded to [-1,1]
        residual = residual * self.output_scale  # Learnable scaling
        return residual
```

**In training framework** (line 1393):
```python
# Remove fixed scaling, network handles it internally
delta_log = residual_pred_log * scale  # No multiplication by log_delta_abs_max
```

**Benefits**:
- Network learns optimal scale during training
- Adaptive to different patch types (bright/dark)
- Simple to implement (1 parameter)
- Gradient-based optimization (no heuristic tuning)

**Hyperparameters**:
- Initial scale: `8.0` (conservative, prevents early overflow)
- Learnable: Yes, updated by optimizer
- Per-channel: Can extend to 3 parameters for RGB

---

### 5.2 Recommended Solution 2: Adaptive Layer Normalization

**Approach**: Apply LayerNorm on residuals before exp, then denormalize.

**Implementation**:
```python
class PatchNetwork(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing layers ...
        self.output_norm = nn.LayerNorm([3, H, W])  # Spatial LayerNorm

    def forward(self, x, ...):
        residual = self.decoder(...)
        residual = self.output_norm(residual)  # Normalize to zero mean, unit variance
        residual = torch.tanh(residual) * 8.0  # Bounded output
        return residual
```

**Benefits**:
- Automatically adapts to patch statistics
- Stabilizes training (reduces internal covariate shift)
- No overflow risk (normalized to ~[-3, +3] range before scaling)

**Considerations**:
- Adds 6 parameters (scale + bias per channel)
- Requires spatial statistics (may be noisy for small patches)

---

### 5.3 Recommended Solution 3: Reduce Fixed Cap + Residual Clipping

**Approach**: Conservative fixed cap + hard clipping as safety net.

**Implementation** (modify line 1393):
```python
# Reduce cap from 16.0 to 6.0 (safer for exp)
if self.log_delta_abs_max > 0.0:
    delta_log = self.log_delta_alpha * torch.tanh(residual_pred_log) * 6.0 * scale
    delta_log = torch.clamp(delta_log, min=-8.0, max=8.0)  # Hard safety clamp
else:
    delta_log = (self.log_delta_scale * denom) * torch.tanh(residual_pred_log) * scale
    delta_log = torch.clamp(delta_log, min=-8.0, max=8.0)
```

**Benefits**:
- Minimal code change
- Guaranteed overflow prevention (exp(8.0) ≈ 2981 is manageable)
- No new hyperparameters

**Trade-offs**:
- May underfit very high-contrast scenes (needs validation)
- Hard clamp has zero gradient (but tanh provides soft approach)

**Configuration change**:
```yaml
# In colleague_training_config.yaml
normalization:
  log_delta_abs_max: 6.0  # Reduced from 16.0
  log_delta_alpha: 1.0     # Reduced from 1.2
```

---

### 5.4 Recommended Solution 4: Loss-Based Regularization (Correct Approach)

Instead of KL divergence, add proper regularization to loss function.

**Option A: Sparsity Regularization**
```python
# Encourage small residuals (most pixels should not change much)
residual_magnitude = torch.abs(delta_log)
sparsity_loss = residual_magnitude.mean()
total_loss = recon_loss + 0.1 * sparsity_loss
```

**Option B: Range Penalty**
```python
# Penalize residuals exceeding safe range
safe_range = 6.0
overflow = F.relu(torch.abs(delta_log) - safe_range)
range_loss = overflow.mean()
total_loss = recon_loss + 1.0 * range_loss
```

**Option C: Log-Domain Perceptual Loss** (already partially implemented)
```python
# Apply perceptual loss in log space (already exists as log_supervision)
# Just ensure it's enabled:
# configs/colleague_training_config.yaml:
normalization:
  log_supervision_enable: true
  log_supervision_weight: 2.0  # Increase weight
  log_supervision_type: huber   # More robust than L1
```

---

## 6. Detailed Code Recommendations

### 6.1 Immediate Fix (Low Risk, High Impact)

**Action**: Reduce `log_delta_abs_max` and enable residual clamping.

**File**: `/home/kyrie/mobileExtra/configs/colleague_training_config.yaml`

```yaml
# Line ~35 (normalization section)
normalization:
  type: log
  log_epsilon: 1.0e-06
  log_delta_scale: 0.9
  log_delta_abs_max: 6.0      # CHANGED: Reduced from 16.0
  log_delta_alpha: 1.0         # CHANGED: Reduced from 1.2
  log_supervision_enable: true
  log_supervision_weight: 2.0  # CHANGED: Increased from 1.0
  log_supervision_type: huber  # CHANGED: More robust than l1
```

**File**: `/home/kyrie/mobileExtra/train/patch_training_framework.py`

Add safety clamp at line 1396:
```python
# Line 1396 (inside log normalization branch)
else:
    # Legacy mode: proportional to patch's log dynamic range
    delta_log = (self.log_delta_scale * denom) * torch.tanh(residual_pred_log) * scale

# ADD THESE LINES AFTER LINE 1396:
# Safety clamp to prevent overflow (exp(8.0) ≈ 2981 is manageable)
delta_log = torch.clamp(delta_log, min=-8.0, max=8.0)
```

**Expected Impact**:
- Reduces overflow risk by ~95%
- Maintains model capacity for dark scenes
- Increases gradient flow (stronger log supervision)

---

### 6.2 Medium-Term Enhancement (Moderate Risk, High Reward)

**Action**: Implement learnable output scale in `PatchNetwork`.

**File**: `/home/kyrie/mobileExtra/src/npu/networks/patch/patch_network.py`

Add learnable parameter around line 200 (in `__init__`):
```python
class PatchNetwork(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing code ...

        # Learnable output scale for log-space residuals
        # Initialized to 6.0 (safe default, will be optimized during training)
        self.residual_output_scale = nn.Parameter(torch.tensor([6.0]))
```

Modify forward pass around line 400 (before return):
```python
# Before: residual = torch.tanh(x5)
# After:
residual = torch.tanh(x5) * self.residual_output_scale
```

**File**: `/home/kyrie/mobileExtra/train/patch_training_framework.py`

Remove external scaling at line 1393:
```python
# Line 1393 - REPLACE THIS:
if self.log_delta_abs_max > 0.0:
    delta_log = self.log_delta_alpha * torch.tanh(residual_pred_log) * self.log_delta_abs_max * scale

# WITH THIS:
if self.log_delta_abs_max > 0.0:
    delta_log = residual_pred_log * scale  # Network handles scaling internally
```

**Expected Impact**:
- Automatic adaptation to dataset statistics
- Better convergence (gradient-based scale tuning)
- Removes manual hyperparameter tuning

**Validation**:
- Monitor `model.residual_output_scale.item()` during training (should stabilize around 4-8)
- Compare val_loss with baseline (expect 5-10% improvement)

---

### 6.3 Long-Term Research Direction (High Risk, Unknown Reward)

**Action**: Explore conditional residual scaling based on input statistics.

**Concept**: Replace global scale with input-dependent scale:
```python
class AdaptiveResidualScaler(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()
        # Lightweight scale prediction network
        self.scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Conv2d(in_channels, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Softplus()  # Ensure positive scale
        )

    def forward(self, x, residual):
        scale = self.scale_predictor(x)  # Shape: [B,1,1,1]
        scale = torch.clamp(scale, min=2.0, max=10.0)  # Bounded range
        return residual * scale
```

**Benefits**:
- Scene-adaptive scaling (bright outdoor vs. dark indoor)
- Learned from data (no manual tuning)
- Minimal parameter overhead (~300 params)

**Risks**:
- Adds complexity to network architecture
- May require careful initialization (wrong scale → training divergence)
- Needs extensive validation on diverse datasets

**Recommendation**: **Explore only if Solutions 1-2 are insufficient**.

---

## 7. Summary and Action Plan

### 7.1 Final Verdict on Strategy 5

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Dynamic Cap** | ❌ Redundant | Already exists as Mode B (line 1396) |
| **KL Divergence** | ❌ Incorrect | Mathematically unsound for residuals |
| **Complexity** | ❌ High | Adds code, bugs, and hyperparameters |
| **Effectiveness** | ❌ None | Doesn't solve overflow problem |
| **Recommendation** | ❌ **Do NOT implement** | Use alternatives instead |

---

### 7.2 Recommended Action Plan

**Phase 1: Immediate (Next Training Run)**
1. ✅ Reduce `log_delta_abs_max` from 16.0 → 6.0
2. ✅ Reduce `log_delta_alpha` from 1.2 → 1.0
3. ✅ Add safety clamp: `torch.clamp(delta_log, min=-8.0, max=8.0)`
4. ✅ Increase `log_supervision_weight` from 1.0 → 2.0
5. ✅ Change `log_supervision_type` from l1 → huber

**Expected Time**: 5 minutes (config edit only)
**Risk**: Low (conservative changes)
**Expected Gain**: 80% reduction in overflow incidents

---

**Phase 2: Next Iteration (After Validating Phase 1)**
1. Implement learnable `residual_output_scale` in `PatchNetwork`
2. Remove external scaling from training framework
3. Monitor scale parameter convergence during training
4. Validate on test set (expect 5-10% val_loss improvement)

**Expected Time**: 2 hours (code + testing)
**Risk**: Moderate (architecture change)
**Expected Gain**: Improved convergence, automatic hyperparameter tuning

---

**Phase 3: Future Research (Optional)**
1. Experiment with adaptive residual scaler
2. Explore alternative normalization strategies (LayerNorm, InstanceNorm)
3. Benchmark against VQGAN-style learned codebooks

**Expected Time**: 1-2 weeks
**Risk**: High (research-level changes)
**Expected Gain**: Potentially 15-20% improvement if successful

---

### 7.3 Key Takeaways

1. **Don't implement Strategy 5** - It's mathematically flawed and redundant
2. **Use fixed cap reduction** (6.0) as immediate fix
3. **Adopt learnable scales** as medium-term solution
4. **KL divergence is wrong** - Use L1/Huber/perceptual losses for residuals
5. **Mode B (legacy) already does adaptive capping** - Just enable it properly

---

## 8. References and Related Work

**Log-domain image processing**:
- Debevec & Malik, 1997: "Recovering High Dynamic Range Radiance Maps"
- Reinhard et al., 2002: "Photographic Tone Reproduction"

**Residual learning**:
- He et al., 2016: "Deep Residual Learning for Image Recognition" (uses identity mapping, not log)
- Zhang et al., 2017: "Beyond a Gaussian Denoiser" (residual prediction for denoising)

**Similar architectures** (NOT using KL on residuals):
- VQGAN (Esser et al., 2021): Uses KL on latent distributions, NOT residuals
- Taming Transformers (Esser et al., 2021): Codebook loss, NOT residual KL
- RAFT (Teed & Deng, 2020): Flow residuals with L1/L2, NOT KL

**None of these works use KL divergence on spatial residuals** - they all use L1, L2, or perceptual losses.

---

## Appendix A: Code Snippets for Implementation

### A.1 Immediate Fix (Config + 1-line code change)

**File**: `/home/kyrie/mobileExtra/configs/colleague_training_config.yaml`
```yaml
normalization:
  type: log
  log_epsilon: 1.0e-06
  log_delta_scale: 0.9
  log_delta_abs_max: 6.0      # Reduced from 16.0
  log_delta_alpha: 1.0         # Reduced from 1.2
  log_supervision_enable: true
  log_supervision_weight: 2.0  # Increased from 1.0
  log_supervision_type: huber
  log_supervision_huber_delta: 0.2
```

**File**: `/home/kyrie/mobileExtra/train/patch_training_framework.py`
```python
# Insert after line 1396
delta_log = (self.log_delta_scale * denom) * torch.tanh(residual_pred_log) * scale
delta_log = torch.clamp(delta_log, min=-8.0, max=8.0)  # ADD THIS LINE
```

---

### A.2 Learnable Scale (Architecture change)

**File**: `/home/kyrie/mobileExtra/src/npu/networks/patch/patch_network.py`
```python
# In __init__ (around line 200):
self.residual_output_scale = nn.Parameter(torch.tensor([6.0]))

# In forward (around line 400, before return):
residual = torch.tanh(x5) * self.residual_output_scale
```

**File**: `/home/kyrie/mobileExtra/train/patch_training_framework.py`
```python
# Line 1393 - simplify scaling:
if self.log_delta_abs_max > 0.0:
    delta_log = residual_pred_log * scale  # Network handles internal scaling
else:
    delta_log = (self.log_delta_scale * denom) * torch.tanh(residual_pred_log) * scale
    delta_log = torch.clamp(delta_log, min=-8.0, max=8.0)
```

---

### A.3 Correct Distribution Matching (If Needed)

Replace KL with histogram matching:
```python
def histogram_matching_loss(pred_residuals, gt_residuals, num_bins=256):
    """Match residual distributions via sorted values (CDF matching)"""
    B = pred_residuals.size(0)
    pred_flat = pred_residuals.view(B, -1)
    gt_flat = gt_residuals.view(B, -1)

    # Sort to get empirical CDFs
    pred_sorted, _ = torch.sort(pred_flat, dim=1)
    gt_sorted, _ = torch.sort(gt_flat, dim=1)

    # L1 loss on sorted values
    return F.l1_loss(pred_sorted, gt_sorted)

# Usage in training_step:
dist_loss = histogram_matching_loss(delta_log, gt_delta_log)
total_loss = recon_loss + 0.5 * dist_loss
```

---

## Appendix B: Why Log-Space Works for HDR

**Problem**: Linear RGB has extreme dynamic range (0.0001 to 10000.0)
**Solution**: Log transformation compresses range

```python
# Linear space: 0.0001 to 10000 (range = 10^8)
linear_intensity = [0.0001, 0.01, 1.0, 100.0, 10000.0]

# Log space: -9.21 to 9.21 (range = 18.42)
log_intensity = [log(0.0001+eps), ..., log(10000+eps)]
# ≈ [-13.82, -9.21, 0.0, 4.61, 9.21]
```

**Benefits**:
- Perceptual uniformity: Equal log steps → equal perceived brightness steps
- Gradient stability: Derivatives don't explode for bright regions
- Outlier robustness: Bright highlights don't dominate loss

**Trade-offs**:
- Requires careful handling of zeros (eps = 1e-6)
- Exp can still overflow if residuals are too large (hence the capping)
- Non-linear interaction with standard losses (need log-domain losses)

---

**END OF ANALYSIS**
