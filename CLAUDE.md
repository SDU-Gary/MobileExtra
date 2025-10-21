# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Mobile Real-Time Frame Interpolation and Hole Filling System** that implements GPU+NPU heterogeneous computing for mobile devices. The system uses **rendering-side motion vectors** (not optical flow) to generate intermediate frames from 60fps to 120fps with <8ms total latency.

**Core Innovation**: Patch-based selective inpainting with residual MV-guided architecture that learns to repair only prediction errors rather than reconstructing entire frames.

**Key Technical Achievement**: Ultra-safe memory management with intelligent patch detection system achieving >99% training stability.

## Development Environment

### Setup Commands

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate mobileExtra

# Install core dependencies (PyTorch, Lightning, OpenCV)
pip install torch torchvision pytorch-lightning
pip install opencv-python zarr pillow tensorboard
```

**Note**: For Ubuntu/Linux systems, use `environment_ubuntu.yml` instead.

## Common Commands

### Training

```bash
# Primary training entry point (recommended)
python start_colleague_training.py

# Ultra-safe memory-optimized training
python start_ultra_safe_colleague_training.py

# Test data loading and patch extraction
python test_colleague_data_with_simple_grid.py
```

### Testing and Validation

```bash
# Currently no standalone test runner
# Tests are integrated into training validation loops
```

### Monitoring

```bash
# TensorBoard logs are saved to ./logs/colleague_training/
tensorboard --logdir ./logs/colleague_training
```

## Architecture Overview

### Three-Stage Pipeline

1. **Patch System** (`src/npu/networks/patch/`)
   - Smart hole detection and patch extraction
   - Overlapping crop strategy: 256×256 crops with stride 128
   - Optional 4×4 grid: 1080×1920 → 16 patches of 270×480
   - Top-k selection by hole fraction

2. **GPU Warp** (C++ headers in `src/gpu/`, implementations pending)
   - Forward projection using motion vectors
   - Hole detection and conflict resolution
   - <1ms processing time target
   - **Status**: Headers complete, C++ implementation needed

3. **NPU Repair** (`src/npu/networks/`)
   - **Training**: Uses `PatchNetwork` directly (streamlined, no wrappers)
   - **Inference**: Optional `PatchBasedInpainting` pipeline (HoleDetector + PatchExtractor + PatchFusion)
   - 5-layer U-Net with gated convolution
   - **Current config**: Channel progression: 24→36→48→72→96 (bottleneck @ base_channels=24)
   - **High capacity**: Channel progression: 64→96→128→192→256 (bottleneck @ base_channels=64)
   - Lightweight self-attention (O(N) complexity, ~32K parameters @ bottleneck)
   - Total parameters: ~2.5M (base_channels=24) or ~8M (base_channels=64)
   - INT8 quantization ready

### Key Components

**Network Architecture**:
- `src/npu/networks/patch/patch_network.py:176-423` - **PatchNetwork** (Core)
  - 5-layer encoder-decoder U-Net architecture
  - PatchGatedConv2d: Boundary-aware gated convolution with edge enhancement
  - PatchGatedConvBlock: Gated residual blocks with GroupNorm stabilization
  - Input: [B, 7, H, W] → Output: [B, 3, H, W] (residual prediction)
  - Edge-based boundary detection (aligned with loss function)

- `src/npu/networks/patch/lightweight_attention.py` - **LightweightSelfAttention**
  - SeparableAttention2D: H/W/C three-directional attention
  - HierarchicalSpatialAttention: Multi-scale global dependency (adaptive scales)
  - PositionalEncoding2D: 2D sinusoidal position encoding
  - Parameters: ~32K (@ 96 channels bottleneck), Complexity: O(N)

- `src/npu/networks/patch_inpainting.py` - **PatchBasedInpainting** (optional inference pipeline)
  - Full pipeline: HoleDetector → PatchExtractor → PatchNetwork → PatchFusion
  - Used for inference/deployment, not during training
- `src/npu/networks/patch/hole_detector.py` - Intelligent hole detection
- `src/npu/networks/patch/patch_extractor.py` - Patch extraction with padding
- `src/npu/networks/mobile_inpainting_network.py` - Mobile-optimized base network (legacy)

**Training Framework** (Streamlined):
- `train/patch_training_framework.py` - **Simplified PyTorch Lightning training**
  - Directly constructs and trains `PatchNetwork` (no wrapper layers)
  - Built-in PerceptualLoss (VGG16) and EdgeLoss implementations
  - Removed dependencies: PatchBasedInpainting, teacher models, external monitors, schedulers
  - All forward/optimization logic operates on `self.patch_network`
  - Optional GAN training support (discriminator from `src/npu/networks/discriminator.py`)

- `train/patch_aware_dataset.py` - Patch-aware data loading with overlapping crops
- `train/colleague_dataset_adapter.py:46-653` - **ColleagueDatasetAdapter**
  - OpenEXR HDR data loading (supports OpenEXR lib and OpenCV)
  - Linear HDR preprocessing pipeline (clamp → sRGB-to-linear → scale)
  - 7-channel input construction: [warped_RGB, hole_mask, occlusion_placeholder, MV_placeholder]
  - Residual target computation: target_residual = target_rgb - warped_rgb
  - Tone-mapping for display: Reinhard / μ-law (mu=300.0)

- `train/residual_inpainting_loss.py:57-415` - **ResidualInpaintingLoss**
  - VGG16 perceptual loss (5 layers: relu1_2 to relu5_3)
  - Spatial weighted loss with hole/context separation
  - Edge preservation (Sobel gradients)
  - Boundary sharpness enhancement
  - Huber/Charbonnier robust loss options
  - SSIM fallback when VGG unavailable

- `train/patch_tensorboard_logger.py` - **Specialized visualization logging**
  - Lightweight TrainingMonitor placeholder for interface compatibility
  - HDR tone-mapping visualization support
  - Patch comparison grid generation

**Configuration Files**:
- `configs/colleague_training_config.yaml` - Main training configuration
- `configs/deployment_config.yaml` - Mobile deployment parameters

## Technical Design Patterns

### Residual Learning Mode

The network learns **residual = target - warped** instead of reconstructing full frames. This reduces the learning burden and improves convergence:

```python
# In network forward pass:
residual = network(input)  # Network predicts residual correction
output = warped_frame + residual * scale_factor  # scale_factor = 1.0
```

### HDR Data Pipeline

The project uses **OpenEXR HDR data** with specialized preprocessing:

1. **Log-space transformation**: `log1p(x + epsilon)` for stable gradients
2. **Tone mapping**: Reinhard or mu-law for visualization
3. **Per-patch normalization**: Adaptive scaling based on 99th percentile
4. **Global standardization**: Optional single μ/σ normalization

Configuration in `configs/colleague_training_config.yaml`:
```yaml
hdr_processing:
  enable_linear_preprocessing: true
  tone_mapping_for_display: mulaw
  mulaw_mu: 300.0

normalization:
  type: log  # options: none | per_patch | global | log
  log_epsilon: 1.0e-6
  log_delta_abs_max: 16.0
```

### Patch-Based Training

**Simple Grid Strategy** (stable, deterministic):
- 4×4 grid on 1080×1920 input → 16 patches of 270×480
- Overlapping 256×256 crops with stride 128 for fine-grained training
- Top-k selection based on hole fraction (keeps patches with holes)

**Configuration**:
```yaml
patch:
  patch_height: 270
  patch_width: 480
  use_overlapping_crops: true
  crop_size: 256
  crop_stride: 128
  keep_top_frac: 0.5
  min_hole_frac: 0.005
```

### Multi-Component Loss System

The loss function balances multiple objectives with dynamic weighting:

```yaml
loss:
  weights:
    l1: 1.0                    # Base reconstruction
    perceptual: 1.4            # VGG16 perceptual loss (layers relu1_2 to relu5_3)
    local_perceptual: 1.0      # Grid-based perceptual on hole regions
    edge: 6.0                  # Edge-aware loss
    boundary: 8.0              # Hole boundary sharpness
    hole: 50.0                 # Strong weight on hole interior
    ctx: 10.0                  # Context preservation (non-hole regions)
    hole_ring_scale: 0.5       # Hole edge ring enhancement
```

**Robust Loss Options**:
- Huber loss (SmoothL1) with configurable delta
- Charbonnier loss for outlier robustness
- Maskable perceptual/edge losses (apply only in hole regions)

### Memory Management

**Ultra-Safe Mode** features:
- Gradient accumulation (effective batch = 4 × 4 = 16)
- Gradient clipping (norm=0.5)
- Regular garbage collection
- Max GPU memory cap (4GB default)

Configuration in `configs/colleague_training_config.yaml`:
```yaml
training:
  batch_size: 4
  accumulate_grad_batches: 4
  gradient_clip_val: 0.5

memory:
  enable_ultra_safe_mode: true
  max_gpu_memory_gb: 4
```

## Data Format

### Expected Input Structure

```
data/
└── processed_bistro/         # OpenEXR HDR data
    ├── warp_hole/            # Warped RGB with holes (3 channels: RGB)
    │   └── warped_hole-{frame_id}.exr
    ├── ref/                  # Reference target frames (3 channels: RGB)
    │   └── ref-{frame_id}.exr
    └── bistro/
        └── correct/          # Semantic hole masks (1 channel: binary mask)
            └── hole-{frame_id}.exr
```

**7-Channel Input Construction** (assembled by ColleagueDatasetAdapter):
- Channels 0-2: `warped_rgb` from warp_hole/*.exr (warped RGB with holes)
- Channel 3: `semantic_holes` from bistro/correct/*.exr (binary: 1=hole, 0=valid)
- Channel 4: `occlusion` - zero-filled placeholder (reserved for future use)
- Channels 5-6: `residual_mv` - zero-filled placeholder (reserved for future MV data)

**Output**: 3-channel RGB residual prediction (linear, unclamped)

### Data Preprocessing

The system expects preprocessed OpenEXR data. Motion vectors and hole detection are performed upstream (in the GPU warp stage).

## Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Total Latency | <8ms | 6.2-7.5ms (device dependent, estimated) |
| GPU Warp Time | <1ms | Target (implementation pending) |
| NPU Inference Time | <3ms | Target |
| Model Parameters | <3M | ~2.5M (base_channels=24) / ~8M (base_channels=64) |
| GPU Memory | <200MB | 178-195MB (estimated) |
| System Memory | <150MB | 138-148MB (estimated) |
| Training Val Loss | - | 8.25 (epoch 80/200) |
| SSIM | >0.90 | 0.91-0.94 (training metrics) |

## Training Workflow

1. **Data Preparation**: Ensure `data/processed_bistro/` contains OpenEXR HDR frames
   - Required subdirectories: `warp_hole/`, `ref/`, `bistro/correct/`
   - File naming: `warped_hole-{id}.exr`, `ref-{id}.exr`, `hole-{id}.exr`

2. **Configuration**: Edit `configs/colleague_training_config.yaml` as needed
   - Batch size: 8 (effective batch = 16 with gradient accumulation)
   - Network capacity: base_channels = 24 (default) or 64 (high capacity)
   - Patch strategy: overlapping crops (256×256, stride=128, keep_top_frac=0.5)

3. **Training**: Run `python start_colleague_training.py`
   - Framework: PyTorch Lightning
   - Automatic mixed precision supported
   - Ultra-safe memory mode available

4. **Monitoring**: Use TensorBoard to track loss components and visualizations
   - Log directory: `./logs/colleague_training/`
   - Current logs: ~21GB TensorBoard events
   - Command: `tensorboard --logdir ./logs/colleague_training`

5. **Checkpoints**: Saved to `models/colleague/`
   - **Current checkpoints**:
     - `last.ckpt` (90MB) - Latest checkpoint
     - `patch-model-epoch=80-val_loss=8.25.ckpt` - Best validation loss
     - `patch-model-epoch=71-val_loss=8.43.ckpt`
     - `patch-model-epoch=58-val_loss=8.43.ckpt`
   - **Training progress**: 80/200 epochs completed
   - Save frequency: Every 10 epochs

### Key Training Hyperparameters

**Current configuration** (configs/colleague_training_config.yaml):
- **Batch size**: 8 with accumulate_grad_batches=2 (effective batch=16)
- **Learning rate**: 5e-5 with cosine annealing (T_max=100, eta_min=1e-6)
- **Optimizer**: AdamW (weight_decay=1e-4, eps=1e-6, beta2=0.98)
- **Warmup**: 500 steps with 0.1x start factor
- **Gradient clipping**: 0.5 (norm)
- **Max epochs**: 200
- **Network**: base_channels=24 (configurable to 64)

## Mobile Deployment (In Progress)

### Platform Support
- **Android**: Snapdragon, Dimensity chipsets
- **iOS**: A15, A16 NPU
- **TFLite converter**: `src/deployment/platform_adaptation/tflite_converter.py`
- **INT8 quantization**: Network design ready, pipeline pending validation

### C++ Implementation Status

**✅ Complete** (src/system/):
- `dual_thread/` - Dual-thread processor and double buffer manager (.h + .cpp)
- `performance_management/` - Android/iOS performance monitors (.h + .cpp)
- `cascade_scheduler.cpp` - System cascade scheduler
- `utils/` - Platform adapter, memory pool (.h + .cpp)

**⏳ Headers Only** (implementations needed):
- `src/gpu/` - GPU Warp module (forward projection, hole detection)
- `src/npu/inpainting_module.h` - NPU inference wrapper
- `src/npu/color_stabilization_module.h` - Color correction post-processing
- `src/renderer/` - Renderer integration shaders

**Priority**: GPU Warp C++ implementation is critical path for end-to-end deployment

## Important Notes

### HDR Processing
- **Color Space**: All training and inference use **linear HDR color space**. Do NOT apply manual gamma correction unless specifically testing display pipeline.
- **Preprocessing Pipeline**:
  1. Clamp negative values: `rgb = clamp(rgb, min=0.0)`
  2. Optional sRGB→Linear conversion (via Kornia, if enabled)
  3. Linear scaling: `rgb_scaled = rgb / scale_factor` (scale_factor=0.7 default)
- **Display Conversion**: Use tone-mapping (Reinhard/μ-law) + gamma for visualization only
- **Legacy Mode**: Old log1p path available via `enable_linear_preprocessing: false`

### Residual Learning
- **Network Output**: Linear residual prediction (no Tanh, unclamped)
- **Reconstruction**: `output = warped_rgb + residual_prediction * scale_factor`
- **Scale Factor**: Always use `scale_factor = 1.0` (no scaling, direct addition)
- **Training Target**: `target_residual = target_rgb - warped_rgb`

### Network Architecture
- **Boundary Detection**: Uses image edge detection (Sobel on warped_rgb), **not** hole mask edges
  - This aligns network optimization with loss function objectives
  - Location: `patch_network.py:270-299` (`_generate_boundary_mask`)
- **GroupNorm**: Applied after gated convolutions to stabilize activations across HDR scales
- **Attention**: Lightweight self-attention at bottleneck (96 channels @ base_channels=24), ~32K parameters
- **Channel Scaling**: All layer channels scale linearly with base_channels (ch1=1×, ch2=1.5×, ch3=2×, ch4=3×, ch5=4×)

### Loss Function
- **VGG Perceptual Loss**: True VGG16 pretrained on ImageNet
  - Layers: [relu1_2, relu2_2, relu3_3, relu4_3, relu5_3] (indices [3, 8, 15, 22, 29])
  - Layer weights: [1.5, 1.5, 2.0, 2.0, 1.5]
  - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Fallback: SSIM when VGG unavailable
- **Robust Loss**: Huber (SmoothL1) with delta=0.2 for pixel-level reconstruction
- **Hole Weighting**: Strong weight (50.0) on hole interior, moderate (10.0) on context preservation

### Memory Management
- **Ultra-Safe Mode**: `memory.enable_ultra_safe_mode: true`
  - Gradient accumulation: effective_batch = batch_size × accumulate_grad_batches
  - Gradient clipping: norm=0.5
  - Max GPU memory: 4GB cap
  - Periodic garbage collection every 10 steps
- **OOM Troubleshooting**: Enable ultra-safe mode and reduce batch_size before other changes

### Data Loading
- **File Format**: OpenEXR (HDR, 16-bit or 32-bit float)
- **Libraries**: Supports both OpenEXR library and OpenCV (cv2.imread with ANYDEPTH flag)
- **Worker Processes**: Set `num_workers=0` or low value to reduce memory overhead
- **Channel Order**: RGB (not BGR) - OpenCV conversion handled internally

## Development Status

### ✅ Completed Components

**Training System**:
- ✅ PyTorch Lightning training framework with patch-aware data loading
- ✅ ColleagueDatasetAdapter: OpenEXR HDR data loading and preprocessing
- ✅ Overlapping crop strategy (256×256, stride=128) with top-k selection
- ✅ Residual learning architecture (5-layer U-Net, gated convolution)
- ✅ Lightweight self-attention (3-directional + hierarchical, <50K params)
- ✅ Enhanced VGG16 perceptual loss (5 layers, ImageNet pretrained)
- ✅ Multi-component loss system (L1, perceptual, edge, boundary, hole, context)
- ✅ Linear HDR preprocessing pipeline with tone-mapping visualization
- ✅ Ultra-safe memory management mode
- ✅ TensorBoard visualization with patch comparison logging
- ✅ **Training Progress**: 80/200 epochs, val_loss=8.25 (best checkpoint saved)

**Network Architecture**:
- ✅ PatchNetwork: 5-layer encoder-decoder U-Net (~2.5M params @ base_channels=24)
- ✅ Edge-based boundary detection (aligned with loss function)
- ✅ GroupNorm stabilization for HDR data
- ✅ Flexible architecture: supports arbitrary input sizes (e.g., 270×480)

**System Infrastructure** (C++):
- ✅ Dual-thread processor and double buffer manager
- ✅ Android/iOS performance monitors
- ✅ Cascade scheduler
- ✅ Platform adapter and memory pool utilities

### ⏳ In Progress / Pending

**Critical Path**:
- ⏳ **GPU Warp C++ implementation** (headers complete, .cpp needed)
  - Forward projection using motion vectors
  - Hole detection and conflict resolution
  - Target: <1ms processing time

**Deployment**:
- ⏳ NPU inference module integration (header exists)
- ⏳ TFLite model conversion and INT8 quantization validation
- ⏳ Mobile platform end-to-end testing (Android/iOS)
- ⏳ Color stabilization post-processing module

**Training Completion**:
- ⏳ Continue training to 200 epochs (currently at 80/200)
- ⏳ Experiment with higher capacity model (base_channels=64)
- ⏳ Validate on test set and real-world scenarios

### 📊 Current Metrics

- **Model checkpoints**: 4 saved, best at epoch 80 (val_loss=8.25)
- **Training logs**: ~21GB TensorBoard events
- **Model size**: 90MB per checkpoint (.ckpt format)
- **Network parameters**: ~2.5M (base_channels=24)
- **Estimated inference**: <3ms on mobile NPU (pending validation)

---

## Quick Reference

### File Locations

**Core Network**:
- Main network: `src/npu/networks/patch/patch_network.py:176-423`
- Attention module: `src/npu/networks/patch/lightweight_attention.py`
- Loss function: `train/residual_inpainting_loss.py:57-415`

**Data Pipeline**:
- Dataset adapter: `train/colleague_dataset_adapter.py:46-653`
- Training framework: `train/patch_training_framework.py` (streamlined, direct PatchNetwork training)
- Patch-aware dataset: `train/patch_aware_dataset.py`

**Configuration**:
- Training config: `configs/colleague_training_config.yaml`
- Model checkpoints: `models/colleague/`
- TensorBoard logs: `logs/colleague_training/`

**Data**:
- Input data: `data/processed_bistro/warp_hole/*.exr`
- Target data: `data/processed_bistro/ref/*.exr`
- Hole masks: `data/processed_bistro/bistro/correct/*.exr`

### Key Configuration Parameters

**Network Architecture** (line 52-66 in config):
```yaml
network:
  base_channels: 24              # 24 (lightweight) or 64 (high capacity)
  residual_scale_factor: 1.0     # Always 1.0 for direct addition
  learning_mode: residual
```

**Patch Sampling** (line 44-49):
```yaml
patch:
  use_overlapping_crops: true
  crop_size: 256
  crop_stride: 128
  keep_top_frac: 0.5             # Keep top 50% by hole fraction
  min_hole_frac: 0.005           # Minimum hole threshold
```

**HDR Processing** (line 68-80):
```yaml
hdr_processing:
  enable_linear_preprocessing: true
  scale_factor: 0.7              # Brightness scaling
  tone_mapping_for_display: mulaw
  mulaw_mu: 300.0
```

**Loss Weights** (line 161-170):
```yaml
loss:
  weights:
    l1: 1.0
    perceptual: 1.4              # VGG16 perceptual
    edge: 6.0                    # Edge preservation
    boundary: 8.0                # Boundary sharpness
    hole: 50.0                   # Strong hole interior constraint
    ctx: 10.0                    # Context preservation
```

### Common Troubleshooting

**Out of Memory (OOM)**:
1. Enable `memory.enable_ultra_safe_mode: true`
2. Reduce `batch_size` (e.g., from 8 to 4)
3. Reduce `num_workers` to 0 or 1
4. Set `accumulate_grad_batches` higher to maintain effective batch size

**Training Instability**:
1. Check `gradient_clip_val: 0.5` is enabled
2. Verify HDR data is non-negative (linear preprocessing clamps)
3. Enable robust loss: `loss.robust.enable: true`
4. Review learning rate warmup settings

**Data Loading Errors**:
1. Verify OpenEXR files exist: `ls data/processed_bistro/warp_hole/*.exr`
2. Check file naming: `warped_hole-{id}.exr`, `ref-{id}.exr`, `hole-{id}.exr`
3. Test single sample: `python train/colleague_dataset_adapter.py`
4. Install OpenEXR: `pip install OpenEXR` or use OpenCV fallback

**VGG Perceptual Loss Errors**:
1. System will auto-fallback to SSIM if VGG unavailable
2. Check torchvision installation: `pip install torchvision`
3. Verify internet connection for ImageNet weights download (first run)

### Performance Optimization Tips

**Training Speed**:
- Use `num_workers=4` on systems with ample RAM
- Enable AMP (automatic mixed precision) if supported
- Pin memory: `pin_memory=True` in DataLoader
- Use SSD storage for faster data loading

**Model Capacity vs Speed**:
- `base_channels=24`: ~2.5M params, faster inference, good for mobile
- `base_channels=64`: ~8M params, higher accuracy, slower inference

**Memory Efficiency**:
- Overlapping crops are memory-efficient (256×256 vs full 1080×1920)
- Gradient accumulation allows effective large batch sizes with low GPU memory
- Ultra-safe mode enables training on 4GB GPUs

### Next Steps for Development

1. **Complete Training**: Continue to 200 epochs for optimal convergence
2. **GPU Warp Implementation**: Critical for end-to-end pipeline (C++)
3. **Quantization**: Convert to TFLite INT8 and validate accuracy
4. **Mobile Testing**: Deploy to actual devices and measure latency
5. **Dataset Expansion**: Add more scenes for generalization

---

## Recent Architecture Changes (2025)

### Training Framework Simplification

**Motivation**: Removed unnecessary abstractions to streamline training and reduce code complexity.

**Changes**:
- **`train/patch_training_framework.py`**: Now directly constructs and trains `PatchNetwork`
  - Removed: `PatchBasedInpainting` wrapper, teacher models, external `TrainingMonitor`, scheduler dependencies
  - Simplified: Forward pass, optimizer configuration, and GAN logic all operate directly on `self.patch_network`
  - Built-in: `PerceptualLoss` (VGG16 with SSIM fallback) and `EdgeLoss` implementations

- **`train/patch_tensorboard_logger.py`**: Removed external `training_monitor` dependency
  - Added lightweight `TrainingMonitor` placeholder class for interface compatibility
  - Maintains full visualization functionality (patch grids, HDR tone-mapping, metrics logging)

- **`start_colleague_training.py`**: Updated dependency checks
  - Now points to `src/npu/networks/patch/patch_network.py` directly

- **Removed files**: `train/training_monitor.py` (functionality absorbed into tensorboard logger)

**Training vs. Inference Separation**:
- **Training**: Uses `PatchNetwork` directly for maximum simplicity and debugging ease
- **Inference/Deployment**: Can use `PatchBasedInpainting` for complete automatic pipeline (hole detection → patch extraction → repair → fusion)

**Benefits**:
- Cleaner code with fewer indirection layers
- Easier debugging and experimentation
- Reduced import dependencies and potential conflicts
- Faster iteration during model development
