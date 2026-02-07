# 🛠️ Model Training & Benchmark Tools

模型训练加速和测试工具集合

---

## 📦 工具列表

### 1. **Checkpoint → PTH 转换器**
`tools/convert_ckpt_to_pth.py`

将PyTorch Lightning checkpoint转换为纯净的state_dict文件。

**特点**:
- ✅ 提取纯净的model weights（移除optimizer/scheduler）
- ✅ 自动处理Lightning的命名空间（`patch_network.xxx`）
- ✅ 可选提取discriminator weights
- ✅ 文件大小压缩：169MB → ~10MB（~17x）

**用法**:
```bash
# 基础用法
python tools/convert_ckpt_to_pth.py \
    --ckpt models/colleague/last.ckpt \
    --output models/patch_network.pth

# 自动命名（保留checkpoint信息）
python tools/convert_ckpt_to_pth.py \
    --ckpt models/colleague/patch-model-epoch=234-val_loss=11.63.ckpt \
    --output models/ \
    --auto-name

# 保存完整模型（包括optimizer）
python tools/convert_ckpt_to_pth.py \
    --ckpt models/colleague/last.ckpt \
    --output models/full_model.pth \
    --full-model

# 同时提取discriminator
python tools/convert_ckpt_to_pth.py \
    --ckpt models/colleague/last.ckpt \
    --output models/patch_network.pth \
    --extract-discriminator
```

**输出示例**:
```
[1/3] Loading checkpoint: models/colleague/last.ckpt
[2/3] Extracting model state_dict...
  - PatchNetwork parameters: 2,547,891
  - Discriminator parameters: 1,234,567
[3/3] Saving to: models/patch_network.pth
  ✅ Saved: models/patch_network.pth (10.23 MB)

============================================================
✅ Conversion Complete!
============================================================
Checkpoint: models/colleague/last.ckpt
Epoch: 234
Global Step: 23400
Output: models/patch_network.pth (10.23 MB)
Parameters: 2,547,891
Compression: 169.00 MB → 10.23 MB (16.5x smaller)
============================================================
```

---

### 2. **Inference Benchmark 工具**
`tools/benchmark_model.py`

全面测试模型推理速度、参数量、内存占用。

**注意**：当前项目已统一 benchmark 口径为 **forward-only**（只测 `model(x)`）。

**特点**:
- ✅ forward-only 统一计时窗口（跨脚本结果可比）
- ✅ 支持 AMP(FP16 autocast) 测试
- ✅ CPU/CUDA设备支持

**用法**:
```bash
# 测试PTH文件（forward-only）
python tools/benchmark_model.py --model models/patch_network.pth --network-type v1

# 测试checkpoint（forward-only）
python tools/benchmark_model.py \
    --model models/colleague/last.ckpt \
    --from-ckpt

# 指定输入尺寸
python tools/benchmark_model.py \
    --model models/patch_network.pth \
    --height 270 --width 480

更多完整流程见：`docs/PIPELINE.md`

# CPU测试
python tools/benchmark_model.py \
    --model models/patch_network.pth \
    --device cpu

# 快速测试（减少迭代）
python tools/benchmark_model.py \
    --model models/patch_network.pth \
    --quick

# 不同base_channels模型
python tools/benchmark_model.py \
    --model models/patch_network_base16.pth \
    --base-channels 16
```

**输出示例**:
```
======================================================================
📊 BENCHMARK REPORT
======================================================================

📁 Model: models/patch_network.pth
  • Total Parameters: 2,547,891
  • Trainable Parameters: 2,547,891
  • Model Size (FP32): 10.23 MB
  • Model Size (FP16): 5.12 MB
  • Model Size (INT8): 2.56 MB

⚡ Inference Speed (Standard Patch):
  • Mean:   2.45 ms  (408.2 FPS)
  • Median: 2.42 ms
  • Min:    2.38 ms
  • Max:    2.89 ms
  • Std:    0.12 ms
  • P95:    2.65 ms
  • P99:    2.78 ms
  • Peak GPU Memory: 178.45 MB (0.174 GB)

======================================================================

🎯 Performance Assessment:
  Standard Patch (256×256): ✅ EXCELLENT - Meets <3ms mobile NPU target

======================================================================
```

---

### 3. **训练加速配置**

两个预配置的YAML文件用于加速训练。

训练建议直接通过 `tools/train.py` + `--presets` + `--set` 覆盖参数完成，不再维护多份独立的“fast/ultra-fast” config 文件。

---

## 🚀 典型工作流

### 场景1: 快速训练并测试小模型

```bash
# 1. 训练50 epochs（示例：学生网络 + 小batch + 关闭resume）
python tools/train.py --presets base,dataset_colleague,train_ultra_safe \
  --set network.type=student_s1 training.max_epochs=50 training.batch_size=2 training.resume=false

# 2. 转换checkpoint为pth
python tools/convert_ckpt_to_pth.py \
    --ckpt models/colleague_ultra_fast/last.ckpt \
    --output models/ultra_fast.pth

# 3. Benchmark测试
python tools/benchmark_model.py \
    --model models/ultra_fast.pth \
    --base-channels 16 \
    --multi-size
```

### 场景2: 测试现有checkpoint性能

```bash
# 直接从checkpoint测试
python tools/benchmark_model.py \
    --model models/colleague/patch-model-epoch=234-val_loss=11.63.ckpt \
    --from-ckpt \
    --multi-size
```

### 场景3: 批量转换所有checkpoints

```bash
# 转换所有checkpoint
for ckpt in models/colleague/*.ckpt; do
    echo "Converting: $ckpt"
    python tools/convert_ckpt_to_pth.py \
        --ckpt "$ckpt" \
        --output models/pth_exports/ \
        --auto-name
done
```

### 场景4: 对比不同base_channels

```bash
通过 `--set network.base_channels=...` 或切换网络类型进行对比。

# 对比benchmark
python tools/benchmark_model.py --model models/ultra_fast.pth --base-channels 16 > bench_16.txt
python tools/benchmark_model.py --model models/fast.pth --base-channels 24 > bench_24.txt
diff bench_16.txt bench_24.txt
```

---

## 📊 性能对比参考

| 配置 | base_channels | 参数量 | 模型大小(FP32) | 预估推理时间 | 训练速度 |
|------|--------------|--------|---------------|-------------|---------|
| **Original** | 24 | ~2.5M | ~10MB | ~2.5ms | 1x (baseline) |
| **Fast** | 24 | ~2.5M | ~10MB | ~2.5ms | 2-3x |
| **Ultra-Fast** | 16 | ~1.1M | ~4.5MB | ~1.5ms | 4-6x |

---

## 💡 性能优化建议

### 训练加速技巧
1. **降低验证频率**: `validation.frequency: 20-30`
2. **禁用GAN**: `gan.enable: false`（如果不需要）
3. **简化loss**: 禁用wavelet/local_perceptual
4. **减少epochs**: 快速实验用50 epochs
5. **调整batch size**: 根据GPU内存适当增加
6. **使用AMP**: PyTorch Lightning自动支持（需在trainer中启用）

### 推理加速技巧
1. **降低base_channels**: 16 > 12 > 8
2. **INT8量化**: TFLite转换（~4x模型压缩，~2x推理加速）
3. **裁剪attention**: 如果性能瓶颈在attention模块
4. **知识蒸馏**: 用大模型教小模型（可选）

---

## 🔧 故障排查

### 问题1: 转换checkpoint失败
**症状**: `KeyError: 'patch_network.xxx'`
**解决**: 检查checkpoint是否来自正确的训练脚本，确保使用`--from-ckpt`标志

### 问题2: Benchmark显示NaN
**症状**: 推理时间或FPS显示NaN
**解决**: 检查模型加载是否成功，验证`--base-channels`参数是否匹配

### 问题3: CUDA out of memory (benchmark)
**症状**: Benchmark时OOM
**解决**: 使用`--batch-size 1`或`--device cpu`测试

### 问题4: 训练仍然很慢
**症状**: 使用fast配置后仍然慢
**解决**:
- 检查是否有数据加载瓶颈（降低`num_workers`）
- 确认GAN/wavelet确实禁用（查看训练日志）
- 考虑使用ultra-fast配置
- 检查磁盘I/O（OpenEXR读取可能慢）

---

## 📝 脚本详细参数

### convert_ckpt_to_pth.py
```
--ckpt PATH              输入checkpoint路径（必需）
--output PATH            输出文件或目录路径（必需）
--auto-name              自动生成输出文件名
--full-model             保存完整模型（包括optimizer）
--extract-discriminator  同时提取discriminator
```

### benchmark_model.py
```
--model PATH             模型文件路径（必需）
--from-ckpt              从checkpoint加载
--network-type STR       v1|v2|student_s1|student_s2|extranet
--base-channels INT      base_channels（v1/v2/extranet使用；student一般忽略）
--height INT             输入高度（默认256）
--width INT              输入宽度（默认256）
--batch INT              batch大小（默认1）
--device {cuda,cpu}      设备
--warmup INT             warm-up迭代数
--iters INT              测试迭代数
--amp                    CUDA autocast FP16
```

---

## 📚 相关文档

- 主项目文档: `CLAUDE.md`
- 训练配置说明: `configs/colleague_training_config.yaml`
- 网络架构文档: `src/npu/networks/patch/patch_network.py`

---

**更新日期**: 2025-01-11
**维护者**: Mobile Frame Interpolation Team
