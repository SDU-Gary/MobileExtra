# NoiseBase数据预处理使用指南

## 📋 概述

本指南介绍如何使用优化版的NoiseBase数据预处理流水线，将NoiseBase的Zarr格式数据转换为适合MobileInpaintingNetwork训练的6通道数据。

## 🚀 快速开始

### 1. 环境准备

确保安装了必要的依赖：

```bash
pip install numpy opencv-python matplotlib tqdm zarr torch torchvision
```

### 2. 数据准备

确保您的NoiseBase数据结构如下：

```
input_data/
└── bistro1/                    # 场景名称
    ├── frame0000.zip           # 帧数据文件
    ├── frame0001.zip
    ├── frame0002.zip
    └── ...
```

每个zip文件应包含Zarr格式的数据，包含以下字段：
- `color`: RGBE格式颜色数据
- `reference`: Ground Truth参考图像
- `motion`: 世界空间运动矢量
- `position`: 世界空间位置
- `camera_position`: 相机位置
- `view_proj_mat`: 视图投影矩阵
- `exposure`: 曝光参数

### 3. 快速测试

首先运行快速测试确保数据格式正确：

```bash
python simple_test.py --input-dir /path/to/your/noisebase/data --scene bistro1
```

如果测试通过，您会看到：
- ✅ 数据加载成功
- ✅ 6通道训练数据生成
- ✅ 可视化结果保存

### 4. 完整预处理

运行完整的数据预处理：

```bash
python improved_noisebase_preprocessor.py \
    --input-dir /path/to/your/noisebase/data \
    --output-dir ./processed_data \
    --scene bistro1 \
    --start-frame 0 \
    --end-frame 100
```

### 5. 验证结果

使用数据集类测试处理结果：

```bash
python simple_dataset.py --data-root ./processed_data --scene bistro1
```

## 📊 数据处理流程详解

### 流程步骤

1. **加载NoiseBase数据** → 从Zarr文件读取各种缓冲区数据
2. **计算屏幕运动矢量** → 使用投影变换计算像素级运动
3. **前向warp投影** → 生成外推帧和覆盖掩码
4. **空洞检测** → 基于覆盖率和颜色差异检测空洞
5. **残差计算** → 计算投影误差的残差运动矢量
6. **6通道拼接** → 组合RGB + Mask + ResidualMV
7. **质量评估** → 自动评估数据质量

### 输出格式

每个处理后的帧生成以下文件：

```
processed_data/
└── bistro1/
    ├── training_data/          # 🎯 主要训练数据
    │   ├── frame_0001.npy     # [6, H, W] 6通道训练数据
    │   ├── frame_0002.npy
    │   └── ...
    ├── rgb/                    # 📸 原始RGB图像
    ├── warped/                 # 🔄 Warp后图像
    ├── masks/                  # 🕳️ 空洞掩码
    ├── residual_mv/           # ➡️ 残差运动矢量
    └── visualization/         # 🎨 可视化结果
```

### 6通道数据格式

生成的训练数据为`[6, H, W]`格式：

- **通道 0-2**: RGB图像 (Ground Truth)
- **通道 3**: 空洞掩码 (1=空洞, 0=有效)
- **通道 4-5**: 残差运动矢量 (X, Y方向)

## 🔧 高级配置

### 自定义参数

在`ImprovedNoiseBasePreprocessor`中可以调整的参数：

```python
# 空洞检测阈值
self.hole_threshold = 0.5      # 覆盖率阈值

# 残差计算阈值  
self.residual_threshold = 2.0  # 颜色差异阈值
```

### 批量处理多个场景

```bash
# 处理多个场景
for scene in bistro1 kitchen living_room; do
    python improved_noisebase_preprocessor.py \
        --input-dir /path/to/noisebase \
        --output-dir ./processed_data \
        --scene $scene
done
```

### 内存优化

对于大数据集，可以分批处理：

```bash
# 分批处理，每次50帧
python improved_noisebase_preprocessor.py \
    --input-dir /path/to/noisebase \
    --output-dir ./processed_data \
    --scene bistro1 \
    --start-frame 0 \
    --end-frame 50

python improved_noisebase_preprocessor.py \
    --input-dir /path/to/noisebase \
    --output-dir ./processed_data \
    --scene bistro1 \
    --start-frame 50 \
    --end-frame 100
```

## 🐛 常见问题

### Q1: "文件不存在"错误

**问题**: `FileNotFoundError: 文件不存在: /path/to/frame0000.zip`

**解决**: 
- 检查输入路径是否正确
- 确认场景名称是否匹配目录名
- 验证zip文件是否存在且可读

### Q2: Zarr加载失败

**问题**: `zarr.errors.ArrayNotFoundError`

**解决**:
- 检查zip文件是否损坏
- 确认Zarr数据格式是否正确
- 尝试手动解压zip文件检查内容

### Q3: 内存不足

**问题**: `MemoryError` 或系统卡死

**解决**:
- 减少处理的帧数范围
- 使用更小的图像分辨率
- 增加系统内存或使用更强的机器

### Q4: 处理速度慢

**问题**: 前向warp处理很慢

**解决**:
- 优化版本已使用向量化操作
- 考虑使用GPU加速（需要修改代码）
- 减少图像分辨率

### Q5: 空洞检测不准确

**问题**: 生成的空洞掩码质量不好

**解决**:
- 调整`hole_threshold`参数
- 修改颜色差异阈值
- 检查运动矢量计算是否正确

## 📈 性能优化建议

### 1. 并行处理

可以修改代码支持多进程处理：

```python
from multiprocessing import Pool

def process_frame_wrapper(args):
    return preprocessor.process_frame_pair(*args)

# 并行处理多个帧对
with Pool(processes=4) as pool:
    results = pool.map(process_frame_wrapper, frame_pairs)
```

### 2. GPU加速

对于大规模数据，考虑使用GPU加速前向warp：

```python
import torch

# 将numpy数组转换为GPU tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
source_tensor = torch.from_numpy(source_image).to(device)
mv_tensor = torch.from_numpy(motion_vectors).to(device)

# 使用torch.nn.functional.grid_sample进行warp
```

### 3. 内存映射

对于超大数据集，使用内存映射：

```python
import numpy as np

# 使用内存映射加载大文件
large_array = np.memmap('large_file.dat', dtype='float32', mode='r')
```

## 🔗 集成到训练流程

### 使用简化数据集类

```python
from simple_dataset import create_simple_dataloader

# 创建训练数据加载器
train_loader = create_simple_dataloader(
    data_root='./processed_data',
    scene_name='bistro1',
    split='train',
    batch_size=16,
    patch_size=64
)

# 在训练循环中使用
for batch in train_loader:
    inputs = batch['input']    # [B, 6, H, W]
    targets = batch['target']  # [B, 3, H, W]
    
    # 训练网络
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

### 与现有训练框架集成

修改您的训练脚本以使用预处理后的数据：

```python
# 在train.py中
from simple_dataset import SimpleNoiseBaseDataset

# 替换原有数据集
dataset = SimpleNoiseBaseDataset(
    data_root='./processed_data',
    scene_name='bistro1',
    split='train'
)
```

## 📞 支持

如果遇到问题，请：

1. 首先运行`simple_test.py`确认基本功能
2. 检查错误日志和堆栈跟踪
3. 验证输入数据格式
4. 查看生成的可视化结果

---

**注意**: 这是优化版的实现，相比原版本有以下改进：
- ✅ 简化的Zarr加载逻辑
- ✅ 向量化的前向warp实现
- ✅ 更好的错误处理
- ✅ 清晰的代码结构
- ✅ 完整的测试框架