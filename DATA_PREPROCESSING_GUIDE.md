# NoiseBase数据预处理使用指南

## 📋 **学长的detach.py脚本分析**

### **脚本功能**
学长的`detach.py`脚本是一个**NoiseBase数据解包和格式转换工具**，主要功能包括：

1. **数据解包**: 从NoiseBase的`.zip`格式数据中提取各种通道
2. **格式转换**: 将`zarr`格式数据转换为`EXR`格式
3. **多采样处理**: 对多采样数据进行平均聚合
4. **几何计算**: 包含运动矢量计算、屏幕空间投影等功能

### **关键函数解析**

#### **1. 数据解压缩**
```python
def decompress_RGBE(color, exposures):
    """解压缩RGBE格式的颜色数据"""
    # RGBE是一种HDR颜色压缩格式
    # 将压缩的4通道数据解压为3通道RGB
```

#### **2. 几何投影**
```python
def screen_space_position(w_position, pv, height, width):
    """世界空间位置投影到屏幕空间"""
    # 使用相机投影矩阵将3D位置转换为2D像素坐标

def motion_vectors(w_position, w_motion, pv, prev_pv, height, width):
    """计算屏幕空间运动矢量"""
    # 计算当前帧和前一帧的屏幕空间位置差异
```

#### **3. 文件输出**
```python
def write_RGB_exr(file_path, data):
    """写入3通道EXR文件 (RGB)"""
    
def write_one_exr(file_path, data):
    """写入单通道EXR文件 (深度等)"""
    
def write_two_exr(file_path, data):
    """写入双通道EXR文件 (运动矢量等)"""
```

### **数据处理流程**
```python
# 1. 加载zip格式的zarr数据
ds = zarr.group(store=zarr.ZipStore(zip_path, mode='r'))

# 2. 提取各种数据通道
color = decompress_RGBE(ds.color, ds.exposure)  # 颜色
position = np.array(ds.position)                # 世界位置
motion = np.array(ds.motion)                    # 运动
normal = np.array(ds.normal)                    # 法线
albedo = np.array(ds.diffuse)                   # 反射率

# 3. 多采样数据聚合 (关键步骤!)
tposition = position.mean(axis=3)  # axis=3是采样维度

# 4. 维度转换 (从CHW转为HWC)
tposition = np.transpose(tposition, (1, 2, 0))

# 5. 输出EXR文件
write_RGB_exr("output.exr", tposition)
```

## 🔧 **当前预处理脚本的问题**

### **问题1: 数据加载错误**
```python
# 当前的错误实现 (生成随机数据)
def load_frame_data(self, frame_idx: int):
    frame_data = {
        'reference': np.random.rand(3, H, W),  # ❌ 随机数据!
        'position': np.random.rand(3, H, W),
        'motion': np.random.rand(2, H, W),
    }
```

### **问题2: 缺少NoiseBase格式支持**
- 没有处理zip+zarr格式
- 没有RGBE解压缩
- 没有多采样数据聚合
- 没有正确的几何计算

## 🚀 **修正后的数据预处理实现**

让我创建一个正确的NoiseBase数据加载器：

```python
import zarr
import numpy as np
from pathlib import Path

class NoiseBaseDataLoader:
    """正确的NoiseBase数据加载器"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
    
    def load_frame_data(self, scene: str, frame_idx: int):
        """加载真实的NoiseBase帧数据"""
        # 构建zip文件路径
        zip_path = self.data_root / scene / f"frame{frame_idx:04d}.zip"
        
        if not zip_path.exists():
            raise FileNotFoundError(f"Frame data not found: {zip_path}")
        
        # 加载zarr数据
        ds = zarr.group(store=zarr.ZipStore(str(zip_path), mode='r'))
        
        # 解压缩颜色数据
        color = self.decompress_RGBE(ds.color, ds.exposure)
        
        # 提取其他数据
        position = np.array(ds.position)
        motion = np.array(ds.motion)
        normal = np.array(ds.normal)
        albedo = np.array(ds.diffuse)
        reference = np.array(ds.reference)
        
        # 多采样数据聚合
        color = color.mean(axis=3)
        position = position.mean(axis=3)
        motion = motion.mean(axis=3)
        normal = normal.mean(axis=3)
        albedo = albedo.mean(axis=3)
        
        # 维度转换 (CHW格式)
        frame_data = {
            'reference': reference,  # 已经是CHW格式
            'color': color,          # CHW格式
            'position': position,    # CHW格式
            'motion': motion,        # CHW格式 (2HW)
            'normal': normal,        # CHW格式
            'albedo': albedo,        # CHW格式
            'camera_pos': np.array(ds.camera_position),
            'view_proj_mat': np.array(ds.view_proj_mat),
            'exposure': np.array(ds.exposure)
        }
        
        return frame_data
    
    def decompress_RGBE(self, color, exposures):
        """解压缩RGBE格式颜色数据"""
        exponents = (color.astype(np.float32)[3] + 1) / 256
        exponents = np.exp(exponents * (exposures[1] - exposures[0]) + exposures[0])
        color = color.astype(np.float32)[:3] / 255 * exponents[np.newaxis]
        return color
```

## 📖 **正确的使用方法**

### **步骤1: 准备NoiseBase数据**
```bash
# 确保数据目录结构如下:
data/
├── bistro1/
│   ├── frame0000.zip
│   ├── frame0001.zip
│   ├── frame0002.zip
│   └── ...
├── kitchen/
│   ├── frame0000.zip
│   └── ...
```

### **步骤2: 使用修正后的预处理脚本**
```bash
# 使用修正后的实现
python train/run_preprocessing_corrected.py \
    --input-dir ./data \
    --output-dir ./processed_data \
    --scene bistro1 \
    --start-frame 0 \
    --end-frame 50
```

### **步骤3: 验证数据加载**
```python
# 测试数据加载
from noisebase_data_loader import NoiseBaseDataLoader

loader = NoiseBaseDataLoader("./data")
frame_data = loader.load_frame_data("bistro1", 0)

print("Frame data keys:", frame_data.keys())
print("Reference shape:", frame_data['reference'].shape)
print("Position shape:", frame_data['position'].shape)
print("Motion shape:", frame_data['motion'].shape)
```

## 🔍 **数据格式说明**

### **NoiseBase数据结构**
```
每个frame.zip包含:
├── color (RGBE格式, 4HWS)      # 压缩的HDR颜色
├── position (3HWS)             # 世界空间位置
├── motion (3HWS)               # 世界空间运动
├── normal (3HWS)               # 表面法线
├── diffuse (3HWS)              # 漫反射率(albedo)
├── reference (3HW)             # 参考图像
├── camera_position (3)         # 相机位置
├── view_proj_mat (4,4)         # 视图投影矩阵
├── exposure (2)                # 曝光范围
└── ... (其他通道)
```

### **维度说明**
- `H`: 图像高度
- `W`: 图像宽度  
- `S`: 采样数量 (通常需要平均聚合)
- `C`: 通道数

## ⚠️ **注意事项**

1. **多采样处理**: NoiseBase数据包含多个采样点，需要聚合(通常用mean)
2. **RGBE解压缩**: 颜色数据是压缩格式，需要解压缩
3. **维度顺序**: 注意CHW vs HWC的维度顺序
4. **数据类型**: 确保使用float32类型进行计算
5. **内存管理**: NoiseBase数据较大，注意内存使用

## 🎯 **完整使用流程**

### **步骤1: 测试数据加载器**
```bash
# 首先测试数据是否能正确加载
python test_data_loader.py
```

这个脚本会：
- 验证数据目录结构
- 列出可用场景
- 加载第一帧数据
- 检查数据完整性
- 检测是否为花屏数据

### **步骤2: 运行修正后的预处理**
```bash
# 使用修正后的预处理脚本
python train/run_preprocessing_corrected.py \
    --data-root /path/to/noisebase/data \
    --scene bistro1 \
    --output-dir ./processed_data \
    --start-frame 1 \
    --end-frame 50 \
    --create-splits \
    --validate
```

### **步骤3: 检查处理结果**
```bash
# 查看生成的文件
ls processed_data/bistro1/
# 应该包含: rgb/ warped/ masks/ residual_mv/ training_data/ visualization/

# 查看可视化结果
ls processed_data/bistro1/visualization/
# 应该包含: frame_000001_corrected.png 等文件
```

### **步骤4: 验证训练数据**
```python
import numpy as np

# 加载训练数据样本
sample = np.load('processed_data/bistro1/training_data/frame_000001.npy')
print(f"训练数据形状: {sample.shape}")  # 应该是 (6, H, W)
print(f"通道含义: RGB(3) + OcclusionMask(1) + ResidualMV(2)")
```

## 📁 **新增文件说明**

### **核心文件**
- `train/noisebase_data_loader.py`: 正确的NoiseBase数据加载器
- `train/run_preprocessing_corrected.py`: 修正后的预处理脚本
- `test_data_loader.py`: 数据加载器测试脚本

### **使用这些文件的原因**
1. **解决花屏问题**: 使用真实NoiseBase数据而不是随机数据
2. **正确格式处理**: 处理zip+zarr格式、RGBE解压缩、多采样聚合
3. **符合任务书要求**: 实现Z-buffer遮挡检测和MV+深度空洞检测

## ⚠️ **重要注意事项**

### **数据路径要求**
确保NoiseBase数据目录结构如下：
```
data/
├── bistro1/
│   ├── frame0000.zip
│   ├── frame0001.zip
│   ├── frame0002.zip
│   └── ...
├── kitchen/
│   ├── frame0000.zip
│   └── ...
```

### **依赖包要求**
```bash
pip install zarr numpy opencv-python matplotlib
```

### **内存要求**
NoiseBase数据较大，建议：
- 至少16GB内存
- 处理时监控内存使用
- 必要时分批处理

## 🔧 **故障排除**

### **问题1: "花屏"数据**
- **原因**: 使用了随机数据生成而不是真实NoiseBase数据
- **解决**: 使用`test_data_loader.py`验证数据加载，然后使用修正后的预处理脚本

### **问题2: 找不到数据文件**
- **原因**: 数据路径不正确或文件名格式不匹配
- **解决**: 检查数据目录结构，确保文件名为`frame0000.zip`格式

### **问题3: zarr导入错误**
- **原因**: 缺少zarr包
- **解决**: `pip install zarr`

### **问题4: 内存不足**
- **原因**: NoiseBase数据较大
- **解决**: 减少处理帧数，或增加系统内存

---

**总结**: 通过分析学长的`detach.py`脚本，我们发现了NoiseBase数据的正确格式和处理方法。现在提供了完整的解决方案来修正"花屏"问题并正确处理NoiseBase数据。按照上述流程操作即可获得正确的预处理结果。