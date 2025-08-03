# NoiseBase数据预处理逻辑详解

## 📋 **整体流程概述**

NoiseBase预处理的核心目标是：**从光线追踪帧生成外推帧的空洞补全训练数据**

```
输入: NoiseBase光线追踪数据 (frame0000.zip, frame0001.zip, ...)
输出: 6通道训练数据 [RGB(3) + Mask(1) + ResidualMV(2)]
```

## 🔄 **7步核心处理流程**

### **步骤1: 数据加载与解压**
```python
def load_frame_data(self, frame_idx: int) -> Dict:
```

**功能**: 从NoiseBase的zip文件中提取关键数据
- **RGBE颜色数据** → 解压为RGB图像
- **世界空间位置** (position)
- **世界空间运动矢量** (motion) 
- **法线数据** (normal)
- **参考图像** (reference) - Ground Truth
- **相机参数** (view_proj_mat, camera_position)

**关键处理**:
```python
# RGBE解压缩
rgb_color = decompress_RGBE(color_rgbe, exposure)

# Monte Carlo样本平均
rgb_avg = safe_mean_samples(rgb_color, 'rgb_color')  # [4,H,W,8] → [3,H,W]
```

### **步骤2: 屏幕空间运动矢量计算**
```python
def compute_screen_motion_vectors(self, curr_frame, prev_frame) -> np.ndarray:
```

**功能**: 将3D世界空间运动转换为2D屏幕空间运动矢量

**核心算法**:
```python
# 使用projective.py中的投影函数
screen_mv = motion_vectors(
    w_position=curr_position,    # 当前帧3D位置
    w_motion=curr_motion,        # 3D运动矢量
    pv=curr_frame['view_proj_mat'],      # 当前帧投影矩阵
    prev_pv=prev_frame['view_proj_mat'],  # 前一帧投影矩阵
    height=height, width=width
)
```

**输出**: `[2, H, W]` 格式的屏幕空间运动矢量

### **步骤3: 前向Warp投影 (核心算法)**
```python
def forward_warp(self, source_image, motion_vectors) -> Tuple[np.ndarray, np.ndarray]:
```

**功能**: 这是生成**外推帧RGB图像**的核心步骤

**算法原理**:
1. **像素重投影**: 根据运动矢量将当前帧像素投影到目标位置
2. **双线性分布**: 每个源像素按双线性权重分布到4个邻居像素
3. **权重累积**: 使用`np.add.at`进行原子累加，避免重叠覆盖

**详细过程**:
```python
# 1. 计算目标位置
target_x = x_coords + motion_vectors[0]  # [H, W]
target_y = y_coords + motion_vectors[1]  # [H, W]

# 2. 双线性权重分布
for (px, py), weight in zip(positions, weights):
    # 原子累加避免竞争条件
    np.add.at(warped_image[c], (py, px), source_pixel * weight)
    np.add.at(coverage_mask, (py, px), weight)

# 3. 归一化
warped_image[c, valid_pixels] /= coverage_mask[valid_pixels]
```

**输出**:
- `warped_image`: 外推后的RGB图像 `[3, H, W]`
- `coverage_mask`: 像素覆盖权重 `[H, W]`

### **步骤4: 空洞检测 (生成遮挡掩码)**
```python
def detect_holes_and_compute_residuals(self, warped_image, target_image, coverage_mask, motion_vectors):
```

**功能**: 生成**遮挡掩码**，标识需要补全的空洞区域

**多层检测策略**:

#### **4.1 基于覆盖的空洞检测**
```python
# 覆盖不足的区域标记为空洞
hole_mask = (coverage_mask < self.hole_threshold).astype(np.float32)
```

#### **4.2 基于颜色差异的空洞细化**
```python
# 计算warp结果与ground truth的差异
color_diff = np.linalg.norm(warped_image - target_image, axis=0)
color_diff_normalized = color_diff / (np.max(color_diff) + 1e-8)

# 颜色差异过大的区域也标记为空洞
additional_holes = (color_diff_normalized > 0.3) & (coverage_mask > 0)
hole_mask = np.maximum(hole_mask, additional_holes.astype(np.float32))
```

#### **4.3 形态学优化**
```python
# 使用形态学操作优化掩码形状
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, kernel)  # 填补小洞
hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel)   # 去除噪点
```

### **步骤5: 残差运动矢量计算**

**功能**: 计算warp误差的补偿信息

```python
# 对于有效区域，计算基于颜色误差的残差
valid_mask = (coverage_mask > self.hole_threshold) & (hole_mask < 0.5)
color_error = np.linalg.norm(warped_image - target_image, axis=0)
error_factor = np.clip(color_error / self.residual_threshold, 0, 1)

# 残差运动矢量与误差成比例
residual_mv[0][valid_mask] = motion_vectors[0][valid_mask] * error_factor[valid_mask] * 0.1
residual_mv[1][valid_mask] = motion_vectors[1][valid_mask] * error_factor[valid_mask] * 0.1
```

### **步骤6: 6通道训练数据组装**
```python
def create_training_sample(self, rgb_image, hole_mask, residual_mv) -> np.ndarray:
```

**功能**: 组装最终的训练数据

```python
training_sample = np.concatenate([
    rgb_image,                           # RGB通道 [3, H, W] - 外推帧RGB
    hole_mask[np.newaxis, :, :],        # 掩码通道 [1, H, W] - 遮挡掩码
    residual_mv                         # 残差MV通道 [2, H, W] - 运动补偿
], axis=0)  # 最终: [6, H, W]
```

### **步骤7: 数据保存与可视化**

**保存内容**:
- `rgb/`: 原始RGB图像
- `warped/`: 外推后RGB图像  
- `masks/`: 空洞掩码
- `residual_mv/`: 残差运动矢量
- `training_data/`: 6通道训练数据 (.npy格式)
- `visualization/`: 可视化结果

## 🎯 **关键技术点**

### **1. 外推帧RGB生成原理**

**问题**: 如何从当前帧生成下一帧的RGB图像？

**解决方案**: 前向Warp投影
```
当前帧像素 + 运动矢量 → 下一帧位置
```

**挑战与解决**:
- **空洞问题**: 运动导致某些区域无像素覆盖 → 空洞检测
- **重叠问题**: 多个像素投影到同一位置 → 权重累积
- **亚像素精度**: 投影位置非整数 → 双线性分布

### **2. 遮挡掩码生成逻辑**

**多重判据**:
1. **几何遮挡**: 覆盖权重不足 (`coverage < threshold`)
2. **语义遮挡**: 颜色差异过大 (`color_diff > threshold`)
3. **形态优化**: 形态学操作平滑掩码

**掩码含义**:
- `1`: 空洞区域，需要补全
- `0`: 有效区域，无需处理

### **3. 残差运动矢量作用**

**目的**: 提供运动补偿信息，帮助网络理解warp误差

**计算方式**: 基于颜色误差的运动矢量缩放
```python
residual = original_mv * color_error_factor * scale
```

## 📊 **数据流示意图**

```
NoiseBase原始数据
       ↓
   [加载解压]
       ↓
Frame_t, Frame_{t-1}
       ↓
  [运动矢量计算]
       ↓
   屏幕空间MV [2,H,W]
       ↓
   [前向Warp]
       ↓
外推RGB + 覆盖掩码
       ↓
   [空洞检测]
       ↓
   遮挡掩码 [1,H,W]
       ↓
  [残差计算]
       ↓
  残差MV [2,H,W]
       ↓
   [数据组装]
       ↓
6通道训练数据 [6,H,W]
```

## 🎯 **训练数据格式**

**输入**: 6通道数据 `[6, H, W]`
- 通道0-2: 外推帧RGB (有空洞)
- 通道3: 遮挡掩码 (标识空洞位置)
- 通道4-5: 残差运动矢量 (运动补偿信息)

**目标**: 3通道Ground Truth `[3, H, W]`
- 完整的RGB图像 (无空洞)

**网络任务**: 学习从有空洞的外推帧 + 辅助信息 → 完整RGB图像的映射

这个预处理流程巧妙地将**时序信息**转换为**空间补全问题**，为深度学习网络提供了丰富的训练数据。