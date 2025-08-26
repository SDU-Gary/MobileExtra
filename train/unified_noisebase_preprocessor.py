#!/usr/bin/env python3
"""
统一的NoiseBase数据预处理脚本 - 移动端实时帧外插系统
作者：AI算法团队
日期：2025-08-03

完整实现从NoiseBase数据到网络训练数据的转换：
1. 正确的zarr+zip数据加载和RGBE解压缩
2. 基于运动矢量的前向投影(Forward Warp)
3. 空洞检测和遮挡掩码生成
4. 残差运动矢量计算
5. 7通道训练数据生成：RGB(3) + HoleMask(1) + OcclusionMask(1) + ResidualMV(2)

使用方法:
python unified_noisebase_preprocessor.py --data-root ./data --scene bistro1 --output ./processed_unified
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import gc  # 垃圾回收

# 设置matplotlib后端为Agg，避免tkinter线程安全问题
import matplotlib
matplotlib.use('Agg')  # 必须在pyplot导入之前设置
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import zipfile
import tempfile
import shutil
import time
import warnings
from tqdm import tqdm

# 依赖库导入和兼容性处理
try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# 添加当前目录到路径以导入项目模块
sys.path.insert(0, str(Path(__file__).parent))

# 尝试导入zarr兼容性模块
try:
    from zarr_compat import load_zarr_group, decompress_RGBE_compat
    HAS_ZARR_COMPAT = True
except ImportError as e:
    HAS_ZARR_COMPAT = False


class UnifiedNoiseBasePreprocessor:
    """
    统一的NoiseBase数据预处理器
    
    整合了所有必要功能：
    - NoiseBase数据加载（zip+zarr格式）
    - RGBE颜色解压缩和多采样聚合
    - Forward Warp外推帧生成
    - 空洞检测和遮挡掩码生成
    - 训练数据格式化和保存
    """
    
    def __init__(self, 
                 data_root: str,
                 output_dir: str,
                 scene_name: str = "bistro1",
                 test_mode: bool = False,
                 **kwargs):
        """
        初始化预处理器
        
        Args:
            data_root: NoiseBase数据根目录
            output_dir: 处理后数据输出目录
            scene_name: 场景名称
            test_mode: 测试模式，跳过数据目录验证
            **kwargs: 其他配置参数
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        self.test_mode = test_mode
        
        # 验证数据目录（测试模式下跳过）
        if not test_mode:
            if not self.data_root.exists():
                raise FileNotFoundError(f"数据目录不存在: {self.data_root}")
            
            scene_dir = self.data_root / scene_name
            if not scene_dir.exists():
                available_scenes = [d.name for d in self.data_root.iterdir() if d.is_dir()]
                raise FileNotFoundError(f"场景 '{scene_name}' 不存在。可用场景: {available_scenes}")
        else:
            pass  # 测试模式：跳过数据目录验证
        
        # 创建输出目录结构
        self._setup_output_dirs()
        
        # 算法参数
        self.hole_threshold = kwargs.get('hole_threshold', 0.05)  # 降低阈值提高敏感度
        self.residual_threshold = kwargs.get('residual_threshold', 2.0)
        self.depth_discontinuity_threshold = kwargs.get('depth_discontinuity_threshold', 0.1)
        self.motion_discontinuity_threshold = kwargs.get('motion_discontinuity_threshold', 1.0)
        
        # 调试选项
        self.debug_occlusion = kwargs.get('debug_occlusion', False)
        
        # 性能优化设置
        self.use_numba = HAS_NUMBA and kwargs.get('use_numba', True)
        self.batch_processing = kwargs.get('batch_processing', False)
        
        # 初始化完成
    
    def _setup_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            'rgb',           # 目标RGB图像
            'warped',        # 前向投影后的图像
            'masks',         # 掩码文件
            'residual_mv',   # 残差运动矢量
            'training_data', # 7通道训练数据
            'visualization', # 可视化结果
            'debug'          # 调试信息
        ]
        
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # ==================== 数据加载模块 ====================
    
    def load_frame_data(self, scene: str, frame_idx: int) -> Optional[Dict]:
        """
        加载指定帧的NoiseBase数据（资源管理版本）
        
        Args:
            scene: 场景名称
            frame_idx: 帧索引
            
        Returns:
            frame_data: 帧数据字典，包含reference, position, motion, depth等
        """
        try:
            zip_path = self.data_root / scene / f"frame{frame_idx:04d}.zip"
            
            if not zip_path.exists():
                return None
            
            # 使用上下文管理器确保资源自动清理
            if HAS_ZARR_COMPAT:
                with load_zarr_group(str(zip_path)) as ds:
                    return self._extract_frame_data(ds)
            else:
                ds = self._load_with_fallback(zip_path)
                if ds is None:
                    return None
                return self._extract_frame_data(ds)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_frame_data(self, ds) -> Optional[Dict]:
        """
        从zarr数据源提取帧数据
        
{{ ... }}
        Args:
            ds: zarr数据源
            
        Returns:
            frame_data: 提取的帧数据字典
        """
        try:
            if ds is None:
                return None
            
            # 提取和处理数据
            frame_data = {}
            
            # 1. 参考图像 - 使用现代zarr API
            reference_success = False
            
            # 获取可用的数据键
            available_keys = []
            try:
                if hasattr(ds, 'keys'):
                    available_keys = list(ds.keys())
                elif hasattr(ds, 'array_keys'):
                    available_keys = list(ds.array_keys())
            except Exception as e:
                pass
            
            # 尝试加载reference数据
            if 'reference' in available_keys:
                try:
                    reference = np.array(ds['reference'])
                    reference = self._process_reference_data(reference)
                    frame_data['reference'] = reference
                    reference_success = True
                except Exception as e:
                    pass
            
            # 回退到color+exposure数据
            if not reference_success and 'color' in available_keys and 'exposure' in available_keys:
                try:
                    color_data = ds['color']
                    exposure_data = ds['exposure']
                    color = self._process_color_data(color_data, exposure_data)
                    frame_data['reference'] = color
                    reference_success = True
                except Exception as e:
                    import traceback
                    traceback.print_exc()
            
            if not reference_success:
                raise ValueError(f"无法获取任何颜色数据 (reference或color+exposure)")
            
            # 2. 世界空间位置数据
            if 'position' in available_keys:
                try:
                    position = self._process_position_data(ds['position'])
                    frame_data['position'] = position
                    # 保存Z分量作为深度，用于空洞检测的梯度计算
                    # 注意：这个深度用于梯度检测，不适合直接用于遮挡检测
                    frame_data['depth'] = position[2:3]  # Z分量作为深度（用于空洞检测）
                except Exception as e:
                    pass
            
            # 3. 运动矢量数据
            if 'motion' in available_keys:
                try:
                    motion = self._process_motion_data(ds['motion'])
                    frame_data['motion'] = motion
                except Exception as e:
                    pass
            
            # 4. 法线数据（用于遮挡检测）
            if 'normal' in available_keys:
                try:
                    normal = self._process_normal_data(ds['normal'])
                    frame_data['normal'] = normal
                except Exception as e:
                    pass
            
            # 5. 相机参数
            frame_data['camera_params'] = self._extract_camera_params(ds, available_keys)
            
            return frame_data
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def _load_with_fallback(self, zip_path: Path):
        """备用的zarr数据加载方法"""
        try:
            if HAS_ZARR:
                import zarr
                return zarr.group(store=zarr.ZipStore(str(zip_path), mode='r'))
            else:
                # 解压到临时目录
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    zip_file.extractall(temp_dir)
                return zarr.open_group(temp_dir, mode='r')
        except Exception as e:
            return None
    
    def _process_color_data(self, color_data, exposure_data) -> np.ndarray:
        """处理RGBE颜色数据 - 修复为使用detach.py的正确处理顺序"""
        try:
            # 将数据转换为numpy数组
            color_data = np.array(color_data)
            exposure_data = np.array(exposure_data)
            
            print(f"   调试: color_data形状: {color_data.shape}, exposure: {exposure_data}")
            
            # 先进行RGBE解压缩（在多采样数据上）
            if HAS_ZARR_COMPAT:
                color = decompress_RGBE_compat(color_data, exposure_data)
            else:
                # 使用修复后的RGBE解压缩实现
                color = self._decompress_RGBE_basic(color_data, exposure_data)
            
            print(f"   调试: RGBE解压后形状: {color.shape}, 范围: [{color.min():.6f}, {color.max():.6f}]")
            
            # 然后进行多采样聚合（按detach.py方式）
            if color.ndim == 4 and color.shape[-1] > 1:
                color = color.mean(axis=-1)
                print(f"   调试: 多采样聚合后形状: {color.shape}")
            
            # 确保格式为CHW（按detach.py的转置方式）
            if color.ndim == 3:
                if color.shape[-1] == 3:  # HWC -> CHW
                    color = color.transpose(2, 0, 1)
                elif color.shape[0] == 3:  # 已经是CHW
                    pass
                else:
                    raise ValueError(f"无法识别的color数据格式: {color.shape}")
            
            print(f"   调试: 最终color形状: {color.shape}")
            
            return color.astype(np.float32)
            
        except Exception as e:
            print(f"颜色数据处理失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _decompress_RGBE_basic(self, color_rgbe: np.ndarray, exposure: np.ndarray) -> np.ndarray:
        """基础RGBE解压缩实现 - 修复为使用detach.py的正确算法"""
        # 使用detach.py中的正确RGBE解压缩算法
        color_rgbe = np.array(color_rgbe)
        exposure = np.array(exposure)
        
        if color_rgbe.shape[0] != 4:
            raise ValueError(f"RGBE数据应该是4通道在第一维，实际: {color_rgbe.shape}")
        
        # 使用detach.py的正确RGBE解压缩算法
        exponents = (color_rgbe.astype(np.float32)[3] + 1) / 256
        exponents = np.exp(exponents * (exposure[1] - exposure[0]) + exposure[0])
        color = color_rgbe.astype(np.float32)[:3] / 255 * exponents[np.newaxis]
        
        return color
    
    def _process_position_data(self, position_data) -> np.ndarray:
        """处理世界空间位置数据"""
        position = np.array(position_data)
        
        # 多采样聚合
        if position.ndim == 4:
            position = position.mean(axis=-1)
        
        # 确保格式为CHW
        if position.shape[-1] == 3:
            position = position.transpose(2, 0, 1)
        
        return position.astype(np.float32)
    
    @staticmethod
    def screen_space_projection(world_pos: np.ndarray, 
                               view_proj_matrix: np.ndarray, 
                               height: int, 
                               width: int) -> np.ndarray:
        """
        将世界空间位置投影到屏幕空间像素坐标
        
        Args:
            world_pos: 世界空间坐标 (3, H, W)
            view_proj_matrix: 视图投影矩阵 (4, 4)
            height, width: 图像尺寸
            
        Returns:
            屏幕空间坐标 (2, H, W)，单位为像素
        """
        C, H, W = world_pos.shape
        if C != 3:
            raise ValueError(f"world_pos 必须是3通道，实际为 {C}")

        # 扩展为齐次坐标
        ones = np.ones((1, H, W), dtype=world_pos.dtype)
        homogeneous_pos = np.concatenate([world_pos, ones], axis=0)

        # 矩阵投影
        projected_pos = np.einsum('ij,jhw->ihw', view_proj_matrix, homogeneous_pos)
        
        # 透视除法
        w = projected_pos[3:4, :, :]
        epsilon = 1e-5
        w = np.where(w < epsilon, epsilon, w)
        ndc_pos = projected_pos[:2, :, :] / w

        # NDC到屏幕坐标转换
        screen_pos_x = (ndc_pos[0] * 0.5 + 0.5) * width
        screen_pos_y = (-ndc_pos[1] * 0.5 + 0.5) * height

        return np.stack([screen_pos_x, screen_pos_y], axis=0)

    def _create_default_projection_matrix(self) -> np.ndarray:
        """创建默认透视投影矩阵"""
        fov = np.pi / 3  # 60度
        aspect = 16.0 / 9.0
        near, far = 0.1, 100.0
        
        f = 1.0 / np.tan(fov / 2)
        
        projection = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0], 
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        return projection

    def _compute_screen_space_mv(self, curr_frame: Dict, prev_frame: Dict) -> np.ndarray:
        """
        【核心修复】计算屏幕空间运动矢量
        基于3D几何投影，将世界空间运动转换为屏幕空间像素运动
        """
        print("   🚀 计算屏幕空间运动矢量（3D->2D投影）")
        
        pos_t = curr_frame['position']
        motion_t = curr_frame['motion'][:3]
        
        # 获取投影矩阵
        vp_mat_t = curr_frame.get('camera_params', {}).get('view_proj_mat')
        vp_mat_prev = prev_frame.get('camera_params', {}).get('view_proj_mat')
        
        if vp_mat_t is None or vp_mat_prev is None:
            print("   ⚠️ 缺少投影矩阵，使用默认矩阵")
            default_proj = self._create_default_projection_matrix()
            vp_mat_t = vp_mat_t or default_proj
            vp_mat_prev = vp_mat_prev or default_proj
        else:
            vp_mat_t = vp_mat_t.T
            vp_mat_prev = vp_mat_prev.T

        H, W = pos_t.shape[1], pos_t.shape[2]

        # 计算前一帧世界坐标
        pos_prev = pos_t - motion_t

        # 投影到屏幕空间
        screen_pos_t = self.screen_space_projection(pos_t, vp_mat_t, H, W)
        screen_pos_prev = self.screen_space_projection(pos_prev, vp_mat_prev, H, W)
        
        # 计算屏幕空间运动矢量
        screen_space_mv = screen_pos_prev - screen_pos_t
        
        # 调试信息
        mv_magnitude = np.sqrt(screen_space_mv[0]**2 + screen_space_mv[1]**2)
        print(f"    屏幕空间MV: 形状={screen_space_mv.shape}")
        print(f"      像素运动统计: 平均={mv_magnitude.mean():.2f}px, 最大={mv_magnitude.max():.2f}px")
        print(f"      非零运动像素比例: {np.mean(mv_magnitude > 0.1):.3f}")
        
        return screen_space_mv.astype(np.float32)

    def _process_motion_data(self, motion_data) -> np.ndarray:
        """处理世界空间运动矢量数据（仅加载和聚合，不进行2D转换）"""
        motion = np.array(motion_data)
        
        # 多采样聚合
        if motion.ndim == 4:
            motion = motion.mean(axis=-1)
        
        # 确保格式为CHW
        if motion.shape[-1] == 3:
            motion = motion.transpose(2, 0, 1)

        print(f"   已加载世界空间motion数据: {motion.shape}")
        return motion.astype(np.float32)
    def _process_normal_data(self, normal_data) -> np.ndarray:
        """处理法线数据"""
        normal = np.array(normal_data)
        
        # 多采样聚合
        if normal.ndim == 4:
            normal = normal.mean(axis=-1)
        
        # 确保格式为CHW
        if normal.shape[-1] == 3:
            normal = normal.transpose(2, 0, 1)
        
        # 法线归一化
        norm = np.sqrt(np.sum(normal**2, axis=0, keepdims=True))
        norm = np.maximum(norm, 1e-8)  # 避免除零
        normal = normal / norm
        
        return normal.astype(np.float32)
    
    def _process_reference_data(self, reference_data) -> np.ndarray:
        """处理reference参考图像数据 - 修复为使用detach.py的正确方式"""
        try:
            reference = np.array(reference_data)
            print(f"   调试: 原始reference形状: {reference.shape}, 范围: [{reference.min():.6f}, {reference.max():.6f}]")
            
            # 按照detach.py的方式处理：直接转置，无需多采样聚合
            # detach.py: ref = np.transpose(reference, (1, 2, 0))
            if reference.ndim == 3 and reference.shape[0] == 3:
                # CHW -> HWC for detach.py compatibility, then back to CHW
                reference_hwc = reference.transpose(1, 2, 0)
                reference = reference_hwc.transpose(2, 0, 1)  # Back to CHW for consistency
                print(f"   调试: reference转置后形状: {reference.shape}")
            elif reference.ndim == 4:
                # 如果有多采样维度，先聚合再转置（但detach.py中reference通常没有多采样）
                reference = reference.mean(axis=-1) if reference.shape[-1] > 1 else reference.squeeze(-1)
                if reference.ndim == 3 and reference.shape[-1] == 3:
                    reference = reference.transpose(2, 0, 1)
                print(f"   调试: reference多采样处理后形状: {reference.shape}")
            elif reference.ndim == 2:
                # 灰度图像，转换为3通道
                reference = np.stack([reference, reference, reference], axis=0)
            
            # 不强制转换到[0,1]范围，保持HDR数据
            # detach.py中没有对reference进行范围限制
            print(f"   调试: 最终reference形状: {reference.shape}, 范围: [{reference.min():.6f}, {reference.max():.6f}]")
            
            return reference.astype(np.float32)
            
        except Exception as e:
            print(f"Reference数据处理详细错误: {e}")
            print(f"Reference数据形状: {reference_data.shape if hasattr(reference_data, 'shape') else 'unknown'}")
            import traceback
            traceback.print_exc()
            raise
    
    def _extract_camera_params(self, ds, available_keys: list) -> Dict:
        """提取相机参数"""
        params = {}
        
        try:
            # 使用现代zarr API
            if 'camera_position' in available_keys:
                params['camera_position'] = np.array(ds['camera_position'])
            if 'proj_mat' in available_keys:
                params['proj_mat'] = np.array(ds['proj_mat'])
            if 'view_proj_mat' in available_keys:
                params['view_proj_mat'] = np.array(ds['view_proj_mat'])
        except Exception as e:
            print(f"提取相机参数时出错: {e}")
        
        return params
    
    # ==================== Forward Warp模块 ====================
    
    def forward_warp_frame(self, 
                          prev_frame: Dict, 
                          motion_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向投影算法实现
        
        Args:
            prev_frame: 前一帧数据
            motion_vectors: 运动矢量 [2, H, W]
            
        Returns:
            warped_image: 投影后图像 [C, H, W]
            coverage_mask: 覆盖掩码 [H, W]
        """
        source_image = prev_frame['reference']
        
        # 确保输入格式正确
        if source_image.ndim == 3 and source_image.shape[2] in [1, 3]:
            source_image = source_image.transpose(2, 0, 1)
        
        if motion_vectors.ndim == 3 and motion_vectors.shape[2] == 2:
            motion_vectors = motion_vectors.transpose(2, 0, 1)
        
        C, H, W = source_image.shape
        
        # 初始化输出
        warped_image = np.zeros_like(source_image)
        coverage_mask = np.zeros((H, W), dtype=np.float32)
        
        # 使用优化的前向投影算法
        if self.use_numba and HAS_NUMBA:
            warped_image, coverage_mask = self._forward_splatting_numba(
                source_image, motion_vectors, warped_image, coverage_mask
            )
        else:
            warped_image, coverage_mask = self._forward_splatting_python(
                source_image, motion_vectors, warped_image, coverage_mask
            )
        
        # 深度冲突解决（如果有深度信息）
        if 'depth' in prev_frame:
            warped_image = self._resolve_depth_conflicts(
                warped_image, coverage_mask, prev_frame['depth'], motion_vectors
            )
        
        return warped_image, coverage_mask
    
    def _forward_splatting_python(self, 
                                source_image: np.ndarray,
                                motion_vectors: np.ndarray,
                                warped_image: np.ndarray,
                                coverage_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """纯Python实现的前向投影"""
        C, H, W = source_image.shape
        
        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        
        # 计算目标坐标
        target_x = x_coords + motion_vectors[0]
        target_y = y_coords + motion_vectors[1]
        
        # 有效像素掩码
        valid_mask = (
            (target_x >= 0) & (target_x < W-1) & 
            (target_y >= 0) & (target_y < H-1)
        )
        
        # 获取有效坐标
        valid_indices = np.where(valid_mask)
        src_y, src_x = valid_indices
        
        # 计算目标坐标（整数）
        tgt_x = np.round(target_x[valid_mask]).astype(np.int32)
        tgt_y = np.round(target_y[valid_mask]).astype(np.int32)
        
        # 投影像素值
        for i in range(len(src_y)):
            sy, sx = src_y[i], src_x[i]
            ty, tx = tgt_y[i], tgt_x[i]
            
            # 累加颜色值和覆盖计数
            warped_image[:, ty, tx] += source_image[:, sy, sx]
            coverage_mask[ty, tx] += 1.0
        
        # 归一化
        valid_coverage = coverage_mask > 0
        warped_image[:, valid_coverage] /= coverage_mask[valid_coverage]
        
        return warped_image, coverage_mask
    
    if HAS_NUMBA:
        @staticmethod
        @jit(nopython=True, parallel=True)
        def _forward_splatting_numba(source_image: np.ndarray,
                                   motion_vectors: np.ndarray,
                                   warped_image: np.ndarray,
                                   coverage_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Numba优化的前向投影"""
            C, H, W = source_image.shape
            
            for y in prange(H):
                for x in range(W):
                    # 计算目标位置
                    target_x = x + motion_vectors[0, y, x]
                    target_y = y + motion_vectors[1, y, x]
                    
                    # 边界检查
                    if 0 <= target_x < W-1 and 0 <= target_y < H-1:
                        # 最近邻投影
                        tx = int(round(target_x))
                        ty = int(round(target_y))
                        
                        if 0 <= tx < W and 0 <= ty < H:
                            # 累加颜色值
                            for c in range(C):
                                warped_image[c, ty, tx] += source_image[c, y, x]
                            coverage_mask[ty, tx] += 1.0
            
            # 归一化
            for y in prange(H):
                for x in range(W):
                    if coverage_mask[y, x] > 0:
                        for c in range(C):
                            warped_image[c, y, x] /= coverage_mask[y, x]
            
            return warped_image, coverage_mask
    else:
        def _forward_splatting_numba(self, *args):
            return self._forward_splatting_python(*args)
    
    def _resolve_depth_conflicts(self, 
                                warped_image: np.ndarray,
                                coverage_mask: np.ndarray,
                                depth: np.ndarray,
                                motion_vectors: np.ndarray) -> np.ndarray:
        """基于深度的冲突解决"""
        # 简化的深度冲突解决 - 保持距离相机最近的像素
        # 在实际GPU实现中，这会通过原子深度测试完成
        return warped_image
    
    # ==================== 空洞检测和遮挡分析模块 ====================
    
    def detect_holes_and_occlusion(self, 
                                  warped_image: np.ndarray,
                                  target_image: np.ndarray,
                                  coverage_mask: np.ndarray,
                                  curr_frame: Dict,
                                  prev_frame: Dict,
                                  motion_vectors: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        检测几何空洞、非几何空洞和语义遮挡
        
        Args:
            warped_image: 前向投影图像
            target_image: 目标图像
            coverage_mask: 覆盖掩码
            curr_frame: 当前帧数据
            prev_frame: 前一帧数据
            motion_vectors: 运动矢量 (2, H, W)，用于真正的遮挡检测
            
        Returns:
            masks: 包含各种掩码的字典
        """
        masks = {}
        
        # 1. 几何空洞检测 - 基于覆盖度
        masks['holes'] = self._detect_geometric_holes(coverage_mask)
        
        # 2. 非几何空洞检测 - 基于深度和运动不连续性（网络训练需要）
        masks['semantic_holes'] = self._detect_semantic_holes(curr_frame, prev_frame)
        
        # 3. 真正的遮挡检测 - 基于深度比较（学长的方法）
        if motion_vectors is not None:
            masks['occlusion'] = self._detect_true_occlusion(
                curr_frame, prev_frame, warped_image, motion_vectors
            )
        else:
            # 备用：使用非几何空洞作为遮挡掩码
            masks['occlusion'] = masks['semantic_holes']
        
        # 4. 静态和动态空洞分类
        masks['static_holes'], masks['dynamic_holes'] = self._classify_holes(
            masks['holes'], curr_frame, prev_frame
        )
        
        print(f"   几何空洞: {np.mean(masks['holes']):.3f} 覆盖率")
        print(f"   非几何空洞: {np.mean(masks['semantic_holes']):.3f} 覆盖率")
        print(f"   遮挡检测: {np.mean(masks['occlusion']):.3f} 覆盖率")
        
        return masks
    
    def _detect_geometric_holes(self, coverage_mask: np.ndarray) -> np.ndarray:
        """检测几何空洞"""
        # 基于覆盖度阈值的空洞检测
        hole_mask = (coverage_mask < self.hole_threshold).astype(np.float32)
        
        # 形态学操作优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, kernel)
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel)
        
        return hole_mask
    
    def _detect_semantic_holes(self, curr_frame: Dict, prev_frame: Dict) -> np.ndarray:
        """
        检测非几何空洞 - 基于深度和运动不连续性（网络训练需要）
        
        这是原来的"语义遮挡"检测方法，实际上是一种非几何的空洞检测
        现在单独提取出来，用于网络训练数据
        
        注意：这里使用Z分量深度进行梯度检测，与遮挡检测使用的相机距离深度不同
        """
        # 基于深度不连续性的检测（使用Z分量深度的梯度）
        depth_holes = np.zeros((curr_frame['reference'].shape[1], curr_frame['reference'].shape[2]), dtype=np.float32)
        
        if 'depth' in curr_frame:
            depth_holes = self._compute_depth_discontinuity(curr_frame['depth'])
        
        # 基于运动不连续性的检测
        motion_holes = np.zeros_like(depth_holes)
        
        if 'motion' in curr_frame:
            motion_holes = self._compute_motion_discontinuity(curr_frame['motion'])
        
        # 结合两种方法
        semantic_holes = np.maximum(depth_holes, motion_holes)
        
        # 形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        semantic_holes = cv2.morphologyEx(semantic_holes, cv2.MORPH_CLOSE, kernel)
        
        return semantic_holes
    
    def _detect_semantic_occlusion(self, 
                                  curr_frame: Dict, 
                                  prev_frame: Dict,
                                  warped_frame: Optional[np.ndarray] = None,
                                  motion_vectors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        检测语义遮挡 - 更新为真正的遮挡检测
        
        注意：这个方法名保持不变以兼容现有代码，但现在实现了真正的遮挡检测
        之前这个方法实际上是空洞检测，现在改为正确的遮挡检测
        """
        # 如果提供了warp结果和运动矢量，使用基于深度比较的真正遮挡检测
        if warped_frame is not None and motion_vectors is not None:
            return self._detect_true_occlusion(curr_frame, prev_frame, warped_frame, motion_vectors)
        else:
            # 备用方法：基于梯度的传统方法
            return self._detect_semantic_occlusion_fallback(curr_frame, prev_frame)
    
    def _compute_depth_discontinuity(self, depth: np.ndarray) -> np.ndarray:
        """
        计算深度不连续性 - 用于空洞检测
        
        注意：这里使用的是Z分量深度，通过Sobel算子检测梯度不连续性
        与遮挡检测中的相机距离深度是不同的概念
        """
        if depth.ndim == 3:
            depth = depth[0]  # 取第一个通道
        
        # 计算梯度
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        
        # 梯度幅度
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 基于百分位数的阈值
        threshold = np.percentile(gradient_magnitude, 98)
        
        discontinuity = (gradient_magnitude > threshold * self.depth_discontinuity_threshold).astype(np.float32)
        
        return discontinuity
    
    def _compute_motion_discontinuity(self, motion: np.ndarray) -> np.ndarray:
        """计算运动不连续性"""
        # 计算运动矢量的梯度
        motion_x, motion_y = motion[0], motion[1]
        
        # X方向运动的梯度
        grad_mx_x = cv2.Sobel(motion_x, cv2.CV_32F, 1, 0, ksize=3)
        grad_mx_y = cv2.Sobel(motion_x, cv2.CV_32F, 0, 1, ksize=3)
        
        # Y方向运动的梯度
        grad_my_x = cv2.Sobel(motion_y, cv2.CV_32F, 1, 0, ksize=3)
        grad_my_y = cv2.Sobel(motion_y, cv2.CV_32F, 0, 1, ksize=3)
        
        # 总梯度幅度
        total_gradient = np.sqrt(grad_mx_x**2 + grad_mx_y**2 + grad_my_x**2 + grad_my_y**2)
        
        # 基于百分位数的阈值
        threshold = np.percentile(total_gradient, 98)
        
        discontinuity = (total_gradient > threshold * self.motion_discontinuity_threshold).astype(np.float32)
        
        return discontinuity
    
    def _calculate_camera_depth(self, world_position: np.ndarray, camera_position: np.ndarray) -> np.ndarray:
        """
        计算世界坐标点到相机位置的距离（学长提供的my_log_depth函数）
        
        Args:
            world_position: 世界坐标位置 (H, W, 3) 或 (3, H, W)
            camera_position: 相机位置 (3,)
            
        Returns:
            depth: 对数深度 (H, W)
        """
        # 确保world_position是(H, W, 3)格式
        if world_position.shape[0] == 3 and len(world_position.shape) == 3:
            # 从(3, H, W)转换为(H, W, 3)
            world_position = world_position.transpose(1, 2, 0)
        
        # 计算欧几里得距离
        d = np.linalg.norm(world_position - camera_position.reshape(1, 1, 3), axis=-1)
        
        # 过滤无效距离：太小或太大的距离都认为无效
        valid_distance = (d > 1e-3) & (d < 1e6) & np.isfinite(d)
        
        # 应用对数变换，避免除零错误
        safe_d = np.maximum(d, 1e-8)
        log_depth = np.log(1 + 1 / safe_d)
        
        # 将无效距离对应的深度设为负值，便于后续过滤
        log_depth[~valid_distance] = -1.0
        
        return log_depth
    
    def _detect_true_occlusion(self, 
                              curr_frame: Dict, 
                              prev_frame: Dict,
                              warped_frame: np.ndarray,
                              motion_vectors: np.ndarray,
                              depth_threshold: float = 0.05) -> np.ndarray:
        """
        检测真正的遮挡掩码 - 基于学长的深度比较方法
        
        Args:
            curr_frame: 当前帧数据
            prev_frame: 前一帧数据  
            warped_frame: warp后的前一帧
            motion_vectors: 运动矢量 (2, H, W)
            depth_threshold: 深度差异阈值
            
        Returns:
            occlusion_mask: 遮挡掩码 (H, W)
        """
        H, W = curr_frame['reference'].shape[1], curr_frame['reference'].shape[2]
        
        # 获取当前帧世界位置和相机位置
        if 'position' not in curr_frame or 'camera_params' not in curr_frame:
            print("   ⚠️ 缺少位置或相机参数，使用基于梯度的遮挡检测")
            return self._detect_semantic_occlusion_fallback(curr_frame, prev_frame)
            
        current_position = curr_frame['position']  # 当前帧世界坐标
        camera_params = curr_frame['camera_params']
        
        if 'camera_position' not in camera_params:
            print("   ⚠️ 缺少相机位置，使用备用遮挡检测")
            return self._detect_semantic_occlusion_fallback(curr_frame, prev_frame)
            
        camera_position = camera_params['camera_position']
        
        # 计算当前帧的相机深度（学长的正确方法）
        # 注意：这里计算的是相机到世界坐标点的欧几里得距离，用于遮挡检测
        # 与空洞检测中使用的Z分量深度不同
        current_depth = self._calculate_camera_depth(current_position, camera_position)
        
        # 计算warp对应的前一帧世界坐标位置
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        prev_x = x + motion_vectors[1]  # X方向运动
        prev_y = y + motion_vectors[0]  # Y方向运动
        
        # 边界检查
        valid_mask = ((prev_x >= 0) & (prev_x < W) & (prev_y >= 0) & (prev_y < H))
        
        # 双线性插值获取前一帧世界坐标
        from scipy.ndimage import map_coordinates
        coords = np.stack([prev_y.ravel(), prev_x.ravel()], axis=-1)
        
        prev_world_pos = np.zeros((H, W, 3))
        if 'position' in prev_frame:
            prev_position = prev_frame['position']
            if prev_position.shape[0] == 3:  # (3, H, W)
                for i in range(3):
                    # 使用 'nearest' 模式避免边界处的零值问题
                    prev_world_pos[..., i] = map_coordinates(
                        prev_position[i], coords.T, order=1, mode='nearest', cval=0
                    ).reshape(H, W)
            
        # 计算前一帧投影到当前相机的期望深度
        expected_depth = self._calculate_camera_depth(prev_world_pos, camera_position)
        
        # 调试信息
        print(f"   🔍 当前深度范围: [{current_depth.min():.3f}, {current_depth.max():.3f}]")
        print(f"   🔍 期望深度范围: [{expected_depth.min():.3f}, {expected_depth.max():.3f}]")
        print(f"   🔍 当前帧世界坐标范围: X[{current_position[0].min():.1f}, {current_position[0].max():.1f}], Y[{current_position[1].min():.1f}, {current_position[1].max():.1f}], Z[{current_position[2].min():.1f}, {current_position[2].max():.1f}]")
        print(f"   🔍 相机位置: [{camera_position[0]:.1f}, {camera_position[1]:.1f}, {camera_position[2]:.1f}]")
        
        # 只考虑有效深度值的区域
        # 学长的对数深度函数: log(1 + 1/d) 应该总是正值，除非d无效
        # 我的函数对无效距离返回-1.0，所以检查条件改为 >= 0
        valid_current = (current_depth >= 0) & np.isfinite(current_depth)
        valid_expected = (expected_depth >= 0) & np.isfinite(expected_depth)
        valid_depth_mask = valid_current & valid_expected
        
        # 深度比较检测遮挡（只在有效区域）
        depth_diff = np.abs(expected_depth - current_depth)
        
        # 遮挡检测逻辑改进：基于学长原始逻辑但更保守
        # 学长的原始逻辑是检测深度不匹配，这里改为更严格的遮挡检测
        
        # 计算相对深度差异（归一化）
        relative_diff = depth_diff / (np.maximum(current_depth, expected_depth) + 1e-8)
        
        # 遮挡条件：
        # 1. 深度差异显著 (相对差异 > 阈值)
        # 2. 且期望深度 > 当前深度 (前一帧的物体被当前帧遮挡)
        # 3. 且都是有效深度
        significant_diff = (relative_diff > depth_threshold)  # 使用相对阈值
        depth_occlusion = (expected_depth > current_depth)   # 期望深度更远 = 被遮挡
        
        occlusion_mask = np.zeros((H, W), dtype=np.float32)
        occlusion_condition = valid_depth_mask & significant_diff & depth_occlusion
        occlusion_mask[occlusion_condition] = 1.0
        
        # 超出边界的区域不算遮挡
        occlusion_mask[~valid_mask] = 0.0
        
        print(f"   🔍 遮挡像素比例: {np.mean(occlusion_mask):.3f}")
        print(f"   🔍 有效深度比例: {np.mean(valid_depth_mask):.3f}")
        print(f"   🔍 当前深度有效像素: {np.mean(valid_current):.3f}")
        print(f"   🔍 期望深度有效像素: {np.mean(valid_expected):.3f}")
        
        if np.any(valid_depth_mask):
            print(f"   🔍 显著深度差异比例: {np.mean(significant_diff[valid_depth_mask]):.3f}")
            print(f"   🔍 深度遮挡条件比例: {np.mean(depth_occlusion[valid_depth_mask]):.3f}")
            # 显示有效区域的深度统计
            print(f"   🔍 有效区域深度差异: min={depth_diff[valid_depth_mask].min():.3f}, max={depth_diff[valid_depth_mask].max():.3f}, mean={depth_diff[valid_depth_mask].mean():.3f}")
        else:
            print(f"   🔍 显著深度差异比例: 无有效像素")
            print(f"   🔍 深度遮挡条件比例: 无有效像素")
        
        print(f"   🔍 边界外像素比例: {np.mean(~valid_mask):.3f}")
        
        # 形态学优化（只有当遮挡区域不为空时）
        if np.any(occlusion_mask > 0):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
        
        # 调试信息（可选）
        if hasattr(self, 'debug_occlusion') and self.debug_occlusion:
            self._debug_occlusion_detection(curr_frame, expected_depth, current_depth, occlusion_mask)
        
        return occlusion_mask
    
    def _debug_occlusion_detection(self, 
                                  curr_frame: Dict, 
                                  expected_depth: np.ndarray,
                                  current_depth: np.ndarray,
                                  occlusion_mask: np.ndarray) -> None:
        """调试遮挡检测结果"""
        print("\n=== 遮挡检测调试信息 ===")
        print(f"当前深度统计: min={current_depth.min():.3f}, max={current_depth.max():.3f}, mean={current_depth.mean():.3f}")
        print(f"期望深度统计: min={expected_depth.min():.3f}, max={expected_depth.max():.3f}, mean={expected_depth.mean():.3f}")
        
        # 检查深度值分布
        current_positive = np.sum(current_depth > 0)
        expected_positive = np.sum(expected_depth > 0)
        print(f"正深度像素: 当前={current_positive}, 期望={expected_positive}, 总像素={current_depth.size}")
        
        # 检查深度差异分布
        valid_mask = (current_depth > 0) & (expected_depth > 0)
        if np.any(valid_mask):
            depth_diff = np.abs(expected_depth - current_depth)[valid_mask]
            print(f"深度差异统计: min={depth_diff.min():.3f}, max={depth_diff.max():.3f}, mean={depth_diff.mean():.3f}")
            
            relative_diff = depth_diff / (np.maximum(current_depth[valid_mask], expected_depth[valid_mask]) + 1e-8)
            print(f"相对差异统计: min={relative_diff.min():.3f}, max={relative_diff.max():.3f}, mean={relative_diff.mean():.3f}")
        
        print(f"最终遮挡掩码: {np.mean(occlusion_mask):.3f} 的像素被标记为遮挡")
        print("=======================\n")
    
    def _detect_semantic_occlusion_fallback(self, curr_frame: Dict, prev_frame: Dict) -> np.ndarray:
        """
        备用遮挡检测方法 - 基于梯度的方法（原来的实现）
        """
        # 这是原来的基于深度和运动不连续性的方法
        depth_occlusion = np.zeros((curr_frame['reference'].shape[1], curr_frame['reference'].shape[2]), dtype=np.float32)
        
        if 'depth' in curr_frame:
            depth_occlusion = self._compute_depth_discontinuity(curr_frame['depth'])
        
        motion_occlusion = np.zeros_like(depth_occlusion)
        
        if 'motion' in curr_frame:
            motion_occlusion = self._compute_motion_discontinuity(curr_frame['motion'])
        
        occlusion_mask = np.maximum(depth_occlusion, motion_occlusion)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
        
        return occlusion_mask
    
    def _classify_holes(self, 
                       hole_mask: np.ndarray, 
                       curr_frame: Dict, 
                       prev_frame: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """分类静态和动态空洞"""
        # 简化的分类：基于运动幅度
        static_holes = hole_mask.copy()
        dynamic_holes = np.zeros_like(hole_mask)
        
        if 'motion' in curr_frame:
            motion_magnitude = np.sqrt(
                curr_frame['motion'][0]**2 + curr_frame['motion'][1]**2
            )
            
            # 动态区域：运动幅度大的区域
            dynamic_threshold = np.percentile(motion_magnitude, 75)
            dynamic_regions = motion_magnitude > dynamic_threshold
            
            # 重新分配空洞类型
            dynamic_holes = hole_mask * dynamic_regions.astype(np.float32)
            static_holes = hole_mask * (~dynamic_regions).astype(np.float32)
        
        return static_holes, dynamic_holes
    
    # ==================== 训练数据生成模块 ====================
    
    def generate_training_data(self, 
                             curr_frame: Dict, 
                             prev_frame: Dict,
                             next_frame_gt: Optional[Dict] = None,
                             extrapolation_factor: float = 1.0) -> Dict:
        """
        生成完整的训练数据 - 【外插修复版】
        
        Args:
            curr_frame: 当前帧数据
            prev_frame: 前一帧数据
            next_frame_gt: 下一帧Ground Truth数据（外插模式）
            extrapolation_factor: 外推系数，默认1.0表示完整的下一帧
            
        Returns:
            training_data: 包含所有训练数据的字典
        """
        print(f"   🔧 生成训练数据，外推系数: {extrapolation_factor}")
        
        # 1. 计算正确的屏幕空间运动矢量
        base_mv = self._compute_screen_space_mv(curr_frame, prev_frame)
        
        # 2. 应用外推系数
        extrapolated_mv = base_mv * extrapolation_factor
        
        # 3. 前向投影
        if next_frame_gt is not None:
            # 外插模式：从当前帧外推到下一帧
            print(f"   外插模式：t -> t+{extrapolation_factor}")
            warped_image, coverage_mask = self.forward_warp_frame(
                curr_frame, extrapolated_mv
            )
            target_rgb = next_frame_gt['reference']
        else:
            # 重投影模式：从前一帧投影到当前帧
            print(f"   重投影模式：t-1 -> t")
            warped_image, coverage_mask = self.forward_warp_frame(
                prev_frame, base_mv
            )
            target_rgb = curr_frame['reference']
        
        # 4. 空洞和遮挡检测
        if next_frame_gt is not None:
            # 外插模式：使用外推运动矢量
            motion_vectors_for_occlusion = extrapolated_mv
        else:
            # 重投影模式：使用基础运动矢量
            motion_vectors_for_occlusion = base_mv
            
        masks = self.detect_holes_and_occlusion(
            warped_image, target_rgb, coverage_mask, curr_frame, prev_frame, motion_vectors_for_occlusion
        )
        
        # 5. 残差运动矢量计算
        if next_frame_gt is not None:
            # 计算真实的t->t+1运动矢量并计算残差
            try:
                gt_mv = self._compute_screen_space_mv(next_frame_gt, curr_frame)
                residual_mv = gt_mv - extrapolated_mv
            except:
                residual_mv = self._compute_residual_motion_vectors(
                    warped_image, target_rgb, extrapolated_mv, masks['holes']
                )
        else:
            residual_mv = self._compute_residual_motion_vectors(
                warped_image, target_rgb, base_mv, masks['holes']
            )
        
        # 6. 组装训练数据
        # 网络训练需要的数据：warped_RGB(3) + 几何空洞(1) + 遮挡掩码(1) + 残差MV(2) = 7通道
        # 修复关键错误：输入应该是带空洞的外推图像，而不是真实RGB！
        training_sample = np.concatenate([
            warped_image,                                  # ✅ 修复：使用带空洞的外推图像 (3)
            masks['holes'][np.newaxis],                    # ✅ 修复：使用几何空洞掩码 (1) - 基于覆盖度的准确空洞检测
            masks['occlusion'][np.newaxis],                # 遮挡掩码 (1)
            residual_mv,                                   # 残差运动矢量 (2)
            target_rgb                                     # 目标RGB (3)
        ], axis=0)
        
        return {
            'target_rgb': target_rgb,
            'warped_image': warped_image,
            'coverage_mask': coverage_mask,
            'hole_mask': masks['holes'],
            'semantic_holes': masks.get('semantic_holes', masks['holes']),  # 非几何空洞检测结果
            'occlusion_mask': masks['occlusion'],
            'static_holes': masks['static_holes'],
            'dynamic_holes': masks['dynamic_holes'],
            'residual_mv': residual_mv,
            'training_sample': training_sample,
            'base_mv': base_mv,
            'extrapolated_mv': extrapolated_mv,
            'extrapolation_factor': extrapolation_factor
        }
    
    def _compute_residual_motion_vectors(self, 
                                       warped_image: np.ndarray,
                                       target_image: np.ndarray,
                                       original_mv: np.ndarray,
                                       hole_mask: np.ndarray) -> np.ndarray:
        """计算残差运动矢量"""
        # 计算外推图像与目标图像的颜色差异
        color_diff = np.mean(np.abs(warped_image - target_image), axis=0)
        
        # 基于颜色差异和空洞掩码计算残差权重
        residual_weight = np.minimum(color_diff / (color_diff.max() + 1e-8), 1.0)
        residual_weight = np.maximum(residual_weight, hole_mask * 0.5)
        
        # 生成残差运动矢量（简化版本）
        residual_mv = np.zeros_like(original_mv)
        
        # 在空洞区域应用更强的残差（移除随机噪声确保训练稳定性）
        residual_mv[0] = original_mv[0] * 0.1 * residual_weight
        residual_mv[1] = original_mv[1] * 0.1 * residual_weight
        
        # 限制残差幅度
        residual_magnitude = np.sqrt(residual_mv[0]**2 + residual_mv[1]**2)
        max_residual = self.residual_threshold
        
        mask = residual_magnitude > max_residual
        if np.any(mask):
            scale = max_residual / (residual_magnitude + 1e-8)
            residual_mv[0] = np.where(mask, residual_mv[0] * scale, residual_mv[0])
            residual_mv[1] = np.where(mask, residual_mv[1] * scale, residual_mv[1])
        
        return residual_mv
    
    # ==================== 主处理流程 ====================
    
    def process_frame_pair(self, frame_idx: int) -> bool:
        """
        处理单个帧对（重投影模式）
        
        Args:
            frame_idx: 当前帧索引（需要前一帧 frame_idx-1）
            
        Returns:
            success: 处理是否成功
        """
        try:
            # 加载帧数据
            curr_frame = self.load_frame_data(self.scene_name, frame_idx)
            prev_frame = self.load_frame_data(self.scene_name, frame_idx - 1)
            
            if curr_frame is None or prev_frame is None:
                return False
            
            # 生成训练数据（不使用外推，用于验证）
            training_data = self.generate_training_data(
                curr_frame, prev_frame, next_frame_gt=None, extrapolation_factor=1.0
            )
            
            # 保存结果
            self._save_results(frame_idx, training_data)
            
            # 创建可视化
            self._create_visualization(frame_idx, training_data)
            
            # 处理完成
            
            # 强制垃圾回收，释放内存和临时文件
            del curr_frame, prev_frame, training_data
            gc.collect()
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # 异常情况下也要尝试垃圾回收
            gc.collect()
            return False

    def process_frame_triplet(self, frame_idx_t: int, extrapolation_factor: float = 1.0) -> bool:
        """
        处理单个帧三元组 (t-1, t, t+1)，为 t->t+x 的外插生成数据。
        
        Args:
            frame_idx_t: 当前帧(t)的索引
            extrapolation_factor: 外推系数，默认1.0表示完整外推到t+1
            
        Returns:
            success: 处理是否成功
        """
        try:
            # 加载三帧数据
            prev_frame = self.load_frame_data(self.scene_name, frame_idx_t - 1)
            curr_frame = self.load_frame_data(self.scene_name, frame_idx_t)
            next_frame_gt = self.load_frame_data(self.scene_name, frame_idx_t + 1)
            
            if prev_frame is None or curr_frame is None or next_frame_gt is None:
                print(f"无法加载帧三元组 {frame_idx_t-1} -> {frame_idx_t} -> {frame_idx_t+1}")
                return False
            
            print(f"处理帧三元组: {frame_idx_t-1} -> {frame_idx_t} (预测) -> {frame_idx_t+1} (GT)")
            
            # 生成外插训练数据
            training_data = self.generate_training_data(
                curr_frame, prev_frame, next_frame_gt, extrapolation_factor
            )
            
            # 保存结果 (保存时以当前帧 t 的索引为名)
            self._save_results(frame_idx_t, training_data)
            
            # 创建可视化
            self._create_visualization(frame_idx_t, training_data)
            
            print(f"帧 {frame_idx_t} 的外插数据处理完成")
            
            # 强制垃圾回收
            del prev_frame, curr_frame, next_frame_gt, training_data
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"处理帧三元组 {frame_idx_t} 失败: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        return False

    def process_scene(self, 
                     start_frame: int = 1,
                     end_frame: Optional[int] = None,
                     max_frames: Optional[int] = None,
                     mode: str = "triplet",
                     extrapolation_factor: float = 1.0) -> Dict:
        """
        处理整个场景
        
        Args:
            start_frame: 起始帧（默认从1开始，因为需要前一帧）
            end_frame: 结束帧
            max_frames: 最大处理帧数
            mode: 处理模式 - "triplet"(外插模式) 或 "pair"(重投影模式)
            extrapolation_factor: 外推系数（仅在triplet模式下使用）
            
        Returns:
            results: 处理结果统计
        """
        print(f"开始处理场景: {self.scene_name}")
        print(f"处理模式: {mode.upper()} ({'外插' if mode == 'triplet' else '重投影'})")
        if mode == "triplet":
            print(f"外推系数: {extrapolation_factor}")
        
        # 确定处理范围
        scene_dir = self.data_root / self.scene_name
        available_frames = sorted([
            int(f.stem.replace('frame', '')) 
            for f in scene_dir.glob('frame*.zip')
        ])
        
        if not available_frames:
            raise ValueError(f"场景 {self.scene_name} 中没有找到帧数据")
        
        if end_frame is None:
            end_frame = max(available_frames)
        
        # 根据模式调整处理范围
        if mode == "triplet":
            # 三元组模式：确保起始帧有前一帧，结束帧有后一帧
            start_frame = max(start_frame, min(available_frames) + 1)
            end_frame = min(end_frame, max(available_frames) - 1)
            print(f"三元组模式：需要t-1, t, t+1三帧数据")
        else:
            # 帧对模式：确保起始帧有前一帧
            start_frame = max(start_frame, min(available_frames) + 1)
            end_frame = min(end_frame, max(available_frames))
            print(f"帧对模式：需要t-1, t两帧数据")
        
        if max_frames:
            end_frame = min(end_frame, start_frame + max_frames - 1)
        
        print(f"处理范围: frame {start_frame} 到 frame {end_frame} (作为时间点 t)")
        print(f"可用帧: {len(available_frames)} 帧")
        
        # 处理统计
        results = {
            'total_frames': end_frame - start_frame + 1,
            'successful_frames': 0,
            'failed_frames': 0,
            'start_time': time.time(),
            'mode': mode,
            'extrapolation_factor': extrapolation_factor
        }
        
        # 逐帧处理
        for frame_idx in tqdm(range(start_frame, end_frame + 1), desc=f"处理帧({mode})"):
            if mode == "triplet":
                success = self.process_frame_triplet(frame_idx, extrapolation_factor)
            else:
                success = self.process_frame_pair(frame_idx)
                
            if success:
                results['successful_frames'] += 1
            else:
                results['failed_frames'] += 1
            
            # 每10帧强制垃圾回收，防止内存累积
            if frame_idx % 10 == 0:
                gc.collect()
                print(f"🧹 第{frame_idx}帧: 执行垃圾回收")
        
        results['end_time'] = time.time()
        results['total_time'] = results['end_time'] - results['start_time']
        
        # 输出处理结果统计
        print(f"\n📊 处理完成统计:")
        print(f"   模式: {mode.upper()} ({'外插' if mode == 'triplet' else '重投影'})")
        if mode == "triplet":
            print(f"   外推系数: {extrapolation_factor}")
        print(f"   总帧数: {results['total_frames']}")
        print(f"   成功: {results['successful_frames']}")
        print(f"   失败: {results['failed_frames']}")
        print(f"   成功率: {results['successful_frames']/results['total_frames']*100:.1f}%")
        print(f"   总时间: {results['total_time']:.1f}秒")
        print(f"   平均每帧: {results['total_time']/results['total_frames']:.2f}秒")
        
        # 保存处理统计
        stats_file = self.output_dir / f'processing_stats_{mode}.json'
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    # ==================== 输出和可视化模块 ====================
    
    def _save_results(self, frame_idx: int, training_data: Dict):
        """保存处理结果"""
        frame_name = f"frame_{frame_idx:06d}"
        
        # 保存RGB图像
        rgb_path = self.output_dir / 'rgb' / f"{frame_name}.npy"
        np.save(rgb_path, training_data['target_rgb'])
        
        # 保存外推图像
        warped_path = self.output_dir / 'warped' / f"{frame_name}.npy"
        np.save(warped_path, training_data['warped_image'])
        
        # 保存掩码
        masks_dir = self.output_dir / 'masks'
        # 保存几何空洞（基于覆盖度的准确空洞检测）
        np.save(masks_dir / f"{frame_name}_holes.npy", training_data['hole_mask'])
        np.save(masks_dir / f"{frame_name}_occlusion.npy", training_data['occlusion_mask'])
        # 保存静态和动态空洞（基于几何空洞的分类，仅供参考）
        np.save(masks_dir / f"{frame_name}_static_holes.npy", training_data['static_holes'])
        np.save(masks_dir / f"{frame_name}_dynamic_holes.npy", training_data['dynamic_holes'])
        
        # 保存残差运动矢量
        residual_path = self.output_dir / 'residual_mv' / f"{frame_name}.npy"
        np.save(residual_path, training_data['residual_mv'])
        
        # 保存7通道训练数据
        training_path = self.output_dir / 'training_data' / f"{frame_name}.npy"
        np.save(training_path, training_data['training_sample'])
        
        # 保存PNG格式的可视化图像
        self._save_visualization_images(frame_idx, training_data)
    
    def _save_visualization_images(self, frame_idx: int, training_data: Dict):
        """保存可视化图像 - 修复为保存HDR格式（EXR）和可视化格式（PNG）"""
        frame_name = f"frame_{frame_idx:06d}"
        
        # 保存HDR格式的RGB图像（EXR）- 按detach.py方式
        try:
            rgb_hdr = training_data['target_rgb'].transpose(1, 2, 0)  # CHW -> HWC
            self._save_exr_image(self.output_dir / 'rgb' / f"{frame_name}.exr", rgb_hdr)
            print(f"   保存HDR RGB: {frame_name}.exr")
        except Exception as e:
            print(f"   HDR RGB保存失败: {e}")
        
        # 保存可视化用的PNG图像（色调映射后）
        try:
            rgb = training_data['target_rgb'].transpose(1, 2, 0)
            # 简单的色调映射而非硬截断
            rgb_mapped = self._tone_map_for_display(rgb)
            rgb_uint8 = (rgb_mapped * 255).astype(np.uint8)
            cv2.imwrite(str(self.output_dir / 'rgb' / f"{frame_name}.png"), 
                       cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"   PNG RGB保存失败: {e}")
        
        # 外推图像也保存HDR和可视化两个版本
        try:
            warped_hdr = training_data['warped_image'].transpose(1, 2, 0)
            self._save_exr_image(self.output_dir / 'warped' / f"{frame_name}.exr", warped_hdr)
            
            warped_mapped = self._tone_map_for_display(warped_hdr)
            warped_uint8 = (warped_mapped * 255).astype(np.uint8)
            cv2.imwrite(str(self.output_dir / 'warped' / f"{frame_name}.png"),
                       cv2.cvtColor(warped_uint8, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"   外推图像保存失败: {e}")
        
        # 掩码图像（保持PNG格式）
        try:
            masks_dir = self.output_dir / 'masks'
            # 保存几何空洞PNG（基于覆盖度的准确空洞检测）
            cv2.imwrite(str(masks_dir / f"{frame_name}_holes.png"), 
                       (training_data['hole_mask'] * 255).astype(np.uint8))
            cv2.imwrite(str(masks_dir / f"{frame_name}_occlusion.png"), 
                       (training_data['occlusion_mask'] * 255).astype(np.uint8))
        except Exception as e:
            print(f"   掩码保存失败: {e}")
    
    def _save_exr_image(self, file_path: Path, image_data: np.ndarray):
        """保存EXR格式图像 - 使用detach.py的方法"""
        try:
            import OpenEXR
            import Imath
            
            # 确保图像数据是HWC格式的float32
            if image_data.shape[0] == 3:  # CHW -> HWC
                image_data = image_data.transpose(1, 2, 0)
            
            image_data = image_data.astype(np.float32)
            
            # 创建EXR文件头
            header = OpenEXR.Header(image_data.shape[1], image_data.shape[0])
            header['channels'] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) 
                                 for name in ['R', 'G', 'B']}
            
            # 创建输出文件
            out_file = OpenEXR.OutputFile(str(file_path), header)
            
            # 写入通道数据
            patch_data = {}
            for i, channel_name in enumerate(['R', 'G', 'B']):
                patch_data[channel_name] = image_data[:, :, i].flatten().astype(np.float32).tobytes()
            
            out_file.writePixels(patch_data)
            out_file.close()
            
        except ImportError:
            print(f"   OpenEXR未安装，跳过EXR保存: {file_path}")
        except Exception as e:
            print(f"   EXR保存失败: {e}")
    
    def _tone_map_for_display(self, hdr_image: np.ndarray) -> np.ndarray:
        """对HDR图像进行色调映射用于显示"""
        # 简单的Reinhard色调映射
        return hdr_image / (1.0 + hdr_image)
    
    def _create_visualization(self, frame_idx: int, training_data: Dict):
        """创建综合可视化"""
        frame_name = f"frame_{frame_idx:06d}"
        
        # 创建多子图可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Frame {frame_idx} Processing Results', fontsize=16)
        
        # 目标RGB
        rgb = training_data['target_rgb'].transpose(1, 2, 0)
        rgb = np.clip(rgb, 0, 1)
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Target RGB')
        axes[0, 0].axis('off')
        
        # 外推图像
        warped = training_data['warped_image'].transpose(1, 2, 0)
        warped = np.clip(warped, 0, 1)
        axes[0, 1].imshow(warped)
        axes[0, 1].set_title('Warped Image')
        axes[0, 1].axis('off')
        
        # 覆盖掩码
        im1 = axes[0, 2].imshow(training_data['coverage_mask'], cmap='viridis')
        axes[0, 2].set_title('Coverage Mask')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)
        
        # 几何空洞
        axes[0, 3].imshow(training_data['hole_mask'], cmap='Reds')
        axes[0, 3].set_title('Geometric Holes')
        axes[0, 3].axis('off')
        
        # 几何空洞检测（网络训练输入）
        axes[1, 0].imshow(training_data['hole_mask'], cmap='Blues')
        axes[1, 0].set_title('Geometric Holes (Network Input)')
        axes[1, 0].axis('off')
        
        # 静态空洞
        axes[1, 1].imshow(training_data['static_holes'], cmap='Oranges')
        axes[1, 1].set_title('Static Holes(Geometric)')
        axes[1, 1].axis('off')
        
        # 动态空洞
        axes[1, 2].imshow(training_data['dynamic_holes'], cmap='Purples')
        axes[1, 2].set_title('Dynamic Holes(Geometric)')
        axes[1, 2].axis('off')
        
        # 残差运动矢量幅度
        residual_mag = np.sqrt(training_data['residual_mv'][0]**2 + training_data['residual_mv'][1]**2)
        im2 = axes[1, 3].imshow(residual_mag, cmap='plasma')
        axes[1, 3].set_title('Residual MV Magnitude')
        axes[1, 3].axis('off')
        plt.colorbar(im2, ax=axes[1, 3], fraction=0.046)
        
        plt.tight_layout()
        vis_path = self.output_dir / 'visualization' / f"{frame_name}_analysis.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Unified NoiseBase Preprocessor')
    parser.add_argument('--data-root', type=str, required=True, 
                       help='NoiseBase数据根目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--scene', type=str, default='bistro1',
                       help='场景名称')
    parser.add_argument('--start-frame', type=int, default=1,
                       help='起始帧')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='结束帧')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大处理帧数')
    parser.add_argument('--hole-threshold', type=float, default=0.3,
                       help='空洞检测阈值')
    parser.add_argument('--use-numba', action='store_true',
                       help='使用Numba加速')
    parser.add_argument('--test-mode', action='store_true',
                       help='测试模式（只处理几帧）')
    
    args = parser.parse_args()
    
    try:
        # 创建预处理器
        preprocessor = UnifiedNoiseBasePreprocessor(
            data_root=args.data_root,
            output_dir=args.output,
            scene_name=args.scene,
            hole_threshold=args.hole_threshold,
            use_numba=args.use_numba
        )
        
        # 测试模式
        if args.test_mode:
            args.max_frames = 3
        
        # 处理场景
        results = preprocessor.process_scene(
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_frames=args.max_frames
        )
        
        if results['successful_frames'] > 0:
            print(f"\n预处理完成！输出目录: {args.output}")
            print(f"训练数据格式: 7通道 [RGB(3) + HoleMask(1) + OcclusionMask(1) + ResidualMV(2)]")
        else:
            print("\n预处理失败，请检查数据和参数")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n预处理出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()