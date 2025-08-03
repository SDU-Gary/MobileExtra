#!/usr/bin/env python3
"""
@file noisebase_preprocessor.py
@brief NoiseBase数据集预处理器

功能描述：
- 从NoiseBase Zarr数据生成网络训练所需的6通道输入
- 实现前向warp投影生成外推帧
- 检测空洞并生成遮挡掩码
- 计算投影残差运动矢量
- 输出适配MobileInpaintingNetwork的训练数据

处理流程：
1. 加载连续帧的NoiseBase数据
2. 使用运动矢量进行前向warp投影
3. 检测warp后的空洞区域
4. 计算投影残差
5. 生成6通道训练数据

输出格式：
- RGB: 原始参考图像 [3, H, W]
- Mask: 空洞掩码 [1, H, W] 
- ResidualMV: 残差运动矢量 [2, H, W]

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import tqdm

# 导入兼容性模块
try:
    # 尝试相对导入
    from .zarr_compat import load_zarr_group, decompress_RGBE_compat as decompress_RGBE
    from .projective import screen_space_position, motion_vectors, log_depth
except ImportError:
    # 尝试从当前目录导入
    import sys
    from pathlib import Path
    
    # 添加training目录到Python路径
    training_dir = Path(__file__).parent
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    
    from zarr_compat import load_zarr_group, decompress_RGBE_compat as decompress_RGBE
    from projective import screen_space_position, motion_vectors, log_depth


class NoiseBasePreprocessor:
    """
    NoiseBase数据集预处理器
    
    将NoiseBase的Zarr格式数据转换为适合MobileInpaintingNetwork训练的格式
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 scene_name: str = "bistro1"):
        """
        初始化预处理器
        
        Args:
            input_dir: NoiseBase数据输入目录
            output_dir: 处理后数据输出目录  
            scene_name: 场景名称
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        
        # 创建输出目录结构
        self.setup_output_dirs()
        
        # Warp参数
        self.warp_method = 'forward_projection'  # 前向投影方法
        self.hole_threshold = 0.5  # 空洞检测阈值
        self.residual_threshold = 2.0  # 残差阈值
        
        print(f"=== NoiseBase Preprocessor ===")
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Scene: {scene_name}")
    
    def setup_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            'rgb',           # 原始RGB图像
            'warped',        # Warp后的图像
            'masks',         # 空洞掩码
            'residual_mv',   # 残差运动矢量
            'training_data', # 最终训练数据
            'visualization'  # 可视化结果
        ]
        
        for dir_name in dirs:
            (self.output_dir / self.scene_name / dir_name).mkdir(parents=True, exist_ok=True)
    
    def load_frame_data(self, frame_idx: int) -> Dict:
        """
        加载单帧NoiseBase数据
        
        Args:
            frame_idx: 帧索引
        
        Returns:
            frame_data: 帧数据字典
        """
        zip_path = self.input_dir / self.scene_name / f"frame{frame_idx:04d}.zip"
        
        if not zip_path.exists():
            raise FileNotFoundError(f"Frame data not found: {zip_path}")
        
        try:
            # 加载Zarr数据（使用兼容性函数）
            ds = load_zarr_group(str(zip_path))
            
            # 检查并提取各种缓冲区数据
            def safe_extract_array(group, key, description):
                """安全提取数组数据"""
                try:
                    # 方法1: 尝试作为属性访问
                    if hasattr(group, key):
                        return np.array(getattr(group, key))
                    
                    # 方法2: 尝试作为字典键访问
                    if hasattr(group, '__getitem__') and key in group:
                        return np.array(group[key])
                    
                    # 方法3: 尝试直接索引访问
                    try:
                        return np.array(group[key])
                    except (KeyError, TypeError):
                        pass
                    
                    # 如果都失败，列出可用键
                    available_keys = []
                    if hasattr(group, 'keys'):
                        available_keys = list(group.keys())
                    elif hasattr(group, '__dict__'):
                        available_keys = [k for k in group.__dict__.keys() if not k.startswith('_')]
                    
                    raise KeyError(f"{description} ('{key}') not found. Available: {available_keys}")
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to extract {description}: {e}")
        
        except Exception as e:
            raise RuntimeError(f"Error loading frame {frame_idx}: {e}")
        
        color_rgbe = safe_extract_array(ds, 'color', 'RGBE格式颜色')
        diffuse = safe_extract_array(ds, 'diffuse', '漫反射')
        normal = safe_extract_array(ds, 'normal', '法线')
        motion = safe_extract_array(ds, 'motion', '世界空间运动矢量')
        position = safe_extract_array(ds, 'position', '世界空间位置')
        reference = safe_extract_array(ds, 'reference', 'Ground Truth参考图像')
        
        # 相机参数
        camera_pos = safe_extract_array(ds, 'camera_position', '相机位置')
        view_proj_mat = safe_extract_array(ds, 'view_proj_mat', '视图投影矩阵')
        exposure = safe_extract_array(ds, 'exposure', '曝光参数')
        
        # 解压缩颜色数据
        rgb_color = decompress_RGBE(color_rgbe, exposure)
        
        # 对Monte Carlo样本求平均的安全函数
        def safe_mean_samples(arr, name):
            """安全地对样本维度求平均"""
            if arr.ndim == 4 and arr.shape[3] > 1:
                # 有样本维度，求平均
                result = arr.mean(axis=3)
                print(f"   {name}: {arr.shape} -> {result.shape} (平均)")
                return result
            elif arr.ndim == 4 and arr.shape[3] == 1:
                # 只有一个样本，移除维度
                result = arr.squeeze(axis=3)
                print(f"   {name}: {arr.shape} -> {result.shape} (squeeze)")
                return result
            elif arr.ndim == 3:
                # 没有样本维度，直接使用
                print(f"   {name}: {arr.shape} (直接使用)")
                return arr
            else:
                print(f"   {name}: 未预期的形状 {arr.shape}")
                return arr
        
        rgb_avg = safe_mean_samples(rgb_color, 'rgb_color')
        reference_avg = safe_mean_samples(reference, 'reference')
        motion_avg = safe_mean_samples(motion, 'motion')
        position_avg = safe_mean_samples(position, 'position')
        normal_avg = safe_mean_samples(normal, 'normal')
        diffuse_avg = safe_mean_samples(diffuse, 'diffuse')
        
        return {
            'rgb': rgb_avg,
            'reference': reference_avg,
            'motion': motion_avg,
            'position': position_avg,
            'normal': normal_avg,
            'diffuse': diffuse_avg,
            'camera_pos': camera_pos,
            'view_proj_mat': view_proj_mat,
            'frame_idx': frame_idx
        }
    
    def compute_screen_motion_vectors(self, 
                                    curr_frame: Dict, 
                                    prev_frame: Dict) -> np.ndarray:
        """
        计算屏幕空间运动矢量
        
        Args:
            curr_frame: 当前帧数据
            prev_frame: 前一帧数据
        
        Returns:
            screen_mv: 屏幕空间运动矢量 [2, H, W]
        """
        # 获取位置和运动数据
        curr_position = curr_frame['position']
        curr_motion = curr_frame['motion']
        
        # 确保数据有样本维度 (projective.py期望3HWS格式)
        if curr_position.ndim == 3:
            # 添加样本维度 [3, H, W] -> [3, H, W, 1]
            curr_position = curr_position[..., np.newaxis]
        if curr_motion.ndim == 3:
            # 添加样本维度 [3, H, W] -> [3, H, W, 1]
            curr_motion = curr_motion[..., np.newaxis]
        
        height, width = curr_position.shape[1:3]
        
        # 使用projective.py中的函数计算运动矢量
        screen_mv = motion_vectors(
            w_position=curr_position,
            w_motion=curr_motion,
            pv=curr_frame['view_proj_mat'],
            prev_pv=prev_frame['view_proj_mat'],
            height=height,
            width=width
        )
        
        # 移除样本维度并返回 [2, H, W, 1] -> [2, H, W]
        if screen_mv.ndim == 4 and screen_mv.shape[3] == 1:
            screen_mv = screen_mv.squeeze(axis=3)
        
        return screen_mv
    
    def forward_warp(self, 
                    source_image: np.ndarray,
                    motion_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量化的前向warp投影（优化版本）
        
        Args:
            source_image: 源图像 [3, H, W]
            motion_vectors: 运动矢量 [2, H, W]
        
        Returns:
            warped_image: warp后图像 [3, H, W]
            coverage_mask: 覆盖掩码 [H, W] (1=有效, 0=空洞)
        """
        C, H, W = source_image.shape
        
        # 初始化输出
        warped_image = np.zeros_like(source_image)
        coverage_mask = np.zeros((H, W), dtype=np.float32)
        
        # 创建坐标网格
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # 计算目标位置
        target_x = x_coords + motion_vectors[0]  # [H, W]
        target_y = y_coords + motion_vectors[1]  # [H, W]
        
        # 有效像素掩码
        valid_mask = (
            (target_x >= 0) & (target_x < W-1) &
            (target_y >= 0) & (target_y < H-1)
        )
        
        if not np.any(valid_mask):
            return warped_image, coverage_mask
        
        # 提取有效像素
        valid_y, valid_x = np.where(valid_mask)
        valid_target_x = target_x[valid_mask]
        valid_target_y = target_y[valid_mask]
        
        # 双线性插值的四个邻居
        x0 = np.floor(valid_target_x).astype(int)
        y0 = np.floor(valid_target_y).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # 确保在边界内
        boundary_mask = (x1 < W) & (y1 < H)
        if not np.any(boundary_mask):
            return warped_image, coverage_mask
        
        # 过滤边界外的点
        valid_y = valid_y[boundary_mask]
        valid_x = valid_x[boundary_mask]
        valid_target_x = valid_target_x[boundary_mask]
        valid_target_y = valid_target_y[boundary_mask]
        x0, y0, x1, y1 = x0[boundary_mask], y0[boundary_mask], x1[boundary_mask], y1[boundary_mask]
        
        # 双线性权重
        wx = valid_target_x - x0
        wy = valid_target_y - y0
        
        weights = [
            (1-wx) * (1-wy),  # (x0, y0)
            wx * (1-wy),      # (x1, y0)
            (1-wx) * wy,      # (x0, y1)
            wx * wy           # (x1, y1)
        ]
        
        positions = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        
        # 向量化分布像素值
        for (px, py), weight in zip(positions, weights):
            valid_weight_mask = weight > 1e-6
            if np.any(valid_weight_mask):
                # 使用np.add.at进行原子累加
                for c in range(C):
                    np.add.at(warped_image[c], (py[valid_weight_mask], px[valid_weight_mask]), 
                             source_image[c, valid_y[valid_weight_mask], valid_x[valid_weight_mask]] * weight[valid_weight_mask])
                np.add.at(coverage_mask, (py[valid_weight_mask], px[valid_weight_mask]), weight[valid_weight_mask])
        
        # 归一化
        valid_pixels = coverage_mask > 1e-6
        for c in range(C):
            warped_image[c, valid_pixels] /= coverage_mask[valid_pixels]
        
        # 生成二值覆盖掩码
        coverage_mask = (coverage_mask > self.hole_threshold).astype(np.float32)
        
        return warped_image, coverage_mask
    
    def detect_holes_and_occlusion(self,
                                 warped_image: np.ndarray,
                                 target_image: np.ndarray,
                                 coverage_mask: np.ndarray,
                                 curr_frame: Dict,
                                 prev_frame: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        分别检测空洞和遮挡掩码
        
        Args:
            warped_image: warp后图像 [3, H, W]
            target_image: 目标图像 [3, H, W]
            coverage_mask: 覆盖掩码 [H, W]
            curr_frame: 当前帧数据
            prev_frame: 前一帧数据
        
        Returns:
            hole_mask: 空洞掩码 [H, W] (1=空洞, 0=有效)
            occlusion_mask: 遮挡掩码 [H, W] (1=遮挡, 0=无遮挡)
        """
        H, W = coverage_mask.shape
        
        # === 方法1: 几何空洞检测 ===
        # 基于覆盖度的纯几何空洞
        hole_mask = (coverage_mask < self.hole_threshold).astype(np.float32)
        
        # 形态学处理优化空洞掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, kernel)
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel)
        
        # === 方法2: 遮挡检测 ===
        occlusion_mask = self.detect_occlusion_mask(curr_frame, prev_frame)
        
        return hole_mask, occlusion_mask
    
    def detect_occlusion_mask(self, curr_frame: Dict, prev_frame: Dict) -> np.ndarray:
        """
        基于深度和几何关系检测遮挡掩码
        
        Args:
            curr_frame: 当前帧数据
            prev_frame: 前一帧数据
            
        Returns:
            occlusion_mask: 遮挡掩码 [H, W]
        """
        H, W = curr_frame['position'].shape[1:3]
        
        # 获取深度信息（从世界空间位置计算）
        curr_depth = self.compute_depth_from_position(
            curr_frame['position'], curr_frame['camera_pos']
        )
        prev_depth = self.compute_depth_from_position(
            prev_frame['position'], prev_frame['camera_pos']
        )
        
        # 方法1: 基于深度不连续性检测遮挡
        depth_gradient = np.gradient(curr_depth)
        depth_discontinuity = np.sqrt(depth_gradient[0]**2 + depth_gradient[1]**2)
        depth_occlusion = (depth_discontinuity > np.percentile(depth_discontinuity, 95))
        
        # 方法2: 基于运动不一致性检测遮挡
        # 计算相邻像素的运动矢量差异
        motion_x = curr_frame['motion'][0]
        motion_y = curr_frame['motion'][1]
        
        motion_grad_x = np.gradient(motion_x)
        motion_grad_y = np.gradient(motion_y)
        motion_discontinuity = np.sqrt(
            motion_grad_x[0]**2 + motion_grad_x[1]**2 + 
            motion_grad_y[0]**2 + motion_grad_y[1]**2
        )
        motion_occlusion = (motion_discontinuity > np.percentile(motion_discontinuity, 90))
        
        # 结合两种方法
        occlusion_mask = (depth_occlusion | motion_occlusion).astype(np.float32)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
        
        return occlusion_mask
    
    def compute_depth_from_position(self, world_position: np.ndarray, camera_pos: np.ndarray) -> np.ndarray:
        """
        从世界空间位置计算深度
        
        Args:
            world_position: 世界空间位置 [3, H, W]
            camera_pos: 相机位置 [3]
            
        Returns:
            depth: 深度图 [H, W]
        """
        # 计算到相机的距离作为深度
        diff = world_position - camera_pos.reshape(3, 1, 1)
        depth = np.linalg.norm(diff, axis=0)
        return depth
    
    def compute_residual_motion_vectors(self,
                                      warped_image: np.ndarray,
                                      target_image: np.ndarray,
                                      coverage_mask: np.ndarray,
                                      motion_vectors: np.ndarray,
                                      hole_mask: np.ndarray) -> np.ndarray:
        """
        计算残差运动矢量
        
        Args:
            warped_image: warp后图像 [3, H, W]
            target_image: 目标图像 [3, H, W]
            coverage_mask: 覆盖掩码 [H, W]
            motion_vectors: 原始运动矢量 [2, H, W]
            hole_mask: 空洞掩码 [H, W]
        
        Returns:
            residual_mv: 残差运动矢量 [2, H, W]
        """
        residual_mv = np.zeros_like(motion_vectors)
        
        # 对于有效区域（非空洞），计算warp误差
        valid_mask = (coverage_mask > self.hole_threshold) & (hole_mask < 0.5)
        if np.any(valid_mask):
            # 基于颜色差异计算残差
            color_error = np.linalg.norm(warped_image - target_image, axis=0)
            error_factor = np.clip(color_error / self.residual_threshold, 0, 1)
            
            # 残差运动矢量与误差成比例
            residual_mv[0][valid_mask] = motion_vectors[0][valid_mask] * error_factor[valid_mask] * 0.1
            residual_mv[1][valid_mask] = motion_vectors[1][valid_mask] * error_factor[valid_mask] * 0.1
        
        return residual_mv
    
    def create_training_sample(self,
                             rgb_image: np.ndarray,
                             hole_mask: np.ndarray,
                             occlusion_mask: np.ndarray,
                             residual_mv: np.ndarray) -> np.ndarray:
        """
        创建7通道训练样本
        
        Args:
            rgb_image: RGB图像 [3, H, W]  
            hole_mask: 空洞掩码 [H, W]
            occlusion_mask: 遮挡掩码 [H, W]
            residual_mv: 残差运动矢量 [2, H, W]
        
        Returns:
            training_sample: 7通道训练数据 [7, H, W]
        """
        # 拼接7通道数据：RGB(3) + HoleMask(1) + OcclusionMask(1) + ResidualMV(2)
        training_sample = np.concatenate([
            rgb_image,                              # RGB通道 [3, H, W]
            hole_mask[np.newaxis, :, :],           # 空洞掩码通道 [1, H, W]  
            occlusion_mask[np.newaxis, :, :],      # 遮挡掩码通道 [1, H, W]
            residual_mv                            # 残差MV通道 [2, H, W]
        ], axis=0)
        
        return training_sample
    
    def save_frame_results(self, 
                          frame_idx: int,
                          rgb_image: np.ndarray,
                          warped_image: np.ndarray,
                          hole_mask: np.ndarray,
                          occlusion_mask: np.ndarray,
                          residual_mv: np.ndarray,
                          training_sample: np.ndarray):
        """
        保存帧处理结果
        
        Args:
            frame_idx: 帧索引
            rgb_image: 原始RGB图像
            warped_image: warp后图像
            hole_mask: 空洞掩码
            occlusion_mask: 遮挡掩码
            residual_mv: 残差运动矢量
            training_sample: 训练样本
        """
        base_path = self.output_dir / self.scene_name
        
        # 转换为可保存格式 (CHW -> HWC, 归一化到[0,1])
        def prepare_for_save(img, is_mask=False):
            if len(img.shape) == 3:
                img = img.transpose(1, 2, 0)  # CHW -> HWC
            if not is_mask:
                img = np.clip((img + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
            return (img * 255).astype(np.uint8)
        
        # 保存各种数据
        frame_name = f"frame_{frame_idx:04d}"
        
        # RGB图像
        rgb_save = prepare_for_save(rgb_image)
        cv2.imwrite(str(base_path / 'rgb' / f"{frame_name}.png"), 
                   cv2.cvtColor(rgb_save, cv2.COLOR_RGB2BGR))
        
        # Warp后图像
        warped_save = prepare_for_save(warped_image)
        cv2.imwrite(str(base_path / 'warped' / f"{frame_name}.png"),
                   cv2.cvtColor(warped_save, cv2.COLOR_RGB2BGR))
        
        # 空洞掩码
        hole_mask_save = (hole_mask * 255).astype(np.uint8)
        cv2.imwrite(str(base_path / 'masks' / f"{frame_name}_holes.png"), hole_mask_save)
        
        # 遮挡掩码
        occlusion_mask_save = (occlusion_mask * 255).astype(np.uint8)
        cv2.imwrite(str(base_path / 'masks' / f"{frame_name}_occlusion.png"), occlusion_mask_save)
        
        # 残差运动矢量（保存为NumPy数组）
        np.save(str(base_path / 'residual_mv' / f"{frame_name}.npy"), residual_mv)
        
        # 训练样本（保存为NumPy数组）
        np.save(str(base_path / 'training_data' / f"{frame_name}.npy"), training_sample)
        
        # 可视化结果
        self.create_visualization(frame_idx, rgb_image, warped_image, hole_mask, occlusion_mask, residual_mv)
    
    def create_visualization(self,
                           frame_idx: int,
                           rgb_image: np.ndarray,
                           warped_image: np.ndarray, 
                           hole_mask: np.ndarray,
                           occlusion_mask: np.ndarray,
                           residual_mv: np.ndarray):
        """
        创建可视化结果
        
        Args:
            frame_idx: 帧索引
            rgb_image: 原始图像
            warped_image: warp后图像
            hole_mask: 空洞掩码
            occlusion_mask: 遮挡掩码
            residual_mv: 残差运动矢量
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 原始图像
        rgb_vis = np.clip((rgb_image.transpose(1, 2, 0) + 1) / 2, 0, 1)
        axes[0, 0].imshow(rgb_vis)
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')
        
        # Warp后图像
        warped_vis = np.clip((warped_image.transpose(1, 2, 0) + 1) / 2, 0, 1)
        axes[0, 1].imshow(warped_vis)
        axes[0, 1].set_title('Warped Image')
        axes[0, 1].axis('off')
        
        # 空洞掩码
        axes[0, 2].imshow(hole_mask, cmap='gray')
        axes[0, 2].set_title('Hole Mask')
        axes[0, 2].axis('off')
        
        # 遮挡掩码
        axes[0, 3].imshow(occlusion_mask, cmap='gray')
        axes[0, 3].set_title('Occlusion Mask')
        axes[0, 3].axis('off')
        
        # 残差运动矢量可视化
        mv_magnitude = np.sqrt(residual_mv[0]**2 + residual_mv[1]**2)
        axes[1, 0].imshow(mv_magnitude, cmap='jet')
        axes[1, 0].set_title('Residual MV Magnitude')
        axes[1, 0].axis('off')
        
        # 运动矢量方向
        mv_angle = np.arctan2(residual_mv[1], residual_mv[0])
        axes[1, 1].imshow(mv_angle, cmap='hsv')
        axes[1, 1].set_title('Residual MV Direction')
        axes[1, 1].axis('off')
        
        # 空洞覆盖结果
        hole_overlay = rgb_vis.copy()
        hole_overlay[hole_mask > 0.5] = [1, 0, 0]  # 红色标记空洞
        axes[1, 2].imshow(hole_overlay)
        axes[1, 2].set_title('Holes Overlay')
        axes[1, 2].axis('off')
        
        # 遮挡覆盖结果
        occlusion_overlay = rgb_vis.copy()
        occlusion_overlay[occlusion_mask > 0.5] = [0, 1, 0]  # 绿色标记遮挡
        axes[1, 3].imshow(occlusion_overlay)
        axes[1, 3].set_title('Occlusion Overlay')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        vis_path = self.output_dir / self.scene_name / 'visualization' / f"vis_{frame_idx:04d}.png"
        plt.savefig(str(vis_path), dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_frame_pair(self, curr_idx: int, prev_idx: int):
        """
        处理帧对
        
        Args:
            curr_idx: 当前帧索引
            prev_idx: 前一帧索引
        """
        # 加载连续两帧数据
        curr_frame = self.load_frame_data(curr_idx)
        prev_frame = self.load_frame_data(prev_idx)
        
        # 计算屏幕空间运动矢量
        screen_mv = self.compute_screen_motion_vectors(curr_frame, prev_frame)
        
        # 前向warp投影
        warped_image, coverage_mask = self.forward_warp(
            curr_frame['reference'], screen_mv
        )
        
        # 分别检测空洞和遮挡掩码
        hole_mask, occlusion_mask = self.detect_holes_and_occlusion(
            warped_image, curr_frame['reference'], coverage_mask, curr_frame, prev_frame
        )
        
        # 计算残差运动矢量
        residual_mv = self.compute_residual_motion_vectors(
            warped_image, curr_frame['reference'], coverage_mask, screen_mv, hole_mask
        )
        
        # 创建训练样本
        training_sample = self.create_training_sample(
            curr_frame['reference'], hole_mask, occlusion_mask, residual_mv
        )
        
        # 保存结果
        self.save_frame_results(
            curr_idx, curr_frame['reference'], warped_image, 
            hole_mask, occlusion_mask, residual_mv, training_sample
        )
        
        return {
            'frame_idx': curr_idx,
            'hole_coverage': np.mean(hole_mask),
            'occlusion_coverage': np.mean(occlusion_mask),
            'mv_magnitude': np.mean(np.sqrt(residual_mv[0]**2 + residual_mv[1]**2))
        }
    
    def process_sequence(self, start_frame: int = 0, end_frame: int = None):
        """
        处理整个序列
        
        Args:
            start_frame: 起始帧
            end_frame: 结束帧
        """
        # 确定处理范围
        if end_frame is None:
            # 自动检测可用帧数
            frame_count = 0
            while (self.input_dir / self.scene_name / f"frame{frame_count:04d}.zip").exists():
                frame_count += 1
            end_frame = frame_count - 1
        
        print(f"Processing frames {start_frame} to {end_frame} ({end_frame-start_frame+1} total)")
        
        # 处理统计
        stats = {
            'processed_frames': 0,
            'total_hole_coverage': 0,
            'total_mv_magnitude': 0
        }
        
        # 逐帧处理
        with tqdm.tqdm(total=end_frame-start_frame, desc="Processing frames") as pbar:
            for i in range(start_frame + 1, end_frame + 1):  # 从第二帧开始
                try:
                    result = self.process_frame_pair(i, i-1)
                    
                    # 更新统计
                    stats['processed_frames'] += 1
                    stats['total_hole_coverage'] += result['hole_coverage']
                    stats['total_mv_magnitude'] += result['mv_magnitude']
                    
                    pbar.set_postfix({
                        'Holes': f"{result['hole_coverage']:.3f}",
                        'MV': f"{result['mv_magnitude']:.3f}"
                    })
                    
                except Exception as e:
                    print(f"Error processing frame {i}: {e}")
                
                pbar.update(1)
        
        # 打印最终统计
        if stats['processed_frames'] > 0:
            avg_hole_coverage = stats['total_hole_coverage'] / stats['processed_frames']
            avg_mv_magnitude = stats['total_mv_magnitude'] / stats['processed_frames']
            
            print(f"\n=== Processing Statistics ===")
            print(f"Processed frames: {stats['processed_frames']}")
            print(f"Average hole coverage: {avg_hole_coverage:.3f}")
            print(f"Average MV magnitude: {avg_mv_magnitude:.3f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NoiseBase Preprocessor')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='NoiseBase data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--scene', type=str, default='bistro1',
                       help='Scene name (bistro1, kitchen)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Start frame index')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='End frame index')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔄 NoiseBase Preprocessing Started")
    print("="*60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Scene: {args.scene}")
    print("="*60)
    
    try:
        # 创建预处理器
        preprocessor = NoiseBasePreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            scene_name=args.scene
        )
        
        # 处理序列
        preprocessor.process_sequence(
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
        
        print("\n🎉 Preprocessing completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()