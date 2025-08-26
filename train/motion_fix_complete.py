#!/usr/bin/env python3
"""
NoiseBase运动矢量修复完整方案
实现正确的3D到2D投影计算和外推系数支持

基于motion计算修正.md中的对话内容实现
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

class MotionVectorFixer:
    """运动矢量修复器 - 实现正确的3D到2D投影"""
    
    def __init__(self, debug: bool = True):
        self.debug = debug
    
    @staticmethod
    def screen_space_projection(world_pos: np.ndarray, 
                               view_proj_matrix: np.ndarray, 
                               height: int, 
                               width: int) -> np.ndarray:
        """
        将世界空间位置 (3, H, W) 投影到屏幕空间像素坐标 (2, H, W)。
        
        Args:
            world_pos: 世界空间坐标，形状 (3, H, W)
            view_proj_matrix: 视图投影矩阵，形状 (4, 4)
            height: 图像高度
            width: 图像宽度
            
        Returns:
            屏幕空间坐标 (U, V)，形状 (2, H, W)
        """
        C, H, W = world_pos.shape
        if C != 3:
            raise ValueError(f"world_pos 必须是3通道，实际为 {C}")

        # 1. 扩展为齐次坐标 (4, H, W)
        ones = np.ones((1, H, W), dtype=world_pos.dtype)
        homogeneous_pos = np.concatenate([world_pos, ones], axis=0)

        # 2. 矩阵投影 (view_proj @ pos) -> (4, H, W)
        # 使用einsum进行高效的批量矩阵向量乘法
        projected_pos = np.einsum('ij,jhw->ihw', view_proj_matrix, homogeneous_pos)
        
        # 3. 透视除法 (w-divide)
        w = projected_pos[3:4, :, :]
        # 避免除以零
        epsilon = 1e-5
        w = np.where(w < epsilon, epsilon, w)
        
        # 只对x, y进行透视除法
        ndc_pos = projected_pos[:2, :, :] / w

        # 4. NDC坐标[-1, 1]转换为屏幕像素坐标[0, W]和[0, H]
        # U = (ndc_x * 0.5 + 0.5) * W
        # V = (-ndc_y * 0.5 + 0.5) * H  (Y轴在NDC和图像坐标系中方向相反)
        screen_pos_x = (ndc_pos[0] * 0.5 + 0.5) * width
        screen_pos_y = (-ndc_pos[1] * 0.5 + 0.5) * height

        return np.stack([screen_pos_x, screen_pos_y], axis=0)
    
    def compute_screen_space_mv(self, 
                               curr_frame: Dict, 
                               prev_frame: Dict) -> np.ndarray:
        """
        【核心修复】根据前后两帧的3D信息计算屏幕空间运动矢量。
        这个MV描述了当前帧的像素来自于前一帧的哪个位置。
        
        Args:
            curr_frame: 当前帧的数据字典
            prev_frame: 前一帧的数据字典
            
        Returns:
            屏幕空间运动矢量，形状 (2, H, W)，单位为像素。
        """
        # 1. 获取必要数据
        pos_t = curr_frame['position']      # 当前帧世界坐标 (3, H, W)
        motion_t = curr_frame['motion'][:3] # 世界空间运动矢量 (3, H, W) (确保为3通道)
        
        # 获取相机矩阵
        if 'camera_params' in curr_frame and 'view_proj_mat' in curr_frame['camera_params']:
            vp_mat_t = curr_frame['camera_params']['view_proj_mat']
        else:
            print("   ⚠️ 使用默认投影矩阵")
            vp_mat_t = self._create_default_projection_matrix()
            
        if 'camera_params' in prev_frame and 'view_proj_mat' in prev_frame['camera_params']:
            vp_mat_prev = prev_frame['camera_params']['view_proj_mat']
        else:
            print("   ⚠️ 前一帧使用默认投影矩阵")
            vp_mat_prev = self._create_default_projection_matrix()

        H, W = pos_t.shape[1], pos_t.shape[2]

        # 2. 计算上一帧的世界坐标
        # motion 定义为 pos_t - pos_{t-1}，所以 pos_{t-1} = pos_t - motion_t
        pos_prev = pos_t - motion_t

        # 3. 将当前帧世界坐标投影到当前帧屏幕
        screen_pos_t = self.screen_space_projection(pos_t, vp_mat_t, H, W)

        # 4. 将上一帧世界坐标投影到上一帧屏幕
        screen_pos_prev = self.screen_space_projection(pos_prev, vp_mat_prev, H, W)
        
        # 5. 计算屏幕空间运动矢量 (MV)
        # MV = 来源位置 - 目标位置 = screen_pos_prev - screen_pos_t
        screen_space_mv = screen_pos_prev - screen_pos_t
        
        # 调试信息
        if self.debug:
            mv_magnitude = np.sqrt(screen_space_mv[0]**2 + screen_space_mv[1]**2)
            print(f"   ✅ 已计算屏幕空间MV: 形状={screen_space_mv.shape}")
            print(f"      像素运动统计: 平均={mv_magnitude.mean():.2f}px, 最大={mv_magnitude.max():.2f}px, 中位数={np.median(mv_magnitude):.2f}px")
            print(f"      非零运动像素比例: {np.mean(mv_magnitude > 0.1):.3f}")

        return screen_space_mv.astype(np.float32)
    
    def _create_default_projection_matrix(self) -> np.ndarray:
        """创建默认透视投影矩阵"""
        # 创建一个基本的透视投影矩阵
        # FOV = 60度, aspect = 16/9, near = 0.1, far = 100
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
    
    def generate_training_data_with_extrapolation(self, 
                                                 curr_frame: Dict, 
                                                 prev_frame: Dict,
                                                 next_frame_gt: Dict,  
                                                 extrapolation_factor: float = 1.0) -> Dict:
        """
        生成完整的训练数据 - 【外插修复版】
        
        Args:
            curr_frame: 当前帧(t)数据
            prev_frame: 前一帧(t-1)数据
            next_frame_gt: 真实的下一帧(t+1)数据，作为Ground Truth
            extrapolation_factor: 外推系数(x)。默认为1.0，表示外推到 t+1。
            
        Returns:
            training_data: 包含所有训练数据的字典
        """
        print(f"   🚀 生成外插训练数据，外推系数: {extrapolation_factor}")
        
        # 1. 计算 t-1 -> t 的基础屏幕空间运动矢量
        base_mv = self.compute_screen_space_mv(curr_frame, prev_frame)

        # 2. 根据外推系数，计算用于 t -> t+x 的运动矢量
        # 核心假设：恒定速度模型
        extrapolated_mv = base_mv * extrapolation_factor
        
        if self.debug:
            base_magnitude = np.sqrt(base_mv[0]**2 + base_mv[1]**2)
            extrap_magnitude = np.sqrt(extrapolated_mv[0]**2 + extrapolated_mv[1]**2)
            print(f"   基础MV幅值: 平均={base_magnitude.mean():.2f}px, 最大={base_magnitude.max():.2f}px")
            print(f"   外推MV幅值: 平均={extrap_magnitude.mean():.2f}px, 最大={extrap_magnitude.max():.2f}px")

        # 3. 前向投影当前帧(t)来生成外推帧(t+x)
        # 这里需要外部提供forward_warp_frame函数
        # warped_image, coverage_mask = self.forward_warp_frame(
        #     curr_frame,       # 注意！源图像是当前帧(t)
        #     extrapolated_mv   # 使用外推后的运动矢量
        # )
        
        # 暂时返回关键数据，实际集成时需要连接到完整的warp和检测流程
        return {
            'base_mv': base_mv,
            'extrapolated_mv': extrapolated_mv, 
            'extrapolation_factor': extrapolation_factor,
            'target_frame_gt': next_frame_gt['reference'] if 'reference' in next_frame_gt else None
        }
    
    def create_integration_patches(self) -> Dict[str, str]:
        """创建集成补丁代码"""
        
        # 补丁1: 添加到UnifiedNoiseBasePreprocessor类中的screen_space_projection方法
        screen_projection_patch = '''
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
        w = np.where(w == 0, 1e-8, w)
        ndc_pos = projected_pos[:2, :, :] / w

        # NDC到屏幕坐标转换
        screen_pos_x = (ndc_pos[0] * 0.5 + 0.5) * width
        screen_pos_y = (-ndc_pos[1] * 0.5 + 0.5) * height

        return np.stack([screen_pos_x, screen_pos_y], axis=0)
'''

        # 补丁2: 替换_process_motion_data方法
        process_motion_patch = '''
    def _process_motion_data(self, motion_data) -> np.ndarray:
        """处理世界空间运动矢量数据（仅加载和聚合）"""
        motion = np.array(motion_data)
        
        # 多采样聚合
        if motion.ndim == 4:
            motion = motion.mean(axis=-1)
        
        # 确保格式为CHW
        if motion.shape[-1] == 3:
            motion = motion.transpose(2, 0, 1)

        print(f"   已加载世界空间motion数据: {motion.shape}")
        return motion.astype(np.float32)
'''

        # 补丁3: 新增_compute_screen_space_mv方法
        compute_mv_patch = '''
    def _compute_screen_space_mv(self, curr_frame: Dict, prev_frame: Dict) -> np.ndarray:
        """
        【核心修复】计算屏幕空间运动矢量
        """
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
        print(f"   ✅ 屏幕空间MV: 平均={mv_magnitude.mean():.2f}px, 最大={mv_magnitude.max():.2f}px")
        
        return screen_space_mv.astype(np.float32)
'''

        # 补丁4: 修改generate_training_data方法
        generate_training_patch = '''
    def generate_training_data(self, 
                             curr_frame: Dict, 
                             prev_frame: Dict,
                             next_frame_gt: Optional[Dict] = None,
                             extrapolation_factor: float = 1.0) -> Dict:
        """
        生成完整的训练数据 - 【外插修复版】
        """
        # 1. 计算正确的屏幕空间运动矢量
        base_mv = self._compute_screen_space_mv(curr_frame, prev_frame)
        
        # 2. 应用外推系数
        extrapolated_mv = base_mv * extrapolation_factor
        
        # 3. 前向投影（使用正确的MV）
        if next_frame_gt is not None:
            # 外插模式：从当前帧外推到下一帧
            warped_image, coverage_mask = self.forward_warp_frame(
                curr_frame, extrapolated_mv
            )
            target_rgb = next_frame_gt['reference']
        else:
            # 重投影模式：从前一帧投影到当前帧
            warped_image, coverage_mask = self.forward_warp_frame(
                prev_frame, base_mv
            )
            target_rgb = curr_frame['reference']
        
        # 4. 空洞和遮挡检测
        masks = self.detect_holes_and_occlusion(
            warped_image, target_rgb, coverage_mask, curr_frame, prev_frame
        )
        
        # 5. 残差运动矢量计算
        if next_frame_gt is not None:
            # 计算真实的t->t+1运动矢量并计算残差
            gt_mv = self._compute_screen_space_mv(next_frame_gt, curr_frame)
            residual_mv = gt_mv - extrapolated_mv
        else:
            residual_mv = self._compute_residual_motion_vectors(
                warped_image, target_rgb, base_mv, masks['holes']
            )
        
        # 6. 组装训练数据
        training_sample = np.concatenate([
            target_rgb,                         # 目标RGB (3)
            masks['holes'][np.newaxis],         # 几何空洞掩码 (1)
            masks['occlusion'][np.newaxis],     # 语义遮挡掩码 (1)
            residual_mv                         # 残差运动矢量 (2)
        ], axis=0)
        
        return {
            'target_rgb': target_rgb,
            'warped_image': warped_image,
            'coverage_mask': coverage_mask,
            'hole_mask': masks['holes'],
            'occlusion_mask': masks['occlusion'],
            'residual_mv': residual_mv,
            'training_sample': training_sample,
            'base_mv': base_mv,
            'extrapolated_mv': extrapolated_mv
        }
'''
        
        return {
            'screen_projection_patch': screen_projection_patch,
            'process_motion_patch': process_motion_patch,
            'compute_mv_patch': compute_mv_patch,
            'generate_training_patch': generate_training_patch
        }

def create_default_projection_matrix() -> np.ndarray:
    """创建默认透视投影矩阵的独立函数"""
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

# 测试函数
def test_motion_vector_calculation():
    """测试运动矢量计算"""
    print("🧪 测试运动矢量修复")
    
    # 创建测试数据
    H, W = 1080, 1920
    
    # 模拟当前帧数据
    curr_frame = {
        'position': np.random.randn(3, H, W).astype(np.float32) * 10,
        'motion': np.random.randn(3, H, W).astype(np.float32) * 0.1,
        'camera_params': {
            'view_proj_mat': create_default_projection_matrix()
        }
    }
    
    # 模拟前一帧数据
    prev_frame = {
        'position': curr_frame['position'] - curr_frame['motion'],
        'motion': curr_frame['motion'],
        'camera_params': {
            'view_proj_mat': create_default_projection_matrix()
        }
    }
    
    # 测试运动矢量计算
    fixer = MotionVectorFixer(debug=True)
    screen_mv = fixer.compute_screen_space_mv(curr_frame, prev_frame)
    
    print(f"✅ 测试完成")
    print(f"   屏幕空间MV形状: {screen_mv.shape}")
    print(f"   MV数值范围: [{screen_mv.min():.3f}, {screen_mv.max():.3f}]")
    
    return screen_mv

if __name__ == "__main__":
    # 运行测试
    test_mv = test_motion_vector_calculation()
    
    # 创建集成补丁
    fixer = MotionVectorFixer()
    patches = fixer.create_integration_patches()
    
    print("\n📋 已生成集成补丁代码")
    print("   可以使用这些补丁来修改unified_noisebase_preprocessor.py")