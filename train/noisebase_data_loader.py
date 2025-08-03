#!/usr/bin/env python3
"""
NoiseBase数据加载器 - 基于学长detach.py脚本的正确实现

根据学长的detach.py脚本分析，NoiseBase数据格式为：
- 存储格式: ZIP文件包含zarr数据
- 数据结构: 多采样数据需要聚合
- 颜色格式: RGBE压缩格式需要解压缩
- 几何数据: 包含位置、运动、法线等信息
"""

import numpy as np
import zarr
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings


class NoiseBaseDataLoader:
    """
    NoiseBase数据集加载器
    
    基于学长detach.py脚本的分析，正确处理NoiseBase数据格式：
    - ZIP + zarr存储格式
    - RGBE颜色解压缩
    - 多采样数据聚合
    - 几何信息计算
    """
    
    def __init__(self, data_root: str):
        """
        初始化数据加载器
        
        Args:
            data_root: NoiseBase数据根目录
        """
        self.data_root = Path(data_root)
        print(f"NoiseBase数据根目录: {self.data_root}")
        
        # 验证数据目录存在
        if not self.data_root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_root}")
    
    def load_frame_data(self, scene: str, frame_idx: int) -> Optional[Dict]:
        """
        加载指定帧的数据
        
        Args:
            scene: 场景名称 (如 'bistro1', 'kitchen')
            frame_idx: 帧索引
            
        Returns:
            frame_data: 帧数据字典，包含所有通道信息
        """
        try:
            # 构建zip文件路径
            zip_path = self.data_root / scene / f"frame{frame_idx:04d}.zip"
            
            if not zip_path.exists():
                print(f"⚠️ 帧数据文件不存在: {zip_path}")
                return None
            
            print(f"📂 加载帧数据: {zip_path}")
            
            # 加载zarr数据
            ds = zarr.group(store=zarr.ZipStore(str(zip_path), mode='r'))
            
            # 打印可用的数据通道
            print(f"📊 可用数据通道: {list(ds.keys())}")
            
            # 提取基础数据
            frame_data = {}
            
            # 1. 参考图像 (通常已经是最终格式)
            if 'reference' in ds:
                reference = np.array(ds.reference)
                frame_data['reference'] = reference
                print(f"   reference: {reference.shape}")
            
            # 2. 颜色数据 (RGBE格式，需要解压缩)
            if 'color' in ds and 'exposure' in ds:
                color_rgbe = np.array(ds.color)
                exposure = np.array(ds.exposure)
                color = self.decompress_RGBE(color_rgbe, exposure)
                
                # 多采样聚合
                if color.ndim == 4:  # CHWS格式
                    color = color.mean(axis=3)
                
                frame_data['color'] = color
                print(f"   color: {color.shape} (解压缩后)")
            
            # 3. 世界空间位置
            if 'position' in ds:
                position = np.array(ds.position)
                
                # 多采样聚合
                if position.ndim == 4:  # CHWS格式
                    position = position.mean(axis=3)
                
                frame_data['position'] = position
                print(f"   position: {position.shape}")
            
            # 4. 世界空间运动
            if 'motion' in ds:
                motion = np.array(ds.motion)
                
                # 多采样聚合
                if motion.ndim == 4:  # CHWS格式
                    motion = motion.mean(axis=3)
                
                frame_data['motion'] = motion
                print(f"   motion: {motion.shape}")
            
            # 5. 表面法线
            if 'normal' in ds:
                normal = np.array(ds.normal)
                
                # 多采样聚合
                if normal.ndim == 4:  # CHWS格式
                    normal = normal.mean(axis=3)
                
                frame_data['normal'] = normal
                print(f"   normal: {normal.shape}")
            
            # 6. 漫反射率 (albedo)
            if 'diffuse' in ds:
                albedo = np.array(ds.diffuse)
                
                # 多采样聚合
                if albedo.ndim == 4:  # CHWS格式
                    albedo = albedo.mean(axis=3)
                
                frame_data['albedo'] = albedo
                print(f"   albedo: {albedo.shape}")
            
            # 7. 相机参数
            if 'camera_position' in ds:
                camera_pos = np.array(ds.camera_position)
                frame_data['camera_pos'] = camera_pos
                print(f"   camera_pos: {camera_pos.shape}")
            
            if 'view_proj_mat' in ds:
                view_proj_mat = np.array(ds.view_proj_mat)
                frame_data['view_proj_mat'] = view_proj_mat
                print(f"   view_proj_mat: {view_proj_mat.shape}")
            
            # 8. 曝光参数
            if 'exposure' in ds:
                exposure = np.array(ds.exposure)
                frame_data['exposure'] = exposure
                print(f"   exposure: {exposure}")
            
            # 计算屏幕空间运动矢量 (如果有必要的数据)
            if all(key in frame_data for key in ['position', 'motion', 'view_proj_mat']):
                screen_motion = self.compute_screen_motion_vectors(
                    frame_data['position'], 
                    frame_data['motion'],
                    frame_data['view_proj_mat']
                )
                frame_data['screen_motion'] = screen_motion
                print(f"   screen_motion: {screen_motion.shape} (计算得出)")
            
            print(f"✅ 成功加载帧 {frame_idx} 数据")
            return frame_data
            
        except Exception as e:
            print(f"❌ 加载帧 {frame_idx} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def decompress_RGBE(self, color: np.ndarray, exposures: np.ndarray) -> np.ndarray:
        """
        解压缩RGBE格式的颜色数据
        
        基于学长detach.py中的实现
        
        Args:
            color: RGBE格式颜色数据 [4, H, W, S]
            exposures: 曝光范围 [min_exposure, max_exposure]
            
        Returns:
            color: 解压缩后的RGB颜色 [3, H, W, S]
        """
        # 计算指数
        exponents = (color.astype(np.float32)[3] + 1) / 256
        exponents = np.exp(exponents * (exposures[1] - exposures[0]) + exposures[0])
        
        # 解压缩RGB通道
        color_rgb = color.astype(np.float32)[:3] / 255 * exponents[np.newaxis]
        
        return color_rgb
    
    def compute_screen_motion_vectors(self, 
                                    world_position: np.ndarray,
                                    world_motion: np.ndarray, 
                                    view_proj_mat: np.ndarray,
                                    prev_view_proj_mat: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算屏幕空间运动矢量
        
        基于学长detach.py中的motion_vectors函数
        
        Args:
            world_position: 世界空间位置 [3, H, W]
            world_motion: 世界空间运动 [3, H, W]
            view_proj_mat: 当前帧视图投影矩阵 [4, 4]
            prev_view_proj_mat: 前一帧视图投影矩阵 [4, 4] (可选)
            
        Returns:
            screen_motion: 屏幕空间运动矢量 [2, H, W]
        """
        if prev_view_proj_mat is None:
            prev_view_proj_mat = view_proj_mat
        
        H, W = world_position.shape[1:3]
        
        # 计算当前位置的屏幕坐标
        current_screen = self.world_to_screen(world_position, view_proj_mat, H, W)
        
        # 计算前一帧位置的屏幕坐标
        prev_world_pos = world_position + world_motion
        prev_screen = self.world_to_screen(prev_world_pos, prev_view_proj_mat, H, W)
        
        # 计算运动矢量
        screen_motion = prev_screen - current_screen
        
        return screen_motion
    
    def world_to_screen(self, 
                       world_position: np.ndarray, 
                       view_proj_mat: np.ndarray,
                       height: int, 
                       width: int) -> np.ndarray:
        """
        世界空间位置投影到屏幕空间
        
        基于学长detach.py中的screen_space_position函数
        
        Args:
            world_position: 世界空间位置 [3, H, W]
            view_proj_mat: 视图投影矩阵 [4, 4]
            height: 图像高度
            width: 图像宽度
            
        Returns:
            screen_position: 屏幕空间位置 [2, H, W]
        """
        # 转换为齐次坐标
        homogeneous = np.concatenate([
            world_position,
            np.ones_like(world_position[0:1])
        ], axis=0)  # [4, H, W]
        
        # 投影变换 (注意：DirectX使用行向量)
        projected = np.einsum('ij, ihw -> jhw', view_proj_mat, homogeneous)
        
        # 透视除法
        projected = np.divide(
            projected[0:2], projected[3],
            out=np.zeros_like(projected[0:2]),
            where=projected[3] != 0
        )
        
        # DirectX像素坐标转换
        projected = projected * np.reshape([0.5 * width, -0.5 * height], (2, 1, 1)).astype(np.float32) \
                   + np.reshape([width / 2, height / 2], (2, 1, 1)).astype(np.float32)
        
        # 翻转为IJ索引 (height, width)
        projected = np.flip(projected, 0)
        
        return projected
    
    def list_available_scenes(self) -> list:
        """
        列出可用的场景
        
        Returns:
            scenes: 场景名称列表
        """
        scenes = []
        for scene_dir in self.data_root.iterdir():
            if scene_dir.is_dir():
                # 检查是否包含frame文件
                frame_files = list(scene_dir.glob("frame*.zip"))
                if frame_files:
                    scenes.append(scene_dir.name)
        
        return sorted(scenes)
    
    def count_frames(self, scene: str) -> int:
        """
        统计指定场景的帧数
        
        Args:
            scene: 场景名称
            
        Returns:
            frame_count: 帧数
        """
        scene_dir = self.data_root / scene
        if not scene_dir.exists():
            return 0
        
        frame_count = 0
        while (scene_dir / f"frame{frame_count:04d}.zip").exists():
            frame_count += 1
        
        return frame_count
    
    def validate_data_integrity(self, scene: str, max_frames: int = 10) -> Dict:
        """
        验证数据完整性
        
        Args:
            scene: 场景名称
            max_frames: 最大检查帧数
            
        Returns:
            validation_result: 验证结果
        """
        result = {
            'scene': scene,
            'total_frames': self.count_frames(scene),
            'valid_frames': 0,
            'invalid_frames': [],
            'common_channels': set(),
            'sample_shapes': {}
        }
        
        print(f"🔍 验证场景 '{scene}' 数据完整性...")
        
        check_frames = min(max_frames, result['total_frames'])
        
        for i in range(check_frames):
            frame_data = self.load_frame_data(scene, i)
            
            if frame_data is not None:
                result['valid_frames'] += 1
                
                # 记录通道信息
                if not result['common_channels']:
                    result['common_channels'] = set(frame_data.keys())
                else:
                    result['common_channels'] &= set(frame_data.keys())
                
                # 记录样本形状
                if not result['sample_shapes']:
                    result['sample_shapes'] = {k: v.shape for k, v in frame_data.items() if hasattr(v, 'shape')}
            else:
                result['invalid_frames'].append(i)
        
        print(f"✅ 验证完成:")
        print(f"   总帧数: {result['total_frames']}")
        print(f"   有效帧数: {result['valid_frames']}/{check_frames}")
        print(f"   公共通道: {sorted(result['common_channels'])}")
        print(f"   样本形状: {result['sample_shapes']}")
        
        return result


def main():
    """测试数据加载器"""
    print("🚀 测试NoiseBase数据加载器...")
    
    # 创建数据加载器 (需要指定实际的数据路径)
    data_root = input("请输入NoiseBase数据根目录路径: ").strip()
    
    if not data_root:
        print("❌ 未指定数据路径，使用默认测试路径")
        data_root = "./data"
    
    try:
        loader = NoiseBaseDataLoader(data_root)
        
        # 列出可用场景
        scenes = loader.list_available_scenes()
        print(f"📋 可用场景: {scenes}")
        
        if not scenes:
            print("❌ 未找到任何场景数据")
            return
        
        # 选择第一个场景进行测试
        test_scene = scenes[0]
        print(f"🎯 测试场景: {test_scene}")
        
        # 统计帧数
        frame_count = loader.count_frames(test_scene)
        print(f"📊 场景 '{test_scene}' 包含 {frame_count} 帧")
        
        if frame_count > 0:
            # 加载第一帧数据
            print(f"\n📂 加载第一帧数据...")
            frame_data = loader.load_frame_data(test_scene, 0)
            
            if frame_data:
                print(f"✅ 成功加载帧数据!")
                print(f"   数据通道: {list(frame_data.keys())}")
                
                # 显示数据统计
                for key, value in frame_data.items():
                    if hasattr(value, 'shape'):
                        print(f"   {key}: {value.shape}, dtype={value.dtype}")
                        if hasattr(value, 'min'):
                            print(f"      范围: [{value.min():.3f}, {value.max():.3f}]")
            
            # 验证数据完整性
            print(f"\n🔍 验证数据完整性...")
            validation = loader.validate_data_integrity(test_scene, max_frames=3)
        
        print(f"\n✅ 数据加载器测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()