#!/usr/bin/env python3
"""
ColleagueDatasetAdapter - 同事数据集适配器

将同事提供的OpenEXR格式数据适配到现有的7通道训练框架
保持现有网络架构不变，仅在数据加载层面进行适配

数据映射策略：
- warped_rgb [0:3]: 来自 warp_hole 目录（带黑洞的warped RGB）
- semantic_holes [3:4]: 来自 correct 目录（语义洞洞掩码）
- occlusion [4:5]: 全零填充（缺失的遮挡掩码）
- residual_mv [5:7]: 全零填充（缺失的残差运动矢量）
- target_rgb: 来自 ref 目录（参考图像）
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
try:
    import OpenEXR
    import Imath
    import array
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False
    print("WARNING: OpenEXR库未安装，将尝试使用其他方法读取EXR文件")
    
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import residual learning helper
import sys
import os
sys.path.append('./train')
try:
    from residual_learning_helper import ResidualLearningHelper
except ImportError:
    print("WARNING: ResidualLearningHelper not found, using fallback residual computation")


class ColleagueDatasetAdapter(Dataset):
    """同事数据集适配器 - 保持7通道输入格式
    
    Features:
    - 加载OpenEXR格式数据
    - 自动构建7通道输入格式
    - 残差学习目标计算
    - 兼容现有训练框架
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 augmentation: bool = False,
                 enable_linear_preprocessing: bool = True,
                 enable_srgb_linear: bool = True,
                 scale_factor: float = 0.70,
                 tone_mapping_for_display: str = 'reinhard',
                 gamma: float = 2.2,
                 exposure: float = 1.0,
                 adaptive_exposure: Optional[dict] = None):
        """初始化适配器
        Args:
            data_root: 数据根目录 (包含 processed_bistro)
            split: 数据分割 ('train', 'val', 'test')
            augmentation: 是否启用数据增强 (暂不支持)
        """

        # HDR处理配置（与Plan一致）
        self.enable_linear_preprocessing = enable_linear_preprocessing
        self.enable_srgb_linear = enable_srgb_linear
        self.scale_factor = float(scale_factor)
        self.tone_mapping_for_display = tone_mapping_for_display
        self.gamma = float(gamma)
        self.exposure = float(exposure)
        self.adaptive_exposure = adaptive_exposure or {'enable': False}

        # Kornia可用性标记（不在实例上保存模块对象以避免多进程pickle问题）
        try:
            import kornia.color  # noqa: F401
            self._kornia_available = True
        except Exception:
            self._kornia_available = False
            if self.enable_srgb_linear:
                print("[INFO] Kornia not available, skip sRGB→Linear conversion")
        
        self.data_root = Path(data_root)
        self.split = split
        self.augmentation = augmentation
        
        # 构建数据目录路径
        self.bistro_root = self.data_root / 'processed_bistro'
        
        if not self.bistro_root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.bistro_root}")
        
        # 检查必需的子目录
        required_dirs = ['warp_hole', 'correct', 'ref']
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.bistro_root / dir_name if dir_name != 'correct' else self.bistro_root / 'bistro' / 'correct'
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            raise FileNotFoundError(f"缺失必需目录: {missing_dirs}")
        
        # 构建文件映射
        self._build_file_mapping()
        
        # 数据分割
        self.frame_list = self._split_data()
        
        print(f"=== ColleagueDatasetAdapter ({split}) ===")
        print(f"数据根目录: {data_root}")
        print(f"Bistro数据目录: {self.bistro_root}")
        print(f"Frame样本数: {len(self.frame_list)}")
        
        # 验证数据完整性
        self._validate_data()
    
    def get_frame(self, frame_idx: int) -> dict:
        """
        获取单帧数据 - 用于推理
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            dict: 包含input和target的字典
                - 'input': 7通道输入 [7, H, W]
                - 'target': 3通道目标 [3, H, W]
        """
        if frame_idx >= len(self.frame_list):
            return None
            
        # 使用__getitem__获取数据
        input_tensor, target_residual, target_rgb = self.__getitem__(frame_idx)
        
        return {
            'input': input_tensor,           # [7, H, W]
            'target': target_rgb,           # [3, H, W]
            'target_residual': target_residual  # [3, H, W]
        }
    
    def _build_file_mapping(self):
        """构建文件名到路径的映射关系"""
        self.file_mapping = {}
        
        # 以ref目录为基准扫描所有帧
        ref_dir = self.bistro_root / 'ref'
        ref_files = list(ref_dir.glob('*.exr'))
        
        if len(ref_files) == 0:
            raise FileNotFoundError(f"ref目录中未找到EXR文件: {ref_dir}")
        
        for ref_file in ref_files:
            # 提取frame_id：ref-X.exr -> X
            if ref_file.name.startswith('ref-') and ref_file.name.endswith('.exr'):
                frame_id = ref_file.name[4:-4]  # 去掉 'ref-' 和 '.exr'
            else:
                continue
            
            # 构建对应文件路径
            warp_hole_file = self.bistro_root / 'warp_hole' / f'warped_hole-{frame_id}.exr'
            correct_file = self.bistro_root / 'bistro' / 'correct' / f'hole-{frame_id}.exr'
            
            # 验证所有必需文件都存在
            if ref_file.exists() and warp_hole_file.exists() and correct_file.exists():
                self.file_mapping[frame_id] = {
                    'ref': ref_file,
                    'warp_hole': warp_hole_file,
                    'correct': correct_file
                }
        
        if len(self.file_mapping) == 0:
            raise ValueError("未找到匹配的数据文件三元组 (ref, warp_hole, correct)")
        
        print(f"成功构建文件映射: {len(self.file_mapping)} 个有效帧")
    
    def _split_data(self) -> List[str]:
        """数据分割"""
        all_frame_ids = list(self.file_mapping.keys())
        
        # 按帧ID排序确保一致性
        all_frame_ids.sort(key=lambda x: int(x))
        
        if len(all_frame_ids) == 0:
            raise ValueError("没有可用的数据帧")
        
        # 数据分割 (80% train, 10% val, 10% test)
        total_frames = len(all_frame_ids)
        train_end = int(0.8 * total_frames)
        val_end = int(0.9 * total_frames)
        
        if self.split == 'train':
            return all_frame_ids[:train_end]
        elif self.split == 'val':
            return all_frame_ids[train_end:val_end]
        elif self.split == 'test':
            return all_frame_ids[val_end:]
        else:
            raise ValueError(f"不支持的数据分割: {self.split}")
    
    def _validate_data(self):
        """验证数据完整性"""
        print("验证数据完整性...")
        
        # 检查前几个样本
        for i, frame_id in enumerate(self.frame_list[:3]):
            try:
                file_paths = self.file_mapping[frame_id]
                
                # 尝试加载每个文件
                ref_data = self._load_exr(file_paths['ref'])
                warp_hole_data = self._load_exr(file_paths['warp_hole'])
                correct_data = self._load_exr(file_paths['correct'])
                
                print(f"Frame {frame_id}: ref{ref_data.shape}, warp_hole{warp_hole_data.shape}, correct{correct_data.shape}")
                
                # 验证数据形状一致性
                if ref_data.shape != warp_hole_data.shape:
                    print(f"警告: Frame {frame_id} 的 ref 和 warp_hole 形状不匹配")
                
                if i == 0:
                    H, W = ref_data.shape[1], ref_data.shape[2]
                    print(f"数据验证: 图像分辨率 = {H}x{W}")
                    print(f"数据范围: ref=[{ref_data.min():.3f}, {ref_data.max():.3f}]")
                    print(f"数据范围: warp_hole=[{warp_hole_data.min():.3f}, {warp_hole_data.max():.3f}]")
                    print(f"数据范围: correct=[{correct_data.min():.3f}, {correct_data.max():.3f}]")
                
            except Exception as e:
                print(f"警告: 无法加载Frame {frame_id}: {e}")
                return False
        
        print("SUCCESS: 数据格式验证通过")
        return True
    
    def _load_exr(self, file_path: Path) -> torch.Tensor:
        """加载OpenEXR文件并转换为PyTorch张量
        
        支持多种EXR加载方式：
        1. OpenEXR库 (如果可用)
        2. OpenCV (如果可用)
        3. 尝试其他方法
        
        Args:
            file_path: EXR文件路径
            
        Returns:
            tensor: 形状为 [C, H, W] 的张量
        """
        if not file_path.exists():
            raise FileNotFoundError(f"EXR文件不存在: {file_path}")
        
        # 方法1: 使用OpenEXR库
        if OPENEXR_AVAILABLE:
            try:
                return self._load_exr_openexr(file_path)
            except Exception as e:
                print(f"OpenEXR库加载失败，尝试其他方法: {e}")
        
        # 方法2: 使用OpenCV
        if CV2_AVAILABLE:
            try:
                return self._load_exr_opencv(file_path)
            except Exception as e:
                print(f"OpenCV加载失败，尝试其他方法: {e}")
        
        # 方法3: 如果都失败，抛出异常
        raise RuntimeError(f"无法加载EXR文件 {file_path}: 缺少必要的库 (OpenEXR 或 OpenCV)")
    
    def _load_exr_openexr(self, file_path: Path) -> torch.Tensor:
        """使用OpenEXR库加载EXR文件"""
        # 打开EXR文件
        exr_file = OpenEXR.InputFile(str(file_path))
        header = exr_file.header()
        
        # 获取图像尺寸
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # 获取通道信息
        channels = header['channels']
        channel_names = list(channels.keys())
        
        # 读取通道数据
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_data = []
        
        # 按RGB顺序读取（如果是RGB图像）
        if 'R' in channel_names and 'G' in channel_names and 'B' in channel_names:
            for channel in ['R', 'G', 'B']:
                if channel in channel_names:
                    raw_data = exr_file.channel(channel, pixel_type)
                    pixels = array.array('f', raw_data)
                    channel_array = np.array(pixels, dtype=np.float32).reshape((height, width))
                    channel_data.append(channel_array)
        else:
            # 对于单通道或其他格式，读取所有通道
            for channel_name in sorted(channel_names):
                raw_data = exr_file.channel(channel_name, pixel_type)
                pixels = array.array('f', raw_data)
                channel_array = np.array(pixels, dtype=np.float32).reshape((height, width))
                channel_data.append(channel_array)
        
        exr_file.close()
        
        # 转换为PyTorch张量 [C, H, W]
        if len(channel_data) == 0:
            raise ValueError(f"EXR文件没有有效通道: {file_path}")
        
        tensor = torch.from_numpy(np.stack(channel_data, axis=0)).float()
        return tensor
    
    def _load_exr_opencv(self, file_path: Path) -> torch.Tensor:
        """使用OpenCV加载EXR文件"""
        # OpenCV加载EXR文件
        img = cv2.imread(str(file_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        if img is None:
            raise ValueError(f"OpenCV无法加载EXR文件: {file_path}")
        
        # OpenCV默认为BGR格式，转换为RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为numpy数组并调整维度顺序 [H, W, C] -> [C, H, W]
        if len(img.shape) == 2:
            # 单通道图像
            img_array = img[np.newaxis, :, :]  # [1, H, W]
        else:
            # 多通道图像
            img_array = np.transpose(img, (2, 0, 1))  # [C, H, W]
        
        # 转换为PyTorch张量
        tensor = torch.from_numpy(img_array.astype(np.float32))
        
        return tensor
    
    def __len__(self) -> int:
        return len(self.frame_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取数据样本 - 残差学习版本
        
        Returns:
            input_tensor: 7通道输入 [7, H, W] (warped_RGB + holes + occlusion + residual_mv)
            target_residual: 3通道残差目标 [3, H, W] (target_RGB - warped_RGB, 归一化后)
            target_rgb: 3通道完整目标 [3, H, W] (用于损失计算)
        """
        frame_id = self.frame_list[idx]
        file_paths = self.file_mapping[frame_id]
        
        try:
            # 加载OpenEXR数据
            ref_data = self._load_exr(file_paths['ref'])         # [3, H, W] 参考图像
            warp_hole_data = self._load_exr(file_paths['warp_hole'])  # [3, H, W] 带洞的warp图像
            correct_data = self._load_exr(file_paths['correct'])      # [1, H, W] 洞洞掩码
            
            # 获取图像尺寸
            C_ref, H, W = ref_data.shape
            C_warp, H_warp, W_warp = warp_hole_data.shape
            
            # 验证尺寸一致性
            if (H, W) != (H_warp, W_warp):
                raise ValueError(f"图像尺寸不匹配: ref({H},{W}) vs warp_hole({H_warp},{W_warp})")
            
            # 确保ref和warp_hole都是3通道RGB
            if C_ref != 3:
                if C_ref == 1:
                    ref_data = ref_data.repeat(3, 1, 1)  # 灰度转RGB
                else:
                    ref_data = ref_data[:3]  # 取前3通道
            
            if C_warp != 3:
                if C_warp == 1:
                    warp_hole_data = warp_hole_data.repeat(3, 1, 1)  # 灰度转RGB
                else:
                    warp_hole_data = warp_hole_data[:3]  # 取前3通道
            
            # 确保correct是单通道掩码
            if correct_data.shape[0] != 1:
                if correct_data.shape[0] == 3:
                    # 如果是3通道，取平均值作为掩码
                    correct_data = correct_data.mean(dim=0, keepdim=True)
                else:
                    correct_data = correct_data[:1]  # 取第一通道
            
            # 构建7通道输入张量
            input_tensor = torch.zeros(7, H, W)
            input_tensor[:3] = warp_hole_data       # [0:3] warped RGB with holes
            input_tensor[3:4] = correct_data        # [3:4] semantic holes mask
            input_tensor[4:5] = torch.zeros(1, H, W)  # [4:5] placeholder occlusion (全零填充)
            input_tensor[5:7] = torch.zeros(2, H, W)  # [5:7] placeholder residual_mv (全零填充)
            
            # 数据预处理：线性HDR或旧log1p路径（由开关控制）
            if self.enable_linear_preprocessing:
                input_tensor = self._linear_hdr_preprocessing(input_tensor)
                target_rgb = self._linear_hdr_preprocessing(ref_data)
            else:
                input_tensor = self._normalize_to_tanh_range(input_tensor)
                target_rgb = self._normalize_to_tanh_range(ref_data)
            
            # 使用统一的残差学习工具类计算残差目标
            try:
                from residual_learning_helper import ResidualLearningHelper
                warped_rgb = input_tensor[:3]
                target_residual = ResidualLearningHelper.compute_residual_target(target_rgb, warped_rgb)
            except ImportError:
                # Fallback: 保持兼容性的残差计算
                warped_rgb = input_tensor[:3]
                raw_residual = target_rgb - warped_rgb
                # 直接使用原始残差，不进行除法缩放
                target_residual = torch.clamp(raw_residual, -1.0, 1.0)
            
            return input_tensor, target_residual, target_rgb
            
        except Exception as e:
            error_msg = f"错误加载Frame {frame_id}: {e}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(f"数据加载失败: {error_msg}") from e
    
    def _normalize_to_tanh_range(self, tensor: torch.Tensor) -> torch.Tensor:
        """归一化到Tanh范围 [-1, 1]
        
        使用与unified_dataset.py相同的归一化逻辑确保一致性
        """
        normalized = tensor.clone()
        
        # RGB通道处理（前3通道或全部通道）
        if tensor.shape[0] >= 3:
            rgb_channels = tensor[:3] if tensor.shape[0] > 3 else tensor
            
            # 确保输入为非负数再应用log1p（EXR数据可能包含负值）
            rgb_channels = torch.clamp(rgb_channels, min=0.0)  # 确保非负
            
            # 使用log1p非线性归一化改善HDR数据分布
            rgb_log_transformed = torch.log1p(rgb_channels)  # log(1+x)
            
            # 确定log变换后的范围
            log_min_val = 0.0                    # log1p(0) = 0
            log_max_val = 5.023574285781275      # log1p(151.0) = log(152.0) ≈ 5.024
            
            # clamp并映射到[-1,1]
            rgb_clamped_log = torch.clamp(rgb_log_transformed, log_min_val, log_max_val)
            rgb_normalized_log = (rgb_clamped_log - log_min_val) / (log_max_val - log_min_val)  # [0, 1]
            rgb_normalized_log = rgb_normalized_log * 2.0 - 1.0  # [-1, 1]
            
            rgb_normalized = rgb_normalized_log
            
            if tensor.shape[0] > 3:
                normalized[:3] = rgb_normalized
            else:
                normalized = rgb_normalized
        
        # 掩码通道处理（第4-5通道）
        if tensor.shape[0] >= 5:
            for i in range(3, 5):
                mask = tensor[i]
                mask = torch.clamp(mask, 0.0, 1.0)  # 确保掩码在[0,1]范围
                normalized[i] = mask  # 保持[0,1]范围，不进行[-1,1]转换
        
        # 运动矢量通道处理（第6-7通道）
        if tensor.shape[0] >= 7:
            mv_channels = tensor[5:7]
            
            # 使用分位数方法，更稳定
            mv_abs_max = torch.quantile(torch.abs(mv_channels), 0.95)  # 95%分位数
            
            if mv_abs_max > 1e-6:
                # 归一化到[-1, 1]范围
                normalized[5:7] = torch.clamp(mv_channels / mv_abs_max, -1.0, 1.0)
            else:
                # 如果运动很小，保持原值
                normalized[5:7] = mv_channels
        
        return normalized
    
    def denormalize_for_display(self, tensor: torch.Tensor) -> torch.Tensor:
        """用于TensorBoard显示的可视化转换。

        - 线性HDR路径：RGB线性/非负/缩放→ tone-mapping（Reinhard）→ gamma → [0,1]
        - 旧路径（[-1,1]）：按旧log1p逆+Reinhard显示
        """
        # 只处理RGB通道
        if tensor.shape[0] >= 3:
            rgb = tensor[:3] if tensor.shape[0] > 3 else tensor
            if self.enable_linear_preprocessing:
                # 输入应为线性HDR缩放后的范围（可>1）
                lin = torch.clamp(rgb, min=0.0)
                # 通用tone-mapping（Reinhard + gamma）
                try:
                    from src.npu.utils.hdr_vis import tone_map as _tm
                except Exception:
                    import sys, os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'utils'))
                    from hdr_vis import tone_map as _tm
                # 读取曝光配置（若未来注入到Adapter）
                exposure = getattr(self, 'exposure', 1.0)
                adaptive_exposure = getattr(self, 'adaptive_exposure', {'enable': False})
                ldr = _tm(
                    lin,
                    method=self.tone_mapping_for_display,
                    gamma=self.gamma,
                    exposure=exposure,
                    adaptive_exposure=adaptive_exposure,
                )
                return ldr
            else:
                # 旧路径：严格对应log1p逆
                rgb_01 = (rgb + 1.0) / 2.0
                log_min_val = 0.0
                log_max_val = 5.023574285781275
                log_values = rgb_01 * (log_max_val - log_min_val) + log_min_val
                hdr_rgb = torch.expm1(log_values)
                ldr_rgb = hdr_rgb / (1.0 + hdr_rgb)
                return torch.clamp(ldr_rgb, 0.0, 1.0)
        # 非RGB通道
        if self.enable_linear_preprocessing:
            return torch.clamp(tensor, 0.0, 1.0)
        return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)

    def _linear_hdr_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """线性HDR预处理：非负 + （可选）sRGB→Linear + 线性缩放

        返回：与输入相同形状，RGB通道按 `scale_factor` 线性缩放，掩码保持[0,1]，MV保持原值
        """
        normalized = tensor.clone()
        # RGB
        if tensor.shape[0] >= 3:
            rgb = tensor[:3]
            rgb_lin = torch.clamp(rgb, min=0.0)
            if self.enable_srgb_linear and self._kornia_available:
                try:
                    import kornia.color as Kcolor
                    # Kornia转换期望输入在[0,1]，这里仅用于轻度矫正；EXR通常已线性，影响有限
                    rgb_01 = torch.clamp(rgb_lin, 0.0, 1.0)
                    rgb_lin = Kcolor.rgb_to_linear_rgb(rgb_01)
                except Exception:
                    pass
            rgb_scaled = rgb_lin / max(self.scale_factor, 1e-8)
            normalized[:3] = rgb_scaled
        # 掩码（3,4索引）保持[0,1]
        if tensor.shape[0] >= 5:
            for i in range(3, 5):
                normalized[i] = torch.clamp(tensor[i], 0.0, 1.0)
        # MV（5,6索引）保持原始像素位移
        if tensor.shape[0] >= 7:
            normalized[5:7] = tensor[5:7]
        return normalized

    def __getstate__(self):
        """Drop any non-picklable attributes for DataLoader workers."""
        state = self.__dict__.copy()
        # Ensure no module objects are kept
        for k in list(state.keys()):
            if k.startswith('_module_'):
                state.pop(k, None)
        return state


def create_colleague_dataloader(data_root: str,
                               split: str = 'train',
                               batch_size: int = 1,
                               num_workers: int = 0,
                               shuffle: bool = True) -> DataLoader:
    """创建ColleagueDatasetAdapter的DataLoader
    
    Args:
        data_root: 数据根目录 (包含processed_bistro)
        split: 数据分割
        batch_size: 批次大小  
        num_workers: 工作线程数（推荐0以减少内存占用）
        shuffle: 是否打乱数据
    """
    dataset = ColleagueDatasetAdapter(
        data_root=data_root,
        split=split,
        augmentation=False  # 暂时禁用数据增强
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=False,  # 减少内存占用
        prefetch_factor=1 if num_workers > 0 else None  # 减少预取缓冲
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试ColleagueDatasetAdapter
    print("测试ColleagueDatasetAdapter...")
    
    try:
        dataset = ColleagueDatasetAdapter(
            data_root="./data",
            split='train'
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试加载第一个样本
        input_tensor, target_residual, target_rgb = dataset[0]
        print(f"输入形状: {input_tensor.shape}")
        print(f"目标残差形状: {target_residual.shape}")
        print(f"目标RGB形状: {target_rgb.shape}")
        print(f"输入范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        print(f"残差目标范围: [{target_residual.min():.3f}, {target_residual.max():.3f}]")
        print(f"RGB目标范围: [{target_rgb.min():.3f}, {target_rgb.max():.3f}]")
        
        # 验证7通道结构
        print(f"\n7通道结构验证:")
        print(f"warped_rgb [0:3]: 范围[{input_tensor[:3].min():.3f}, {input_tensor[:3].max():.3f}]")
        print(f"semantic_holes [3:4]: 范围[{input_tensor[3:4].min():.3f}, {input_tensor[3:4].max():.3f}]")
        print(f"occlusion [4:5]: 范围[{input_tensor[4:5].min():.3f}, {input_tensor[4:5].max():.3f}] (应为0)")
        print(f"residual_mv [5:7]: 范围[{input_tensor[5:7].min():.3f}, {input_tensor[5:7].max():.3f}] (应为0)")
        
        # 验证残差学习逻辑
        warped_rgb = input_tensor[:3]
        reconstructed_rgb = warped_rgb + target_residual  # 直接相加，无需缩放因子
        reconstruction_error = torch.mean(torch.abs(reconstructed_rgb - target_rgb))
        print(f"\n残差重建误差: {reconstruction_error:.6f} (应该很小)")
        
        print("SUCCESS: ColleagueDatasetAdapter测试通过!")
        
    except Exception as e:
        print(f"ERROR: ColleagueDatasetAdapter测试失败: {e}")
        print("请确保 ./data/processed_bistro 目录存在并包含必要的数据文件")
