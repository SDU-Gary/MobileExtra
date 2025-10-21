#!/usr/bin/env python3
"""
Patch-Aware Dataset - Intelligent dataset supporting patch-based training

Key features:
- Built on ColleagueDatasetAdapter for OpenEXR data
- Dynamic patch generation based on hole masks
- Efficient patch caching to avoid recomputation  
- Patch-level augmentations and boundary handling
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import random
from dataclasses import dataclass

#  FIX: 移除unified_dataset依赖，只支持colleague数据集
# Import base dataset - colleague数据集处理
try:
    from .colleague_dataset_adapter import ColleagueDatasetAdapter
except ImportError:
    from colleague_dataset_adapter import ColleagueDatasetAdapter

# Import patch modules
import sys
import os

# Add paths to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'npu', 'networks'))

try:
    from src.npu.networks.patch import HoleDetector, PatchExtractor, PatchInfo, PatchPosition
except ImportError:
    try:
        from patch import HoleDetector, PatchExtractor, PatchInfo, PatchPosition
    except ImportError:
        # Fallback: direct module import
        patch_dir = os.path.join(project_root, 'src', 'npu', 'networks', 'patch')
        sys.path.insert(0, patch_dir)
        from hole_detector import HoleDetector, PatchInfo
        from patch_extractor import PatchExtractor, PatchPosition

#  Import optimized patch system
try:
    from optimized_patch_strategy import OptimizedPatchDetector, AdaptivePatchInfo, create_optimized_config
    from adaptive_patch_extractor import AdaptivePatchExtractor
    OPTIMIZED_PATCH_AVAILABLE = True
except ImportError:
    OPTIMIZED_PATCH_AVAILABLE = False

#  Import simple grid patch system
try:
    from simple_patch_extractor import SimplePatchExtractor, SimpleGridConfig
    SIMPLE_GRID_AVAILABLE = True
except ImportError:
    SIMPLE_GRID_AVAILABLE = False


@dataclass
class PatchTrainingConfig:
    """Patch training configuration"""
    enable_patch_mode: bool = True
    patch_size: int = 128
    patch_mode_probability: float = 0.7
    
    min_patches_per_image: int = 1
    max_patches_per_image: int = 8
    patch_overlap_threshold: float = 0.3
    
    enable_patch_cache: bool = True
    cache_size: int = 1000
    cache_hit_threshold: float = 0.8
    
    patch_augmentation: bool = True
    augmentation_probability: float = 0.5
    
    min_hole_area: int = 100
    max_hole_area: int = 10000
    boundary_margin: int = 16
    
    #  Patch extraction strategy configuration
    use_optimized_patches: bool = True      # Enable optimized patch system
    use_simple_grid_patches: bool = False   #  NEW: Use simple 4x4 grid strategy (most stable)
    
    # Optimized patch configuration (used when use_simple_grid_patches=False)
    optimized_tight_fitting: bool = True    # Enable tight fitting (5% oversizing)
    optimized_shape_analysis: bool = True   # Enable shape-aware patches
    optimized_waste_limit: float = 0.35     # Max 35% background waste
    optimized_coverage_target: float = 0.92 # Target 92% hole coverage
    
    #  Simple grid patch configuration (used when use_simple_grid_patches=True)
    simple_grid_rows: int = 4               # Grid rows (short side)
    simple_grid_cols: int = 4               # Grid columns (long side)
    simple_expected_height: int = 1080      # Expected input image height
    simple_expected_width: int = 1920       # Expected input image width


@dataclass
class PatchSample:
    """Patch sample data structure"""
    patch_input: torch.Tensor        # [7, 128, 128]
    patch_target: torch.Tensor       # [3, 128, 128]
    
    patch_info: PatchInfo
    position: PatchPosition
    source_index: int
    
    hole_area: int
    boundary_mask: Optional[torch.Tensor] = None
    quality_score: float = 1.0


class PatchCache:
    """Patch cache system with LRU eviction and hit rate tracking"""
    
    def __init__(self, config: PatchTrainingConfig):
        self.config = config
        self.cache = {}  # {cache_key: PatchSample}
        self.access_order = []  # LRU访问顺序
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_patches': 0
        }
    
    def _generate_cache_key(self, index: int, patch_info: PatchInfo) -> str:
        """Generate cache key"""
        return f"{index}_{patch_info.center_x}_{patch_info.center_y}_{patch_info.hole_area}"
    
    def get(self, cache_key: str) -> Optional[PatchSample]:
        """Get cached patch"""
        if cache_key in self.cache:
            # Update access order
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            self.stats['hits'] += 1
            return self.cache[cache_key]
        else:
            self.stats['misses'] += 1
            return None
    
    def put(self, cache_key: str, patch_sample: PatchSample):
        """Store patch in cache"""
        # Check cache capacity
        if len(self.cache) >= self.config.cache_size:
            # LRU eviction
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            self.stats['evictions'] += 1
        
        # Store new patch
        self.cache[cache_key] = patch_sample
        self.access_order.append(cache_key)
        self.stats['total_patches'] += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(total_requests, 1)
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'cache_size': len(self.cache),
            'evictions': self.stats['evictions']
        }


class PatchAwareDataset(Dataset):
    """Patch-aware dataset for efficient patch-based training
    
    Features:
    - Built on ColleagueDatasetAdapter for OpenEXR data
    - Intelligent patch generation based on hole distribution  
    - Efficient caching to avoid repeated patch extraction
    - Patch-level data augmentations
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 config: Optional[PatchTrainingConfig] = None,
                 base_augmentation: bool = False):
        """Initialize PatchAwareDataset
        
        Args:
            data_root: Data root directory
            split: Data split ('train', 'val', 'test')
            config: Patch training configuration
            base_augmentation: Enable base augmentations
        """
        self.data_root = data_root
        self.split = split
        self.config = config or PatchTrainingConfig()
        self.base_augmentation = base_augmentation
        
        #  FIX: 初始化colleague数据集适配器
        self.base_dataset = ColleagueDatasetAdapter(
            data_root=data_root,
            split=split
        )
        
        # Initialize patch components
        if self.config.enable_patch_mode:
            #  Strategy 1: Simple Grid Patches (Most Stable for Training)
            if self.config.use_simple_grid_patches and SIMPLE_GRID_AVAILABLE:
                print(" Using SimplePatchExtractor for maximum training stability")
                simple_config = SimpleGridConfig(
                    grid_rows=self.config.simple_grid_rows,
                    grid_cols=self.config.simple_grid_cols,
                    expected_height=self.config.simple_expected_height,
                    expected_width=self.config.simple_expected_width,
                    enable_debug_info=False  # Disable debug for training
                )
                self.hole_detector = None  # No hole detection needed
                self.patch_extractor = SimplePatchExtractor(simple_config)
                self._using_simple_grid = True
                self._using_optimized_patches = False
                print(f"   Grid: {self.config.simple_grid_rows}x{self.config.simple_grid_cols} = {self.config.simple_grid_rows*self.config.simple_grid_cols} patches per image")
                
            #  Strategy 2: Optimized Patch Detection (Complex but Adaptive)
            elif OPTIMIZED_PATCH_AVAILABLE and self.config.use_optimized_patches:
                print(" Using OptimizedPatchDetector for better hole coverage and efficiency")
                optimized_config = create_optimized_config()
                
                # Override with training config parameters
                optimized_config.tight_fitting = self.config.optimized_tight_fitting
                optimized_config.enable_shape_analysis = self.config.optimized_shape_analysis
                optimized_config.background_waste_limit = self.config.optimized_waste_limit
                optimized_config.hole_coverage_target = self.config.optimized_coverage_target
                optimized_config.max_patches_per_image = self.config.max_patches_per_image
                optimized_config.min_hole_area = self.config.min_hole_area
                
                self.hole_detector = OptimizedPatchDetector(optimized_config)
                self.patch_extractor = AdaptivePatchExtractor()
                self._using_optimized_patches = True
                self._using_simple_grid = False
                
            #  Strategy 3: Original Patch Detection (Fallback)
            else:
                print("⚠️  Using original patch detector (fallback)")
                self.hole_detector = HoleDetector()
                self.patch_extractor = PatchExtractor()
                self._using_optimized_patches = False
                self._using_simple_grid = False
            
            # Initialize patch cache
            if self.config.enable_patch_cache:
                self.patch_cache = PatchCache(self.config)
        
        # Statistics
        self.stats = {
            'patch_mode_count': 0,
            'total_patches_generated': 0,
            'cache_hits': 0,
            'avg_patches_per_image': 0.0,
            'optimized_patch_efficiency': 0.0,     # Track optimized patch efficiency
            'optimized_waste_reduction': 0.0,      # Track waste reduction
            'adaptive_patch_sizes': []             # Track adaptive patch sizes
        }
        
        # Dataset initialized successfully
    
    def __len__(self) -> int:
        """Dataset length"""
        return len(self.base_dataset)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get training sample in patch mode
        
        Returns:
            Dict containing:
            - 'input': patch input data [N, 7, 128, 128]  
            - 'target_residual': patch target residual data [N, 3, 128, 128]
            - 'target_rgb': patch target RGB data [N, 3, 128, 128]
            - 'mode': 'patch'
            - 'metadata': additional metadata
        """
        #  FIX: 获取colleague数据 - ColleagueDatasetAdapter返回三元组
        base_sample = self.base_dataset[index]
        
        #  统一的残差学习三元组适配
        try:
            from residual_learning_helper import ResidualLearningHelper
        except ImportError:
            sys.path.append('./train')
            from residual_learning_helper import ResidualLearningHelper
        
        input_data, target_residual, target_rgb = self._get_unified_sample(base_sample)
        
        # Use patch mode with residual learning data
        return self._get_patch_sample(index, input_data, target_residual, target_rgb)
    
    def _get_unified_sample(self, base_sample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        统一处理不同格式的基础数据样本
        确保返回一致的三元组格式: (input_data, target_residual, target_rgb)
        """
        try:
            from residual_learning_helper import ResidualLearningHelper
        except ImportError:
            sys.path.append('./train')
            from residual_learning_helper import ResidualLearningHelper
            
        # 情况1: UnifiedDataset返回的三元组 (推荐路径)
        if isinstance(base_sample, tuple) and len(base_sample) == 3:
            input_data, target_residual, target_rgb = base_sample
            
            #  验证数据一致性
            if not ResidualLearningHelper.validate_residual_data(input_data, target_residual, target_rgb):
                # 重新计算以确保一致性
                warped_rgb = input_data[:3]
                target_residual = ResidualLearningHelper.compute_residual_target(target_rgb, warped_rgb)
            
            return input_data, target_residual, target_rgb
        
        # 情况2: 旧格式二元组兼容性处理
        elif isinstance(base_sample, tuple) and len(base_sample) == 2:
            input_data, target_rgb = base_sample
            warped_rgb = input_data[:3]
            #  使用统一的残差计算
            target_residual = ResidualLearningHelper.compute_residual_target(target_rgb, warped_rgb)
            return input_data, target_residual, target_rgb
        
        # 情况3: 字典格式或其他格式处理
        else:
            data = base_sample['data'] if isinstance(base_sample, dict) else base_sample
            
            # 构建input_data [7, H, W]
            input_data = torch.cat([
                data[:3],    # warped_rgb
                data[3:4],   # semantic_holes  
                data[4:5],   # occlusion
                data[5:7]    # residual_mv
            ], dim=0)
            
            target_rgb = data[7:10]  # target_rgb [3, H, W]
            warped_rgb = input_data[:3]
            
            #  使用统一的残差计算
            target_residual = ResidualLearningHelper.compute_residual_target(target_rgb, warped_rgb)
            return input_data, target_residual, target_rgb
    
    def _get_patch_sample(self, index: int, input_data: torch.Tensor, target_residual: torch.Tensor, target_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get patch mode sample"""
        
        #  Strategy routing: Simple Grid vs Complex Detection
        if self._using_simple_grid:
            return self._get_simple_grid_patches(index, input_data, target_residual, target_rgb)
        else:
            return self._get_complex_detection_patches(index, input_data, target_residual, target_rgb)
    
    def _get_simple_grid_patches(self, index: int, input_data: torch.Tensor, target_residual: torch.Tensor, target_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get patches using simple 4x4 grid strategy"""
        # Convert tensors to numpy for patch extraction
        input_numpy = input_data.numpy()          # [7, H, W]
        target_residual_numpy = target_residual.numpy()  # [3, H, W]
        target_rgb_numpy = target_rgb.numpy()     # [3, H, W]
        
        # Extract 4x4 grid patches - 16 patches total
        try:
            input_patches, input_positions = self.patch_extractor.extract_patches(input_numpy)
            target_residual_patches, _ = self.patch_extractor.extract_patches(target_residual_numpy)
            target_rgb_patches, _ = self.patch_extractor.extract_patches(target_rgb_numpy)
        except Exception as e:
            print(f"⚠️  Simple grid extraction failed: {e}")
            # Fallback to center patch
            return self._get_fallback_center_patch(input_data, target_residual, target_rgb, index)
        
        # Convert patches to tensors
        patches_input = []
        patches_target_residual = []
        patches_target_rgb = []
        patches_metadata = []
        
        for i, (input_patch, residual_patch, rgb_patch, position) in enumerate(zip(
            input_patches, target_residual_patches, target_rgb_patches, input_positions
        )):
            # Convert to tensors and ensure correct shape [C, H, W]
            patch_input_tensor = torch.from_numpy(input_patch).float()                    # [7, patch_h, patch_w]
            patch_target_residual_tensor = torch.from_numpy(residual_patch).float()      # [3, patch_h, patch_w]  
            patch_target_rgb_tensor = torch.from_numpy(rgb_patch).float()                # [3, patch_h, patch_w]
            
            # Resize to standard patch size (128x128) if needed
            if patch_input_tensor.shape[1] != 128 or patch_input_tensor.shape[2] != 128:
                patch_input_tensor = torch.nn.functional.interpolate(
                    patch_input_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
                ).squeeze(0)
                patch_target_residual_tensor = torch.nn.functional.interpolate(
                    patch_target_residual_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
                ).squeeze(0)
                patch_target_rgb_tensor = torch.nn.functional.interpolate(
                    patch_target_rgb_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
                ).squeeze(0)
            
            patches_input.append(patch_input_tensor)
            patches_target_residual.append(patch_target_residual_tensor)
            patches_target_rgb.append(patch_target_rgb_tensor)
            
            # Simple grid metadata
            patches_metadata.append({
                'patch_info': None,  # No complex patch info needed
                'position': position,
                'hole_area': 0,      # Not applicable for grid patches
                'from_cache': False,
                'grid_patch_id': i,
                'grid_position': (position.x, position.y),
                'original_patch_size': (position.height, position.width)
            })
        
        # Convert to tensors
        batch_input = torch.stack(patches_input)                    # [N, 7, 128, 128]
        batch_target_residual = torch.stack(patches_target_residual)  # [N, 3, 128, 128]
        batch_target_rgb = torch.stack(patches_target_rgb)          # [N, 3, 128, 128]
        
        # Update statistics
        self.stats['patch_mode_count'] += 1
        self.stats['total_patches_generated'] += len(patches_input)
        self.stats['avg_patches_per_image'] = (
            self.stats['total_patches_generated'] / 
            max(self.stats['patch_mode_count'], 1)
        )
        
        return {
            'input': batch_input,
            'target_residual': batch_target_residual,  #  残差目标
            'target_rgb': batch_target_rgb,            #  RGB目标  
            'mode': 'patch',
            'extraction_strategy': 'simple_grid',
            'metadata': {
                'patches_count': len(patches_input),
                'patches_info': patches_metadata,
                'original_shape': input_data.shape[1:],
                'source_index': index,
                'grid_info': {
                    'rows': self.config.simple_grid_rows,
                    'cols': self.config.simple_grid_cols,
                    'total_patches': len(patches_input)
                }
            }
        }
    
    def _get_complex_detection_patches(self, index: int, input_data: torch.Tensor, target_residual: torch.Tensor, target_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get patches using complex hole detection strategy"""
        # Extract hole mask for patch detection
        holes_mask = input_data[3].numpy()  # [H, W]
        
        # Detect patch centers
        patch_infos = self.hole_detector.detect_patch_centers(holes_mask)
        
        if len(patch_infos) == 0:
            # Generate fallback patch if no valid patches detected
            H, W = holes_mask.shape
            # Generate center patch
            center_x = W // 2
            center_y = H // 2
            
            if self._using_optimized_patches:
                # Create AdaptivePatchInfo for fallback
                patch_infos = [AdaptivePatchInfo(
                    center_x=center_x, center_y=center_y, hole_area=1000, 
                    patch_id=0, patch_size=128, boundary_valid=True
                )]
            else:
                from src.npu.networks.patch.hole_detector import PatchInfo
                patch_infos = [PatchInfo(center_x=center_x, center_y=center_y, hole_area=1000, patch_id=0)]
        
        # Limit patch count
        max_patches = min(len(patch_infos), self.config.max_patches_per_image)
        selected_patches = patch_infos[:max_patches]
        
        # Extract patches
        patches_input = []
        patches_target_residual = []
        patches_target_rgb = []
        patches_metadata = []
        
        for patch_info in selected_patches:
            # Try cache first - 注意：缓存暂时禁用，避免残差学习数据不匹配
            # cache_key = self._generate_cache_key(index, patch_info) if self.config.enable_patch_cache else None
            # cached_patch = self.patch_cache.get(cache_key) if cache_key else None
            cached_patch = None  # 暂时禁用缓存直到适配残差学习
            
            if cached_patch is not None:
                # Use cached patch (需要适配残差学习数据)
                patches_input.append(cached_patch.patch_input)
                patches_target_residual.append(cached_patch.patch_target)  # 假设缓存的是残差
                patches_target_rgb.append(cached_patch.patch_target)       # 需要修正
                patches_metadata.append({
                    'patch_info': cached_patch.patch_info,
                    'position': cached_patch.position,
                    'hole_area': cached_patch.hole_area,
                    'from_cache': True
                })
                self.stats['cache_hits'] += 1
            else:
                # Extract new patch -  残差学习: 同时提取残差和RGB patches
                input_patches, input_positions = self.patch_extractor.extract_patches(
                    input_data.numpy(), [patch_info]
                )
                target_residual_patches, _ = self.patch_extractor.extract_patches(
                    target_residual.numpy(), [patch_info]
                )
                target_rgb_patches, _ = self.patch_extractor.extract_patches(
                    target_rgb.numpy(), [patch_info]
                )
                
                if len(input_patches) > 0 and len(target_residual_patches) > 0 and len(target_rgb_patches) > 0:
                    patch_input_tensor = torch.from_numpy(input_patches[0])           # [7, 128, 128]
                    patch_target_residual_tensor = torch.from_numpy(target_residual_patches[0])  # [3, 128, 128]
                    patch_target_rgb_tensor = torch.from_numpy(target_rgb_patches[0])           # [3, 128, 128]
                    
                    # Apply augmentation if enabled
                    # Note: augmentation disabled to avoid cache mapping issues
                    if self.config.patch_augmentation and self.config.augmentation_probability > random.random():
                        patch_input_tensor, patch_target_rgb_tensor = self._apply_patch_augmentation(
                            patch_input_tensor, patch_target_rgb_tensor
                        )
                        # 重新计算残差（增强后）
                        warped_rgb = patch_input_tensor[:3]
                        raw_residual = patch_target_rgb_tensor - warped_rgb
                        patch_target_residual_tensor = torch.clamp(raw_residual / 2.0, -1.0, 1.0)
                    
                    patches_input.append(patch_input_tensor)
                    patches_target_residual.append(patch_target_residual_tensor)
                    patches_target_rgb.append(patch_target_rgb_tensor)
                    
                    #  Calculate efficiency for optimized patches
                    patch_metadata = {
                        'patch_info': patch_info,
                        'position': input_positions[0],
                        'hole_area': patch_info.hole_area,
                        'from_cache': False
                    }
                    
                    if self._using_optimized_patches and hasattr(patch_info, 'patch_size'):
                        # Track adaptive patch sizes
                        self.stats['adaptive_patch_sizes'].append(patch_info.patch_size)
                        
                        # Calculate patch efficiency (hole pixels / total pixels)
                        patch_holes = holes_mask[
                            max(0, patch_info.center_y - patch_info.patch_size//2):
                            min(holes_mask.shape[0], patch_info.center_y + patch_info.patch_size//2),
                            max(0, patch_info.center_x - patch_info.patch_size//2):
                            min(holes_mask.shape[1], patch_info.center_x + patch_info.patch_size//2)
                        ]
                        efficiency = np.sum(patch_holes) / max(patch_holes.size, 1)
                        patch_metadata['efficiency'] = efficiency
                        patch_metadata['patch_size'] = patch_info.patch_size
                        
                        # Update efficiency stats
                        if hasattr(self, '_efficiency_samples'):
                            self._efficiency_samples.append(efficiency)
                        else:
                            self._efficiency_samples = [efficiency]
                    
                    patches_metadata.append(patch_metadata)
                    
                    # Cache patch - 暂时禁用
                    # if cache_key:
                    #     patch_sample = PatchSample(
                    #         patch_input=patch_input_tensor,
                    #         patch_target=patch_target_residual_tensor,  # 缓存残差
                    #         patch_info=patch_info,
                    #         position=input_positions[0],
                    #         source_index=index,
                    #         hole_area=patch_info.hole_area
                    #     )
                    #     self.patch_cache.put(cache_key, patch_sample)
        
        if len(patches_input) == 0:
            # Fallback to center patch if extraction failed
            return self._get_fallback_center_patch(input_data, target_residual, target_rgb, index)
        
        # Convert to tensors -  残差学习: 返回残差和RGB
        batch_input = torch.stack(patches_input)                    # [N, 7, 128, 128]
        batch_target_residual = torch.stack(patches_target_residual)  # [N, 3, 128, 128]
        batch_target_rgb = torch.stack(patches_target_rgb)          # [N, 3, 128, 128]
        
        # Update statistics
        self.stats['patch_mode_count'] += 1
        self.stats['total_patches_generated'] += len(patches_input)
        self.stats['avg_patches_per_image'] = (
            self.stats['total_patches_generated'] / 
            max(self.stats['patch_mode_count'], 1)
        )
        
        #  Update optimized patch statistics
        if self._using_optimized_patches and hasattr(self, '_efficiency_samples'):
            if self._efficiency_samples:
                self.stats['optimized_patch_efficiency'] = np.mean(self._efficiency_samples)
                self.stats['optimized_waste_reduction'] = 1.0 - self.stats['optimized_patch_efficiency']
        
        return {
            'input': batch_input,
            'target_residual': batch_target_residual,  #  残差目标
            'target_rgb': batch_target_rgb,            #  RGB目标  
            'mode': 'patch',
            'extraction_strategy': 'complex_detection',
            'metadata': {
                'patches_count': len(patches_input),
                'patches_info': patches_metadata,
                'original_shape': input_data.shape[1:],
                'source_index': index
            }
        }
    
    def _get_fallback_center_patch(self, input_data: torch.Tensor, target_residual: torch.Tensor, target_rgb: torch.Tensor, index: int) -> Dict[str, torch.Tensor]:
        """Fallback to center patch if grid extraction fails"""
        H, W = input_data.shape[1:]
        center_patch_input = input_data[:, H//2-64:H//2+64, W//2-64:W//2+64]        # [7, 128, 128]
        center_patch_target_residual = target_residual[:, H//2-64:H//2+64, W//2-64:W//2+64]  # [3, 128, 128]
        center_patch_target_rgb = target_rgb[:, H//2-64:H//2+64, W//2-64:W//2+64]   # [3, 128, 128]
        
        # Ensure correct patch size
        if center_patch_input.shape[1] != 128 or center_patch_input.shape[2] != 128:
            center_patch_input = torch.nn.functional.interpolate(
                center_patch_input.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
            ).squeeze(0)
            center_patch_target_residual = torch.nn.functional.interpolate(
                center_patch_target_residual.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
            ).squeeze(0)
            center_patch_target_rgb = torch.nn.functional.interpolate(
                center_patch_target_rgb.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
            ).squeeze(0)
        
        batch_input = center_patch_input.unsqueeze(0)                    # [1, 7, 128, 128]
        batch_target_residual = center_patch_target_residual.unsqueeze(0)  # [1, 3, 128, 128]
        batch_target_rgb = center_patch_target_rgb.unsqueeze(0)          # [1, 3, 128, 128]
        
        return {
            'input': batch_input,
            'target_residual': batch_target_residual,
            'target_rgb': batch_target_rgb,
            'mode': 'patch',
            'extraction_strategy': 'fallback_center',
            'metadata': {
                'patches_count': 1,
                'patches_info': [{
                    'patch_info': None,
                    'position': None,
                    'hole_area': 0,
                    'from_cache': False,
                    'is_fallback_patch': True
                }],
                'original_shape': input_data.shape[1:],
                'source_index': index
            }
        }
    
    
    def _apply_patch_augmentation(self, patch_input: torch.Tensor, patch_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply patch-level data augmentation
        
        Includes random flips, rotations, and color transforms (RGB only)
        """
        # Random horizontal flip
        if random.random() < 0.5:
            patch_input = torch.flip(patch_input, [2])
            patch_target = torch.flip(patch_target, [2])
        
        # Random vertical flip
        if random.random() < 0.3:
            patch_input = torch.flip(patch_input, [1])
            patch_target = torch.flip(patch_target, [1])
        
        # Random 90-degree rotation
        if random.random() < 0.25:
            k = random.randint(1, 3)
            patch_input = torch.rot90(patch_input, k, [1, 2])
            patch_target = torch.rot90(patch_target, k, [1, 2])
        
        # RGB color enhancement
        if random.random() < 0.3:
            # Apply same color transform to input RGB and target RGB
            brightness_factor = 0.9 + random.random() * 0.2  # [0.9, 1.1]
            contrast_factor = 0.9 + random.random() * 0.2    # [0.9, 1.1]
            
            # Input RGB channels
            patch_input[:3] = torch.clamp(
                patch_input[:3] * contrast_factor + (brightness_factor - 1.0), 
                -1.0, 1.0
            )
            
            # Target RGB
            patch_target = torch.clamp(
                patch_target * contrast_factor + (brightness_factor - 1.0),
                -1.0, 1.0
            )
        
        return patch_input, patch_target
    
    
    def _generate_cache_key(self, index: int, patch_info: PatchInfo) -> str:
        """Generate cache key"""
        return f"{index}_{patch_info.center_x}_{patch_info.center_y}_{patch_info.hole_area}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        total_samples = self.stats['patch_mode_count']
        
        stats = {
            'total_samples': total_samples,
            'patch_mode_count': self.stats['patch_mode_count'],
            'total_patches_generated': self.stats['total_patches_generated'],
            'avg_patches_per_image': self.stats['avg_patches_per_image'],
        }
        
        # Add cache statistics
        if self.config.enable_patch_cache:
            cache_stats = self.patch_cache.get_stats()
            stats.update({f'cache_{k}': v for k, v in cache_stats.items()})
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'patch_mode_count': 0,
            'total_patches_generated': 0,
            'cache_hits': 0,
            'avg_patches_per_image': 0.0
        }


def patch_aware_collate_fn(batch):
    """Custom collate function for patch batch data -  残差学习版本
    
    Note: Module-level function required for multiprocessing serialization
    """
    # All samples are patch mode - merge all patches
    all_patch_inputs = []
    all_patch_target_residuals = []
    all_patch_target_rgbs = []
    patch_metadata = []
    
    for sample in batch:
        all_patch_inputs.append(sample['input'])             # [N, 7, 128, 128]
        all_patch_target_residuals.append(sample['target_residual'])  # [N, 3, 128, 128] 
        all_patch_target_rgbs.append(sample['target_rgb'])   # [N, 3, 128, 128]
        patch_metadata.extend(sample['metadata']['patches_info'])
    
    result = {
        'patch_input': torch.cat(all_patch_inputs, dim=0),
        'patch_target_residual': torch.cat(all_patch_target_residuals, dim=0),  #  残差目标
        'patch_target_rgb': torch.cat(all_patch_target_rgbs, dim=0),            #  RGB目标
        'patch_metadata': patch_metadata,
        'modes': [sample['mode'] for sample in batch],
        'batch_info': {
            'patch_count': len(batch),
            'total_patches': len(patch_metadata)
        }
    }
    
    return result


def create_patch_aware_dataloader(data_root: str,
                                split: str = 'train',
                                batch_size: int = 4,
                                config: Optional[PatchTrainingConfig] = None,
                                num_workers: int = 4,
                                **kwargs) -> DataLoader:
    """Create PatchAware DataLoader with custom collate function
    
    Note: Custom collate needed for dynamic batch sizes in patch mode
    """
    
    dataset = PatchAwareDataset(
        data_root=data_root,
        split=split,
        config=config,
        base_augmentation=False  # DISABLED: avoid color mapping issues
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=patch_aware_collate_fn,  # Module-level function
        pin_memory=True,
        persistent_workers=False,  # Windows compatibility
        **kwargs
    )


def test_patch_aware_dataset():
    """Test PatchAwareDataset"""
    # Configuration
    config = PatchTrainingConfig(
        enable_patch_mode=True,
        patch_mode_probability=0.8,
        enable_patch_cache=True,
        cache_size=100
    )
    
    # Create dataset
    try:
        dataset = PatchAwareDataset(
            data_root="./output_motion_fix",
            split='train',
            config=config
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"Mode: {sample['mode']}")
            print(f"Input shape: {sample['input'].shape}")
            print(f"Target shape: {sample['target'].shape}")
            print(f"Metadata: {sample['metadata']}")
        
        # Output statistics
        stats = dataset.get_stats()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test
    success = test_patch_aware_dataset()
    print(f"\n{'[PASS] Test passed!' if success else '[FAIL] Test failed!'}")