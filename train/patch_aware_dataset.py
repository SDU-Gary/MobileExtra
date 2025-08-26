#!/usr/bin/env python3
"""
Patch-Aware Dataset - Intelligent dataset supporting patch-based training

Key features:
- Built on UnifiedNoiseBaseDataset with full compatibility
- Dynamic patch generation based on hole masks
- Efficient patch caching to avoid recomputation  
- Patch-level augmentations and boundary handling
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any
import random
import cv2
from dataclasses import dataclass
import time

# Import base dataset
try:
    from .unified_dataset import UnifiedNoiseBaseDataset
except ImportError:
    from unified_dataset import UnifiedNoiseBaseDataset

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
        from hole_detector import HoleDetector, HoleDetectorConfig, PatchInfo
        from patch_extractor import PatchExtractor, PatchExtractorConfig, PatchPosition, PaddingMode


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
    - Built on UnifiedNoiseBaseDataset with full compatibility
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
        
        # Initialize base dataset
        self.base_dataset = UnifiedNoiseBaseDataset(
            data_root=data_root,
            split=split,
            augmentation=base_augmentation
        )
        
        # Initialize patch components
        if self.config.enable_patch_mode:
            self.hole_detector = HoleDetector()
            self.patch_extractor = PatchExtractor()
            
            # Initialize patch cache
            if self.config.enable_patch_cache:
                self.patch_cache = PatchCache(self.config)
        
        # Statistics
        self.stats = {
            'patch_mode_count': 0,
            'total_patches_generated': 0,
            'cache_hits': 0,
            'avg_patches_per_image': 0.0
        }
        
        print(f"[INFO] PatchAwareDataset ({split}) initialized")
        print(f"Base dataset size: {len(self.base_dataset)}")
        print(f"Patch mode enabled: {self.config.enable_patch_mode}")
        print(f"Patch mode probability: {self.config.patch_mode_probability}")
        if self.config.enable_patch_cache:
            print(f"Patch cache enabled: {self.config.cache_size} slots")
    
    def __len__(self) -> int:
        """Dataset length"""
        return len(self.base_dataset)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get training sample in patch mode
        
        Returns:
            Dict containing:
            - 'input': patch input data [N, 7, 128, 128]  
            - 'target': patch target data [N, 3, 128, 128]
            - 'mode': 'patch'
            - 'metadata': additional metadata
        """
        # Get base data
        base_sample = self.base_dataset[index]
        
        # UnifiedNoiseBaseDataset返回的是元组 (input_tensor, target_tensor)
        if isinstance(base_sample, tuple) and len(base_sample) == 2:
            input_data, target_data = base_sample  # [7, H, W], [3, H, W]
        else:
            # Fallback: assume dict format
            data = base_sample['data'] if isinstance(base_sample, dict) else base_sample
            # Separate input and target
            input_data = torch.cat([
                data[:3],    # warped_rgb
                data[3:4],   # semantic_holes  
                data[4:5],   # occlusion
                data[5:7]    # residual_mv
            ], dim=0)  # [7, H, W]
            target_data = data[7:10]  # target_rgb [3, H, W]
        
        # Use patch mode directly
        return self._get_patch_sample(index, input_data, target_data)
    
    
    def _get_patch_sample(self, index: int, input_data: torch.Tensor, target_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get patch mode sample"""
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
            from src.npu.networks.patch.hole_detector import PatchInfo
            patch_infos = [PatchInfo(center_x=center_x, center_y=center_y, hole_area=1000, patch_id=0)]
        
        # Limit patch count
        max_patches = min(len(patch_infos), self.config.max_patches_per_image)
        selected_patches = patch_infos[:max_patches]
        
        # Extract patches
        patches_input = []
        patches_target = []
        patches_metadata = []
        
        for patch_info in selected_patches:
            # Try cache first
            cache_key = self._generate_cache_key(index, patch_info) if self.config.enable_patch_cache else None
            cached_patch = self.patch_cache.get(cache_key) if cache_key else None
            
            if cached_patch is not None:
                # Use cached patch
                patches_input.append(cached_patch.patch_input)
                patches_target.append(cached_patch.patch_target)
                patches_metadata.append({
                    'patch_info': cached_patch.patch_info,
                    'position': cached_patch.position,
                    'hole_area': cached_patch.hole_area,
                    'from_cache': True
                })
                self.stats['cache_hits'] += 1
            else:
                # Extract new patch
                input_patches, input_positions = self.patch_extractor.extract_patches(
                    input_data.numpy(), [patch_info]
                )
                target_patches, target_positions = self.patch_extractor.extract_patches(
                    target_data.numpy(), [patch_info]
                )
                
                if len(input_patches) > 0:
                    patch_input_tensor = torch.from_numpy(input_patches[0])  # [7, 128, 128]
                    patch_target_tensor = torch.from_numpy(target_patches[0])  # [3, 128, 128]
                    
                    # Apply augmentation if enabled
                    # Note: augmentation disabled to avoid cache mapping issues
                    if self.config.patch_augmentation and self.config.augmentation_probability > random.random():
                        patch_input_tensor, patch_target_tensor = self._apply_patch_augmentation(
                            patch_input_tensor, patch_target_tensor
                        )
                    
                    patches_input.append(patch_input_tensor)
                    patches_target.append(patch_target_tensor)
                    patches_metadata.append({
                        'patch_info': patch_info,
                        'position': input_positions[0],
                        'hole_area': patch_info.hole_area,
                        'from_cache': False
                    })
                    
                    # Cache patch
                    if cache_key:
                        patch_sample = PatchSample(
                            patch_input=patch_input_tensor,
                            patch_target=patch_target_tensor,
                            patch_info=patch_info,
                            position=input_positions[0],
                            source_index=index,
                            hole_area=patch_info.hole_area
                        )
                        self.patch_cache.put(cache_key, patch_sample)
        
        if len(patches_input) == 0:
            # Fallback to center patch if extraction failed
            H, W = input_data.shape[1:]
            center_patch_input = input_data[:, H//2-64:H//2+64, W//2-64:W//2+64]  # [7, 128, 128]
            center_patch_target = target_data[:, H//2-64:H//2+64, W//2-64:W//2+64]  # [3, 128, 128]
            
            # Ensure correct patch size
            if center_patch_input.shape[1] != 128 or center_patch_input.shape[2] != 128:
                center_patch_input = torch.nn.functional.interpolate(
                    center_patch_input.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
                ).squeeze(0)
                center_patch_target = torch.nn.functional.interpolate(
                    center_patch_target.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
                ).squeeze(0)
            
            patches_input = [center_patch_input]
            patches_target = [center_patch_target]
            patches_metadata = [{
                'patch_info': None,
                'position': None,
                'hole_area': 0,
                'from_cache': False,
                'is_default_patch': True
            }]
        
        # Convert to tensors
        batch_input = torch.stack(patches_input)    # [N, 7, 128, 128]
        batch_target = torch.stack(patches_target)  # [N, 3, 128, 128]
        
        # Update statistics
        self.stats['patch_mode_count'] += 1
        self.stats['total_patches_generated'] += len(patches_input)
        self.stats['avg_patches_per_image'] = (
            self.stats['total_patches_generated'] / 
            max(self.stats['patch_mode_count'], 1)
        )
        
        return {
            'input': batch_input,
            'target': batch_target,
            'mode': 'patch',
            'metadata': {
                'patches_count': len(patches_input),
                'patches_info': patches_metadata,
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
    """Custom collate function for patch batch data
    
    Note: Module-level function required for multiprocessing serialization
    """
    # All samples are patch mode - merge all patches
    all_patch_inputs = []
    all_patch_targets = []
    patch_metadata = []
    
    for sample in batch:
        all_patch_inputs.append(sample['input'])    # [N, 7, 128, 128]
        all_patch_targets.append(sample['target'])  # [N, 3, 128, 128]
        patch_metadata.extend(sample['metadata']['patches_info'])
    
    result = {
        'patch_input': torch.cat(all_patch_inputs, dim=0),
        'patch_target': torch.cat(all_patch_targets, dim=0),
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