"""
Memory-safe dataset implementation with multi-resolution support and memory optimization.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any
import cv2
import gc


class MemorySafeDataset(Dataset):
    """Memory-safe dataset with dynamic resolution, memory monitoring, and cache management."""
    
    def __init__(self, 
                 data_root: str,
                 target_resolution: Tuple[int, int] = (270, 480),
                 normalize_method: str = "none",
                 max_cache_size: int = 10,
                 split: str = "train",
                 augmentation: bool = False):
        """Initialize dataset with memory-safe loading and caching."""
        self.data_root = data_root
        self.target_resolution = target_resolution
        self.normalize_method = normalize_method
        self.max_cache_size = max_cache_size
        self.split = split
        self.augmentation = augmentation
        
        # Setup data paths
        self.training_dir = os.path.join(data_root, "training_data")
        if not os.path.exists(self.training_dir):
            raise FileNotFoundError(f"Training data directory not found: {self.training_dir}")
        
        # Load data files
        self.data_files = [f for f in os.listdir(self.training_dir) if f.endswith('.npy')]
        self.data_files.sort()
        
        if len(self.data_files) == 0:
            raise ValueError(f"No data files found in: {self.training_dir}")
        
        # Dataset split
        total_files = len(self.data_files)
        split_idx = int(total_files * 0.8)
        
        if split == "train":
            self.data_files = self.data_files[:split_idx]
        else:  # val
            self.data_files = self.data_files[split_idx:]
        
        print(f"[LOADED] {split} dataset loaded: {len(self.data_files)} samples")
        print(f"[INFO] Target resolution: {target_resolution[0]}×{target_resolution[1]}")
        
        # Cache system
        self.cache = {}
        self.cache_order = []
        
        # Memory monitoring
        self.memory_warning_threshold = 0.8
        
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single sample: input[7,H,W], target[3,H,W]."""
        # Check cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Load data
        data_path = os.path.join(self.training_dir, self.data_files[idx])
        
        try:
            # Load 10-channel data
            full_data = np.load(data_path).astype(np.float32)
            
            # Validate format
            if full_data.shape[0] != 10:
                raise ValueError(f"Invalid data format: expected 10 channels, got {full_data.shape[0]}")
            
            # Split input/target
            input_data = full_data[:7]    # [7, H, W]
            target_data = full_data[7:10] # [3, H, W]
            
            # Resize if needed
            if self.target_resolution != input_data.shape[1:]:
                input_data = self._resize_data(input_data, self.target_resolution)
                target_data = self._resize_data(target_data, self.target_resolution)
            
            # Normalize data
            input_data = self._normalize_data(input_data)
            target_data = self._normalize_data(target_data)
            
            # Convert to tensors (float32)
            input_tensor = torch.from_numpy(input_data).float()
            target_tensor = torch.from_numpy(target_data).float()
            
            # Apply augmentation if enabled
            if self.augmentation and self.split == "train":
                input_tensor, target_tensor = self._apply_augmentation(input_tensor, target_tensor)
            
            # Update cache
            result = (input_tensor, target_tensor)
            self._update_cache(idx, result)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Failed to load data {data_path}: {e}")
            # Return fallback zero data
            return self._get_fallback_data()
    
    def _resize_data(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize data with motion vector scaling."""
        if len(data.shape) == 3:  # [C, H, W]
            # Convert to [H, W, C] for OpenCV
            data_hwc = np.transpose(data, (1, 2, 0))
            
            # Process channels separately to handle motion vectors correctly
            channels = []
            for c in range(data_hwc.shape[2]):
                if c >= 5:  # motion vector channels
                    # Special handling for motion vectors
                    resized = cv2.resize(data_hwc[:, :, c], (target_size[1], target_size[0]), 
                                       interpolation=cv2.INTER_LINEAR)
                    # Scale motion vectors for new resolution
                    if c == 5:  # x direction
                        resized *= target_size[1] / data_hwc.shape[1]
                    else:  # y direction
                        resized *= target_size[0] / data_hwc.shape[0]
                else:
                    # Other channels use bilinear interpolation
                    resized = cv2.resize(data_hwc[:, :, c], (target_size[1], target_size[0]), 
                                       interpolation=cv2.INTER_LINEAR)
                channels.append(resized)
            
            # Recombine and convert back to [C, H, W]
            resized_hwc = np.stack(channels, axis=2)
            return np.transpose(resized_hwc, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using specified method."""
        if self.normalize_method == "none" or self.normalize_method == "hdr_preserve":
            return data  # Preserve original HDR range
        if self.normalize_method == "robust_quantile":
            # Robust quantile normalization
            q1, q99 = np.percentile(data, [1, 99])
            data = np.clip(data, q1, q99)
            data = (data - q1) / (q99 - q1 + 1e-8)
        elif self.normalize_method == "minmax":
            # Simple min-max normalization
            data_min, data_max = data.min(), data.max()
            data = (data - data_min) / (data_max - data_min + 1e-8)
        else:
            # Default: divide by 255 (assuming 0-255 range)
            data = data / 255.0
        
        return data
    
    def _apply_augmentation(self, input_tensor: torch.Tensor, 
                          target_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation (simplified to avoid motion vector issues)."""
        # Only vertical flip supported (horizontal flip affects motion vector direction)
        if torch.rand(1) < 0.5:
            input_tensor = torch.flip(input_tensor, [1])  # Vertical flip
            target_tensor = torch.flip(target_tensor, [1])
            # Adjust y-direction motion vector
            if input_tensor.shape[0] >= 7:  # Ensure MV channels exist
                input_tensor[6] = -input_tensor[6]  # Invert y-direction MV
        
        return input_tensor, target_tensor
    
    def _update_cache(self, idx: int, data: Tuple[torch.Tensor, torch.Tensor]):
        """Update cache with LRU policy."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest cache entry
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]
        
        self.cache[idx] = data
        self.cache_order.append(idx)
    
    def _get_fallback_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get fallback zero data when loading fails."""
        h, w = self.target_resolution
        input_tensor = torch.zeros(7, h, w, dtype=torch.float32)
        target_tensor = torch.zeros(3, h, w, dtype=torch.float32)
        return input_tensor, target_tensor
    
    def clear_cache(self):
        """Clear cache and trigger garbage collection."""
        self.cache.clear()
        self.cache_order.clear()
        gc.collect()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        cache_size = len(self.cache)
        if cache_size > 0:
            # Estimate single sample size
            sample_input, sample_target = next(iter(self.cache.values()))
            sample_size = (sample_input.numel() + sample_target.numel()) * 4 / 1024 / 1024  # FP32
            total_cache_mb = cache_size * sample_size
        else:
            sample_size = 0
            total_cache_mb = 0
        
        return {
            'cached_samples': cache_size,
            'max_cache_size': self.max_cache_size,
            'sample_size_mb': sample_size,
            'total_cache_mb': total_cache_mb,
            'target_resolution': self.target_resolution
        }


def create_memory_safe_dataloader(dataset: MemorySafeDataset, 
                                config: Dict[str, Any]) -> DataLoader:
    """Create memory-safe DataLoader with optimized settings."""
    
    training_config = config.get('training', {})
    
    # Extract parameters
    num_workers = training_config.get('num_workers', 0)
    batch_size = training_config.get('batch_size', 1)
    pin_memory = training_config.get('dataloader_pin_memory', False)
    persistent_workers = training_config.get('persistent_workers', False)
    
    # Build DataLoader arguments
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': (dataset.split == 'train'),
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True,
    }
    
    # Set multiprocessing parameters only when num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = training_config.get('prefetch_factor', 2)
        dataloader_kwargs['persistent_workers'] = persistent_workers
    
    return DataLoader(dataset, **dataloader_kwargs)


def test_memory_safe_dataset():
    """Test memory-safe dataset functionality."""
    print("[TEST] Testing memory-safe dataset")
    
    try:
        # Create test dataset
        dataset = MemorySafeDataset(
            data_root="./output_motion_fix",
            target_resolution=(270, 480),  # 1/4 resolution
            normalize_method="robust_quantile",
            max_cache_size=5,
            split="train",
            augmentation=False
        )
        
        print(f"[SUCCESS] Dataset created successfully: {len(dataset)} samples")
        
        # Test data loading functionality
        input_data, target_data = dataset[0]
        print(f"[SUCCESS] Data loading successful:")
        print(f"   Input shape: {input_data.shape}")
        print(f"   Target shape: {target_data.shape}")
        print(f"   Input range: [{input_data.min():.3f}, {input_data.max():.3f}]")
        print(f"   Target range: [{target_data.min():.3f}, {target_data.max():.3f}]")
        
        # Memory info
        memory_info = dataset.get_memory_info()
        print(f"[INFO] Memory info:")
        for key, value in memory_info.items():
            print(f"   {key}: {value}")
        
        # Test DataLoader
        test_config = {
            'training': {
                'batch_size': 2,
                'num_workers': 0,
                'dataloader_pin_memory': False,
                'prefetch_factor': 1,
                'persistent_workers': False
            }
        }
        
        dataloader = create_memory_safe_dataloader(dataset, test_config)
        
        # Test one batch
        batch_input, batch_target = next(iter(dataloader))
        print(f"[SUCCESS] DataLoader test successful:")
        print(f"   Batch input shape: {batch_input.shape}")
        print(f"   Batch target shape: {batch_target.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Dataset test failed: {e}")
        return False


if __name__ == "__main__":
    test_memory_safe_dataset()