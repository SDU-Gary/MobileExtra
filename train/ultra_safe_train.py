#!/usr/bin/env python3
"""
Ultra Safe Training Script - Memory overflow prevention training
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import gc
import time
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "train"))

try:
    from memory_safe_dataset import MemorySafeDataset, create_memory_safe_dataloader
    from src.npu.networks.residual_mv_guided_network import create_residual_mv_guided_network
    from residual_inpainting_loss import create_residual_inpainting_loss
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    print("Please ensure all required files exist")
    sys.exit(1)


class MemoryMonitor:
    """Simplified memory monitor for GPU usage tracking"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.last_warning_time = 0
    
    def check_gpu_memory(self) -> Tuple[float, bool]:
        """Check GPU memory usage rate"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            usage_rate = allocated / total
            
            current_time = time.time()
            
            if usage_rate > self.critical_threshold:
                print(f"GPU memory critical: {usage_rate:.1%} ({allocated:.1f}GB/{total:.1f}GB)")
                return usage_rate, True
            elif usage_rate > self.warning_threshold and current_time - self.last_warning_time > 10:
                print(f"GPU memory warning: {usage_rate:.1%} ({allocated:.1f}GB/{total:.1f}GB)")
                self.last_warning_time = current_time
            
            return usage_rate, False
        
        return 0.0, False
    
    def force_cleanup(self):
        """Force memory cleanup"""
        print("Force cleaning GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class UltraSafeTrainer:
    """Ultra safe trainer with memory management"""
    
    def __init__(self, config_path: str):
        """Initialize ultra safe trainer"""
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.multi_loss = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        self.log_dir = Path(self.config['logging']['save_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print("Ultra safe trainer initialization complete")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Config file loaded successfully: {config_path}")
            return config
        except Exception as e:
            print(f"Config file loading failed: {e}")
            raise
    
    def setup_model(self):
        """Setup model"""
        try:
            model_config = self.config['model']
            architecture = model_config.get('architecture', 'simplified_unet')
            
            if architecture == 'residual_mv_guided':
                print("Creating residual MV-guided network (task-driven architecture)...")
                self.model = create_residual_mv_guided_network(self.config)
                
                # Print residual MV network config
                residual_config = model_config.get('residual_mv_config', {})
                print("Residual MV-guided network config:")
                print(f"  Encoder channels: {residual_config.get('encoder_channels', [32, 64, 128, 256, 512])}")
                print(f"  MV feature channels: {residual_config.get('mv_feature_channels', 32)}")
                print(f"  Gated conv: {residual_config.get('use_gated_conv', True)}")
                
                attention_config = residual_config.get('attention_config', {})
                print(f"  Gated attention: {attention_config.get('use_gated_attention', True)}")
                print(f"  MV sensitivity: {attention_config.get('mv_sensitivity', 0.05)}")
                
                residual_learning = residual_config.get('residual_learning', {})
                print(f"  Residual composition: {residual_learning.get('enable_residual_composition', True)}")
                print(f"  MV range clamp: {residual_learning.get('clamp_mv_range', [-100, 100])}")
                
            else:
                print("Unrecognized network architecture, falling back to residual MV-guided network")
                print("Creating residual MV-guided network (default architecture)...")
                self.model = create_residual_mv_guided_network(self.config)
                
                print("Note: Other architectures moved to archive directory")
                print("Current project focuses on residual MV-guided architecture for better performance")
            
            self.model = self.model.to(self.device)
            
            # Print model info
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Network created successfully")
            print(f"Trainable parameters: {param_count:,}")
            
            # Memory usage estimation
            test_input_shape = (1, 7, *self.config['datasets']['preprocessing']['target_resolution'])
            memory_info = self.model.get_memory_usage(test_input_shape)
            print(f"Estimated memory usage: {memory_info['total_estimated_mb']:.1f} MB")
            
        except Exception as e:
            print(f"Model setup failed: {e}")
            raise
    
    def setup_optimizer(self):
        """Setup optimizer"""
        try:
            optimizer_config = self.config['optimizer']
            
            # Ensure numeric parameters are correct types
            lr = float(optimizer_config['lr'])
            weight_decay = float(optimizer_config['weight_decay'])
            eps = float(optimizer_config['eps'])
            betas = [float(b) for b in optimizer_config['betas']]
            
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
            
            print("Optimizer setup complete")
            
        except Exception as e:
            print(f"Optimizer setup failed: {e}")
            raise
    
    def setup_loss_functions(self):
        """Setup loss functions"""
        try:
            model_config = self.config['model']
            architecture = model_config.get('architecture', 'simplified_unet')
            
            if architecture == 'residual_mv_guided':
                print("Setting up residual inpainting loss function...")
                self.multi_loss = create_residual_inpainting_loss(self.config, self.device)
                
                print("Residual inpainting loss function setup complete")
                print("Residual loss function weights:")
                for loss_name, weight in self.multi_loss.get_loss_weights().items():
                    print(f"  {loss_name}: {weight}")
                
            else:
                print("Unrecognized loss function type, falling back to residual inpainting loss")
                print("Setting up residual inpainting loss function (default)...")
                self.multi_loss = create_residual_inpainting_loss(self.config, self.device)
                
                print("Note: Traditional multi-loss functions archived, using residual inpainting loss")
                print("Residual loss function optimized for selective repair tasks")
                
                print("Residual inpainting loss function setup complete")
                print("Loss function weights:")
                for loss_name, weight in self.multi_loss.get_loss_weights().items():
                    print(f"  {loss_name}: {weight}")
            
            print(f"Loss function device: {self.device}")
            
        except Exception as e:
            print(f"Loss function setup failed: {e}")
            raise
    
    def setup_dataloaders(self):
        """Setup data loaders"""
        try:
            dataset_config = self.config['datasets']
            preprocessing_config = dataset_config['preprocessing']
            
            # Training dataset
            train_dataset = MemorySafeDataset(
                data_root=dataset_config['train_data_root'],
                target_resolution=tuple(preprocessing_config['target_resolution']),
                normalize_method=preprocessing_config['normalize_method'],
                max_cache_size=5,  # Limited cache size
                split="train",
                augmentation=preprocessing_config['augmentation']
            )
            
            # Validation dataset
            val_dataset = MemorySafeDataset(
                data_root=dataset_config['train_data_root'],
                target_resolution=tuple(preprocessing_config['target_resolution']),
                normalize_method=preprocessing_config['normalize_method'],
                max_cache_size=3,  # Smaller validation cache
                split="val",
                augmentation=False
            )
            
            # Data loaders
            self.train_loader = create_memory_safe_dataloader(train_dataset, self.config)
            self.val_loader = create_memory_safe_dataloader(val_dataset, self.config)
            
            print(f"SUCCESS: Data loaders setup complete")
            print(f"INFO: Training samples: {len(train_dataset)}")
            print(f"INFO: Validation samples: {len(val_dataset)}")
            
        except Exception as e:
            print(f"ERROR: Data loader setup failed: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging"""
        try:
            if self.config['tensorboard']['enabled']:
                self.writer = SummaryWriter(self.log_dir / "tensorboard")
                print("SUCCESS: TensorBoard logging setup complete")
            
        except Exception as e:
            print(f"WARNING: Logging setup failed: {e}")
    
    def log_images_to_tensorboard(self, input_data: torch.Tensor, output_data: torch.Tensor, 
                                target_data: torch.Tensor, epoch: int, phase: str = "train", 
                                max_images: int = 4):
        """记录图像到TensorBoard"""
        if not self.writer:
            return
        
        try:
            # Check if using residual MV-guided network
            model_config = self.config['model']
            architecture = model_config.get('architecture', 'simplified_unet')
            is_residual_mv_guided = (architecture == 'residual_mv_guided')
            
            # Limit image count to avoid excessive display
            batch_size = min(input_data.size(0), max_images)
            
            # Process input images (7 channels)
            input_batch = input_data[:batch_size]
            
            # Extract different channels for visualization
            warped_rgb = input_batch[:, :3, :, :]  # First 3 channels: warped RGB
            holes_mask = input_batch[:, 3:4, :, :].repeat(1, 3, 1, 1)  # Holes mask (replicated to 3 channels)
            occlusion_mask = input_batch[:, 4:5, :, :].repeat(1, 3, 1, 1)  # Occlusion mask (replicated to 3 channels)
            
            
            # Motion vector visualization (convert to RGB visualization)
            motion_vectors = input_batch[:, 5:7, :, :]  # MV_x, MV_y
            mv_magnitude = torch.sqrt(motion_vectors[:, 0:1, :, :]**2 + motion_vectors[:, 1:2, :, :]**2 + 1e-8)
            mv_rgb = torch.cat([
                motion_vectors[:, 0:1, :, :],  # R: MV_x
                motion_vectors[:, 1:2, :, :],  # G: MV_y  
                mv_magnitude                    # B: magnitude
            ], dim=1)
            
            # Process output and target images
            output_batch = output_data[:batch_size]
            target_batch = target_data[:batch_size]
            
            # Fix HDR display conversion: check if input normalization is enabled
            model_config = self.config['model']
            enable_normalization = model_config.get('enable_input_normalization', False)
            
            if enable_normalization and hasattr(self.model, 'input_normalizer'):
                # Use input normalizer's HDR conversion functionality
                normalizer = self.model.input_normalizer
                
                # Input images (already normalized) use improved display conversion
                warped_rgb_display = normalizer.hdr_to_ldr_for_display(warped_rgb, "adaptive_reinhard", 1.8)
                target_batch_display = normalizer.hdr_to_ldr_for_display(target_batch, "adaptive_reinhard", 1.8)
                
                # Network output needs denormalization before display format conversion
                output_batch_display = normalizer.prepare_for_tensorboard(output_batch, data_type="rgb", is_normalized=True)
                
                # Fix: Use normalizer to correctly handle Mask and MV data display
                # Mask data: Should already be in [0,1] range according to new strategy
                holes_mask_display = normalizer.prepare_for_tensorboard(holes_mask, "mask")
                occlusion_mask_display = normalizer.prepare_for_tensorboard(occlusion_mask, "mask")
                
                # MV data: Use specialized MV visualization (convert to pseudo-color)
                mv_rgb_display = normalizer.prepare_for_tensorboard(motion_vectors, "mv")
            else:
                # Fix: Fallback display processing method 
                def hdr_to_display(tensor):
                    """Convert HDR image to displayable LDR image"""
                    tensor = torch.clamp(tensor, min=0.0)
                    tone_mapped = tensor / (1.0 + tensor)
                    gamma = 1.8  # Lower gamma to increase brightness
                    tone_mapped = torch.pow(tone_mapped, 1.0 / gamma)
                    return torch.clamp(tone_mapped, 0.0, 1.0)
                
                def mask_to_display(mask_tensor):
                    """Mask display processing: should be in [0,1] range, clip directly"""
                    return torch.clamp(mask_tensor, 0.0, 1.0)
                
                def mv_to_display(mv_tensor):
                    """Motion vector display processing: convert to visualizable RGB"""
                    # MV may have negative values, needs special handling
                    mv_normalized = torch.tanh(mv_tensor * 0.1) * 0.5 + 0.5  # Map to [0,1]
                    if mv_tensor.shape[1] == 2:  # Dual-channel MV
                        mv_magnitude = torch.sqrt(mv_tensor[:, 0:1]**2 + mv_tensor[:, 1:2]**2 + 1e-8)
                        mv_magnitude_norm = torch.tanh(mv_magnitude * 0.1)
                        return torch.cat([
                            mv_normalized[:, 0:1], mv_normalized[:, 1:2], mv_magnitude_norm
                        ], dim=1)
                    else:
                        return mv_normalized
                
                warped_rgb_display = hdr_to_display(warped_rgb)
                holes_mask_display = mask_to_display(holes_mask)
                occlusion_mask_display = mask_to_display(occlusion_mask)
                mv_rgb_display = mv_to_display(motion_vectors)
                output_batch_display = hdr_to_display(output_batch)
                target_batch_display = hdr_to_display(target_batch)
            
            # Basic image logging
            self.writer.add_images(f'{phase}/1_input_warped_rgb', warped_rgb_display, epoch)
            self.writer.add_images(f'{phase}/2_input_holes_mask', holes_mask_display, epoch)
            self.writer.add_images(f'{phase}/3_input_occlusion_mask', occlusion_mask_display, epoch)
            self.writer.add_images(f'{phase}/4_input_motion_vectors', mv_rgb_display, epoch)
            self.writer.add_images(f'{phase}/5_output_predicted', output_batch_display, epoch)
            self.writer.add_images(f'{phase}/6_target_ground_truth', target_batch_display, epoch)
            
            # Special visualization for residual MV-guided network
            if is_residual_mv_guided and hasattr(self.model, 'get_intermediate_outputs'):
                try:
                    with torch.no_grad():
                        intermediate = self.model.get_intermediate_outputs(input_batch)
                    
                    # Spatial attention visualization
                    if 'spatial_attention' in intermediate:
                        spatial_attention = intermediate['spatial_attention'].repeat(1, 3, 1, 1)
                        if enable_normalization and hasattr(self.model, 'input_normalizer'):
                            spatial_attention_display = torch.clamp(spatial_attention, 0.0, 1.0)
                        else:
                            spatial_attention_display = hdr_to_display(spatial_attention)
                        self.writer.add_images(f'{phase}/7_spatial_attention', spatial_attention_display, epoch)
                    
                    # MV urgency visualization
                    if 'mv_urgency' in intermediate:
                        mv_urgency = intermediate['mv_urgency'].repeat(1, 3, 1, 1)
                        if enable_normalization and hasattr(self.model, 'input_normalizer'):
                            mv_urgency_display = torch.clamp(mv_urgency, 0.0, 1.0)
                        else:
                            mv_urgency_display = hdr_to_display(mv_urgency)
                        self.writer.add_images(f'{phase}/8_mv_urgency', mv_urgency_display, epoch)
                    
                    # Correction residual visualization
                    if 'correction_residual' in intermediate:
                        correction_residual = intermediate['correction_residual']
                        # Residual may have negative values, needs special handling
                        residual_normalized = torch.tanh(correction_residual * 5.0) * 0.5 + 0.5  # Normalize to [0,1]
                        self.writer.add_images(f'{phase}/9_correction_residual', residual_normalized, epoch)
                    
                    # Residual comparison: target - warped vs predicted - warped
                    target_residual = target_batch - warped_rgb
                    predicted_residual = output_batch - warped_rgb
                    
                    # Normalize residuals for display
                    target_residual_display = torch.tanh(target_residual * 5.0) * 0.5 + 0.5
                    predicted_residual_display = torch.tanh(predicted_residual * 5.0) * 0.5 + 0.5
                    
                    self.writer.add_images(f'{phase}/10_target_residual', target_residual_display, epoch)
                    self.writer.add_images(f'{phase}/11_predicted_residual', predicted_residual_display, epoch)
                    
                    # Residual error visualization
                    residual_error = torch.abs(predicted_residual - target_residual)
                    residual_error_display = torch.tanh(residual_error * 10.0)  # Emphasize error regions
                    residual_error_display = residual_error_display.repeat(1, 3, 1, 1) if residual_error_display.shape[1] == 1 else residual_error_display
                    self.writer.add_images(f'{phase}/12_residual_error', residual_error_display, epoch)
                    
                    print(f"INFO: Logged {phase} residual learning visualization to TensorBoard (epoch {epoch})")
                    
                except Exception as residual_viz_error:
                    print(f"WARNING: Residual visualization failed: {residual_viz_error}")
            
            # Create comparison images (side-by-side display)
            comparison = torch.cat([
                warped_rgb_display, output_batch_display, target_batch_display
            ], dim=3)  # Horizontal concatenation
            
            final_index = 13 if is_residual_mv_guided else 7
            self.writer.add_images(f'{phase}/{final_index}_comparison_warped_pred_gt', comparison, epoch)
            
            print(f"INFO: Logged {phase} images to TensorBoard (epoch {epoch})")
            
        except Exception as e:
            print(f"WARNING: Image logging failed: {e}")
    
    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        epoch_total_loss = 0.0
        num_batches = 0
        
        # Get configuration
        accumulate_grad_batches = self.config['trainer']['accumulate_grad_batches']
        gradient_clip_val = self.config['trainer']['gradient_clip_val']
        
        for batch_idx, (input_data, target_data) in enumerate(self.train_loader):
            try:
                # 检查内存
                usage_rate, is_critical = self.memory_monitor.check_gpu_memory()
                if is_critical:
                    print("CRITICAL: Memory critical, skipping this batch")
                    self.memory_monitor.force_cleanup()
                    continue
                
                # 数据移到设备
                input_data = input_data.to(self.device, non_blocking=True)
                target_data = target_data.to(self.device, non_blocking=True)
                
                # Forward propagation (memory-optimized version)
                spatial_attention = None
                if hasattr(self.model, 'forward') and 'return_attention' in str(self.model.forward.__code__.co_varnames):
                    # 残差MV引导网络：一次前向传播获取output和attention
                    output, spatial_attention = self.model(input_data, return_attention=True)
                else:
                    # 其他网络：普通前向传播
                    output = self.model(input_data)
                
                # 记录第一个batch的图像到TensorBoard
                if batch_idx == 0 and self.writer:
                    # DEBUG: Check mask data range and non-zero values
                    if self.current_epoch % 10 == 0:  # 每10个epoch打印一次
                        holes_mask = input_data[:, 3:4, :, :]
                        occlusion_mask = input_data[:, 4:5, :, :]
                        holes_nonzero = (holes_mask > 0).sum().item()
                        occlusion_nonzero = (occlusion_mask > 0).sum().item()
                        print(f"[DEBUG] Epoch {self.current_epoch}: holes_mask nonzero pixels={holes_nonzero}, range=[{holes_mask.min():.3f}, {holes_mask.max():.3f}]")
                        print(f"[DEBUG] Epoch {self.current_epoch}: occlusion_mask nonzero pixels={occlusion_nonzero}, range=[{occlusion_mask.min():.3f}, {occlusion_mask.max():.3f}]")
                    
                    self.log_images_to_tensorboard(
                        input_data, output, target_data, 
                        self.current_epoch, phase="train"
                    )
                
                # 计算多损失函数（传入网络学习的spatial_attention）
                if spatial_attention is not None:
                    total_loss, loss_dict = self.multi_loss(output, target_data, input_data, spatial_attention)
                else:
                    total_loss, loss_dict = self.multi_loss(output, target_data, input_data)
                
                # 梯度缩放（用于累积）
                scaled_loss = total_loss / accumulate_grad_batches
                
                # 反向传播
                scaled_loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % accumulate_grad_batches == 0:
                    # 梯度裁剪
                    if gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                    
                    # 优化器步骤
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_total_loss += total_loss.item()
                
                # 记录详细损失（第一个batch）
                if batch_idx == 0:
                    self.last_train_losses = loss_dict
                
                num_batches += 1
                
                # Periodic cleanup
                if batch_idx % 10 == 0:
                    del input_data, target_data, output
                    torch.cuda.empty_cache()
                
                # 进度报告
                if batch_idx % 20 == 0:
                    avg_loss = epoch_total_loss / max(num_batches, 1)
                    print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {avg_loss:.6f}, "
                          f"GPU: {usage_rate:.1%}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CRITICAL: GPU memory insufficient, skipping batch {batch_idx}")
                    self.memory_monitor.force_cleanup()
                    continue
                else:
                    raise e
        
        return epoch_total_loss / max(num_batches, 1)
    
    def validate(self) -> float:
        """Validation"""
        self.model.eval()
        val_total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (input_data, target_data) in enumerate(self.val_loader):
                try:
                    # 检查内存
                    usage_rate, is_critical = self.memory_monitor.check_gpu_memory()
                    if is_critical:
                        self.memory_monitor.force_cleanup()
                        continue
                    
                    # 数据移到设备
                    input_data = input_data.to(self.device, non_blocking=True)
                    target_data = target_data.to(self.device, non_blocking=True)
                    
                    # Forward propagation (memory-optimized version)
                    spatial_attention = None
                    if hasattr(self.model, 'forward') and 'return_attention' in str(self.model.forward.__code__.co_varnames):
                        # 残差MV引导网络：一次前向传播获取output和attention
                        output, spatial_attention = self.model(input_data, return_attention=True)
                    else:
                        # 其他网络：普通前向传播
                        output = self.model(input_data)
                    
                    # Log first batch validation images to TensorBoard
                    if batch_idx == 0 and self.writer:
                        self.log_images_to_tensorboard(
                            input_data, output, target_data, 
                            self.current_epoch, phase="val"
                        )
                    
                    # 计算多损失函数（传入网络学习的spatial_attention）
                    if spatial_attention is not None:
                        total_loss, loss_dict = self.multi_loss(output, target_data, input_data, spatial_attention)
                    else:
                        total_loss, loss_dict = self.multi_loss(output, target_data, input_data)
                    
                    val_total_loss += total_loss.item()
                    
                    # 记录详细损失（第一个batch）
                    if batch_idx == 0:
                        self.last_val_losses = loss_dict
                    
                    num_batches += 1
                    
                    # Cleanup
                    del input_data, target_data, output
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CRITICAL: GPU memory insufficient during validation, skipping batch {batch_idx}")
                        self.memory_monitor.force_cleanup()
                        continue
                    else:
                        raise e
        
        return val_total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'config': self.config
            }
            
            checkpoint_path = self.log_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.log_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                print(f"SUCCESS: Saved best model: {val_loss:.6f}")
            
        except Exception as e:
            print(f"WARNING: Checkpoint save failed: {e}")
    
    def train(self):
        """Main training loop"""
        print("INFO: Starting ultra safe training")
        print("=" * 50)
        
        try:
            # Setup all components
            self.setup_model()
            self.setup_optimizer()
            self.setup_loss_functions()
            self.setup_dataloaders()
            self.setup_logging()
            
            # Training loop
            max_epochs = self.config['training']['max_epochs']
            
            for epoch in range(max_epochs):
                self.current_epoch = epoch
                
                print(f"\nINFO: Epoch {epoch+1}/{max_epochs}")
                print("-" * 30)
                
                # Memory check
                usage_rate, _ = self.memory_monitor.check_gpu_memory()
                
                # Training
                start_time = time.time()
                train_loss = self.train_epoch()
                train_time = time.time() - start_time
                
                # Validation
                start_time = time.time()
                val_loss = self.validate()
                val_time = time.time() - start_time
                
                # Logging
                print(f"SUCCESS: Training loss: {train_loss:.6f} ({train_time:.1f}s)")
                print(f"SUCCESS: Validation loss: {val_loss:.6f} ({val_time:.1f}s)")
                
                # Detailed loss information
                if hasattr(self, 'last_train_losses') and hasattr(self, 'last_val_losses'):
                    print("INFO: Detailed loss analysis:")
                    
                    # Check if using residual MV-guided network
                    model_config = self.config['model']
                    architecture = model_config.get('architecture', 'simplified_unet')
                    
                    if architecture == 'residual_mv_guided':
                        # Residual loss function loss names
                        loss_names = ['residual_mse', 'residual_l1', 'spatial_weighted', 
                                    'preservation', 'edge_preservation', 'perceptual']
                    else:
                        # Traditional multi-loss function loss names
                        loss_names = ['mse', 'l1', 'ssim', 'edge', 'hole_aware', 'perceptual']
                    
                    for loss_name in loss_names:
                        if loss_name in self.last_train_losses and loss_name in self.last_val_losses:
                            train_val = self.last_train_losses[loss_name]
                            val_val = self.last_val_losses[loss_name]
                            print(f"   {loss_name.upper()}: Train={train_val:.6f}, Val={val_val:.6f}")
                
                # TensorBoard logging
                if self.writer:
                    # Total loss
                    self.writer.add_scalar('Loss/Train_Total', train_loss, epoch)
                    self.writer.add_scalar('Loss/Val_Total', val_loss, epoch)
                    
                    # Detailed loss logging
                    if hasattr(self, 'last_train_losses'):
                        for loss_name, loss_value in self.last_train_losses.items():
                            if loss_name != 'total':  # Avoid duplicate total loss logging
                                self.writer.add_scalar(f'Loss/Train_{loss_name.upper()}', loss_value, epoch)
                    
                    if hasattr(self, 'last_val_losses'):
                        for loss_name, loss_value in self.last_val_losses.items():
                            if loss_name != 'total':
                                self.writer.add_scalar(f'Loss/Val_{loss_name.upper()}', loss_value, epoch)
                    
                    # System monitoring
                    self.writer.add_scalar('Memory/GPU_Usage', usage_rate, epoch)
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch, val_loss)
                
                # Early stopping check
                if self._should_early_stop(val_loss):
                    print("INFO: Early stopping triggered")
                    break
                
                # Periodic cleanup
                if epoch % 5 == 0:
                    self.memory_monitor.force_cleanup()
            
            print("\nSUCCESS: Training completed!")
            
        except KeyboardInterrupt:
            print("\nWARNING: Training interrupted by user")
        except Exception as e:
            print(f"\nERROR: Training failed: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            if self.writer:
                self.writer.close()
            self.memory_monitor.force_cleanup()
    
    def _should_early_stop(self, val_loss: float) -> bool:
        """Early stopping check (simplified version)"""
        early_stop_config = self.config.get('early_stopping', {})
        patience = early_stop_config.get('patience', 10)
        
        # Simple check here, should track patience in practice
        return False  # Temporarily disable early stopping


def main():
    """Main function"""
    import argparse
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Ultra safe mobile frame interpolation training")
    parser.add_argument('--config', type=str, 
                       default="./configs/residual_mv_guided_config.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    config_path = args.config
    
    print("INFO: Ultra safe training startup")
    print(f"📄 Configuration file: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file does not exist: {config_path}")
        print("Available configuration files:")
        print("  - ./configs/residual_mv_guided_config.yaml (Residual MV-guided network)")
        print("  - ./configs/heterogeneous_network_config.yaml (Heterogeneous input network)")
        print("  - ./configs/enhanced_network_config.yaml (Enhanced network)")
        print("  - ./configs/ultra_safe_training_config.yaml (Test version)")
        print("  - ./configs/full_resolution_config.yaml (Full resolution)")
        return
    
    try:
        # Create trainer
        trainer = UltraSafeTrainer(config_path)
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()