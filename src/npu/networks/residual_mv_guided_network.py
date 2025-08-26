#!/usr/bin/env python3
"""Residual MV-guided selective inpainting network"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math
from .input_normalizer import UnifiedInputNormalizer


class GatedConv2d(nn.Module):
    """Lightweight gated convolution for irregular hole regions"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dilation: int = 1):
        super().__init__()
        
        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride, padding, dilation, bias=False
        )
        self.gate_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride, padding, dilation, bias=False
        )
        
        self.feature_bn = nn.BatchNorm2d(out_channels)
        self.gate_bn = nn.BatchNorm2d(out_channels)
        
        self.feature_activation = nn.ReLU(inplace=True)
        self.gate_activation = nn.Sigmoid()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.feature_conv, self.gate_conv]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_conv(x)
        features = self.feature_bn(features)
        features = self.feature_activation(features)
        
        gates = self.gate_conv(x)
        gates = self.gate_bn(gates)
        gates = self.gate_activation(gates)
        
        return features * gates


class SpatialAttentionGenerator(nn.Module):
    """Spatial attention generator based on masks and residual MV"""
    
    def __init__(self, use_gated_conv: bool = True):
        super().__init__()
        if use_gated_conv:
            self.attention_processor = nn.Sequential(
                GatedConv2d(3, 16, 3, padding=1),
                GatedConv2d(16, 8, 3, padding=1),
                nn.Conv2d(8, 1, 1),
                nn.Sigmoid()
            )
        else:
            self.attention_processor = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, 1),
                nn.Sigmoid()
            )
    
    def forward(self, holes_mask: torch.Tensor, occlusion_mask: torch.Tensor, 
                residual_mv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mv_magnitude = torch.sqrt(residual_mv[:, 0:1]**2 + residual_mv[:, 1:2]**2 + 1e-8)
        mv_urgency = torch.tanh(mv_magnitude * 0.05)
        
        attention_input = torch.cat([holes_mask, occlusion_mask, mv_urgency], dim=1)
        spatial_attention = self.attention_processor(attention_input)
        
        return spatial_attention, mv_urgency


class BackboneFeatureExtractor(nn.Module):
    """Multi-scale feature extractor for warped RGB"""
    
    def __init__(self, input_channels: int = 3, 
                 encoder_channels: Tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        
        # 输入卷积
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # 编码器层
        self.encoders = nn.ModuleList()
        in_ch = encoder_channels[0]
        
        for out_ch in encoder_channels[1:]:
            self.encoders.append(self._make_encoder_block(in_ch, out_ch))
            in_ch = out_ch
    
    def _make_encoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, warped_rgb: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_conv(warped_rgb)
        features = [x]
        
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        
        return features


class ResidualMVProcessor(nn.Module):
    """Residual MV processor for motion guidance features"""
    
    def __init__(self, mv_channels: int = 2, output_channels: int = 32, 
                 use_gated_conv: bool = True):
        super().__init__()
        
        if use_gated_conv:
            self.mv_processor = nn.Sequential(
                nn.Conv2d(mv_channels, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                GatedConv2d(16, output_channels//2, 3, padding=1),
                GatedConv2d(output_channels//2, output_channels, 3, padding=1)
            )
        else:
            self.mv_processor = nn.Sequential(
                nn.Conv2d(mv_channels, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, output_channels//2, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels//2, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, residual_mv: torch.Tensor, 
                spatial_attention: torch.Tensor) -> torch.Tensor:
        processed_mv = residual_mv
        mv_features = self.mv_processor(processed_mv)
        attended_mv_features = mv_features * spatial_attention
        
        return attended_mv_features


class GatedDecoderBlock(nn.Module):
    """Gated decoder block"""
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0,
                 use_gated_conv: bool = True):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.upsample_bn = nn.BatchNorm2d(out_channels)
        
        fusion_in_channels = out_channels + skip_channels
        if use_gated_conv:
            self.fusion = nn.Sequential(
                GatedConv2d(fusion_in_channels, out_channels, 3, padding=1),
                GatedConv2d(out_channels, out_channels, 3, padding=1)
            )
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(fusion_in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        x = self.upsample_conv(x)
        x = self.upsample_bn(x)
        x = F.relu(x, inplace=True)
        
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.fusion(x)
        
        return x


class GatedSelectiveDecoder(nn.Module):
    """Gated selective decoder for residual correction"""
    
    def __init__(self, encoder_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
                 mv_feature_channels: int = 32, use_gated_conv: bool = True):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        decoder_channels = encoder_channels[::-1]  # 反转顺序
        
        self.decoder_blocks = nn.ModuleList()
        
        in_ch = decoder_channels[0] + mv_feature_channels
        out_ch = decoder_channels[1]
        self.decoder_blocks.append(
            GatedDecoderBlock(in_ch, out_ch, encoder_channels[-2], use_gated_conv)
        )
        
        for i in range(1, len(decoder_channels)-1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i+1]
            skip_ch = encoder_channels[-(i+2)]
            self.decoder_blocks.append(
                GatedDecoderBlock(in_ch, out_ch, skip_ch, use_gated_conv)
            )
        
        # 残差输出头
        self.residual_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1]//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[-1]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1]//2, 3, 1)  # 输出RGB残差
        )
    
    def forward(self, backbone_features: List[torch.Tensor], 
                mv_features: torch.Tensor, 
                spatial_attention: torch.Tensor) -> torch.Tensor:
        """
        门控解码生成残差修正
        
        Args:
            backbone_features: 多尺度backbone特征
            mv_features: 运动指导特征
            spatial_attention: 空间注意力权重
            
        Returns:
            correction_residual: [B, 3, H, W] 残差修正
        """
        
        # 1. 融合最深层特征和MV指导
        deepest_features = backbone_features[-1]  # 最深层特征
        
        # 调整MV特征尺寸匹配
        if mv_features.shape[2:] != deepest_features.shape[2:]:
            mv_features = F.interpolate(
                mv_features, size=deepest_features.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        x = torch.cat([deepest_features, mv_features], dim=1)
        
        # 2. 门控解码
        for i, decoder in enumerate(self.decoder_blocks):
            skip_idx = -(i+2)
            skip_features = backbone_features[skip_idx] if -skip_idx <= len(backbone_features) else None
            x = decoder(x, skip_features)
        
        # 3. 生成残差修正
        correction_residual = self.residual_head(x)
        
        # 4. 只在需要修复的区域应用修正（关键！）
        masked_residual = correction_residual * spatial_attention
        
        return masked_residual


class ResidualMVGuidedNetwork(nn.Module):
    """Residual MV-guided selective inpainting network"""
    
    def __init__(self, 
                 input_channels: int = 7,
                 output_channels: int = 3,
                 encoder_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
                 mv_feature_channels: int = 32,
                 use_gated_conv: bool = True,
                 dropout_rate: float = 0.1,
                 enable_input_normalization: bool = True):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.enable_input_normalization = enable_input_normalization
        
        if enable_input_normalization:
            self.input_normalizer = UnifiedInputNormalizer(
                rgb_method='hdr_to_ldr',
                tone_mapping='reinhard',
                normalize_masks=False,
                normalize_mv=False,
                gamma=2.2
            )
        else:
            self.input_normalizer = None
        
        self.attention_generator = SpatialAttentionGenerator(use_gated_conv)
        
        self.backbone_extractor = BackboneFeatureExtractor(3, encoder_channels)
        
        self.mv_processor = ResidualMVProcessor(2, mv_feature_channels, use_gated_conv)
        
        self.selective_decoder = GatedSelectiveDecoder(
            encoder_channels, mv_feature_channels, use_gated_conv
        )
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        
        if self.input_normalizer is not None:
            x_processed = self.input_normalizer(x, update_stats=self.training)
        else:
            x_processed = x
        
        warped_rgb = x_processed[:, 0:3, :, :]
        holes_mask = x_processed[:, 3:4, :, :]
        occlusion_mask = x_processed[:, 4:5, :, :]
        residual_mv = x_processed[:, 5:7, :, :]
        
        spatial_attention, mv_urgency = self.attention_generator(
            holes_mask, occlusion_mask, residual_mv
        )
        
        backbone_features = self.backbone_extractor(warped_rgb)
        
        mv_features = self.mv_processor(residual_mv, spatial_attention)
        
        mv_features = self.dropout(mv_features)
        correction_residual = self.selective_decoder(
            backbone_features, mv_features, spatial_attention
        )
        
        refined_output = warped_rgb + correction_residual
        
        if return_attention:
            return refined_output, spatial_attention
        else:
            return refined_output
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_usage(self, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        batch_size, channels, height, width = input_shape
        
        input_size = batch_size * channels * height * width * 4 / 1024 / 1024
        
        param_count = self.get_parameter_count()
        param_size = param_count * 4 / 1024 / 1024
        
        max_channels = max(self.encoder_channels)
        activation_size = batch_size * max_channels * height * width * 4 / 1024 / 1024
        
        return {
            'input_mb': input_size,
            'parameters_mb': param_size,
            'activations_mb': activation_size,
            'total_estimated_mb': input_size + param_size + activation_size * 2,
            'parameter_count': param_count
        }
    
    def get_intermediate_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        warped_rgb = x[:, 0:3, :, :]
        holes_mask = x[:, 3:4, :, :]
        occlusion_mask = x[:, 4:5, :, :]
        residual_mv = x[:, 5:7, :, :]
        
        spatial_attention, mv_urgency = self.attention_generator(
            holes_mask, occlusion_mask, residual_mv
        )
        
        backbone_features = self.backbone_extractor(warped_rgb)
        
        mv_features = self.mv_processor(residual_mv, spatial_attention)
        
        correction_residual = self.selective_decoder(
            backbone_features, mv_features, spatial_attention
        )
        
        refined_output = warped_rgb + correction_residual
        
        return {
            'warped_rgb': warped_rgb,
            'spatial_attention': spatial_attention,
            'mv_urgency': mv_urgency,
            'mv_features': mv_features,
            'correction_residual': correction_residual,
            'refined_output': refined_output
        }


def create_residual_mv_guided_network(config: Dict) -> ResidualMVGuidedNetwork:
    
    model_config = config.get('model', {})
    
    encoder_channels = tuple(model_config.get('encoder_channels', [32, 64, 128, 256, 512]))
    mv_feature_channels = model_config.get('mv_feature_channels', 32)
    use_gated_conv = model_config.get('use_gated_conv', True)
    dropout_rate = model_config.get('dropout_rate', 0.1)
    
    enable_input_normalization = model_config.get('enable_input_normalization', True)
    
    network = ResidualMVGuidedNetwork(
        input_channels=model_config.get('input_channels', 7),
        output_channels=model_config.get('output_channels', 3),
        encoder_channels=encoder_channels,
        mv_feature_channels=mv_feature_channels,
        use_gated_conv=use_gated_conv,
        dropout_rate=dropout_rate,
        enable_input_normalization=enable_input_normalization
    )
    
    return network


if __name__ == "__main__":
    test_config = {
        'model': {
            'input_channels': 7,
            'output_channels': 3,
            'encoder_channels': [32, 64, 128, 256, 512],
            'mv_feature_channels': 32,
            'use_gated_conv': True,
            'dropout_rate': 0.1
        }
    }
    
    model = create_residual_mv_guided_network(test_config)
    test_input = torch.randn(1, 7, 270, 480)
    
    with torch.no_grad():
        output = model(test_input)
        intermediate = model.get_intermediate_outputs(test_input)
    
    param_count = model.get_parameter_count()
    memory_info = model.get_memory_usage(test_input.shape)
    
    print(f"Network created successfully")
    print(f"Parameters: {param_count:,}")
    print(f"Input: {test_input.shape}, Output: {output.shape}")
    print(f"Memory usage: {memory_info['total_estimated_mb']:.1f}MB")