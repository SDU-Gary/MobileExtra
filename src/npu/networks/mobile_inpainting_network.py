"""Mobile inpainting network with U-Net + Gated Convolution + FFC architecture"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedConv2d(nn.Module):
    """Gated convolution with feature and gate branches"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(GatedConv2d, self).__init__()
        
        # Feature and gate branches
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.feature_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.gate_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.feature_conv.bias is not None:
            nn.init.constant_(self.feature_conv.bias, 0)
        if self.gate_conv.bias is not None:
            nn.init.constant_(self.gate_conv.bias, 0)
    
    def forward(self, x):
        features = self.feature_conv(x)
        gates = torch.sigmoid(self.gate_conv(x))
        gated_features = features * gates
        
        return gated_features


class GatedConvBlock(nn.Module):
    """Gated convolution residual block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(GatedConvBlock, self).__init__()
        
        self.gated_conv1 = GatedConv2d(in_channels, out_channels, stride=stride)
        self.prelu1 = nn.PReLU(out_channels)
        
        self.gated_conv2 = GatedConv2d(out_channels, out_channels)
        
        # Shortcut projection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.gated_conv1(x)
        out = self.prelu1(out)
        out = self.gated_conv2(out)
        
        # Residual connection
        out = out + identity
        
        return out


class FFCResNetBlock(nn.Module):
    """Fast Fourier Convolution residual block with local/global path split"""
    
    def __init__(self, channels, ratio_gin=0.75, ratio_gout=0.75):
        super(FFCResNetBlock, self).__init__()
        
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        
        # Channel split calculation
        self.global_in_num = int(channels * ratio_gin)
        self.local_in_num = channels - self.global_in_num
        
        self.global_out_num = int(channels * ratio_gout) 
        self.local_out_num = channels - self.global_out_num
        
        # Local path processing
        if self.local_in_num > 0:
            self.local_block = GatedConvBlock(self.local_in_num, self.local_out_num)
        
        # Global path processing
        if self.global_in_num > 0:
            self.global_block = nn.Sequential(
                nn.Conv2d(self.global_in_num, self.global_out_num, 1, bias=False),
                nn.PReLU(self.global_out_num)
            )
        
        # Feature fusion
        if self.local_out_num > 0 and self.global_out_num > 0:
            self.fusion = GatedConv2d(self.local_out_num + self.global_out_num, channels, 1, padding=0)
        elif self.local_out_num > 0:
            self.fusion = GatedConv2d(self.local_out_num, channels, 1, padding=0)
        else:
            self.fusion = GatedConv2d(self.global_out_num, channels, 1, padding=0)
    
    def forward(self, x):
        identity = x
        
        if self.ratio_gin == 0:
            # Local only
            local_feat = self.local_block(x)
            out = self.fusion(local_feat)
        elif self.ratio_gin == 1:
            # Global only
            global_feat = self._global_processing(x)
            out = self.fusion(global_feat)
        else:
            # Mixed path
            local_feat, global_feat = torch.split(x, [self.local_in_num, self.global_in_num], dim=1)
            
            # Local processing
            local_feat = self.local_block(local_feat)
            
            # Global frequency processing
            global_feat = self._global_processing(global_feat)
            
            # Feature fusion
            if self.local_out_num > 0 and self.global_out_num > 0:
                fused_feat = torch.cat([local_feat, global_feat], dim=1)
            elif self.local_out_num > 0:
                fused_feat = local_feat
            else:
                fused_feat = global_feat
            
            out = self.fusion(fused_feat)
        
        # Residual connection
        return out + identity
    
    def _global_processing(self, x):
        """Global frequency domain processing with cuFFT precision fix"""
        B, C, H, W = x.shape
        
        # Preserve dtype
        original_dtype = x.dtype
        
        # cuFFT fix for non-power-of-2 sizes in FP16
        if x.dtype == torch.float16 and (not self._is_power_of_two(H) or not self._is_power_of_two(W)):
            x = x.to(torch.float32)
        
        # FFT to frequency domain
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # Process real and imaginary parts
        real_part = x_fft.real
        imag_part = x_fft.imag
        
        processed_real = self.global_block(real_part)
        processed_imag = self.global_block(imag_part)
        
        # Reconstruct complex tensor
        x_fft = processed_real + 1j * processed_imag
        
        # IFFT back to spatial domain
        x_spatial = torch.fft.irfft2(x_fft, s=(H, W), dim=(-2, -1), norm='ortho')
        
        # Restore original dtype
        if x_spatial.dtype != original_dtype:
            x_spatial = x_spatial.to(original_dtype)
        
        return x_spatial
    
    def _is_power_of_two(self, n):
        return n > 0 and (n & (n - 1)) == 0


class MobileInpaintingNetwork(nn.Module):
    """Mobile inpainting network: 7ch input (WarPed+Holes+Occlusion+ResidualMV) -> 3ch RGB"""
    
    def __init__(self, input_channels=7, output_channels=3):
        super(MobileInpaintingNetwork, self).__init__()
        
        # Input projection
        self.input_proj = GatedConv2d(input_channels, 32, 3, 1, 1)
        
        # Encoder path
        self.encoder1 = GatedConvBlock(32, 32)
        self.down1 = GatedConv2d(32, 64, 3, 2, 1)
        self.encoder2 = GatedConvBlock(64, 64)
        self.down2 = GatedConv2d(64, 128, 3, 2, 1)
        self.encoder3 = GatedConvBlock(128, 128)
        
        # FFC bottleneck
        self.bottleneck = FFCResNetBlock(128, ratio_gin=0.75, ratio_gout=0.75)
        
        # Decoder path with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = GatedConv2d(128 + 64, 64, 3, 1, 1)
        self.decoder1 = GatedConvBlock(64, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = GatedConv2d(64 + 32, 32, 3, 1, 1)
        self.decoder2 = GatedConvBlock(32, 32)
        
        # Output layer
        self.output_conv = nn.Conv2d(32, output_channels, 1, 1, 0)
        self.output_activation = nn.Tanh()  # [-1, 1] range
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass: [B, 7, H, W] -> [B, 3, H, W]"""
        # Input projection
        x = self.input_proj(x)
        
        # Encoder path with skip connection storage
        enc1 = self.encoder1(x)
        down1 = self.down1(enc1)
        enc2 = self.encoder2(down1)
        down2 = self.down2(enc2)
        enc3 = self.encoder3(down2)
        
        # FFC bottleneck processing
        bottleneck = self.bottleneck(enc3)
        
        # Decoder path with skip connections
        up1 = self.up1(bottleneck)
        # Size matching for skip connections
        if up1.shape[-2:] != enc2.shape[-2:]:
            up1 = F.interpolate(up1, size=enc2.shape[-2:], mode='nearest')
        
        up1 = torch.cat([up1, enc2], dim=1)
        up1 = self.up_conv1(up1)
        dec1 = self.decoder1(up1)
        up2 = self.up2(dec1)
        
        if up2.shape[-2:] != enc1.shape[-2:]:
            up2 = F.interpolate(up2, size=enc1.shape[-2:], mode='nearest')
        
        up2 = torch.cat([up2, enc1], dim=1)
        up2 = self.up_conv2(up2)
        dec2 = self.decoder2(up2)
        
        # Output
        output = self.output_conv(dec2)
        output = self.output_activation(output)
        
        return output
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb


def create_mobile_inpainting_network(input_channels=7, output_channels=3):
    """Create mobile inpainting network instance"""
    model = MobileInpaintingNetwork(input_channels, output_channels)
    
    # Model info
    param_count = model.get_parameter_count()
    model_size = model.get_model_size_mb()
    
    print(f"=== Mobile Inpainting Network ===")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Architecture: U-Net + Gated Convolution + FFC")
    print(f"Input Channels: {input_channels} (WarPedRGB + SemanticHoles + Occlusion + ResidualMV)")
    print(f"Output Channels: {output_channels} (RGB)")
    
    return model


if __name__ == "__main__":
    model = create_mobile_inpainting_network()
    
    # Test forward pass
    input_tensor = torch.randn(4, 7, 64, 64)
    print(f"\nTesting forward pass...")
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print(f"\nNetwork test completed successfully!")