"""
@file mobile_inpainting_network.py
@brief 移动端补全网络PyTorch实现

网络架构设计：
- 参数预算: 3.0M (更新架构)
- 输入: 6通道 (RGB + Mask + ResidualMV)
- 输出: 3通道 RGB修复结果
- 结构: U-Net骨架 + Gated Convolution + FFC瓶颈层

技术特点：
- GatedConvBlock: Gated Convolution残差块，智能处理不规则空洞
- FFCResNetBlock: FFC瓶颈层，同时处理局部和全局特征
- U-Net架构: 编码器-解码器结构，跳跃连接保持细节
- 频域处理: 通过FFT/IFFT实现全局感受野

核心创新：
- 无空洞分类机制，简化网络设计
- Gated Convolution替代标准卷积，柔性处理空洞边缘
- FFC模块在频域处理全局纹理和结构信息

性能目标：
- 训练收敛: <100 epochs
- 验证SSIM: >0.90
- 推理延迟: <4ms (NPU)
- 量化友好: 支持INT8部署

@author AI算法团队
@date 2025-07-28
@version 1.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedConv2d(nn.Module):
    """
    门控卷积层 - 网络的基础组件
    
    核心思想：
    - 对于每个卷积操作，同时生成特征图和门控掩码
    - 门控掩码决定哪些特征应该被传递，哪些应该被抑制
    - 特别适合处理不规则空洞区域
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(GatedConv2d, self).__init__()
        
        # 特征卷积分支：产生主要特征
        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        
        # 门控卷积分支：产生门控掩码
        self.gate_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.kaiming_normal_(self.feature_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.gate_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.feature_conv.bias is not None:
            nn.init.constant_(self.feature_conv.bias, 0)
        if self.gate_conv.bias is not None:
            nn.init.constant_(self.gate_conv.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            gated_features: 门控后的特征图 [B, out_C, H', W']
        """
        # 计算特征和门控
        features = self.feature_conv(x)  # [B, out_C, H', W']
        gates = torch.sigmoid(self.gate_conv(x))  # [B, out_C, H', W']
        
        # 门控机制：特征与门控相乘
        gated_features = features * gates
        
        return gated_features


class GatedConvBlock(nn.Module):
    """
    门控卷积残差块 - 编码器和解码器的基础模块
    
    结构：
    1. GatedConv + PReLU
    2. GatedConv  
    3. 残差连接
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(GatedConvBlock, self).__init__()
        
        self.gated_conv1 = GatedConv2d(in_channels, out_channels, stride=stride)
        self.prelu1 = nn.PReLU(out_channels)
        
        self.gated_conv2 = GatedConv2d(out_channels, out_channels)
        
        # 残差连接的投影层（当通道数不同时）
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        """前向传播"""
        identity = self.shortcut(x)
        
        out = self.gated_conv1(x)
        out = self.prelu1(out)
        out = self.gated_conv2(out)
        
        # 残差连接
        out = out + identity
        
        return out


class FFCResNetBlock(nn.Module):
    """
    FFC残差网络块 - 网络的核心模块
    
    功能：
    - 通道分裂：25%局部路径 + 75%全局路径
    - 局部路径：标准门控卷积处理
    - 全局路径：频域处理（FFT -> 1x1卷积 -> IFFT）
    - 特征融合与残差连接
    """
    
    def __init__(self, channels, ratio_gin=0.75, ratio_gout=0.75):
        super(FFCResNetBlock, self).__init__()
        
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        
        # 计算全局和局部通道数
        self.global_in_num = int(channels * ratio_gin)
        self.local_in_num = channels - self.global_in_num
        
        self.global_out_num = int(channels * ratio_gout) 
        self.local_out_num = channels - self.global_out_num
        
        # 局部路径：标准门控卷积块
        if self.local_in_num > 0:
            self.local_block = GatedConvBlock(self.local_in_num, self.local_out_num)
        
        # 全局路径：频域处理
        if self.global_in_num > 0:
            self.global_block = nn.Sequential(
                nn.Conv2d(self.global_in_num, self.global_out_num, 1, bias=False),
                nn.PReLU(self.global_out_num)
            )
        
        # 特征融合层
        if self.local_out_num > 0 and self.global_out_num > 0:
            self.fusion = GatedConv2d(self.local_out_num + self.global_out_num, channels, 1, padding=0)
        elif self.local_out_num > 0:
            self.fusion = GatedConv2d(self.local_out_num, channels, 1, padding=0)
        else:
            self.fusion = GatedConv2d(self.global_out_num, channels, 1, padding=0)
    
    def forward(self, x):
        """前向传播"""
        identity = x
        
        if self.ratio_gin == 0:
            # 仅局部路径
            local_feat = self.local_block(x)
            out = self.fusion(local_feat)
        elif self.ratio_gin == 1:
            # 仅全局路径
            global_feat = self._global_processing(x)
            out = self.fusion(global_feat)
        else:
            # 混合路径
            local_feat, global_feat = torch.split(x, [self.local_in_num, self.global_in_num], dim=1)
            
            # 局部路径处理
            local_feat = self.local_block(local_feat)
            
            # 全局路径处理（频域）
            global_feat = self._global_processing(global_feat)
            
            # 特征融合
            if self.local_out_num > 0 and self.global_out_num > 0:
                fused_feat = torch.cat([local_feat, global_feat], dim=1)
            elif self.local_out_num > 0:
                fused_feat = local_feat
            else:
                fused_feat = global_feat
            
            out = self.fusion(fused_feat)
        
        # 残差连接
        return out + identity
    
    def _global_processing(self, x):
        """全局特征处理（频域）"""
        B, C, H, W = x.shape
        
        # FFT变换到频域
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # 频域特征处理
        x_fft = x_fft.real * self.global_block(x.unsqueeze(-1).real).squeeze(-1) + \
                1j * x_fft.imag * self.global_block(x.unsqueeze(-1).imag).squeeze(-1)
        
        # 逆FFT变换回空间域
        x_spatial = torch.fft.irfft2(x_fft, s=(H, W), dim=(-2, -1), norm='ortho')
        
        return x_spatial


class MobileInpaintingNetwork(nn.Module):
    """
    移动端补全网络 - 主网络架构
    
    架构设计：
    - 输入：6通道 (RGB 3 + Mask 1 + ResidualMV 2)
    - 输出：3通道 RGB
    - 结构：U-Net + Gated Convolution + FFC
    - 参数预算：3M
    
    U-Net结构：
    - 编码器：32 -> 64 -> 128 通道
    - 瓶颈层：FFCResNetBlock (128通道)
    - 解码器：128 -> 64 -> 32 通道，带跳跃连接
    """
    
    def __init__(self, input_channels=6, output_channels=3):
        super(MobileInpaintingNetwork, self).__init__()
        
        # 输入投影层
        self.input_proj = GatedConv2d(input_channels, 32, 3, 1, 1)
        
        # 编码器（降采样路径）
        self.encoder1 = GatedConvBlock(32, 32)          # 32x32 -> 32x32
        self.down1 = GatedConv2d(32, 64, 3, 2, 1)      # 32x32 -> 64x16
        
        self.encoder2 = GatedConvBlock(64, 64)          # 64x64 -> 64x64  
        self.down2 = GatedConv2d(64, 128, 3, 2, 1)     # 64x64 -> 128x32
        
        self.encoder3 = GatedConvBlock(128, 128)        # 128x128 -> 128x128
        
        # 瓶颈层（核心FFC模块）
        self.bottleneck = FFCResNetBlock(128, ratio_gin=0.75, ratio_gout=0.75)
        
        # 解码器（上采样路径）
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = GatedConv2d(128 + 64, 64, 3, 1, 1)  # 跳跃连接
        self.decoder1 = GatedConvBlock(64, 64)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest') 
        self.up_conv2 = GatedConv2d(64 + 32, 32, 3, 1, 1)   # 跳跃连接
        self.decoder2 = GatedConvBlock(32, 32)
        
        # 输出层
        self.output_conv = nn.Conv2d(32, output_channels, 1, 1, 0)
        self.output_activation = nn.Tanh()  # 输出范围 [-1, 1]
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [B, 6, H, W] - RGB(3) + Mask(1) + ResidualMV(2)
        Returns:
            output: 修复后的RGB图像 [B, 3, H, W]
        """
        # 输入投影
        x = self.input_proj(x)  # [B, 32, H, W]
        
        # 编码器路径（保存跳跃连接）
        enc1 = self.encoder1(x)         # [B, 32, H, W]
        down1 = self.down1(enc1)        # [B, 64, H/2, W/2]
        
        enc2 = self.encoder2(down1)     # [B, 64, H/2, W/2]
        down2 = self.down2(enc2)        # [B, 128, H/4, W/4]
        
        enc3 = self.encoder3(down2)     # [B, 128, H/4, W/4]
        
        # 瓶颈层（FFC处理）
        bottleneck = self.bottleneck(enc3)  # [B, 128, H/4, W/4]
        
        # 解码器路径（带跳跃连接）
        up1 = self.up1(bottleneck)      # [B, 128, H/2, W/2]
        up1 = torch.cat([up1, enc2], dim=1)  # [B, 128+64, H/2, W/2]
        up1 = self.up_conv1(up1)        # [B, 64, H/2, W/2]
        dec1 = self.decoder1(up1)       # [B, 64, H/2, W/2]
        
        up2 = self.up2(dec1)            # [B, 64, H, W]
        up2 = torch.cat([up2, enc1], dim=1)  # [B, 64+32, H, W]
        up2 = self.up_conv2(up2)        # [B, 32, H, W]
        dec2 = self.decoder2(up2)       # [B, 32, H, W]
        
        # 输出层
        output = self.output_conv(dec2)     # [B, 3, H, W]
        output = self.output_activation(output)  # [B, 3, H, W], range [-1, 1]
        
        return output
    
    def get_parameter_count(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """计算模型大小（MB）"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb


def create_mobile_inpainting_network(input_channels=6, output_channels=3):
    """
    创建移动端补全网络实例
    
    Args:
        input_channels: 输入通道数，默认6 (RGB + Mask + ResidualMV)
        output_channels: 输出通道数，默认3 (RGB)
    
    Returns:
        model: MobileInpaintingNetwork实例
    """
    model = MobileInpaintingNetwork(input_channels, output_channels)
    
    # 打印模型信息
    param_count = model.get_parameter_count()
    model_size = model.get_model_size_mb()
    
    print(f"=== Mobile Inpainting Network ===")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Architecture: U-Net + Gated Convolution + FFC")
    print(f"Input Channels: {input_channels} (RGB + Mask + ResidualMV)")
    print(f"Output Channels: {output_channels} (RGB)")
    
    return model


if __name__ == "__main__":
    # 测试网络
    model = create_mobile_inpainting_network()
    
    # 测试前向传播
    batch_size = 4
    height, width = 64, 64
    input_tensor = torch.randn(batch_size, 6, height, width)
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print(f"\nNetwork test completed successfully!")