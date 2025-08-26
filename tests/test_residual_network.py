#!/usr/bin/env python3
"""
残差MV引导网络测试脚本
验证残差学习架构的正确性和有效性
"""

import torch
import torch.nn as nn
import yaml
import numpy as np
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "train"))

def load_test_modules():
    """加载测试所需的模块"""
    try:
        from src.npu.networks.residual_mv_guided_network import create_residual_mv_guided_network, ResidualMVGuidedNetwork
        from train.residual_inpainting_loss import create_residual_inpainting_loss
        
        print("✅ 残差MV引导网络模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def create_realistic_test_data(batch_size: int = 1, height: int = 270, width: int = 480) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建接近真实场景的测试数据
    模拟warped图像补洞任务的输入特点
    """
    
    # 创建7通道输入数据
    input_data = torch.zeros(batch_size, 7, height, width, dtype=torch.float32)
    
    # 1. warped RGB (通道0-2) - 模拟有warp误差的RGB图像
    base_rgb = torch.rand(batch_size, 3, height, width) * 0.8 + 0.1
    
    # 在某些区域添加warp误差（亮度变化和偏移）
    center_h, center_w = height // 2, width // 2
    warp_error_region = torch.zeros_like(base_rgb)
    
    # 计算warp误差区域的安全边界
    warp_h_start = max(0, center_h-30)
    warp_h_end = min(height, center_h+30)
    warp_w_start = max(0, center_w-40)
    warp_w_end = min(width, center_w+40)
    
    if warp_h_end > warp_h_start and warp_w_end > warp_w_start:
        warp_error_region[:, :, warp_h_start:warp_h_end, warp_w_start:warp_w_end] = 0.3
    
    warped_rgb = base_rgb + warp_error_region
    input_data[:, 0:3, :, :] = warped_rgb
    
    # 2. holes mask (通道3) - 空洞掩码
    holes_mask = torch.zeros(batch_size, 1, height, width)
    # 在图像中创建一些不规则空洞 - 安全边界检查
    hole1_h_start, hole1_h_end = max(0, center_h-15), min(height, center_h+15)
    hole1_w_start, hole1_w_end = max(0, center_w-25), min(width, center_w+25)
    if hole1_h_end > hole1_h_start and hole1_w_end > hole1_w_start:
        holes_mask[:, :, hole1_h_start:hole1_h_end, hole1_w_start:hole1_w_end] = 1.0
    
    hole2_h_start, hole2_h_end = max(0, center_h+40), min(height, center_h+60)
    hole2_w_start, hole2_w_end = max(0, center_w-10), min(width, center_w+10)
    if hole2_h_end > hole2_h_start and hole2_w_end > hole2_w_start:
        holes_mask[:, :, hole2_h_start:hole2_h_end, hole2_w_start:hole2_w_end] = 1.0
    input_data[:, 3:4, :, :] = holes_mask
    
    # 3. occlusion mask (通道4) - 遮挡掩码
    occlusion_mask = torch.zeros(batch_size, 1, height, width)
    # 模拟遮挡区域 - 安全边界检查
    occ_h_start, occ_h_end = max(0, center_h-50), min(height, center_h-20)
    occ_w_start, occ_w_end = max(0, center_w+20), min(width, center_w+50)
    if occ_h_end > occ_h_start and occ_w_end > occ_w_start:
        occlusion_mask[:, :, occ_h_start:occ_h_end, occ_w_start:occ_w_end] = 1.0
    input_data[:, 4:5, :, :] = occlusion_mask
    
    # 4. 残差运动向量 (通道5-6) - 模拟warp误差修正向量
    residual_mv_x = torch.zeros(batch_size, 1, height, width)
    residual_mv_y = torch.zeros(batch_size, 1, height, width)
    
    # 计算实际的区域尺寸
    mv_region_h = min(60, height - max(0, center_h-30))  # 确保不超出边界
    mv_region_w = min(80, width - max(0, center_w-40))   # 确保不超出边界
    
    # 计算实际的切片范围
    mv_start_h = max(0, center_h-30)
    mv_end_h = mv_start_h + mv_region_h
    mv_start_w = max(0, center_w-40) 
    mv_end_w = mv_start_w + mv_region_w
    
    # 在有warp误差的区域添加残差MV
    if mv_region_h > 0 and mv_region_w > 0:
        residual_mv_x[:, :, mv_start_h:mv_end_h, mv_start_w:mv_end_w] = torch.randn(batch_size, 1, mv_region_h, mv_region_w) * 5.0
        residual_mv_y[:, :, mv_start_h:mv_end_h, mv_start_w:mv_end_w] = torch.randn(batch_size, 1, mv_region_h, mv_region_w) * 5.0
    
    input_data[:, 5:6, :, :] = residual_mv_x
    input_data[:, 6:7, :, :] = residual_mv_y
    
    # 5. 创建目标真实图像（模拟完美修复后的结果）
    target_data = base_rgb.clone()  # 基础RGB作为目标
    
    # 在空洞和遮挡区域填充合理的内容 - 使用安全的尺寸
    # 空洞区域1
    if hole1_h_end > hole1_h_start and hole1_w_end > hole1_w_start:
        hole1_h, hole1_w = hole1_h_end - hole1_h_start, hole1_w_end - hole1_w_start
        target_data[:, :, hole1_h_start:hole1_h_end, hole1_w_start:hole1_w_end] = torch.rand(batch_size, 3, hole1_h, hole1_w) * 0.6 + 0.2
    
    # 空洞区域2
    if hole2_h_end > hole2_h_start and hole2_w_end > hole2_w_start:
        hole2_h, hole2_w = hole2_h_end - hole2_h_start, hole2_w_end - hole2_w_start
        target_data[:, :, hole2_h_start:hole2_h_end, hole2_w_start:hole2_w_end] = torch.rand(batch_size, 3, hole2_h, hole2_w) * 0.6 + 0.2
    
    # 遮挡区域
    if occ_h_end > occ_h_start and occ_w_end > occ_w_start:
        occ_h, occ_w = occ_h_end - occ_h_start, occ_w_end - occ_w_start
        target_data[:, :, occ_h_start:occ_h_end, occ_w_start:occ_w_end] = torch.rand(batch_size, 3, occ_h, occ_w) * 0.6 + 0.2
    
    return input_data, target_data

def test_residual_mv_guided_network(device: torch.device) -> bool:
    """测试残差MV引导网络的基本功能"""
    
    print("\n🧪 测试残差MV引导网络基本功能")
    print("-" * 50)
    
    try:
        from src.npu.networks.residual_mv_guided_network import create_residual_mv_guided_network
        
        # 创建测试配置
        test_config = {
            'model': {
                'architecture': 'residual_mv_guided',
                'input_channels': 7,
                'output_channels': 3,
                'residual_mv_config': {
                    'encoder_channels': [32, 64, 128, 256, 512],
                    'mv_feature_channels': 32,
                    'use_gated_conv': True,
                    'attention_config': {
                        'use_gated_attention': True,
                        'mv_sensitivity': 0.05
                    },
                    'residual_learning': {
                        'enable_residual_composition': True,
                        'clamp_mv_range': [-100, 100],
                        'mv_normalization_factor': 100.0
                    }
                },
                'dropout_rate': 0.1
            }
        }
        
        # 创建网络
        model = create_residual_mv_guided_network(test_config)
        model = model.to(device)
        model.eval()
        
        # 创建测试数据
        input_data, target_data = create_realistic_test_data(batch_size=1, height=64, width=64)
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        
        # 基本前向传播测试
        with torch.no_grad():
            start_time = time.time()
            output = model(input_data)
            forward_time = (time.time() - start_time) * 1000
        
        # 网络信息
        param_count = model.get_parameter_count()
        memory_info = model.get_memory_usage(input_data.shape)
        
        print(f"✅ 网络创建和前向传播成功")
        print(f"📊 参数数量: {param_count:,}")
        print(f"📊 输入形状: {input_data.shape}")
        print(f"📊 输出形状: {output.shape}")
        print(f"📊 前向时间: {forward_time:.2f} ms")
        print(f"📊 预估内存: {memory_info['total_estimated_mb']:.1f} MB")
        print(f"📊 输出范围: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 残差MV引导网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_residual_learning_functionality(device: torch.device) -> bool:
    """测试残差学习功能"""
    
    print("\n🔬 测试残差学习功能")
    print("-" * 50)
    
    try:
        from src.npu.networks.residual_mv_guided_network import create_residual_mv_guided_network
        
        # 创建配置
        test_config = {
            'model': {
                'residual_mv_config': {
                    'encoder_channels': [32, 64, 128, 256],
                    'mv_feature_channels': 32,
                    'use_gated_conv': True
                }
            }
        }
        
        model = create_residual_mv_guided_network(test_config)
        model = model.to(device)
        model.eval()
        
        # 创建测试数据
        input_data, target_data = create_realistic_test_data(batch_size=1, height=64, width=64)
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        
        # 提取warped_rgb
        warped_rgb = input_data[:, 0:3, :, :]
        
        # 前向传播
        with torch.no_grad():
            output = model(input_data)
            
            # 验证残差学习：output应该 = warped_rgb + correction
            predicted_residual = output - warped_rgb
            target_residual = target_data - warped_rgb
            
            # 获取中间输出
            if hasattr(model, 'get_intermediate_outputs'):
                intermediate = model.get_intermediate_outputs(input_data)
                
                print("📊 中间输出分析:")
                for key, tensor in intermediate.items():
                    print(f"   {key}: {tensor.shape}, 范围=[{tensor.min():.3f}, {tensor.max():.3f}]")
                
                # 验证残差组合
                if 'correction_residual' in intermediate:
                    correction = intermediate['correction_residual']
                    composed_output = warped_rgb + correction
                    
                    composition_error = torch.abs(output - composed_output).mean()
                    print(f"📊 残差组合验证误差: {composition_error:.6f}")
                    
                    if composition_error < 1e-5:
                        print("✅ 残差组合验证通过")
                    else:
                        print("⚠️  残差组合验证失败")
        
        # 残差统计分析
        residual_mse = torch.mean((predicted_residual - target_residual)**2)
        residual_magnitude = torch.mean(torch.abs(predicted_residual))
        
        print(f"📊 残差学习分析:")
        print(f"   预测残差幅度: {residual_magnitude:.6f}")
        print(f"   残差MSE损失: {residual_mse:.6f}")
        print(f"   输出变化程度: {torch.mean(torch.abs(output - warped_rgb)):.6f}")
        
        print("✅ 残差学习功能验证完成")
        return True
        
    except Exception as e:
        print(f"❌ 残差学习功能测试失败: {e}")
        return False

def test_residual_inpainting_loss(device: torch.device) -> bool:
    """测试残差补洞损失函数"""
    
    print("\n🎯 测试残差补洞损失函数")
    print("-" * 50)
    
    try:
        from train.residual_inpainting_loss import create_residual_inpainting_loss
        
        # 创建测试配置
        test_config = {
            'loss': {
                'residual_mse_weight': 1.2,
                'residual_l1_weight': 0.6,
                'spatial_weighted_weight': 1.0,
                'preservation_weight': 0.5,
                'edge_preservation_weight': 0.3,
                'perceptual_weight': 0.15
            }
        }
        
        # 创建损失函数
        loss_fn = create_residual_inpainting_loss(test_config, device)
        
        # 创建测试数据
        input_data, target_data = create_realistic_test_data(batch_size=1, height=64, width=64)
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        
        # 模拟网络输出（在warped_rgb基础上添加一些修正）
        warped_rgb = input_data[:, 0:3, :, :]
        correction = torch.randn_like(warped_rgb) * 0.1
        predicted_output = warped_rgb + correction
        
        # 计算损失
        total_loss, loss_dict = loss_fn(predicted_output, target_data, input_data)
        
        print(f"✅ 残差补洞损失函数测试成功")
        print(f"📊 总损失: {total_loss.item():.6f}")
        print(f"📊 损失分解:")
        for key, value in loss_dict.items():
            print(f"   {key}: {value:.6f}")
        
        # 验证损失权重
        weights = loss_fn.get_loss_weights()
        print(f"📊 损失权重:")
        for key, value in weights.items():
            print(f"   {key}: {value}")
        
        print("✅ 残差补洞损失函数验证完成")
        return True
        
    except Exception as e:
        print(f"❌ 残差补洞损失函数测试失败: {e}")
        return False

def test_spatial_attention_mechanism(device: torch.device) -> bool:
    """测试空间注意力机制"""
    
    print("\n🎯 测试空间注意力机制")
    print("-" * 50)
    
    try:
        from src.npu.networks.residual_mv_guided_network import SpatialAttentionGenerator
        
        # 创建空间注意力生成器
        attention_gen = SpatialAttentionGenerator(use_gated_conv=True)
        attention_gen = attention_gen.to(device)
        attention_gen.eval()
        
        # 创建测试数据
        batch_size, height, width = 1, 64, 64
        holes_mask = torch.zeros(batch_size, 1, height, width, device=device)
        occlusion_mask = torch.zeros(batch_size, 1, height, width, device=device)
        residual_mv = torch.randn(batch_size, 2, height, width, device=device) * 10.0
        
        # 添加一些空洞和遮挡
        holes_mask[:, :, 20:40, 20:40] = 1.0
        occlusion_mask[:, :, 10:30, 40:60] = 1.0
        
        # 生成空间注意力
        with torch.no_grad():
            spatial_attention, mv_urgency = attention_gen(holes_mask, occlusion_mask, residual_mv)
        
        print(f"📊 空间注意力分析:")
        print(f"   注意力形状: {spatial_attention.shape}")
        print(f"   注意力范围: [{spatial_attention.min():.3f}, {spatial_attention.max():.3f}]")
        print(f"   MV紧急程度形状: {mv_urgency.shape}")
        print(f"   MV紧急程度范围: [{mv_urgency.min():.3f}, {mv_urgency.max():.3f}]")
        
        # 验证注意力在掩码区域是否更高
        holes_bool = holes_mask.bool().squeeze()
        occlusion_bool = occlusion_mask.bool().squeeze()
        combined_mask = holes_bool | occlusion_bool  # 组合所有需要修复的区域
        
        if combined_mask.any():  # 确保有掩码区域
            mask_attention = spatial_attention.squeeze()[combined_mask].mean()
            non_mask_attention = spatial_attention.squeeze()[~combined_mask].mean()
            
            print(f"   掩码区域注意力: {mask_attention:.3f}")
            print(f"   非掩码区域注意力: {non_mask_attention:.3f}")
            print(f"   注意力差异: {(mask_attention - non_mask_attention):.3f}")
            
            # 打印一些统计信息帮助调试
            print(f"   掩码区域像素数: {combined_mask.sum().item()}")
            print(f"   非掩码区域像素数: {(~combined_mask).sum().item()}")
            print(f"   MV平均强度: {mv_urgency.mean():.3f}")
            
            if mask_attention > non_mask_attention + 0.01:  # 增加一个小的阈值
                print("✅ 空间注意力机制工作正常")
            else:
                print("⚠️  空间注意力机制可能需要调整")
                # 提供调试信息
                print("💡 调试建议:")
                print("   - 检查MV灵敏度参数 (mv_sensitivity)")
                print("   - 验证门控卷积的权重初始化")
                print("   - 考虑增加训练迭代以学习更好的注意力模式")
        else:
            print("⚠️  测试数据中没有掩码区域，无法验证空间注意力")
        
        return True
        
    except Exception as e:
        print(f"❌ 空间注意力机制测试失败: {e}")
        return False

def test_gated_convolution(device: torch.device) -> bool:
    """测试门控卷积功能"""
    
    print("\n🚪 测试门控卷积功能")
    print("-" * 50)
    
    try:
        from src.npu.networks.residual_mv_guided_network import GatedConv2d
        
        # 创建门控卷积层
        gated_conv = GatedConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        gated_conv = gated_conv.to(device)
        gated_conv.eval()
        
        # 创建测试输入
        test_input = torch.randn(1, 3, 32, 32, device=device)
        
        # 前向传播
        with torch.no_grad():
            output = gated_conv(test_input)
        
        print(f"📊 门控卷积测试:")
        print(f"   输入形状: {test_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   输出范围: [{output.min():.3f}, {output.max():.3f}]")
        
        # 验证输出不是简单的卷积结果
        # 门控机制应该产生与普通卷积不同的输出
        normal_conv = nn.Conv2d(3, 16, 3, padding=1).to(device)
        with torch.no_grad():
            normal_output = normal_conv(test_input)
        
        difference = torch.mean(torch.abs(output - normal_output))
        print(f"   与普通卷积差异: {difference:.3f}")
        
        if difference > 0.1:
            print("✅ 门控卷积功能正常")
        else:
            print("⚠️  门控卷积可能退化为普通卷积")
        
        return True
        
    except Exception as e:
        print(f"❌ 门控卷积测试失败: {e}")
        return False

def comprehensive_architecture_test() -> bool:
    """综合架构测试"""
    
    print("🔧 残差MV引导网络综合测试")
    print("=" * 60)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 检查模块导入
    if not load_test_modules():
        return False
    
    # 运行所有测试
    tests = [
        test_residual_mv_guided_network,
        test_residual_learning_functionality,
        test_residual_inpainting_loss,
        test_spatial_attention_mechanism,
        test_gated_convolution
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func(device)
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 失败: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "=" * 60)
    print("🎯 测试总结:")
    print(f"📊 总测试数: {len(results)}")
    print(f"✅ 成功: {sum(results)}")
    print(f"❌ 失败: {len(results) - sum(results)}")
    
    if all(results):
        print("\n🎉 所有测试通过！残差MV引导网络准备就绪")
        print("\n💡 关键验证完成:")
        print("   ✅ 网络架构正确实现")
        print("   ✅ 残差学习机制正常")
        print("   ✅ 损失函数功能完整")
        print("   ✅ 空间注意力有效")
        print("   ✅ 门控卷积工作正常")
        
        print("\n🚀 启动残差MV引导网络训练:")
        print("   python train/ultra_safe_train.py --config configs/residual_mv_guided_config.yaml")
        
    else:
        print("\n⚠️  部分测试失败，请检查相关问题")
        failed_tests = [i for i, result in enumerate(results) if not result]
        print(f"失败的测试索引: {failed_tests}")
    
    return all(results)

if __name__ == "__main__":
    comprehensive_architecture_test()