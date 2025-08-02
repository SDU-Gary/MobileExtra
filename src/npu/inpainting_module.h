/**
 * @file inpainting_module.h
 * @brief NPU补全网络模块头文件
 * 
 * 核心功能：
 * - 接收GPU传输的空洞区域数据
 * - 执行基于Gated Convolution的智能修复
 * - 利用FFC模块进行全局特征处理
 * - 支持批处理和并行推理
 * 
 * 技术约束：
 * - 网络参数: ≤3.0M (U-Net+GatedConv+FFC)
 * - 推理延迟: <4ms (64x64 patch)
 * - 修复质量: SSIM>0.90
 * - 内存占用: <100MB
 * 
 * 支持特性：
 * - INT8量化部署
 * - 多平台NPU适配
 * - 动态批处理大小
 * - 置信度评估
 * 
 * @author AI算法团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现InpaintingModule类定义
// 参考design.md中的InpaintingModule接口设计