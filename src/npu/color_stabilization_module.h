/**
 * @file color_stabilization_module.h
 * @brief 色彩稳定模块头文件
 * 
 * 核心功能：
 * - 生成时序一致的3D LUT参数
 * - 抑制帧间色彩闪烁和漂移
 * - 实现轻量级跨帧色彩稳定
 * - 支持全局色调映射
 * 
 * 技术特点：
 * - 8x8x8紧凑LUT设计 (1536参数)
 * - 时序GRU建模
 * - 分层自适应LUT策略
 * - 时间一致性正则化
 * 
 * 性能指标：
 * - 网络参数: ≤0.7M
 * - LUT生成延迟: <1ms
 * - 色彩稳定性提升: ≥30%
 * - 参数减少: 89.6% vs 17³ LUT
 * 
 * @author AI算法团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现ColorStabilizationModule类定义
// 包含LUT生成和时序建模功能