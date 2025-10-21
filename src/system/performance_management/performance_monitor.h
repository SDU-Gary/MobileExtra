/**
 * @file performance_monitor.h
 * @brief 性能监控器头文件
 * 
 * 核心功能：
 * - 实时监控系统资源使用
 * - 跨平台硬件指标采集
 * - 智能预警和阈值检测
 * - 为降级决策提供数据支持
 * 
 * 监控指标：
 * - 热管理: 设备/CPU/GPU温度
 * - 电源管理: 电池电量/功耗/低功耗模式
 * - 计算资源: NPU/GPU/CPU利用率
 * - 内存带宽: GPU显存/系统内存/带宽利用率
 * - 性能指标: 帧延迟/平均FPS/丢帧统计
 * 
 * 平台支持：
 * - Android: Thermal API, PowerManager
 * - iOS: IOKit, Metal Performance
 * - 通用: CPU/GPU性能计数器
 * 
 * @author 系统监控团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现PerformanceManager类定义
// 包含SystemMetrics结构体和跨平台监控接口