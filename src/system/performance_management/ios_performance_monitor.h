/**
 * @file ios_performance_monitor.h
 * @brief iOS平台性能监控器头文件
 * 
 * 核心功能：
 * - IOKit框架系统信息获取
 * - Metal Performance Shaders监控
 * - ProcessInfo系统状态检测
 * - 电池和温度监控
 * 
 * 监控能力：
 * - 设备温度: IOKit thermal services
 * - 电池状态: UIDevice batteryLevel
 * - 内存使用: mach task_info
 * - CPU使用率: host_processor_info
 * - GPU利用率: Metal性能计数器
 * 
 * 系统集成：
 * - 低功耗模式检测
 * - 热保护状态监控  
 * - App状态变化响应
 * - 后台运行优化
 * 
 * 兼容性：
 * - iOS 13.0+ 
 * - 支持所有A系列芯片
 * - iPhone/iPad全系列
 * 
 * @author 系统监控团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现iOSPerformanceMonitor类定义
// 包含iOS特定的监控接口和IOKit集成