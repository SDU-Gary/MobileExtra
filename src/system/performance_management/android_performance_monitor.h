/**
 * @file android_performance_monitor.h
 * @brief Android平台性能监控器头文件
 * 
 * 核心功能：
 * - Android Thermal API集成
 * - PowerManager电源管理
 * - ActivityManager内存监控
 * - GPU性能计数器访问
 * 
 * 监控能力：
 * - CPU温度: /sys/class/thermal/
 * - GPU温度: vendor specific APIs
 * - 电池状态: BatteryManager
 * - 内存使用: /proc/meminfo
 * - CPU使用率: /proc/stat
 * 
 * 权限要求：
 * - DEVICE_POWER: 电池信息
 * - SYSTEM_ALERT_WINDOW: 温度访问
 * - READ_EXTERNAL_STORAGE: 系统文件
 * 
 * 兼容性：
 * - Android 7.0+ (API 24)
 * - 主流厂商适配: 三星/华为/小米/OPPO
 * - SOC支持: 高通/联发科/华为/三星
 * 
 * @author 系统监控团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现AndroidPerformanceMonitor类定义
// 包含Android特定的监控接口