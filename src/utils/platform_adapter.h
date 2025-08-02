/**
 * @file platform_adapter.h
 * @brief 跨平台适配器头文件
 * 
 * 核心功能：
 * - 不同移动平台的硬件抽象
 * - 统一的API接口封装
 * - 平台特定功能适配
 * - 配置文件自动选择
 * 
 * 支持平台：
 * - Android: 高通/联发科/华为/三星
 * - iOS: Apple A系列芯片
 * - 通用: OpenGL ES 3.1+支持
 * 
 * 适配功能：
 * - NPU推理引擎: SNPE/CoreML/NNAPI/APU
 * - GPU计算: OpenGL/Metal Compute
 * - 系统监控: 平台特定API
 * - 文件系统: 沙盒和权限处理
 * 
 * 配置管理：
 * - 自动设备识别
 * - 最优配置选择
 * - 运行时参数调整
 * - 兼容性fallback
 * 
 * @author 平台适配团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现PlatformAdapter类定义
// 包含平台检测和配置选择逻辑