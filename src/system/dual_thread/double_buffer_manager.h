/**
 * @file double_buffer_manager.h
 * @brief 双缓冲管理器头文件
 * 
 * 核心功能：
 * - 帧数据双缓冲结构管理
 * - GPU-NPU数据传输优化
 * - PBO异步传输实现
 * - 内存池管理和复用
 * 
 * 缓冲策略：
 * - 前缓冲: 当前处理帧数据
 * - 后缓冲: 下一帧预备数据
 * - 交换机制: 无拷贝指针交换
 * - 状态同步: 原子状态标记
 * 
 * 数据类型：
 * - GPU纹理: RGB, Depth, MV, Mask
 * - NPU张量: Patches, Features, Results
 * - 元数据: 时间戳, 质量指标, 状态
 * 
 * 内存优化：
 * - 预分配内存池
 * - 对齐访问优化
 * - 缓存友好布局
 * - 垃圾回收机制
 * 
 * @author 系统架构团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现DoubleBufferManager类定义
// 包含FrameData结构体和缓冲管理接口