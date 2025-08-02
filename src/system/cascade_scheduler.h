/**
 * @file cascade_scheduler.h
 * @brief 级联调度器头文件
 * 
 * 核心功能：
 * - 管理内外插融合策略
 * - 实现多帧并行生成
 * - 动态性能调整
 * - 5级降级策略控制
 * 
 * 调度状态：
 * - OPTIMAL_MODE: 外插+内插全开
 * - HIGH_PERFORMANCE: 外插优先，选择性内插
 * - BALANCED_MODE: 仅双帧外插
 * - CONSERVATIVE_MODE: 单帧外插
 * - EMERGENCY_MODE: 停用插帧
 * 
 * 性能指标：
 * - 状态切换延迟: <0.1ms
 * - 降级触发准确率: >99%
 * - 多帧生成效率: ≥2倍提升
 * 
 * @author 系统架构团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现CascadeScheduler类定义
// 参考design.md中的级联调度器设计