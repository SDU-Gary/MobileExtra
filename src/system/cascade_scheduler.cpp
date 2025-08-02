/**
 * @file cascade_scheduler.cpp
 * @brief 级联调度器实现文件
 * 
 * 实现内外插融合的完整调度逻辑：
 * - 帧请求优先级队列管理
 * - 5状态调度策略执行
 * - 性能监控集成和决策
 * - 动态策略切换机制
 * 
 * 调度策略：
 * - 最优模式: t+1, t+2外插 + t+0.5, t+1.5内插
 * - 高性能: 外插优先，空余时间内插
 * - 平衡模式: 双帧外插(t+1, t+2)
 * - 保守模式: 单帧外插(t+1)
 * - 紧急模式: 禁用插帧
 * 
 * 容错机制：
 * - NPU超时处理
 * - 队列溢出保护
 * - 状态异常恢复
 * 
 * @author 系统架构团队
 * @date 2025-07-28
 * @version 1.0
 */

#include "cascade_scheduler.h"

// TODO: 实现CascadeScheduler类方法
// 重点实现ScheduleFrame和UpdateStrategy方法