/**
 * @file warp_module.h
 * @brief GPU前向投影模块头文件
 * 
 * 核心功能：
 * - 基于运动矢量执行前向Splatting投影
 * - 解决像素冲突并生成空洞掩码
 * - 计算投影残差用于后续校正
 * - 实现统一空洞检测（无分类）
 * 
 * 性能约束：
 * - 处理延迟: <1ms (1080p) - 关键约束
 * - 内存占用: <50MB GPU显存
 * - 像素冲突解决准确率: >98%
 * 
 * 技术特点：
 * - 原子操作深度测试
 * - 最近帧优先原则
 * - 双帧MV融合
 * - 统一空洞检测和残差计算
 * 
 * @author GPU计算团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现WarpModule类定义
// 参考design.md中的WarpModule接口设计