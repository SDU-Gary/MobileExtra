/**
 * @file gbuffer_renderer.h
 * @brief OpenGL G-Buffer渲染器头文件
 * 
 * 负责构建简单但完整的3D渲染管线，生成插帧所需的RGB、深度、运动矢量数据。
 * 实现多重渲染目标(MRT)输出和低分辨率深度缓冲优化技术。
 * 
 * 主要功能：
 * - G-Buffer多重渲染目标管理
 * - 运动矢量计算和输出
 * - 低分辨率深度生成
 * - 帧缓冲管理和交换
 * 
 * 性能要求：
 * - 渲染延迟: <2ms (1080p)
 * - 内存占用: <100MB GPU显存
 * 
 * @author 图形引擎团队
 * @date 2025-07-28
 * @version 1.0
 */

#pragma once

// TODO: 实现GBufferRenderer类定义
// 参考design.md中的GBufferRenderer接口设计