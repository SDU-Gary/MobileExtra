/**
 * @file gbuffer.frag
 * @brief G-Buffer片元着色器
 * 
 * 功能描述：
 * - 计算简单Phong光照模型
 * - 输出RGB颜色到MRT目标0
 * - 输出线性深度到MRT目标1
 * - 计算并输出运动矢量到MRT目标2
 * - 存储上一帧世界坐标到MRT目标3
 * 
 * 输入变量：
 * - v_WorldPos: 世界坐标位置
 * - v_Normal: 插值后法线
 * - v_TexCoord: 纹理坐标
 * - v_CurrentClipPos: 当前帧裁剪空间位置
 * - v_PreviousClipPos: 上一帧裁剪空间位置
 * 
 * 输出目标：
 * - o_RGB: RGBA8 格式RGB颜色
 * - o_Depth: R32F 格式线性深度
 * - o_MotionVector: RG16F 格式运动矢量
 * - o_PreviousPos: RGB32F 格式上一帧坐标
 * 
 * @author 图形引擎团队
 * @date 2025-07-28
 * @version 1.0
 */

#version 450 core

// TODO: 实现完整的片元着色器
// 参考design.md中的gbuffer.frag实现