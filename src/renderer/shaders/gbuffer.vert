/**
 * @file gbuffer.vert
 * @brief G-Buffer顶点着色器
 * 
 * 功能描述：
 * - 顶点变换到裁剪空间
 * - 计算当前帧和上一帧的裁剪空间位置
 * - 输出世界坐标、法线、纹理坐标
 * - 为运动矢量计算提供位置信息
 * 
 * 输入属性：
 * - a_Position: 顶点位置 (vec3)
 * - a_Normal: 顶点法线 (vec3)  
 * - a_TexCoord: 纹理坐标 (vec2)
 * 
 * 输出变量：
 * - v_WorldPos: 世界坐标位置
 * - v_Normal: 变换后法线
 * - v_TexCoord: 纹理坐标
 * - v_CurrentClipPos: 当前帧裁剪空间位置
 * - v_PreviousClipPos: 上一帧裁剪空间位置
 * 
 * @author 图形引擎团队
 * @date 2025-07-28
 * @version 1.0
 */

#version 450 core

// TODO: 实现完整的顶点着色器
// 参考design.md中的gbuffer.vert实现