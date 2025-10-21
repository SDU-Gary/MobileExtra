# CMakeLists.txt
# 移动端实时帧外插与空洞补全系统构建脚本
#
# 项目配置：
# - C++17标准支持
# - OpenGL/OpenGL ES集成
# - 跨平台编译支持
# - 依赖库管理
#
# 支持平台：
# - Windows: Visual Studio 2019+
# - Linux: GCC 9+, Clang 10+
# - Android: NDK r25+
# - iOS: Xcode 12+
#
# 依赖库：
# - OpenGL/GLEW: GPU计算
# - GLM: 数学库
# - Eigen3: 线性代数
# - OpenCV: 图像处理(可选)
#
# 构建选项：
# - BUILD_TESTS: 构建测试套件
# - BUILD_TOOLS: 构建评估工具
# - ENABLE_PROFILING: 性能分析支持
# - PLATFORM_ANDROID/IOS: 平台特定构建
#
# @author 构建团队
# @date 2025-07-28
# @version 1.0

cmake_minimum_required(VERSION 3.16)
project(MobileFrameInterpolation VERSION 1.0.0 LANGUAGES CXX)

# TODO: 实现完整的CMake构建脚本
# 包含跨平台编译和依赖管理