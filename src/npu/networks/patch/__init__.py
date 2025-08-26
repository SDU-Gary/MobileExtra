#!/usr/bin/env python3
"""
Patch-Based架构模块 - 模块初始化

导出核心组件，提供统一的接口

作者：AI算法团队
日期：2025-08-24
"""

from .hole_detector import (
    HoleDetector,
    HoleDetectorConfig,
    PatchInfo
)

from .patch_extractor import (
    PatchExtractor,
    PatchExtractorConfig,
    PatchPosition,
    PaddingMode
)

from .patch_network import (
    PatchNetwork,
    PatchGatedConv2d,
    PatchGatedConvBlock,
    PatchFFCBlock
)

__all__ = [
    # HoleDetector
    'HoleDetector',
    'HoleDetectorConfig', 
    'PatchInfo',
    
    # PatchExtractor
    'PatchExtractor',
    'PatchExtractorConfig',
    'PatchPosition',
    'PaddingMode',
    
    # PatchNetwork
    'PatchNetwork',
    'PatchGatedConv2d',
    'PatchGatedConvBlock', 
    'PatchFFCBlock',
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'AI算法团队'