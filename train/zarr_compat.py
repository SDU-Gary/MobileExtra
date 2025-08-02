#!/usr/bin/env python3
"""
@file zarr_compat.py
@brief Zarr版本兼容性处理

功能描述：
- 处理不同版本Zarr的API差异
- 提供统一的ZipStore接口
- 兼容Zarr 2.x和3.x版本

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import zipfile
import tempfile
import os
import shutil
from pathlib import Path
import zarr
import numpy as np


def get_zarr_version():
    """获取Zarr版本"""
    try:
        return zarr.__version__
    except AttributeError:
        return "unknown"


def create_zip_store(zip_path, mode='r'):
    """
    创建兼容的ZipStore
    
    Args:
        zip_path: zip文件路径
        mode: 打开模式
        
    Returns:
        store: Zarr存储对象
    """
    zarr_version = get_zarr_version()
    
    # 确保使用正确的只读模式
    if mode in ['r+', 'w', 'a']:
        mode = 'r'  # zip文件通常是只读的
    
    try:
        # 尝试Zarr 2.x的方式
        from zarr import ZipStore
        store = ZipStore(zip_path, mode=mode)
        print(f"✅ 使用Zarr 2.x ZipStore加载: {zip_path}")
        return store
    except ImportError:
        try:
            # 尝试Zarr 3.x的方式
            from zarr.storage import ZipStore
            store = ZipStore(zip_path, mode=mode)
            print(f"✅ 使用Zarr 3.x ZipStore加载: {zip_path}")
            return store
        except ImportError:
            # 如果都不行，使用自定义实现
            print(f"⚠️ 使用自定义ZipStore实现: {zip_path}")
            return CustomZipStore(zip_path, mode=mode)
    except Exception as e:
        print(f"⚠️ 标准ZipStore失败，使用自定义实现: {e}")
        return CustomZipStore(zip_path, mode=mode)


class CustomZipStore:
    """自定义ZipStore实现，兼容不同Zarr版本"""
    
    def __init__(self, zip_path, mode='r'):
        """
        初始化自定义ZipStore
        
        Args:
            zip_path: zip文件路径
            mode: 打开模式
        """
        self.zip_path = str(zip_path)
        self.mode = mode
        self._temp_dir = None
        self._extracted_files = {}
        
        if mode == 'r':
            self._extract_zip()
    
    def _extract_zip(self):
        """提取zip文件到临时目录"""
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")
        
        # 创建临时目录
        self._temp_dir = tempfile.mkdtemp()
        
        # 提取zip文件
        with zipfile.ZipFile(self.zip_path, 'r') as zip_file:
            zip_file.extractall(self._temp_dir)
        
        print(f"提取zip文件到: {self._temp_dir}")
    
    def __getitem__(self, key):
        """获取数据项"""
        if self._temp_dir is None:
            raise RuntimeError("Store not properly initialized")
        
        # 查找文件
        file_path = Path(self._temp_dir) / key
        
        if file_path.exists():
            # 读取二进制数据
            with open(file_path, 'rb') as f:
                return f.read()
        else:
            raise KeyError(f"Key not found: {key}")
    
    def __contains__(self, key):
        """检查是否包含key"""
        if self._temp_dir is None:
            return False
        
        file_path = Path(self._temp_dir) / key
        return file_path.exists()
    
    def keys(self):
        """获取所有key"""
        if self._temp_dir is None:
            return []
        
        keys = []
        for root, dirs, files in os.walk(self._temp_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), self._temp_dir)
                keys.append(rel_path.replace('\\', '/'))  # 使用正斜杠
        
        return keys
    
    def __del__(self):
        """清理临时文件"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass  # 静默失败


def load_zarr_group(zip_path):
    """
    简化的Zarr组加载方法
    
    Args:
        zip_path: zip文件路径
        
    Returns:
        group: Zarr组对象
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"文件不存在: {zip_path}")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 解压zip文件
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(temp_dir)
        
        # 直接从目录加载zarr
        group = zarr.open_group(temp_dir, mode='r')
        
        # 简单验证
        if not _validate_zarr_group(group):
            raise ValueError(f"Zarr数据验证失败: {zip_path}")
        
        print(f"✅ Zarr加载成功: {Path(zip_path).name}")
        return group
        
    except Exception as e:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"Zarr加载失败: {zip_path}, 错误: {e}")


def _validate_zarr_group(group):
    """验证zarr组是否包含必要的数据"""
    required_keys = ['color', 'position', 'reference']
    
    try:
        if hasattr(group, 'keys'):
            available_keys = list(group.keys())
        else:
            # 尝试通过属性访问
            available_keys = [key for key in required_keys if hasattr(group, key)]
        
        missing_keys = [key for key in required_keys if key not in available_keys and not hasattr(group, key)]
        
        if missing_keys:
            print(f"   缺少必要键: {missing_keys}, 可用: {available_keys}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   验证失败: {e}")
        return False


def _load_with_zipstore(zip_path):
    """使用ZipStore加载"""
    store = create_zip_store(zip_path, mode='r')
    group = zarr.group(store=store)
    
    # 调试：打印组结构
    debug_zarr_structure(group, zip_path)
    
    return group


def _load_with_direct_zip(zip_path):
    """直接解压zip文件加载"""
    import tempfile
    import shutil
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 解压zip文件
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(temp_dir)
        
        # 直接从目录加载zarr
        group = zarr.open_group(temp_dir, mode='r')
        
        # 调试：打印组结构
        debug_zarr_structure(group, zip_path)
        
        return group
    except Exception as e:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e


def debug_zarr_structure(group, zip_path):
    """调试zarr组结构"""
    try:
        print(f"🔍 调试zarr结构 ({zip_path}):")
        
        if hasattr(group, 'keys'):
            keys = list(group.keys())
            print(f"   组键: {keys}")
            
            # 检查每个键的类型
            for key in keys[:5]:  # 只检查前5个
                try:
                    item = group[key]
                    if hasattr(item, 'shape'):
                        print(f"   {key}: 数组 {item.shape} {item.dtype}")
                    else:
                        print(f"   {key}: {type(item)}")
                except Exception as e:
                    print(f"   {key}: 访问失败 - {e}")
        else:
            print(f"   组类型: {type(group)}")
            print(f"   组属性: {dir(group)}")
            
    except Exception as e:
        print(f"   调试失败: {e}")


def load_zarr_fallback(zip_path):
    """
    备用的Zarr加载方案
    
    Args:
        zip_path: zip文件路径
        
    Returns:
        group: 模拟的Zarr组对象
    """
    print(f"使用备用方案加载: {zip_path}")
    
    # 创建临时目录并提取文件
    temp_dir = tempfile.mkdtemp()
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(temp_dir)
    
    # 尝试直接从目录加载
    try:
        return zarr.open_group(temp_dir, mode='r')
    except Exception:
        # 如果还是失败，返回一个模拟对象
        return FallbackZarrGroup(temp_dir)


class FallbackZarrGroup:
    """备用的Zarr组实现"""
    
    def __init__(self, base_path):
        """
        初始化备用Zarr组
        
        Args:
            base_path: 数据基础路径
        """
        self.base_path = Path(base_path)
        self._arrays = {}
        self._scan_arrays()
    
    def _scan_arrays(self):
        """扫描数组文件"""
        # 查找.zarr目录或直接的数组文件
        for item in self.base_path.rglob('*'):
            if item.is_file() and item.suffix in ['.zarr', '.dat', '.npy']:
                # 简化的数组名
                rel_path = item.relative_to(self.base_path)
                array_name = str(rel_path.with_suffix('').as_posix())
                
                try:
                    if item.suffix == '.npy':
                        self._arrays[array_name] = np.load(item)
                    else:
                        # 尝试加载为zarr数组
                        self._arrays[array_name] = zarr.open_array(str(item), mode='r')
                except Exception as e:
                    print(f"无法加载数组 {array_name}: {e}")
    
    def __getattr__(self, name):
        """获取数组属性"""
        if name in self._arrays:
            return self._arrays[name]
        
        # 尝试动态加载
        array_path = self.base_path / f"{name}.zarr"
        if array_path.exists():
            try:
                array = zarr.open_array(str(array_path), mode='r')
                self._arrays[name] = array
                return array
            except Exception:
                pass
        
        # 尝试.npy文件
        npy_path = self.base_path / f"{name}.npy"
        if npy_path.exists():
            try:
                array = np.load(npy_path)
                self._arrays[name] = array
                return array
            except Exception:
                pass
        
        raise AttributeError(f"数组不存在: {name}")
    
    def keys(self):
        """获取所有数组名"""
        return list(self._arrays.keys())


# 便捷函数
def decompress_RGBE_compat(color, exposures):
    """
    简化版RGBE解压缩
    
    Args:
        color: RGBE颜色数据 [4, H, W] 或 [4, H, W, S]
        exposures: 曝光参数 [2]
        
    Returns:
        rgb: RGB颜色数据 [3, H, W]
    """
    try:
        # 确保输入是numpy数组
        color = np.array(color, dtype=np.float32)
        exposures = np.array(exposures, dtype=np.float32)
        
        print(f"🎨 RGBE解压缩: color形状={color.shape}")
        
        # 处理样本维度（如果存在）
        if color.ndim == 4 and color.shape[-1] > 1:
            color = color.mean(axis=-1)  # 对样本求平均
        elif color.ndim == 4 and color.shape[-1] == 1:
            color = color.squeeze(axis=-1)
        
        # 检查维度
        if color.ndim != 3 or color.shape[0] < 4:
            raise ValueError(f"期望color形状为[4,H,W]，实际: {color.shape}")
        
        # 提取E通道并计算指数
        e_channel = color[3]
        exponents = np.exp((e_channel / 255.0) * (exposures[1] - exposures[0]) + exposures[0])
        
        # 应用到RGB通道
        rgb = color[:3] / 255.0
        rgb = rgb * exponents[np.newaxis, :, :]
        
        print(f"🎨 解压缩成功: {rgb.shape}")
        return rgb
        
    except Exception as e:
        print(f"❌ RGBE解压缩失败: {e}")
        # 返回安全默认值
        if hasattr(color, 'shape') and len(color.shape) >= 2:
            H, W = color.shape[-2], color.shape[-1]
            return np.zeros((3, H, W), dtype=np.float32)
        else:
            return np.zeros((3, 64, 64), dtype=np.float32)


if __name__ == "__main__":
    # 测试兼容性
    print(f"Zarr版本: {get_zarr_version()}")
    
    # 测试ZipStore创建
    try:
        # 创建一个测试zip store（这会失败，但可以看到错误信息）
        print("测试ZipStore创建...")
        store = create_zip_store("test.zip", mode='r')
        print("✅ ZipStore创建成功")
    except Exception as e:
        print(f"⚠️ ZipStore测试失败（预期）: {e}")
    
    print("兼容性模块加载完成")