#!/usr/bin/env python3
"""
Zarr version compatibility layer for handling API differences between versions.
"""

import zipfile
import tempfile
import os
import shutil
from pathlib import Path
import zarr
import numpy as np


def get_zarr_version():
    """Get current Zarr version"""
    try:
        return zarr.__version__
    except AttributeError:
        return "unknown"


def create_zip_store(zip_path, mode='r'):
    """Create compatible ZipStore for different Zarr versions"""
    zarr_version = get_zarr_version()
    
    # Force read-only mode for zip files
    if mode in ['r+', 'w', 'a']:
        mode = 'r'
    
    try:
        # Try Zarr 2.x approach
        from zarr import ZipStore
        store = ZipStore(zip_path, mode=mode)
        print(f"[INFO] Using Zarr 2.x ZipStore: {zip_path}")
        return store
    except ImportError:
        try:
            # Try Zarr 3.x approach
            from zarr.storage import ZipStore
            store = ZipStore(zip_path, mode=mode)
            print(f"[INFO] Using Zarr 3.x ZipStore: {zip_path}")
            return store
        except ImportError:
            # Fallback to custom implementation
            print(f"[WARN] Using custom ZipStore fallback: {zip_path}")
            return CustomZipStore(zip_path, mode=mode)
    except Exception as e:
        print(f"[WARN] Standard ZipStore failed, using fallback: {e}")
        return CustomZipStore(zip_path, mode=mode)


class CustomZipStore:
    """Custom ZipStore implementation compatible with different Zarr versions"""
    
    def __init__(self, zip_path, mode='r'):
        """Initialize custom ZipStore"""
        self.zip_path = str(zip_path)
        self.mode = mode
        self._temp_dir = None
        self._extracted_files = {}
        
        if mode == 'r':
            self._extract_zip()
    
    def _extract_zip(self):
        """Extract zip file to temporary directory"""
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")
        
        # Create temporary directory and extract
        self._temp_dir = tempfile.mkdtemp()
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_file:
            zip_file.extractall(self._temp_dir)
        
        print(f"Extracted to: {self._temp_dir}")
    
    def __getitem__(self, key):
        """Get data item by key"""
        if self._temp_dir is None:
            raise RuntimeError("Store not properly initialized")
        
        file_path = Path(self._temp_dir) / key
        
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return f.read()
        else:
            raise KeyError(f"Key not found: {key}")
    
    def __contains__(self, key):
        """Check if key exists"""
        if self._temp_dir is None:
            return False
        
        file_path = Path(self._temp_dir) / key
        return file_path.exists()
    
    def keys(self):
        """Get all available keys"""
        if self._temp_dir is None:
            return []
        
        keys = []
        for root, dirs, files in os.walk(self._temp_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), self._temp_dir)
                keys.append(rel_path.replace('\\', '/'))  # Use forward slashes
        
        return keys
    
    def __del__(self):
        """Cleanup temporary files"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass  # Silent cleanup


class ManagedZarrGroup:
    """Managed Zarr group wrapper with temporary directory cleanup"""
    
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.temp_dir = None
        self.group = None
        self._load_group()
    
    def _load_group(self):
        """Load zarr group and create temporary directory"""
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"File not found: {self.zip_path}")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract zip file
            with zipfile.ZipFile(self.zip_path, 'r') as zip_file:
                zip_file.extractall(self.temp_dir)
            
            # Load zarr from directory
            self.group = zarr.open_group(self.temp_dir, mode='r')
            
            # Simple validation
            if not _validate_zarr_group(self.group):
                raise ValueError(f"Zarr validation failed: {self.zip_path}")
            
            print(f"[SUCCESS] Zarr loaded successfully: {Path(self.zip_path).name}")
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Zarr loading failed: {self.zip_path}, error: {e}")
    
    def cleanup(self):
        """Immediately cleanup temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"[CLEANUP] Cleaned temporary directory: {os.path.basename(self.temp_dir)}")
            except Exception as e:
                print(f"[WARN] Cleanup failed: {e}")
            finally:
                self.temp_dir = None
                self.group = None
    
    def __getattr__(self, name):
        """Proxy to zarr group object"""
        if self.group is None:
            raise RuntimeError("Zarr group not initialized or already cleaned")
        return getattr(self.group, name)
    
    def __getitem__(self, key):
        """Proxy to zarr group object"""
        if self.group is None:
            raise RuntimeError("Zarr group not initialized or already cleaned")
        return self.group[key]
    
    def __contains__(self, key):
        """Proxy to zarr group object"""
        if self.group is None:
            raise RuntimeError("Zarr group not initialized or already cleaned")
        return key in self.group
    
    def keys(self):
        """Proxy to zarr group object"""
        if self.group is None:
            raise RuntimeError("Zarr group not initialized or already cleaned")
        return self.group.keys()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        self.cleanup()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


def load_zarr_group(zip_path):
    """Load Zarr group (compatibility function)
    
    Args:
        zip_path: Path to zip file
        
    Returns:
        ManagedZarrGroup: Managed Zarr group object
        
    Warning:
        Returned object requires cleanup() call or use with statement
    """
    return ManagedZarrGroup(zip_path)


def _validate_zarr_group(group):
    """Validate zarr group contains necessary data"""
    # Basic channels - need at least some of these
    basic_channels = ['color', 'position', 'motion', 'normal', 'diffuse', 'reference']
    
    try:
        if hasattr(group, 'keys'):
            available_keys = list(group.keys())
        else:
            # Try attribute access
            available_keys = [key for key in basic_channels if hasattr(group, key)]
        
        print(f"   Available channels: {available_keys}")
        
        # Check for minimum basic channels
        found_channels = [key for key in basic_channels if key in available_keys or hasattr(group, key)]
        
        if len(found_channels) < 2:  # Need at least 2 channels
            print(f"   Warning: Only found {len(found_channels)} basic channels: {found_channels}")
            return False
        
        print(f"   Validation successful: found {len(found_channels)} valid channels")
        return True
        
    except Exception as e:
        print(f"   Validation failed: {e}")
        return False


def _load_with_zipstore(zip_path):
    """Load using ZipStore"""
    store = create_zip_store(zip_path, mode='r')
    group = zarr.group(store=store)
    
    # Debug: print group structure
    debug_zarr_structure(group, zip_path)
    
    return group


def _load_with_direct_zip(zip_path):
    """Load by directly extracting zip file"""
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(temp_dir)
        
        # Load zarr directly from directory
        group = zarr.open_group(temp_dir, mode='r')
        
        # Debug: print group structure
        debug_zarr_structure(group, zip_path)
        
        return group
    except Exception as e:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e


def debug_zarr_structure(group, zip_path):
    """Debug zarr group structure"""
    try:
        print(f"[DEBUG] Debug zarr structure ({zip_path}):")
        
        if hasattr(group, 'keys'):
            keys = list(group.keys())
            print(f"   Group keys: {keys}")
            
            # Check type of each key
            for key in keys[:5]:  # Only check first 5
                try:
                    item = group[key]
                    if hasattr(item, 'shape'):
                        print(f"   {key}: array {item.shape} {item.dtype}")
                    else:
                        print(f"   {key}: {type(item)}")
                except Exception as e:
                    print(f"   {key}: access failed - {e}")
        else:
            print(f"   Group type: {type(group)}")
            print(f"   Group attributes: {dir(group)}")
            
    except Exception as e:
        print(f"   Debug failed: {e}")


def load_zarr_fallback(zip_path):
    """Fallback Zarr loading method"""
    print(f"Using fallback method: {zip_path}")
    
    temp_dir = tempfile.mkdtemp()
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(temp_dir)
    
    try:
        return zarr.open_group(temp_dir, mode='r')
    except Exception:
        return FallbackZarrGroup(temp_dir)


class FallbackZarrGroup:
    """Fallback Zarr group implementation"""
    
    def __init__(self, base_path):
        """Initialize fallback Zarr group"""
        self.base_path = Path(base_path)
        self._arrays = {}
        self._scan_arrays()
    
    def _scan_arrays(self):
        """Scan array files"""
        # Find .zarr directories or array files
        for item in self.base_path.rglob('*'):
            if item.is_file() and item.suffix in ['.zarr', '.dat', '.npy']:
                rel_path = item.relative_to(self.base_path)
                array_name = str(rel_path.with_suffix('').as_posix())
                
                try:
                    if item.suffix == '.npy':
                        self._arrays[array_name] = np.load(item)
                    else:
                        self._arrays[array_name] = zarr.open_array(str(item), mode='r')
                except Exception as e:
                    print(f"Cannot load array {array_name}: {e}")
    
    def __getattr__(self, name):
        """Get array attribute"""
        if name in self._arrays:
            return self._arrays[name]
        
        # Try dynamic loading
        array_path = self.base_path / f"{name}.zarr"
        if array_path.exists():
            try:
                array = zarr.open_array(str(array_path), mode='r')
                self._arrays[name] = array
                return array
            except Exception:
                pass
        
        npy_path = self.base_path / f"{name}.npy"
        if npy_path.exists():
            try:
                array = np.load(npy_path)
                self._arrays[name] = array
                return array
            except Exception:
                pass
        
        raise AttributeError(f"Array not found: {name}")
    
    def keys(self):
        """Get all array names"""
        return list(self._arrays.keys())


# Utility functions
def decompress_RGBE_compat(color, exposures):
    """Simplified RGBE decompression for HDR data"""
    try:
        # Ensure inputs are numpy arrays
        color = np.array(color, dtype=np.float32)
        exposures = np.array(exposures, dtype=np.float32)
        
        print(f"[INFO] RGBE decompression: color shape={color.shape}")
        
        # Handle sample dimension
        if color.ndim == 4 and color.shape[-1] > 1:
            color = color.mean(axis=-1)
        elif color.ndim == 4 and color.shape[-1] == 1:
            color = color.squeeze(axis=-1)
        
        if color.ndim != 3 or color.shape[0] < 4:
            raise ValueError(f"Expected color shape [4,H,W], got: {color.shape}")
        
        # Extract E channel and compute exponents
        e_channel = color[3]
        exponents = np.exp((e_channel / 255.0) * (exposures[1] - exposures[0]) + exposures[0])
        
        # Apply to RGB channels
        rgb = color[:3] / 255.0
        rgb = rgb * exponents[np.newaxis, :, :]
        
        print(f"[SUCCESS] Decompression successful: {rgb.shape}")
        return rgb
        
    except Exception as e:
        print(f"[ERROR] RGBE decompression failed: {e}")
        # Return safe default
        if hasattr(color, 'shape') and len(color.shape) >= 2:
            H, W = color.shape[-2], color.shape[-1]
            return np.zeros((3, H, W), dtype=np.float32)
        else:
            return np.zeros((3, 64, 64), dtype=np.float32)


if __name__ == "__main__":
    # Test compatibility
    print(f"Zarr version: {get_zarr_version()}")
    
    # Test ZipStore creation
    try:
        print("Testing ZipStore creation...")
        store = create_zip_store("test.zip", mode='r')
        print("[SUCCESS] ZipStore created successfully")
    except Exception as e:
        print(f"[WARN] ZipStore test failed (expected): {e}")
    
    print("Compatibility module loaded successfully")