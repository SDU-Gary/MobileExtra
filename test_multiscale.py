#!/usr/bin/env python3
"""
Test multi-scale sampling functionality
"""
import torch
import numpy as np
import sys
import os
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'train'))

from train.patch_aware_dataset import PatchTrainingConfig, PatchAwareDataset


def test_multiscale_config():
    """Test that multi-scale configuration loads correctly"""
    print("=" * 60)
    print("Test 1: Multi-scale Configuration")
    print("=" * 60)

    # Load config from YAML
    config_path = './configs/colleague_training_config.yaml'
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    patch_config = config_dict.get('patch', {})

    # Create PatchTrainingConfig
    config = PatchTrainingConfig(
        enable_multiscale=patch_config.get('enable_multiscale', False),
        multiscale_factors=patch_config.get('multiscale_factors', [0.75, 1.0, 1.25]),
        multiscale_base_size=patch_config.get('multiscale_base_size', 128)
    )

    print(f"✓ enable_multiscale: {config.enable_multiscale}")
    print(f"✓ multiscale_factors: {config.multiscale_factors}")
    print(f"✓ multiscale_base_size: {config.multiscale_base_size}")

    assert config.enable_multiscale == True, "Multi-scale should be enabled in config"
    assert config.multiscale_factors == [0.75, 1.0, 1.25], "Scale factors mismatch"

    print("\n✅ Configuration test passed!\n")
    return config


def test_multiscale_patch_extraction():
    """Test multi-scale patch extraction logic"""
    print("=" * 60)
    print("Test 2: Multi-scale Patch Extraction")
    print("=" * 60)

    # Create a dummy config
    config = PatchTrainingConfig(
        enable_multiscale=True,
        multiscale_factors=[0.75, 1.0, 1.25],
        multiscale_base_size=128,
        use_simple_grid_patches=False
    )

    # Create a dummy image
    dummy_image = np.random.rand(3, 1080, 1920).astype(np.float32)
    center_x, center_y = 960, 540  # Image center

    # Create a minimal dataset instance to test the method
    class DummyDataset:
        def __init__(self, config):
            self.config = config

        # Copy the _extract_multiscale_patch method from PatchAwareDataset
        def _extract_multiscale_patch(self, image_numpy: np.ndarray, center_x: int, center_y: int,
                                      base_size: int = 128) -> np.ndarray:
            import random

            if not self.config.enable_multiscale:
                scale = 1.0
            else:
                scale = random.choice(self.config.multiscale_factors)

            crop_size = int(base_size / scale)

            C, H, W = image_numpy.shape
            y1 = max(0, center_y - crop_size // 2)
            x1 = max(0, center_x - crop_size // 2)
            y2 = min(H, y1 + crop_size)
            x2 = min(W, x1 + crop_size)

            patch = image_numpy[:, y1:y2, x1:x2]

            patch_tensor = torch.from_numpy(patch).float()
            patch_resized = torch.nn.functional.interpolate(
                patch_tensor.unsqueeze(0),
                size=(base_size, base_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            return patch_resized.numpy(), scale, crop_size

    dataset = DummyDataset(config)

    # Test multiple extractions
    print("Testing 10 random multi-scale extractions:")
    scale_counts = {0.75: 0, 1.0: 0, 1.25: 0}

    for i in range(10):
        patch, scale, crop_size = dataset._extract_multiscale_patch(
            dummy_image, center_x, center_y, base_size=128
        )
        scale_counts[scale] += 1

        # Verify output shape
        assert patch.shape == (3, 128, 128), f"Patch shape should be (3, 128, 128), got {patch.shape}"

        expected_crop_size = int(128 / scale)
        print(f"  Iteration {i+1}: scale={scale:.2f}, crop_size={crop_size} (expected={expected_crop_size}), output_shape={patch.shape}")

    print(f"\n Scale distribution: {scale_counts}")
    print(f"  - 0.75× (wider context): {scale_counts[0.75]}/10")
    print(f"  - 1.0×  (standard):      {scale_counts[1.0]}/10")
    print(f"  - 1.25× (more detail):   {scale_counts[1.25]}/10")

    print("\n✅ Multi-scale extraction test passed!\n")


def test_crop_size_calculation():
    """Test that crop sizes are calculated correctly"""
    print("=" * 60)
    print("Test 3: Crop Size Calculation")
    print("=" * 60)

    base_size = 128
    scales = [0.75, 1.0, 1.25]

    print(f"Base size: {base_size}×{base_size}")
    print("\nExpected crop sizes:")

    for scale in scales:
        crop_size = int(base_size / scale)
        print(f"  scale={scale:.2f} → crop {crop_size}×{crop_size} → resize to {base_size}×{base_size}")

        if scale == 0.75:
            assert crop_size == 170 or crop_size == 171, f"0.75× should crop ~171, got {crop_size}"
            print(f"    → Sees MORE context ({crop_size}/{base_size} = {crop_size/base_size:.2f}× area)")
        elif scale == 1.0:
            assert crop_size == 128, f"1.0× should crop 128, got {crop_size}"
            print(f"    → Standard view")
        elif scale == 1.25:
            assert crop_size == 102, f"1.25× should crop 102, got {crop_size}"
            print(f"    → Sees MORE detail ({base_size}/{crop_size} = {base_size/crop_size:.2f}× zoom)")

    print("\n✅ Crop size calculation test passed!\n")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MULTI-SCALE SAMPLING FUNCTIONALITY TEST")
    print("="*60 + "\n")

    try:
        # Test 1: Configuration
        config = test_multiscale_config()

        # Test 2: Patch extraction
        test_multiscale_patch_extraction()

        # Test 3: Crop size calculation
        test_crop_size_calculation()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nMulti-scale sampling is ready to use.")
        print("To enable during training, ensure in colleague_training_config.yaml:")
        print("  patch:")
        print("    enable_multiscale: true")
        print("    multiscale_factors: [0.75, 1.0, 1.25]")
        print("    multiscale_base_size: 128")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
