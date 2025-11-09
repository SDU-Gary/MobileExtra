#!/usr/bin/env python3
"""
Test wavelet loss functionality
"""
import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'train'))

from residual_inpainting_loss import ResidualInpaintingLoss, PTWT_AVAILABLE


def test_wavelet_loss_disabled():
    """Test 1: Wavelet loss disabled (default behavior)"""
    print("=" * 60)
    print("Test 1: Wavelet Loss Disabled (Default)")
    print("=" * 60)

    device = torch.device('cpu')

    # Config with wavelet disabled
    config = {
        'loss': {
            'wavelet': {
                'enable': False
            }
        }
    }

    loss_fn = ResidualInpaintingLoss(device, config)

    # Test data
    pred = torch.randn(2, 3, 128, 128)
    target = torch.randn(2, 3, 128, 128)
    input_data = torch.randn(2, 7, 128, 128)

    # Compute loss
    total_loss, loss_dict = loss_fn(pred, target, input_data)

    print(f"✓ Wavelet enabled: {loss_fn.wavelet_enabled}")
    print(f"✓ Wavelet loss: {loss_dict['wavelet']:.6f}")
    assert loss_dict['wavelet'] == 0.0, "Wavelet loss should be 0 when disabled"
    print("✅ Test passed: Wavelet loss is 0 when disabled\n")


def test_wavelet_loss_enabled():
    """Test 2: Wavelet loss enabled"""
    print("=" * 60)
    print("Test 2: Wavelet Loss Enabled")
    print("=" * 60)

    if not PTWT_AVAILABLE:
        print("⚠️  ptwt not available, skipping wavelet enabled test")
        return

    device = torch.device('cpu')

    # Config with wavelet enabled
    config = {
        'loss': {
            'wavelet': {
                'enable': True,
                'wavelet_type': 'haar',
                'level': 1,
                'low_freq_weight': 1.0,
                'high_freq_weight': 0.5
            },
            'weights': {
                'wavelet': 1.0  # Enable wavelet weight
            }
        }
    }

    loss_fn = ResidualInpaintingLoss(device, config)

    # Test data
    pred = torch.randn(2, 3, 128, 128)
    target = torch.randn(2, 3, 128, 128)
    input_data = torch.randn(2, 7, 128, 128)

    # Compute loss
    total_loss, loss_dict = loss_fn(pred, target, input_data)

    print(f"✓ Wavelet enabled: {loss_fn.wavelet_enabled}")
    print(f"✓ Wavelet type: {loss_fn.wavelet_type}")
    print(f"✓ Wavelet level: {loss_fn.wavelet_level}")
    print(f"✓ Low-freq weight: {loss_fn.wavelet_low_freq_weight}")
    print(f"✓ High-freq weight: {loss_fn.wavelet_high_freq_weight}")
    print(f"✓ Wavelet loss: {loss_dict['wavelet']:.6f}")

    assert loss_dict['wavelet'] > 0.0, "Wavelet loss should be > 0 when enabled"
    print("✅ Test passed: Wavelet loss computed successfully\n")


def test_wavelet_decomposition():
    """Test 3: Verify wavelet decomposition works correctly"""
    print("=" * 60)
    print("Test 3: Wavelet Decomposition Verification")
    print("=" * 60)

    if not PTWT_AVAILABLE:
        print("⚠️  ptwt not available, skipping decomposition test")
        return

    import ptwt

    # Create test image
    test_image = torch.randn(1, 3, 128, 128)

    # Decompose
    low_freq, high_freq = ptwt.wavedec2(test_image, 'haar', level=1, mode='reflect')

    print(f"✓ Input shape: {test_image.shape}")
    print(f"✓ Low-freq shape: {low_freq.shape}")
    print(f"✓ High-freq bands: {len(high_freq)} (LH, HL, HH)")

    for i, band in enumerate(high_freq):
        print(f"  Band {i+1} shape: {band.shape}")

    # Verify shapes
    assert low_freq.shape == (1, 3, 64, 64), "Low-freq should be half size"
    assert len(high_freq) == 3, "Should have 3 high-freq bands"
    print("✅ Test passed: Wavelet decomposition works correctly\n")


def test_frequency_separation():
    """Test 4: Verify low/high frequency separation"""
    print("=" * 60)
    print("Test 4: Frequency Separation Verification")
    print("=" * 60)

    if not PTWT_AVAILABLE:
        print("⚠️  ptwt not available, skipping frequency separation test")
        return

    import ptwt

    # Create structured test image (low-freq)
    low_freq_image = torch.zeros(1, 3, 128, 128)
    low_freq_image[:, :, 40:88, 40:88] = 1.0  # Large block

    # Create detailed test image (high-freq)
    high_freq_image = torch.zeros(1, 3, 128, 128)
    for i in range(0, 128, 2):
        high_freq_image[:, :, i, :] = 1.0  # Thin stripes

    # Decompose both
    low_decomp, high_decomp = ptwt.wavedec2(low_freq_image, 'haar', level=1)
    high_low_decomp, high_high_decomp = ptwt.wavedec2(high_freq_image, 'haar', level=1)

    # Low-freq image should have more energy in low-freq component
    low_energy_ratio = low_decomp.abs().mean() / (low_decomp.abs().mean() + sum(b.abs().mean() for b in high_decomp))
    high_energy_ratio = high_low_decomp.abs().mean() / (high_low_decomp.abs().mean() + sum(b.abs().mean() for b in high_high_decomp))

    print(f"✓ Low-freq image → Low component energy ratio: {low_energy_ratio:.3f}")
    print(f"✓ High-freq image → Low component energy ratio: {high_energy_ratio:.3f}")

    assert low_energy_ratio > high_energy_ratio, "Low-freq image should have higher low-component ratio"
    print("✅ Test passed: Frequency separation works as expected\n")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("WAVELET LOSS FUNCTIONALITY TEST")
    print("="*60 + "\n")

    print(f"ptwt available: {PTWT_AVAILABLE}")
    if PTWT_AVAILABLE:
        print("ptwt installed successfully\n")
    else:
        print("⚠️  ptwt not installed. Some tests will be skipped.\n")

    try:
        # Test 1: Disabled
        test_wavelet_loss_disabled()

        # Test 2: Enabled
        test_wavelet_loss_enabled()

        # Test 3: Decomposition
        test_wavelet_decomposition()

        # Test 4: Frequency separation
        test_frequency_separation()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nWavelet loss is ready to use.")
        print("To enable during training, set in colleague_training_config.yaml:")
        print("  loss:")
        print("    wavelet:")
        print("      enable: true")
        print("      wavelet_type: haar")
        print("      level: 1")
        print("      low_freq_weight: 1.0")
        print("      high_freq_weight: 0.5")
        print("    weights:")
        print("      wavelet: 1.0  # Adjust weight as needed")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
