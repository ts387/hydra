#!/usr/bin/env python3
"""
Test script to verify Metal GPU support in Hydra.

This script tests the device detection and synchronization utilities
to ensure proper support for CUDA, MPS (Metal), and CPU backends.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_device_detection():
    """Test device detection across CUDA, MPS, and CPU."""
    print("="*60)
    print("Testing Hydra Metal GPU Support")
    print("="*60)

    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"Python version: {sys.version}")

        # Import device utilities
        from src import device_utils

        # Print device info
        device_utils.print_device_info()

        # Test device detection
        device, device_type = device_utils.get_device()
        print(f"\nSelected device: {device}")
        print(f"Device type: {device_type}")
        print(f"Device name: {device_utils.get_device_name(device)}")

        # Test GPU availability
        is_gpu = device_utils.is_gpu_available()
        print(f"GPU available: {is_gpu}")

        # Test device count
        device_count = device_utils.get_device_count()
        print(f"Device count: {device_count}")

        # Test tensor operations
        print("\n" + "="*60)
        print("Testing Tensor Operations")
        print("="*60)

        # Create a simple tensor
        x = torch.randn(100, 100)
        print(f"\nCreated tensor on CPU: {x.shape}")

        # Move to device
        x_device = x.to(device)
        print(f"Moved tensor to {device}: {x_device.shape}")

        # Perform computation
        y = torch.matmul(x_device, x_device.T)
        print(f"Matrix multiplication result: {y.shape}")

        # Test synchronization
        device_utils.synchronize_device(device)
        print("Device synchronized successfully")

        # Test empty cache
        device_utils.empty_cache(device)
        print("Device cache cleared successfully")

        # Check MPS compatibility warnings
        if device_type == 'mps':
            print("\n" + "="*60)
            print("MPS Compatibility Check")
            print("="*60)

            warnings = device_utils.check_mps_compatibility()
            if warnings:
                print("\nCompatibility warnings:")
                for warning in warnings:
                    print(f"  ! {warning}")
            else:
                print("No compatibility warnings detected")

            # Validate MPS operations
            print("\nValidating MPS operations...")
            results = device_utils.validate_mps_operations()
            for op_name, (success, error) in results.items():
                status = "OK" if success else f"FAILED: {error}"
                print(f"  {op_name}: {status}")

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)

        return True

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPyTorch is not installed. Please install it with:")
        print("  pip install torch>=1.12")
        return False
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_device_detection()
    sys.exit(0 if success else 1)
