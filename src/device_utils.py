"""
Device utilities for cross-platform GPU support (CUDA, Metal MPS, CPU).

This module provides unified device detection and management for PyTorch,
supporting CUDA (NVIDIA GPUs), Metal Performance Shaders (Apple Silicon),
and CPU fallback.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(device_id=0):
    """
    Get the best available PyTorch device.

    Priority order: CUDA > MPS > CPU

    Args:
        device_id (int): Device ID for CUDA (ignored for MPS/CPU)

    Returns:
        torch.device: The selected device
        str: Device type name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        device_type = 'cuda'
        logger.info(f"Using CUDA device: {device_id}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'
        logger.info("Using Metal Performance Shaders (MPS) device")
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        logger.info("Using CPU device (no GPU acceleration available)")

    return device, device_type


def get_device_count():
    """
    Get the number of available GPU devices.

    Returns:
        int: Number of available GPU devices (0 for CPU-only)
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS currently supports single device
        return 1
    else:
        return 0


def synchronize_device(device=None):
    """
    Synchronize the device (ensure all operations are complete).

    This is useful for accurate timing and ensuring operations have finished
    before measuring performance or proceeding with dependent operations.

    Args:
        device (torch.device or None): Device to synchronize. If None,
                                       synchronizes current device.
    """
    if device is None:
        # Try to synchronize current device
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS synchronization (available in PyTorch >= 1.12)
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        # MPS synchronization
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
    # CPU doesn't need synchronization


def is_gpu_available():
    """
    Check if any GPU acceleration is available.

    Returns:
        bool: True if CUDA or MPS is available, False otherwise
    """
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return cuda_available or mps_available


def get_device_name(device):
    """
    Get a human-readable name for the device.

    Args:
        device (torch.device): The device

    Returns:
        str: Human-readable device name
    """
    if device.type == 'cuda':
        return f"CUDA GPU {device.index}: {torch.cuda.get_device_name(device)}"
    elif device.type == 'mps':
        return "Apple Metal Performance Shaders (M-Series GPU)"
    else:
        return "CPU"


def empty_cache(device=None):
    """
    Empty the GPU memory cache.

    Args:
        device (torch.device or None): Device to clear cache for
    """
    if device is None or device.type == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif device.type == 'mps':
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


def print_device_info():
    """
    Print detailed information about available devices.
    """
    print("\n" + "="*60)
    print("Device Information")
    print("="*60)

    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA Available: No")

    # MPS info
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            print(f"MPS (Metal) Available: Yes")
            print(f"  - Apple Silicon GPU acceleration enabled")
        else:
            print("MPS (Metal) Available: No")
            if torch.backends.mps.is_built():
                print("  - MPS backend is built but not available (may need macOS 12.3+)")
    else:
        print("MPS (Metal) Available: No (PyTorch version < 1.12)")

    # PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")

    # Selected device
    device, device_type = get_device()
    print(f"Selected Device: {get_device_name(device)}")
    print("="*60 + "\n")
