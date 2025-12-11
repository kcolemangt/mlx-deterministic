"""
Batch-Invariant RMS Normalization for MLX

This module implements RMSNorm with batch-invariant variance computation,
ensuring deterministic outputs regardless of batch size.

Based on Thinking Machines Labs research:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Research approach: Data-parallel processing where each sample's reduction
is computed atomically, avoiding any batch-dependent reduction patterns.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class BatchInvariantRMSNorm(nn.Module):
    """
    Batch-invariant Root Mean Square Layer Normalization.

    This implementation ensures that the output for a given sample is identical
    regardless of the batch size it's processed in.

    Research-aligned approach: Each sample's mean(x²) is computed as a single
    atomic reduction over all features. This matches the TML research's
    "data-parallel" approach where "one batch element per core, keeping entire
    reduction within single core."

    Args:
        dims (int): The feature dimension to normalize over (last dimension)
        eps (float): A small constant for numerical stability
        chunk_size (int): DEPRECATED - kept for API compatibility, ignored internally.
                         The research-aligned implementation uses atomic per-sample
                         reduction instead of chunking.
        use_metal_kernel (bool): If True, use custom Metal kernel for bitwise-identical
                                determinism. Default: False
    """

    def __init__(
        self,
        dims: int,
        eps: float = 1e-6,
        chunk_size: int = 64,
        use_metal_kernel: bool = False
    ):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.chunk_size = chunk_size  # Kept for API compatibility, not used
        self.use_metal_kernel = use_metal_kernel

        # Learnable scale parameter
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply batch-invariant RMS normalization.

        Research-aligned: Each sample's mean(x²) is computed as a single atomic
        reduction. No chunking or padding is used, ensuring:
        1. Mathematically correct variance for ANY dims value
        2. Identical reduction pattern regardless of batch size
        3. No padding-related numerical errors

        Args:
            x: Input tensor of shape (..., dims)

        Returns:
            Normalized tensor of same shape as input
        """
        # Use Metal kernel for bitwise determinism if requested
        if self.use_metal_kernel:
            from .metal_rms_norm import rms_norm_metal
            return rms_norm_metal(x, self.weight, self.eps)

        # Store original dtype and shape for restoration
        original_dtype = x.dtype
        original_shape = x.shape

        # Upcast to float32 for numerical stability (avoids overflow in float16)
        x = x.astype(mx.float32)

        # Flatten to [batch, dims] for consistent processing
        x_flat = x.reshape(-1, self.dims)

        # Compute mean squared value per sample (atomic reduction along features)
        # This is the key to batch invariance: each sample's reduction is independent
        x_squared = x_flat * x_flat
        mean_sq = mx.mean(x_squared, axis=-1, keepdims=True)  # [batch, 1]

        # Compute RMS normalization: x / sqrt(mean_sq + eps)
        rms = mx.sqrt(mean_sq + self.eps)
        normalized = x_flat / rms

        # Apply learned weight (upcast weight to float32 for consistency)
        weight = self.weight.astype(mx.float32)
        result = normalized * weight

        # Restore original shape and dtype
        result = result.reshape(original_shape)
        return result.astype(original_dtype)


def rms_norm_batch_invariant(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
    chunk_size: int = 64
) -> mx.array:
    """
    Functional batch-invariant RMS normalization.

    Research-aligned: Uses atomic per-sample reduction instead of chunking.

    Args:
        x: Input tensor of shape (..., dims)
        weight: Scale parameters of shape (dims,)
        eps: Small constant for numerical stability
        chunk_size: DEPRECATED - kept for API compatibility, ignored internally

    Returns:
        Normalized tensor of same shape as input
    """
    dims = x.shape[-1]
    original_dtype = x.dtype
    original_shape = x.shape

    # Upcast to float32 for numerical stability (avoids overflow in float16)
    x = x.astype(mx.float32)

    # Flatten to [batch, dims] for consistent processing
    x_flat = x.reshape(-1, dims)

    # Compute mean squared value per sample (atomic reduction)
    x_squared = x_flat * x_flat
    mean_sq = mx.mean(x_squared, axis=-1, keepdims=True)

    # Apply RMS normalization
    rms = mx.sqrt(mean_sq + eps)
    normalized = x_flat / rms

    # Apply weight (upcast to float32) and restore shape
    weight = weight.astype(mx.float32)
    result = normalized * weight
    result = result.reshape(original_shape)
    return result.astype(original_dtype)
