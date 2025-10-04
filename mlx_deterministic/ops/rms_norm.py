"""
Batch-Invariant RMS Normalization for MLX

This module implements RMSNorm with batch-invariant variance computation,
ensuring deterministic outputs regardless of batch size.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class BatchInvariantRMSNorm(nn.Module):
    """
    Batch-invariant Root Mean Square Layer Normalization.

    This implementation ensures that the output for a given sample is identical
    regardless of the batch size it's processed in. This is achieved by using
    fixed-size chunks for variance computation.

    Args:
        dims (int): The feature dimension to normalize over (last dimension)
        eps (float): A small constant for numerical stability
        chunk_size (int): Fixed chunk size for variance computation.
                         Must be power of 2. Default: 64
    """

    def __init__(self, dims: int, eps: float = 1e-6, chunk_size: int = 64):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.chunk_size = chunk_size

        # Learnable scale parameter
        self.weight = mx.ones((dims,))

        # Verify chunk_size is power of 2
        assert chunk_size > 0 and (chunk_size & (chunk_size - 1)) == 0, \
            "chunk_size must be a power of 2"

    def _compute_variance_batch_invariant(self, x: mx.array) -> mx.array:
        """
        Compute variance using fixed-size chunks for batch invariance.

        RMSNorm computes variance PER SAMPLE across the feature dimension.
        Batch invariance is achieved by using a fixed reduction pattern when
        computing mean(x^2) across features, regardless of batch size.

        The strategy:
        1. Pad feature dimension to multiple of chunk_size
        2. Compute chunk-wise means across features
        3. Average chunks with fixed pattern

        Args:
            x: Input tensor of shape (..., dims)

        Returns:
            Mean squared value per sample (same shape as input without last dim)
        """
        # Compute squared values
        x_squared = x * x  # Shape: (..., dims)

        # Pad last dimension (features) to multiple of chunk_size
        pad_size = (self.chunk_size - (self.dims % self.chunk_size)) % self.chunk_size
        if pad_size > 0:
            # Pad along last dimension
            pad_shape = list(x_squared.shape)
            pad_shape[-1] = pad_size
            padding = mx.zeros(pad_shape, dtype=x_squared.dtype)
            x_squared_padded = mx.concatenate([x_squared, padding], axis=-1)
        else:
            x_squared_padded = x_squared

        # Reshape to chunk features: (..., num_chunks, chunk_size)
        new_shape = list(x_squared_padded.shape[:-1]) + [-1, self.chunk_size]
        x_chunked = x_squared_padded.reshape(new_shape)

        # Compute mean within each chunk across chunk_size dimension
        chunk_means = mx.mean(x_chunked, axis=-1)  # Shape: (..., num_chunks)

        # Average across chunks to get final variance
        variance = mx.mean(chunk_means, axis=-1, keepdims=True)  # Shape: (..., 1)

        return variance

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply batch-invariant RMS normalization.

        Args:
            x: Input tensor of shape (..., dims)

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute batch-invariant mean squared value
        mean_sq = self._compute_variance_batch_invariant(x)  # Shape: (..., 1)

        # Compute RMS normalization: x / sqrt(mean_sq + eps) * weight
        rms = mx.sqrt(mean_sq + self.eps)
        normalized = x / rms

        # Apply learned weight
        return normalized * self.weight


def rms_norm_batch_invariant(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
    chunk_size: int = 64
) -> mx.array:
    """
    Functional batch-invariant RMS normalization.

    Args:
        x: Input tensor of shape (..., dims)
        weight: Scale parameters of shape (dims,)
        eps: Small constant for numerical stability
        chunk_size: Fixed chunk size for variance computation

    Returns:
        Normalized tensor of same shape as input
    """
    dims = x.shape[-1]

    # Compute squared values
    x_squared = x * x

    # Pad last dimension (features) to multiple of chunk_size
    pad_size = (chunk_size - (dims % chunk_size)) % chunk_size
    if pad_size > 0:
        pad_shape = list(x_squared.shape)
        pad_shape[-1] = pad_size
        padding = mx.zeros(pad_shape, dtype=x_squared.dtype)
        x_squared_padded = mx.concatenate([x_squared, padding], axis=-1)
    else:
        x_squared_padded = x_squared

    # Reshape to chunk features: (..., num_chunks, chunk_size)
    new_shape = list(x_squared_padded.shape[:-1]) + [-1, chunk_size]
    x_chunked = x_squared_padded.reshape(new_shape)

    # Compute mean within each chunk
    chunk_means = mx.mean(x_chunked, axis=-1)

    # Average across chunks to get final mean squared
    mean_sq = mx.mean(chunk_means, axis=-1, keepdims=True)

    # Apply RMS normalization
    rms = mx.sqrt(mean_sq + eps)
    normalized = x / rms

    return normalized * weight
