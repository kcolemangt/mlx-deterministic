"""
Batch-Invariant Matrix Multiplication for MLX

This module implements matrix multiplication with batch-invariant reduction,
ensuring deterministic outputs regardless of batch size.
"""

import mlx.core as mx
from typing import Optional


def batch_invariant_matmul(
    a: mx.array,
    b: mx.array,
    tile_size: int = 128,
) -> mx.array:
    """
    Batch-invariant matrix multiplication.

    Ensures deterministic results by using fixed-size tiles for the reduction
    dimension (K), regardless of batch size.

    The key insight: Standard matmul computes C[i,j] = sum_k(A[i,k] * B[k,j]).
    The order of this summation can vary with batch size due to different
    computation patterns. We enforce a fixed reduction tree by:
    1. Padding K dimension to multiple of tile_size
    2. Computing partial products for each tile
    3. Summing tiles in a fixed order

    Args:
        a: Left matrix of shape [..., M, K] or [M, K]
        b: Right matrix of shape [K, N] or [..., K, N]
        tile_size: Size of tiles for K dimension. Must be > 0. Default: 128

    Returns:
        Result matrix of shape [..., M, N]

    Supported configurations:
        - 2D x 2D: [M, K] @ [K, N] -> [M, N]
        - 3D x 2D: [B, M, K] @ [K, N] -> [B, M, N]  (batched left)
        - 3D x 3D: [B, M, K] @ [B, K, N] -> [B, M, N]  (both batched)
    """
    assert tile_size > 0, "tile_size must be positive"

    # Get shapes
    a_shape = a.shape
    b_shape = b.shape

    # Validate dimensions
    assert len(a_shape) >= 2, "Matrix a must be at least 2D"
    assert len(b_shape) >= 2, "Matrix b must be at least 2D"

    K_a = a_shape[-1]
    K_b = b_shape[-2]
    assert K_a == K_b, f"Incompatible dimensions: {K_a} != {K_b}"

    K = K_a
    M = a_shape[-2]
    N = b_shape[-1]

    # Pad K dimension to multiple of tile_size
    K_padded = ((K + tile_size - 1) // tile_size) * tile_size
    pad_size = K_padded - K

    if pad_size > 0:
        # Pad matrix a along last dimension (K)
        pad_shape_a = list(a_shape)
        pad_shape_a[-1] = pad_size
        a_padding = mx.zeros(pad_shape_a, dtype=a.dtype)
        a_padded = mx.concatenate([a, a_padding], axis=-1)

        # Pad matrix b along second-to-last dimension (K)
        pad_shape_b = list(b_shape)
        pad_shape_b[-2] = pad_size
        b_padding = mx.zeros(pad_shape_b, dtype=b.dtype)
        b_padded = mx.concatenate([b, b_padding], axis=-2)
    else:
        a_padded = a
        b_padded = b

    # Number of tiles
    num_tiles = K_padded // tile_size

    # Compute partial products for each tile and sum
    # This ensures fixed reduction order regardless of batch size
    accumulator = None

    for tile_idx in range(num_tiles):
        # Extract tile from a and b
        k_start = tile_idx * tile_size
        k_end = k_start + tile_size

        # Slice tiles: a[..., :, k_start:k_end] @ b[..., k_start:k_end, :]
        a_tile = a_padded[..., :, k_start:k_end]
        b_tile = b_padded[..., k_start:k_end, :]

        # Compute partial product for this tile
        partial_product = mx.matmul(a_tile, b_tile)

        # Accumulate
        if accumulator is None:
            accumulator = partial_product
        else:
            accumulator = accumulator + partial_product

    return accumulator


def batch_invariant_addmm(
    bias: mx.array,
    a: mx.array,
    b: mx.array,
    alpha: float = 1.0,
    beta: float = 1.0,
    tile_size: int = 128,
) -> mx.array:
    """
    Batch-invariant addmm: result = beta * bias + alpha * (a @ b)

    Args:
        bias: Bias tensor that will be added. Shape: [M, N] or broadcastable
        a: Left matrix of shape [..., M, K]
        b: Right matrix of shape [K, N] or [..., K, N]
        alpha: Multiplier for matmul result
        beta: Multiplier for bias
        tile_size: Size of tiles for K dimension

    Returns:
        Result matrix of shape [..., M, N]
    """
    # Compute batch-invariant matmul
    mm_result = batch_invariant_matmul(a, b, tile_size=tile_size)

    # Apply alpha and beta scaling and add bias
    result = beta * bias + alpha * mm_result

    return result


class BatchInvariantLinear:
    """
    Batch-invariant linear layer (equivalent to nn.Linear but deterministic).

    This implements: y = x @ W^T + b (if bias exists)
    using batch-invariant matrix multiplication.

    Args:
        weight: Weight matrix of shape [out_features, in_features]
        bias: Optional bias vector of shape [out_features]
        tile_size: Size of tiles for batch-invariant matmul
    """

    def __init__(
        self,
        weight: mx.array,
        bias: Optional[mx.array] = None,
        tile_size: int = 128
    ):
        self.weight = weight
        self.bias = bias
        self.tile_size = tile_size

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply linear transformation with batch-invariant matmul.

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features]
        """
        # Compute x @ W^T using batch-invariant matmul
        # Need to transpose weight for standard linear layer convention
        output = batch_invariant_matmul(x, self.weight.T, tile_size=self.tile_size)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output
