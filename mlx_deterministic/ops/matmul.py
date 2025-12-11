"""
Batch-Invariant Matrix Multiplication for MLX

This module implements matrix multiplication with batch-invariant reduction,
ensuring deterministic outputs regardless of batch size.

Based on Thinking Machines Labs research:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Python path approach: Use fixed tiling over the K (reduction) dimension and
accumulate partial products in a deterministic order. This prevents MLX from
choosing batch-dependent split-K heuristics for large reductions.

Metal path (`use_metal_kernel=True`) provides bitwise determinism using custom
2D output-tiling kernels.
"""

import mlx.core as mx
from typing import Optional


def batch_invariant_matmul(
    a: mx.array,
    b: mx.array,
    tile_size: int = 64,
    use_metal_kernel: bool = False,
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    """
    Batch-invariant matrix multiplication using fixed K-dimension tiling.

    We pad and split the reduction (K) dimension into fixed-size tiles, compute
    partial products per tile, and sum them in a fixed order. This enforces a
    deterministic reduction tree regardless of batch size.

    Args:
        a: Left matrix of shape [..., M, K] or [M, K]
        b: Right matrix of shape [K, N] or [..., K, N]
        tile_size: Tile size for the K (reduction) dimension. Default: 64
        use_metal_kernel: If True, use custom Metal kernel for bitwise-identical
                         determinism. If False (default), use Python tiling which
                         has ~1e-5 tolerance due to MLX's internal matmul variance.
        dtype: Optional output dtype (only used with use_metal_kernel=True).
               If provided, inputs are cast to this dtype. Supports mx.float32
               and mx.float16. If None (default), uses the input dtype.

    Returns:
        Result matrix of shape [..., M, N]

    Supported configurations:
        - 2D x 2D: [M, K] @ [K, N] -> [M, N]
        - 3D x 2D: [B, M, K] @ [K, N] -> [B, M, N]  (batched left)
        - 3D x 3D: [B, M, K] @ [B, K, N] -> [B, M, N]  (both batched)
    """
    # Use Metal kernel for bitwise determinism
    if use_metal_kernel:
        from .metal_matmul import deterministic_matmul_metal
        return deterministic_matmul_metal(a, b, dtype=dtype)
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

    # Pad K dimension to a multiple of tile_size
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

    num_tiles = K_padded // tile_size
    accumulator = None

    for tile_idx in range(num_tiles):
        k_start = tile_idx * tile_size
        k_end = k_start + tile_size

        a_tile = a_padded[..., :, k_start:k_end]
        b_tile = b_padded[..., k_start:k_end, :]

        partial_product = mx.matmul(a_tile, b_tile)
        accumulator = partial_product if accumulator is None else accumulator + partial_product

    return accumulator


def batch_invariant_addmm(
    bias: mx.array,
    a: mx.array,
    b: mx.array,
    alpha: float = 1.0,
    beta: float = 1.0,
    tile_size: int = 64,
) -> mx.array:
    """
    Batch-invariant addmm: result = beta * bias + alpha * (a @ b)

    Args:
        bias: Bias tensor that will be added. Shape: [M, N] or broadcastable
        a: Left matrix of shape [..., M, K]
        b: Right matrix of shape [K, N] or [..., K, N]
        alpha: Multiplier for matmul result
        beta: Multiplier for bias
        tile_size: Tile size for the K (reduction) dimension

    Returns:
        Result matrix of shape [..., M, N]
    """
    # Compute batch-invariant matmul with fixed K tiling
    mm_result = batch_invariant_matmul(a, b, tile_size=tile_size)

    # Apply alpha and beta scaling and add bias
    result = beta * bias + alpha * mm_result

    return result


class BatchInvariantLinear:
    """
    Batch-invariant linear layer (equivalent to nn.Linear but deterministic).

    This implements: y = x @ W^T + b (if bias exists)
    using batch-invariant matrix multiplication with fixed K tiling.

    Args:
        weight: Weight matrix of shape [out_features, in_features]
        bias: Optional bias vector of shape [out_features]
        tile_size: Tile size for the K (reduction) dimension
    """

    def __init__(
        self,
        weight: mx.array,
        bias: Optional[mx.array] = None,
        tile_size: int = 64
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
        # Compute x @ W^T using batch-invariant matmul with fixed K tiling
        output = batch_invariant_matmul(x, self.weight.T, tile_size=self.tile_size)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output
