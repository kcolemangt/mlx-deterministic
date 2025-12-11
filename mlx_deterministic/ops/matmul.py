"""
Batch-Invariant Matrix Multiplication for MLX

This module implements matrix multiplication with batch-invariant reduction,
ensuring deterministic outputs regardless of batch size.

Based on Thinking Machines Labs research:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Research approach: Use 2D output tiling (M×N) instead of split-K. Each output
tile computes a complete K-reduction, ensuring fixed computation order.
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
    Batch-invariant matrix multiplication using 2D output tiling.

    Research-aligned approach: Instead of split-K (splitting the reduction
    dimension), we use 2D output tiling where each output tile computes a
    complete K-reduction. This matches the TML research's approach of
    "split output into 2D tiles, assign each to different core."

    The key insight: Each output tile C[i:i+tile, j:j+tile] = A[i:i+tile, :] @ B[:, j:j+tile]
    computes the full K-dimension reduction within the tile, avoiding any
    inter-tile reduction that could vary with batch size.

    Args:
        a: Left matrix of shape [..., M, K] or [M, K]
        b: Right matrix of shape [K, N] or [..., K, N]
        tile_size: Size of output tiles (used for both M and N dimensions).
                   Previously was K-dimension tile; now repurposed for 2D tiling.
                   Default: 64
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

    M = a_shape[-2]
    N = b_shape[-1]

    # For small matrices, use standard matmul (single tile = deterministic)
    if M <= tile_size and N <= tile_size:
        return mx.matmul(a, b)

    # 2D output tiling: each tile computes complete K-reduction
    tile_m = tile_size
    tile_n = tile_size

    # Compute number of tiles
    num_tiles_m = (M + tile_m - 1) // tile_m
    num_tiles_n = (N + tile_n - 1) // tile_n

    # Process tiles and collect results
    # Each tile computes: C[m_start:m_end, n_start:n_end] = A[m_start:m_end, :] @ B[:, n_start:n_end]
    rows = []
    for i in range(num_tiles_m):
        m_start = i * tile_m
        m_end = min(m_start + tile_m, M)

        # Extract rows of A for this tile row
        a_row = a[..., m_start:m_end, :]  # [..., tile_m, K]

        cols = []
        for j in range(num_tiles_n):
            n_start = j * tile_n
            n_end = min(n_start + tile_n, N)

            # Extract columns of B for this tile
            b_col = b[..., :, n_start:n_end]  # [..., K, tile_n]

            # Complete K-reduction for this output tile
            # This is the full dot product - no partial sums across tiles
            tile_result = mx.matmul(a_row, b_col)  # [..., tile_m, tile_n]
            cols.append(tile_result)

        # Concatenate columns for this row
        row_result = mx.concatenate(cols, axis=-1) if len(cols) > 1 else cols[0]
        rows.append(row_result)

    # Concatenate all rows
    result = mx.concatenate(rows, axis=-2) if len(rows) > 1 else rows[0]
    return result


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
        tile_size: Size of output tiles for 2D tiling

    Returns:
        Result matrix of shape [..., M, N]
    """
    # Compute batch-invariant matmul with 2D output tiling
    mm_result = batch_invariant_matmul(a, b, tile_size=tile_size)

    # Apply alpha and beta scaling and add bias
    result = beta * bias + alpha * mm_result

    return result


class BatchInvariantLinear:
    """
    Batch-invariant linear layer (equivalent to nn.Linear but deterministic).

    This implements: y = x @ W^T + b (if bias exists)
    using batch-invariant matrix multiplication with 2D output tiling.

    Args:
        weight: Weight matrix of shape [out_features, in_features]
        bias: Optional bias vector of shape [out_features]
        tile_size: Size of output tiles for 2D tiling
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
        # Compute x @ W^T using batch-invariant matmul with 2D output tiling
        output = batch_invariant_matmul(x, self.weight.T, tile_size=self.tile_size)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output
