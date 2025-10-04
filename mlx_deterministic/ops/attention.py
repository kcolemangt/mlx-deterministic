"""
Batch-Invariant Attention for MLX

This module implements attention mechanisms with batch-invariant operations,
ensuring deterministic outputs regardless of batch size.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math


def batch_invariant_softmax(
    x: mx.array,
    axis: int = -1,
    chunk_size: int = 128
) -> mx.array:
    """
    Batch-invariant softmax with fixed reduction pattern.

    Standard softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    The max and sum reductions must use fixed patterns for batch invariance.

    Args:
        x: Input tensor
        axis: Axis along which to compute softmax
        chunk_size: Fixed chunk size for reductions

    Returns:
        Softmax probabilities
    """
    # For batch invariance, we need fixed reduction trees for both max and sum
    # We'll use chunked reductions similar to RMSNorm

    # Get the dimension size along the reduction axis
    dim_size = x.shape[axis]

    # Pad to multiple of chunk_size
    pad_size = (chunk_size - (dim_size % chunk_size)) % chunk_size

    if pad_size > 0:
        # Create padding shape
        pad_shape = list(x.shape)
        pad_shape[axis] = pad_size

        # Pad with -inf for max computation (won't affect result)
        padding = mx.full(pad_shape, -float('inf'), dtype=x.dtype)

        # Concatenate along the reduction axis
        if axis == -1:
            x_padded = mx.concatenate([x, padding], axis=-1)
        else:
            # Handle other axes if needed
            x_padded = mx.concatenate([x, padding], axis=axis)
    else:
        x_padded = x

    # Reshape to create chunks: move reduction axis to second-to-last position
    # and create chunk dimension
    if axis == -1:
        # Shape: (..., dim_padded) -> (..., num_chunks, chunk_size)
        new_shape = list(x_padded.shape[:-1]) + [-1, chunk_size]
        x_chunked = x_padded.reshape(new_shape)

        # Find max within each chunk
        chunk_max = mx.max(x_chunked, axis=-1, keepdims=True)  # (..., num_chunks, 1)

        # Find global max across chunks
        global_max = mx.max(chunk_max, axis=-2, keepdims=True)  # (..., 1, 1)

        # Subtract max for numerical stability
        x_shifted = x - mx.squeeze(global_max, axis=(-2, -1))[..., None]

        # Compute exp
        exp_x = mx.exp(x_shifted)

        # Pad exp_x same way
        if pad_size > 0:
            padding_exp = mx.zeros(pad_shape, dtype=exp_x.dtype)
            exp_x_padded = mx.concatenate([exp_x, padding_exp], axis=-1)
        else:
            exp_x_padded = exp_x

        # Reshape to chunks
        exp_x_chunked = exp_x_padded.reshape(new_shape)

        # Sum within each chunk
        chunk_sum = mx.sum(exp_x_chunked, axis=-1, keepdims=True)  # (..., num_chunks, 1)

        # Global sum across chunks
        global_sum = mx.sum(chunk_sum, axis=-2, keepdims=True)  # (..., 1, 1)

        # Compute softmax
        result = exp_x / mx.squeeze(global_sum, axis=(-2, -1))[..., None]

        return result

    else:
        # For simplicity, only support last axis for now
        raise NotImplementedError("Only axis=-1 is currently supported")


class BatchInvariantAttention(nn.Module):
    """
    Batch-invariant multi-head attention.

    Implements scaled dot-product attention with batch-invariant operations:
    - Batch-invariant matmul for Q@K^T and attention@V
    - Batch-invariant softmax for attention weights

    Args:
        dims (int): Model dimension
        num_heads (int): Number of attention heads
        query_input_dims (Optional[int]): Query input dimension (defaults to dims)
        key_input_dims (Optional[int]): Key/Value input dimension (defaults to dims)
        value_dims (Optional[int]): Value dimension (defaults to dims)
        value_output_dims (Optional[int]): Output dimension (defaults to dims)
        bias (bool): Whether to use bias in projections
        matmul_tile_size (int): Tile size for batch-invariant matmul
        softmax_chunk_size (int): Chunk size for batch-invariant softmax
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
        matmul_tile_size: int = 128,
        softmax_chunk_size: int = 128,
    ):
        super().__init__()

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.matmul_tile_size = matmul_tile_size
        self.softmax_chunk_size = softmax_chunk_size

        head_dim = dims // num_heads
        self.scale = head_dim ** -0.5

        # Projection layers (using standard MLX layers for now)
        # In full integration, these would use batch-invariant matmul
        self.query_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.key_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.value_proj = nn.Linear(key_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Apply batch-invariant multi-head attention.

        Args:
            queries: Query tensor [..., seq_len, dims]
            keys: Key tensor [..., seq_len, dims]
            values: Value tensor [..., seq_len, dims]
            mask: Optional attention mask
            cache: Optional (cached_keys, cached_values) for incremental decoding

        Returns:
            Attention output [..., seq_len, dims]
        """
        # Import here to avoid circular dependency
        from .matmul import batch_invariant_matmul

        # Project queries, keys, values
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        # Handle cache for incremental decoding
        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=-2)
            values = mx.concatenate([value_cache, values], axis=-2)

        # Get dimensions
        *batch_dims, seq_len, dims = queries.shape
        _, kv_seq_len, _ = keys.shape
        head_dim = dims // self.num_heads

        # Reshape for multi-head attention
        # [..., seq_len, dims] -> [..., seq_len, num_heads, head_dim]
        queries = queries.reshape(*batch_dims, seq_len, self.num_heads, head_dim)
        keys = keys.reshape(*batch_dims, kv_seq_len, self.num_heads, head_dim)
        values = values.reshape(*batch_dims, kv_seq_len, self.num_heads, head_dim)

        # Transpose to [..., num_heads, seq_len, head_dim]
        queries = queries.transpose(0, 2, 1, 3) if len(batch_dims) == 1 else \
                  queries.transpose(*range(len(batch_dims)), -2, -3, -1)
        keys = keys.transpose(0, 2, 1, 3) if len(batch_dims) == 1 else \
               keys.transpose(*range(len(batch_dims)), -2, -3, -1)
        values = values.transpose(0, 2, 1, 3) if len(batch_dims) == 1 else \
                 values.transpose(*range(len(batch_dims)), -2, -3, -1)

        # Compute attention scores: Q @ K^T
        # Need to transpose keys: [..., num_heads, head_dim, kv_seq_len]
        keys_t = keys.transpose(*range(len(keys.shape) - 2), -1, -2)

        # Batch-invariant matmul for attention scores
        scores = batch_invariant_matmul(
            queries, keys_t, tile_size=self.matmul_tile_size
        )  # [..., num_heads, seq_len, kv_seq_len]

        # Scale scores
        scores = scores * self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Batch-invariant softmax
        # For attention, we apply softmax over the key sequence dimension (last dim)
        attn_weights = batch_invariant_softmax(
            scores, axis=-1, chunk_size=self.softmax_chunk_size
        )

        # Apply attention to values: attn_weights @ V
        # attn_weights: [..., num_heads, seq_len, kv_seq_len]
        # values: [..., num_heads, kv_seq_len, head_dim]
        attn_output = batch_invariant_matmul(
            attn_weights, values, tile_size=self.matmul_tile_size
        )  # [..., num_heads, seq_len, head_dim]

        # Transpose back: [..., seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3) if len(batch_dims) == 1 else \
                      attn_output.transpose(*range(len(batch_dims)), -2, -3, -1)

        # Reshape to [..., seq_len, dims]
        attn_output = attn_output.reshape(*batch_dims, seq_len, dims)

        # Output projection
        output = self.out_proj(attn_output)

        return output


def create_causal_mask(seq_len: int, kv_seq_len: Optional[int] = None) -> mx.array:
    """
    Create a causal attention mask.

    Args:
        seq_len: Query sequence length
        kv_seq_len: Key/value sequence length (defaults to seq_len)

    Returns:
        Mask of shape [seq_len, kv_seq_len] with 0 for allowed positions
        and -inf for masked positions
    """
    kv_seq_len = kv_seq_len or seq_len

    # Create indices
    q_idx = mx.arange(seq_len).reshape(-1, 1)
    kv_idx = mx.arange(kv_seq_len).reshape(1, -1)

    # Causal mask: query position i can only attend to key positions <= i
    mask = (kv_idx > q_idx).astype(mx.float32) * -1e9

    return mask
