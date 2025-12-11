"""
Batch-Invariant Attention for MLX

This module implements attention mechanisms with batch-invariant operations,
ensuring deterministic outputs regardless of batch size.

Based on Thinking Machines Labs research:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Research approach: FlashAttention-style with FIXED SPLIT-SIZE (not fixed number
of splits). This ensures the reduction pattern is independent of batch/sequence
length variations.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math


def flash_attention_fixed_split(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    split_size: int = 256,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """
    FlashAttention with FIXED SPLIT-SIZE for batch invariance.

    Research-aligned approach: Use fixed split_size, not fixed number of splits.
    This ensures identical reduction patterns regardless of sequence length.

    Supports Grouped Query Attention (GQA) where n_kv_heads < n_heads.

    Key insight from TML research:
    - Old approach: KV_len=1000, 4 splits → 250 elements each (varies with KV_len)
    - New approach: KV_len=1000, split_size=256 → 4 full + 1 partial (fixed pattern)

    The online softmax algorithm processes KV in fixed-size blocks, accumulating
    results with numerically stable rescaling.

    Args:
        q: Query tensor [B, H_q, N, D] or [H_q, N, D]
        k: Key tensor [B, H_kv, S, D] or [H_kv, S, D] (H_kv may differ from H_q for GQA)
        v: Value tensor [B, H_kv, S, D] or [H_kv, S, D]
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        split_size: Fixed size for KV splits (default: 256)
        mask: Optional attention mask [N, S] or broadcastable

    Returns:
        Attention output with same shape as q
    """
    # Handle the case where input is unbatched
    original_shape = q.shape
    if len(q.shape) == 3:
        # Add batch dimension
        q = q[None, ...]
        k = k[None, ...]
        v = v[None, ...]
        if mask is not None and len(mask.shape) < 4:
            mask = mask[None, ...]

    B, H_q, N, D = q.shape
    _, H_kv, S, _ = k.shape

    # Handle Grouped Query Attention (GQA) by repeating KV heads
    if H_kv != H_q:
        # Number of query heads per KV head
        n_rep = H_q // H_kv
        # Repeat K and V: [B, H_kv, S, D] -> [B, H_q, S, D]
        k = mx.repeat(k, n_rep, axis=1)
        v = mx.repeat(v, n_rep, axis=1)

    # Initialize online softmax accumulators
    # m_i: running maximum (for numerical stability)
    # l_i: running sum of exp(x - max)
    # o_i: running weighted sum
    m_i = mx.full((B, H_q, N, 1), -1e9, dtype=q.dtype)
    l_i = mx.zeros((B, H_q, N, 1), dtype=q.dtype)
    o_i = mx.zeros((B, H_q, N, D), dtype=q.dtype)

    # Fixed number of blocks based on split_size
    num_blocks = (S + split_size - 1) // split_size

    for j in range(num_blocks):
        start = j * split_size
        end = min(start + split_size, S)

        # Extract KV block
        k_j = k[:, :, start:end, :]  # [B, H_q, block_len, D]
        v_j = v[:, :, start:end, :]  # [B, H_q, block_len, D]

        # Compute attention scores for this block: Q @ K_j^T
        # q: [B, H_q, N, D], k_j: [B, H_q, block_len, D]
        s_ij = (q @ k_j.swapaxes(-1, -2)) * scale  # [B, H_q, N, block_len]

        # Apply mask if provided
        if mask is not None:
            if mask.shape[-1] > end:
                mask_block = mask[..., start:end]
            else:
                mask_block = mask
            s_ij = s_ij + mask_block

        # Online softmax update (numerically stable)
        # Find max in this block
        m_ij = mx.max(s_ij, axis=-1, keepdims=True)  # [B, H_q, N, 1]

        # New running max
        m_new = mx.maximum(m_i, m_ij)

        # Rescale factors
        alpha = mx.exp(m_i - m_new)  # Rescale old accumulator
        beta = mx.exp(s_ij - m_new)  # New block contributions

        # Update running sum
        l_new = alpha * l_i + mx.sum(beta, axis=-1, keepdims=True)

        # Update output accumulator
        # o_i = alpha * o_i + beta @ v_j
        o_i = alpha * o_i + (beta @ v_j)

        # Update running values
        m_i = m_new
        l_i = l_new

    # Final normalization
    result = o_i / l_i

    # Remove batch dimension if input was unbatched
    if len(original_shape) == 3:
        result = result[0]

    return result


def batch_invariant_softmax(
    x: mx.array,
    axis: int = -1,
    chunk_size: int = 128,
    use_metal_kernel: bool = False
) -> mx.array:
    """
    Batch-invariant softmax using atomic per-row reduction.

    Research-aligned: Each row's softmax is computed independently with
    a single max and sum reduction, ensuring batch invariance.

    Args:
        x: Input tensor
        axis: Axis along which to compute softmax (must be -1)
        chunk_size: DEPRECATED - kept for API compatibility
        use_metal_kernel: If True, use custom Metal kernel for bitwise-identical
                         determinism. Default: False

    Returns:
        Softmax probabilities
    """
    if axis != -1:
        raise NotImplementedError("Only axis=-1 is currently supported")

    # Use Metal kernel for bitwise determinism if requested
    if use_metal_kernel:
        from .metal_softmax import softmax_metal
        return softmax_metal(x, axis=axis)

    # Simple atomic softmax per row - no chunking needed
    # Each row's max and sum are computed independently
    x_max = mx.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max
    exp_x = mx.exp(x_shifted)
    sum_exp = mx.sum(exp_x, axis=-1, keepdims=True)

    return exp_x / sum_exp


class BatchInvariantAttention(nn.Module):
    """
    Batch-invariant multi-head attention using FlashAttention algorithm.

    Research-aligned approach: Uses FlashAttention with fixed split-size
    instead of naive O(n²) attention. This ensures:
    1. Memory efficient: O(N) instead of O(N²) for attention matrix
    2. Deterministic: Fixed split pattern regardless of batch/sequence length
    3. Numerically stable: Online softmax with running max

    Args:
        dims (int): Model dimension
        num_heads (int): Number of attention heads
        query_input_dims (Optional[int]): Query input dimension (defaults to dims)
        key_input_dims (Optional[int]): Key/Value input dimension (defaults to dims)
        value_dims (Optional[int]): Value dimension (defaults to dims)
        value_output_dims (Optional[int]): Output dimension (defaults to dims)
        bias (bool): Whether to use bias in projections
        matmul_tile_size (int): DEPRECATED - kept for API compatibility
        softmax_chunk_size (int): Now used as KV split_size for FlashAttention
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

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads

        # Repurpose softmax_chunk_size as split_size for FlashAttention
        self.split_size = softmax_chunk_size

        # Keep for API compatibility (not used in FlashAttention path)
        self.matmul_tile_size = matmul_tile_size
        self.softmax_chunk_size = softmax_chunk_size

        self.scale = self.head_dim ** -0.5

        # Projection layers
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
        Apply batch-invariant multi-head attention using FlashAttention.

        Args:
            queries: Query tensor [..., seq_len, dims]
            keys: Key tensor [..., seq_len, dims]
            values: Value tensor [..., seq_len, dims]
            mask: Optional attention mask
            cache: Optional (cached_keys, cached_values) for incremental decoding

        Returns:
            Attention output [..., seq_len, dims]
        """
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
        kv_seq_len = keys.shape[-2]

        # Reshape for multi-head attention
        # [..., seq_len, dims] -> [..., seq_len, num_heads, head_dim]
        queries = queries.reshape(*batch_dims, seq_len, self.num_heads, self.head_dim)
        keys = keys.reshape(*batch_dims, kv_seq_len, self.num_heads, self.head_dim)
        values = values.reshape(*batch_dims, kv_seq_len, self.num_heads, self.head_dim)

        # Transpose to [..., num_heads, seq_len, head_dim]
        if len(batch_dims) == 1:
            queries = queries.transpose(0, 2, 1, 3)
            keys = keys.transpose(0, 2, 1, 3)
            values = values.transpose(0, 2, 1, 3)
        else:
            # Handle unbatched or multi-batch-dim case
            # [..., seq_len, num_heads, head_dim] -> [..., num_heads, seq_len, head_dim]
            queries = queries.swapaxes(-3, -2)
            keys = keys.swapaxes(-3, -2)
            values = values.swapaxes(-3, -2)

        # Use FlashAttention with fixed split-size
        attn_output = flash_attention_fixed_split(
            queries, keys, values,
            scale=self.scale,
            split_size=self.split_size,
            mask=mask
        )

        # Transpose back: [..., seq_len, num_heads, head_dim]
        if len(batch_dims) == 1:
            attn_output = attn_output.transpose(0, 2, 1, 3)
        else:
            attn_output = attn_output.swapaxes(-3, -2)

        # Reshape to [..., seq_len, dims]
        attn_output = attn_output.reshape(*batch_dims, seq_len, dims)

        # Output projection
        output = self.out_proj(attn_output)

        return output


def scaled_dot_product_attention_deterministic(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache,
    scale: float,
    mask,
    sinks=None,
    split_size: int = 256,
) -> mx.array:
    """
    Batch-invariant replacement for mlx_lm.models.base.scaled_dot_product_attention.

    This function is designed to be a drop-in replacement that uses
    flash_attention_fixed_split for deterministic attention computation.

    Args:
        queries: Query tensor [B, H, N, D]
        keys: Key tensor [B, H, S, D] or quantized tuple
        values: Value tensor [B, H, S, D] or quantized tuple
        cache: KV cache (may have .bits for quantized models)
        scale: Attention scale factor
        mask: Attention mask (None, "causal", bool array, or float array)
        sinks: Attention sinks (not supported, will warn if provided)
        split_size: KV split size for flash attention

    Returns:
        Attention output [B, H, N, D]
    """
    # Handle quantized cache - pass through to original quantized path
    if cache is not None and hasattr(cache, "bits"):
        from mlx_lm.models.base import quantized_scaled_dot_product_attention
        return quantized_scaled_dot_product_attention(
            queries, keys, values,
            scale=scale, mask=mask,
            group_size=cache.group_size, bits=cache.bits,
        )

    # Warn about unsupported attention sinks
    if sinks is not None:
        import warnings
        warnings.warn("Attention sinks not supported in deterministic mode, ignoring")

    # Handle string mask ("causal") - convert to additive mask array
    if isinstance(mask, str) and mask == "causal":
        N = queries.shape[-2]  # query sequence length
        S = keys.shape[-2]     # key sequence length
        # Create indices for causal masking
        q_indices = mx.arange(S - N, S)[:, None]  # [N, 1]
        k_indices = mx.arange(S)[None, :]          # [1, S]
        # Causal: query i can attend to keys <= i
        causal_mask = q_indices >= k_indices       # [N, S] bool
        # Convert to additive mask (0 for attend, -inf for masked)
        mask = mx.where(causal_mask,
                       mx.zeros((N, S), dtype=queries.dtype),
                       mx.full((N, S), -1e9, dtype=queries.dtype))
    elif mask is not None and hasattr(mask, 'dtype') and mask.dtype == mx.bool_:
        # Convert bool mask to additive mask
        mask = mx.where(mask,
                       mx.zeros(mask.shape, dtype=queries.dtype),
                       mx.full(mask.shape, -1e9, dtype=queries.dtype))

    # Use our batch-invariant flash attention
    return flash_attention_fixed_split(
        queries, keys, values,
        scale=scale,
        split_size=split_size,
        mask=mask,
    )


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
