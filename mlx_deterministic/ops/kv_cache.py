"""
Deterministic KV Cache for MLX

This module implements KV cache with deterministic layout guarantees,
ensuring consistent memory layout regardless of batch size or sequence length.

Based on Thinking Machines Labs research:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Research requirement: "Update the KV cache and page table BEFORE the attention
kernel itself, ensuring that your keys and values are always consistently
laid out regardless of how many tokens are being processed."
"""

import mlx.core as mx
from typing import Tuple, Optional


class DeterministicKVCache:
    """
    KV Cache with deterministic layout for batch-invariant attention.

    This cache ensures that KV tensors are always laid out consistently,
    which is critical for deterministic attention computation. The key
    insight from the TML research is that cache layout must be finalized
    BEFORE the attention kernel runs.

    Features:
    1. Pre-attention cache updates
    2. Optional alignment to split boundaries
    3. Consistent layout regardless of batch size

    Args:
        align_to: Alignment boundary (should match attention split_size).
                  If > 0, cache will be padded to multiples of this value.
                  Default: 0 (no alignment padding)
    """

    def __init__(self, align_to: int = 0):
        self.align_to = align_to
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self._seq_len: int = 0

    @property
    def seq_len(self) -> int:
        """Current sequence length in cache (excluding padding)."""
        return self._seq_len

    def reset(self):
        """Clear the cache."""
        self.keys = None
        self.values = None
        self._seq_len = 0

    def update(
        self,
        new_keys: mx.array,
        new_values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache and return consistently-laid-out KV tensors.

        IMPORTANT: This must be called BEFORE the attention kernel runs
        to ensure deterministic layout.

        Args:
            new_keys: New key tensor [..., new_seq_len, dims]
            new_values: New value tensor [..., new_seq_len, dims]

        Returns:
            Tuple of (keys, values) with consistent layout
        """
        new_seq_len = new_keys.shape[-2]

        if self.keys is None:
            # First update - initialize cache
            self.keys = new_keys
            self.values = new_values
            self._seq_len = new_seq_len
        else:
            # Append new tokens
            self.keys = mx.concatenate([self.keys[..., :self._seq_len, :], new_keys], axis=-2)
            self.values = mx.concatenate([self.values[..., :self._seq_len, :], new_values], axis=-2)
            self._seq_len += new_seq_len

        # Return aligned tensors if alignment is specified
        if self.align_to > 0:
            return self._get_aligned()
        else:
            return self.keys, self.values

    def _get_aligned(self) -> Tuple[mx.array, mx.array]:
        """
        Get KV tensors aligned to split boundaries.

        Padding with zeros ensures that attention split boundaries
        are consistent regardless of actual sequence length.
        """
        if self.keys is None:
            raise ValueError("Cache is empty")

        current_len = self._seq_len
        aligned_len = ((current_len + self.align_to - 1) // self.align_to) * self.align_to

        if aligned_len > current_len:
            pad_len = aligned_len - current_len
            pad_shape = list(self.keys.shape)
            pad_shape[-2] = pad_len

            # Pad with zeros (will be masked in attention)
            k_pad = mx.zeros(pad_shape, dtype=self.keys.dtype)
            v_pad = mx.zeros(pad_shape, dtype=self.values.dtype)

            # Return padded tensors (don't modify stored cache)
            return (
                mx.concatenate([self.keys[..., :self._seq_len, :], k_pad], axis=-2),
                mx.concatenate([self.values[..., :self._seq_len, :], v_pad], axis=-2)
            )

        return self.keys[..., :self._seq_len, :], self.values[..., :self._seq_len, :]

    def get_mask_for_aligned(self, query_len: int) -> mx.array:
        """
        Create attention mask that accounts for padding in aligned cache.

        Args:
            query_len: Number of query tokens

        Returns:
            Mask of shape [query_len, aligned_kv_len] with -inf for padded positions
        """
        if self.align_to <= 0:
            # No alignment, no mask needed for padding
            return None

        aligned_len = ((self._seq_len + self.align_to - 1) // self.align_to) * self.align_to

        if aligned_len == self._seq_len:
            return None

        # Create mask: 0 for real tokens, -inf for padding
        kv_idx = mx.arange(aligned_len)
        mask = (kv_idx >= self._seq_len).astype(mx.float32) * -1e9

        # Broadcast to [query_len, aligned_len]
        mask = mx.broadcast_to(mask, (query_len, aligned_len))

        return mask


class RotatingDeterministicKVCache:
    """
    Rotating KV cache with deterministic layout for sliding window attention.

    This cache maintains a fixed-size window of KV pairs, rotating out
    old tokens as new ones are added. The layout is always consistent
    regardless of how many tokens have been processed.

    Args:
        max_size: Maximum number of tokens to cache
        align_to: Alignment boundary for attention splits
    """

    def __init__(self, max_size: int, align_to: int = 0):
        self.max_size = max_size
        self.align_to = align_to
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self._write_idx: int = 0
        self._total_tokens: int = 0

    @property
    def seq_len(self) -> int:
        """Current number of tokens in cache."""
        return min(self._total_tokens, self.max_size)

    def reset(self):
        """Clear the cache."""
        self.keys = None
        self.values = None
        self._write_idx = 0
        self._total_tokens = 0

    def update(
        self,
        new_keys: mx.array,
        new_values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Update rotating cache with new tokens.

        Args:
            new_keys: New key tensor [..., new_seq_len, dims]
            new_values: New value tensor [..., new_seq_len, dims]

        Returns:
            Tuple of (keys, values) for attention
        """
        new_seq_len = new_keys.shape[-2]
        batch_shape = new_keys.shape[:-2]
        dims = new_keys.shape[-1]

        if self.keys is None:
            # Initialize fixed-size buffer
            buffer_shape = list(batch_shape) + [self.max_size, dims]
            self.keys = mx.zeros(buffer_shape, dtype=new_keys.dtype)
            self.values = mx.zeros(buffer_shape, dtype=new_values.dtype)

        # Write new tokens to buffer (with wrap-around)
        for i in range(new_seq_len):
            idx = (self._write_idx + i) % self.max_size
            self.keys = self.keys.at[..., idx, :].set(new_keys[..., i, :])
            self.values = self.values.at[..., idx, :].set(new_values[..., i, :])

        self._write_idx = (self._write_idx + new_seq_len) % self.max_size
        self._total_tokens += new_seq_len

        # Return properly ordered cache
        return self._get_ordered()

    def _get_ordered(self) -> Tuple[mx.array, mx.array]:
        """Get cache contents in correct temporal order."""
        if self._total_tokens <= self.max_size:
            # Haven't filled buffer yet
            return self.keys[..., :self._total_tokens, :], self.values[..., :self._total_tokens, :]

        # Buffer is full, need to reorder
        # Oldest token is at _write_idx, newest at _write_idx - 1
        indices = [(self._write_idx + i) % self.max_size for i in range(self.max_size)]
        indices = mx.array(indices)

        ordered_keys = self.keys[..., indices, :]
        ordered_values = self.values[..., indices, :]

        return ordered_keys, ordered_values
