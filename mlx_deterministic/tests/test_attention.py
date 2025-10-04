"""
Test suite for Batch-Invariant Attention
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_deterministic.ops.attention import (
    batch_invariant_softmax,
    BatchInvariantAttention,
    create_causal_mask
)


def test_softmax_batch_invariance():
    """
    Test batch-invariant softmax.
    """
    seq_len = 256
    chunk_size = 64

    mx.random.seed(42)
    x_batch = mx.random.normal((16, seq_len))

    # Single vs batch
    output_single = batch_invariant_softmax(x_batch[0:1], axis=-1, chunk_size=chunk_size)
    output_batch = batch_invariant_softmax(x_batch, axis=-1, chunk_size=chunk_size)

    diff = mx.abs(output_single - output_batch[0:1])
    max_diff = mx.max(diff).item()

    assert max_diff == 0.0, f"Softmax batch invariance violated by {max_diff}"


def test_softmax_correctness():
    """
    Test softmax correctness vs standard implementation.
    """
    seq_len = 128
    chunk_size = 64

    mx.random.seed(42)
    x = mx.random.normal((8, seq_len))

    # Batch-invariant softmax
    result_bi = batch_invariant_softmax(x, axis=-1, chunk_size=chunk_size)

    # Standard softmax
    result_std = mx.softmax(x, axis=-1)

    diff = mx.abs(result_bi - result_std)
    max_diff = mx.max(diff).item()

    assert max_diff < 1e-4, f"Softmax correctness: max diff {max_diff} too large"


def test_softmax_sums_to_one():
    """
    Test that softmax outputs sum to 1.
    """
    seq_len = 256
    chunk_size = 64

    mx.random.seed(42)
    x = mx.random.normal((8, seq_len))

    result = batch_invariant_softmax(x, axis=-1, chunk_size=chunk_size)

    sums = mx.sum(result, axis=-1)

    # All sums should be 1
    diff_from_one = mx.abs(sums - 1.0)
    max_diff = mx.max(diff_from_one).item()

    assert max_diff < 1e-5, f"Softmax doesn't sum to 1: max diff {max_diff}"


def test_attention_batch_invariance_simple():
    """
    Test: Single vs batched attention computation.
    Expected: Identical outputs.
    """
    dims = 128
    num_heads = 4
    seq_len = 32

    mx.random.seed(42)

    # Create attention module
    attn = BatchInvariantAttention(
        dims=dims,
        num_heads=num_heads,
        matmul_tile_size=64,
        softmax_chunk_size=64
    )

    # Create batched inputs
    queries = mx.random.normal((8, seq_len, dims))
    keys = mx.random.normal((8, seq_len, dims))
    values = mx.random.normal((8, seq_len, dims))

    # Single sample
    output_single = attn(queries[0:1], keys[0:1], values[0:1])

    # Full batch
    output_batch = attn(queries, keys, values)

    diff = mx.abs(output_single - output_batch[0:1])
    max_diff = mx.max(diff).item()

    # Allow small FP tolerance
    assert max_diff < 1e-4, f"Attention batch invariance violated by {max_diff}"


def test_attention_batch_invariance_varied_seq_lengths():
    """
    Test: Different sequence lengths should still be deterministic.
    """
    dims = 128
    num_heads = 4

    mx.random.seed(42)
    attn = BatchInvariantAttention(dims=dims, num_heads=num_heads)

    seq_lengths = [16, 32, 64]

    for seq_len in seq_lengths:
        queries = mx.random.normal((8, seq_len, dims))
        keys = mx.random.normal((8, seq_len, dims))
        values = mx.random.normal((8, seq_len, dims))

        # Test batch invariance
        output_single = attn(queries[0:1], keys[0:1], values[0:1])
        output_batch = attn(queries, keys, values)

        diff = mx.abs(output_single - output_batch[0:1])
        max_diff = mx.max(diff).item()

        assert max_diff < 1e-4, (
            f"Seq len {seq_len}: batch invariance violated by {max_diff}"
        )


def test_attention_with_mask():
    """
    Test: Causal and padding masks.
    Expected: Correct masking behavior, deterministic.
    """
    dims = 128
    num_heads = 4
    seq_len = 32

    mx.random.seed(42)
    attn = BatchInvariantAttention(dims=dims, num_heads=num_heads)

    queries = mx.random.normal((4, seq_len, dims))
    keys = mx.random.normal((4, seq_len, dims))
    values = mx.random.normal((4, seq_len, dims))

    # Create causal mask
    mask = create_causal_mask(seq_len)

    # Test batch invariance with mask
    output_single = attn(queries[0:1], keys[0:1], values[0:1], mask=mask)
    output_batch = attn(queries, keys, values, mask=mask)

    diff = mx.abs(output_single - output_batch[0:1])
    max_diff = mx.max(diff).item()

    assert max_diff < 1e-4, f"Masked attention batch invariance violated by {max_diff}"


def test_attention_multihead():
    """
    Test: Multiple attention heads.
    Expected: All heads produce deterministic results.
    """
    dims = 256
    seq_len = 32

    head_counts = [1, 2, 4, 8]

    for num_heads in head_counts:
        mx.random.seed(42)
        attn = BatchInvariantAttention(dims=dims, num_heads=num_heads)

        queries = mx.random.normal((8, seq_len, dims))
        keys = mx.random.normal((8, seq_len, dims))
        values = mx.random.normal((8, seq_len, dims))

        # Test batch invariance
        output_single = attn(queries[0:1], keys[0:1], values[0:1])
        output_batch = attn(queries, keys, values)

        diff = mx.abs(output_single - output_batch[0:1])
        max_diff = mx.max(diff).item()

        assert max_diff < 1e-4, (
            f"Num heads {num_heads}: batch invariance violated by {max_diff}"
        )


def test_attention_vs_standard():
    """
    Test: Compare against MLX standard attention (rough comparison).
    Expected: Numerically similar results.
    """
    dims = 128
    num_heads = 4
    seq_len = 32

    mx.random.seed(42)

    # Batch-invariant attention
    bi_attn = BatchInvariantAttention(dims=dims, num_heads=num_heads)

    queries = mx.random.normal((1, seq_len, dims))
    keys = mx.random.normal((1, seq_len, dims))
    values = mx.random.normal((1, seq_len, dims))

    output_bi = bi_attn(queries, keys, values)

    # Standard attention (for comparison - won't be exact due to different implementations)
    std_attn = nn.MultiHeadAttention(dims=dims, num_heads=num_heads)

    # Copy weights to make fair comparison
    std_attn.query_proj.weight = bi_attn.query_proj.weight
    std_attn.key_proj.weight = bi_attn.key_proj.weight
    std_attn.value_proj.weight = bi_attn.value_proj.weight
    std_attn.out_proj.weight = bi_attn.out_proj.weight

    output_std = std_attn(queries, keys, values)

    # Should be reasonably close (not exact due to different softmax implementation)
    diff = mx.abs(output_bi - output_std)
    max_diff = mx.max(diff).item()

    # Allow larger tolerance since implementations differ
    assert max_diff < 0.1, f"Attention vs standard: max diff {max_diff} too large"


def test_causal_mask():
    """
    Test causal mask creation.
    """
    seq_len = 4
    mask = create_causal_mask(seq_len)

    # Check shape
    assert mask.shape == (seq_len, seq_len)

    # Check that it's lower triangular (allowing only past tokens)
    # Position i can attend to positions 0...i
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert mask[i, j].item() == 0.0, f"Position ({i},{j}) should be 0"
            else:
                assert mask[i, j].item() < -1e8, f"Position ({i},{j}) should be masked"


def test_attention_output_shape():
    """
    Test that attention produces correct output shapes.
    """
    dims = 128
    num_heads = 4
    seq_len = 32
    batch_size = 8

    attn = BatchInvariantAttention(dims=dims, num_heads=num_heads)

    queries = mx.random.normal((batch_size, seq_len, dims))
    keys = mx.random.normal((batch_size, seq_len, dims))
    values = mx.random.normal((batch_size, seq_len, dims))

    output = attn(queries, keys, values)

    assert output.shape == (batch_size, seq_len, dims), (
        f"Output shape {output.shape} doesn't match expected {(batch_size, seq_len, dims)}"
    )


if __name__ == "__main__":
    # Run all tests
    test_softmax_batch_invariance()
    print("✓ test_softmax_batch_invariance passed")

    test_softmax_correctness()
    print("✓ test_softmax_correctness passed")

    test_softmax_sums_to_one()
    print("✓ test_softmax_sums_to_one passed")

    test_attention_batch_invariance_simple()
    print("✓ test_attention_batch_invariance_simple passed")

    test_attention_batch_invariance_varied_seq_lengths()
    print("✓ test_attention_batch_invariance_varied_seq_lengths passed")

    test_attention_with_mask()
    print("✓ test_attention_with_mask passed")

    test_attention_multihead()
    print("✓ test_attention_multihead passed")

    test_attention_vs_standard()
    print("✓ test_attention_vs_standard passed")

    test_causal_mask()
    print("✓ test_causal_mask passed")

    test_attention_output_shape()
    print("✓ test_attention_output_shape passed")

    print("\nAll attention tests passed!")
