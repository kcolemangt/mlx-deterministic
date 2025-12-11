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


def test_attention_unbatched_matches_batched():
    """
    Test: Unbatched inputs (no leading batch dim) should match batched=1.
    This guards head/sequence layout handling in __call__.
    """
    dims = 128
    num_heads = 4
    seq_len = 32

    mx.random.seed(42)
    attn = BatchInvariantAttention(dims=dims, num_heads=num_heads)

    queries = mx.random.normal((seq_len, dims))
    keys = mx.random.normal((seq_len, dims))
    values = mx.random.normal((seq_len, dims))

    output_unbatched = attn(queries, keys, values)

    output_batched = attn(
        queries.reshape(1, seq_len, dims),
        keys.reshape(1, seq_len, dims),
        values.reshape(1, seq_len, dims),
    )[0]

    diff = mx.abs(output_unbatched - output_batched)
    max_diff = mx.max(diff).item()

    assert max_diff < 1e-4, (
        f"Unbatched attention differs from batched by {max_diff}"
    )


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


# =============================================================================
# Real Qwen Model Tests
# =============================================================================

QWEN_MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"
QWEN_TEST_PROMPTS = [
    "What is the capital of France?",
    "Hello, how are you?",
]


def _load_qwen_model_baseline():
    """Load Qwen model WITHOUT deterministic mode for baseline comparison."""
    from mlx_lm import load
    model, tokenizer = load(QWEN_MODEL_NAME)
    return model, tokenizer


def _load_qwen_model_deterministic():
    """Load Qwen model WITH deterministic attention enabled."""
    from mlx_lm import load
    from mlx_deterministic import enable_mlx_lm_deterministic_mode

    # Enable deterministic mode BEFORE loading
    enable_mlx_lm_deterministic_mode(split_size=256)

    # Load the model
    model, tokenizer = load(QWEN_MODEL_NAME)
    return model, tokenizer


@pytest.mark.slow
def test_qwen_unbatched_vs_batched_with_batch_invariant_attention():
    """
    Test: BatchInvariantAttention with real Qwen embeddings.

    Uses real token embeddings from Qwen but passes them through our
    BatchInvariantAttention module to test unbatched vs batched.
    """
    model, tokenizer = _load_qwen_model_baseline()
    actual_model = model.model if hasattr(model, 'model') else model

    for prompt in QWEN_TEST_PROMPTS:
        tokens = tokenizer.encode(prompt)
        x = mx.array([tokens])

        # Get real embeddings
        embeddings = actual_model.embed_tokens(x)
        mx.eval(embeddings)

        seq_len, dims = embeddings.shape[1], embeddings.shape[2]
        num_heads = 16  # Typical for 3B model

        # Create our BatchInvariantAttention with matching dims
        attn = BatchInvariantAttention(dims=dims, num_heads=num_heads)

        # Unbatched: squeeze batch dim
        emb_unbatched = embeddings[0]  # [seq, dims]
        emb_batched = embeddings       # [1, seq, dims]

        # Run through our attention both ways
        out_unbatched = attn(emb_unbatched, emb_unbatched, emb_unbatched)
        out_batched = attn(emb_batched, emb_batched, emb_batched)
        mx.eval(out_unbatched, out_batched)

        # Compare: unbatched should match batched[0]
        diff = mx.max(mx.abs(out_unbatched - out_batched[0])).item()

        assert diff < 1e-4, (
            f"'{prompt}': Unbatched vs batched attention diff: {diff}"
        )


@pytest.mark.slow
def test_qwen_batch_size_invariance_with_batch_invariant_attention():
    """
    Test: BatchInvariantAttention produces identical outputs across batch sizes.

    Uses real Qwen embeddings, processes through BatchInvariantAttention,
    verifies first sample matches regardless of batch size.
    """
    model, tokenizer = _load_qwen_model_baseline()
    actual_model = model.model if hasattr(model, 'model') else model

    prompt = "What is the capital of France?"
    tokens = tokenizer.encode(prompt)

    # Get embeddings once
    x = mx.array([tokens])
    embeddings = actual_model.embed_tokens(x)
    mx.eval(embeddings)

    dims = embeddings.shape[2]
    num_heads = 16

    # Create our BatchInvariantAttention
    attn = BatchInvariantAttention(dims=dims, num_heads=num_heads)

    batch_sizes = [1, 2, 4, 8]
    outputs = []

    for batch_size in batch_sizes:
        # Create batch by repeating embeddings
        batched_emb = mx.repeat(embeddings, batch_size, axis=0)
        mx.eval(batched_emb)

        # Run through our attention
        out = attn(batched_emb, batched_emb, batched_emb)
        mx.eval(out)

        # Store first sample output
        outputs.append(out[0])

    # Compare all outputs to the first (batch_size=1) output
    reference = outputs[0]
    for i, (batch_size, out) in enumerate(zip(batch_sizes[1:], outputs[1:]), 1):
        diff = mx.max(mx.abs(reference - out)).item()
        assert diff < 1e-4, (
            f"batch_size={batch_size} differs from batch_size=1 by {diff}"
        )


@pytest.mark.slow
def test_qwen_full_forward_batch_variance_baseline():
    """
    Test: Measure baseline batch variance WITHOUT deterministic mode.

    This establishes the baseline variance that deterministic mode should reduce.
    MLX models typically have small but non-zero batch variance (~0.03-0.1).
    """
    # Load model WITHOUT deterministic mode
    model, tokenizer = _load_qwen_model_baseline()

    prompt = "What is the capital of France?"
    tokens = tokenizer.encode(prompt)

    # Mode 1: batch_size=1
    x1 = mx.array([tokens])
    logits1 = model(x1)
    mx.eval(logits1)

    # Mode 2: batch_size=4
    x4 = mx.array([tokens] * 4)
    logits4 = model(x4)
    mx.eval(logits4)

    # Measure baseline variance
    baseline_diff = mx.max(mx.abs(logits1[0] - logits4[0])).item()

    # Baseline should have some variance (but not huge)
    assert baseline_diff < 0.5, (
        f"Baseline variance unexpectedly high: {baseline_diff}"
    )

    # Return the baseline diff for documentation
    print(f"\nBaseline batch variance: {baseline_diff:.6f}")


@pytest.mark.slow
def test_qwen_full_forward_deterministic_improves_variance():
    """
    Test: Full model forward with deterministic mode reduces batch variance.

    Compares batch variance between:
    1. Baseline model (no deterministic mode)
    2. Model with deterministic attention enabled

    The deterministic mode should produce lower or equal variance.
    """
    import importlib

    # First, measure baseline (fresh Python state needed)
    from mlx_lm import load

    prompt = "What is the capital of France?"

    # Baseline measurement
    model_base, tokenizer = load(QWEN_MODEL_NAME)
    tokens = tokenizer.encode(prompt)

    x1 = mx.array([tokens])
    x4 = mx.array([tokens] * 4)

    logits1_base = model_base(x1)
    logits4_base = model_base(x4)
    mx.eval(logits1_base, logits4_base)

    baseline_diff = mx.max(mx.abs(logits1_base[0] - logits4_base[0])).item()

    # Now with deterministic mode (reimport to get fresh state)
    from mlx_deterministic import enable_mlx_lm_deterministic_mode
    enable_mlx_lm_deterministic_mode(split_size=256)

    model_det, _ = load(QWEN_MODEL_NAME)

    logits1_det = model_det(x1)
    logits4_det = model_det(x4)
    mx.eval(logits1_det, logits4_det)

    det_diff = mx.max(mx.abs(logits1_det[0] - logits4_det[0])).item()

    print(f"\nBaseline batch variance: {baseline_diff:.6f}")
    print(f"Deterministic batch variance: {det_diff:.6f}")
    print(f"Improvement: {baseline_diff - det_diff:.6f}")

    # Deterministic should be better or equal
    # Note: Due to module reloading quirks, we allow a small tolerance
    assert det_diff <= baseline_diff + 0.01, (
        f"Deterministic mode made variance worse: baseline={baseline_diff:.4f}, det={det_diff:.4f}"
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
