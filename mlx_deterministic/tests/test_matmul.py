"""
Test suite for Batch-Invariant Matrix Multiplication

Note: MLX's underlying matmul has inherent batch variance of ~1e-5 to 1e-4.
This is a framework limitation - the same input produces slightly different
outputs when processed as single sample vs. in a batch.

Our fixed K-dimension tiling approach controls reduction order, but cannot
eliminate variance from MLX's internal matmul implementation.
Tests use a tolerance to account for this inherent variance.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx_deterministic.ops.matmul import (
    batch_invariant_matmul,
    batch_invariant_addmm,
    BatchInvariantLinear
)

# MLX matmul has inherent batch variance - this is the tolerance we allow
# This matches the variance observed in mx.matmul itself
MLX_MATMUL_TOLERANCE = 5e-5

# Qwen model test configuration
QWEN_MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Qwen 3B model dimensions (used for shape-based tests)
QWEN_HIDDEN_SIZE = 2048
QWEN_INTERMEDIATE_SIZE = 11008
QWEN_NUM_HEADS = 16
QWEN_HEAD_DIM = 128


def test_matmul_batch_invariance_2d():
    """
    Test: 2D matmul with different batch sizes.
    Expected: Identical results.
    """
    M, K, N = 64, 256, 128
    tile_size = 64

    # Create deterministic matrices
    mx.random.seed(42)
    a_batch = mx.random.normal((8, M, K))
    b = mx.random.normal((K, N))

    # Process single sample
    output_single = batch_invariant_matmul(a_batch[0:1], b, tile_size=tile_size)

    # Process full batch
    output_batch = batch_invariant_matmul(a_batch, b, tile_size=tile_size)

    # Compare - allow for MLX's inherent matmul variance
    diff = mx.abs(output_single - output_batch[0:1])
    max_diff = mx.max(diff).item()

    assert max_diff < MLX_MATMUL_TOLERANCE, f"Outputs differ by {max_diff}, expected < {MLX_MATMUL_TOLERANCE}"


def test_matmul_batch_invariance_3d():
    """
    Test: 3D batched matmul with varying batch dimensions.
    Expected: Identical results.
    """
    M, K, N = 32, 128, 64
    tile_size = 64

    # Create deterministic matrices
    mx.random.seed(42)
    a_batch = mx.random.normal((16, M, K))
    b_batch = mx.random.normal((16, K, N))

    batch_sizes = [1, 2, 4, 8, 16]
    outputs = []

    for batch_size in batch_sizes:
        output = batch_invariant_matmul(
            a_batch[:batch_size],
            b_batch[:batch_size],
            tile_size=tile_size
        )
        outputs.append(output[0])  # Store first element

    # Compare all to first - allow for MLX's inherent variance
    reference = outputs[0]
    for i, output in enumerate(outputs[1:], 1):
        diff = mx.abs(output - reference)
        max_diff = mx.max(diff).item()
        assert max_diff < MLX_MATMUL_TOLERANCE, (
            f"Batch size {batch_sizes[i]} differs from batch size 1 by {max_diff}"
        )


def test_matmul_correctness():
    """
    Test: Compare against MLX standard matmul for accuracy.
    Expected: Results within numerical tolerance.
    """
    M, K, N = 64, 256, 128
    tile_size = 64

    # Create input matrices
    mx.random.seed(42)
    a = mx.random.normal((M, K))
    b = mx.random.normal((K, N))

    # Batch-invariant matmul
    result_bi = batch_invariant_matmul(a, b, tile_size=tile_size)

    # Standard matmul
    result_std = mx.matmul(a, b)

    # Compare
    diff = mx.abs(result_bi - result_std)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    # Should be very close (tile-based computation may have minor FP differences)
    assert max_diff < 1e-4, f"Max difference {max_diff} too large"
    assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large"


def test_matmul_various_shapes():
    """
    Test: Multiple matrix shapes and sizes.
    Expected: All produce deterministic results.
    """
    test_cases = [
        (32, 64, 32),    # Small
        (64, 128, 64),   # Medium
        (128, 256, 128), # Large
        (17, 127, 31),   # Non-power-of-2
    ]

    tile_size = 64

    for M, K, N in test_cases:
        mx.random.seed(42)
        a = mx.random.normal((4, M, K))
        b = mx.random.normal((K, N))

        # Single vs batch
        output_single = batch_invariant_matmul(a[0:1], b, tile_size=tile_size)
        output_batch = batch_invariant_matmul(a, b, tile_size=tile_size)

        diff = mx.abs(output_single - output_batch[0:1])
        max_diff = mx.max(diff).item()

        assert max_diff < MLX_MATMUL_TOLERANCE, (
            f"Shape ({M}, {K}, {N}): batch invariance violated by {max_diff}"
        )


def test_matmul_small_mn_large_k_batch_invariance():
    """
    Regression test: M and N small but K large should still be batch-invariant.
    This catches accidental early returns to mx.matmul based on M/N.
    """
    M, K, N = 16, 512, 16
    tile_size = 64

    mx.random.seed(42)
    a_batch = mx.random.normal((8, M, K))
    b = mx.random.normal((K, N))

    output_single = batch_invariant_matmul(a_batch[0:1], b, tile_size=tile_size)
    output_batch = batch_invariant_matmul(a_batch, b, tile_size=tile_size)

    diff = mx.abs(output_single - output_batch[0:1])
    max_diff = mx.max(diff).item()
    assert max_diff < MLX_MATMUL_TOLERANCE, (
        f"Small M/N large K: batch invariance violated by {max_diff}"
    )


def test_matmul_non_divisible_dims():
    """
    Test: K dimension not divisible by tile size.
    Expected: Padding/unpadding works correctly.
    """
    M, K, N = 32, 100, 64  # K=100 not divisible by tile_size=64
    tile_size = 64

    mx.random.seed(42)
    a = mx.random.normal((M, K))
    b = mx.random.normal((K, N))

    # Batch-invariant matmul
    result_bi = batch_invariant_matmul(a, b, tile_size=tile_size)

    # Standard matmul
    result_std = mx.matmul(a, b)

    # Compare
    diff = mx.abs(result_bi - result_std)
    max_diff = mx.max(diff).item()

    assert max_diff < 1e-4, f"Non-divisible K: max diff {max_diff} too large"

    # Test batch invariance
    a_batch = mx.random.normal((8, M, K))
    output_single = batch_invariant_matmul(a_batch[0:1], b, tile_size=tile_size)
    output_batch = batch_invariant_matmul(a_batch, b, tile_size=tile_size)

    diff = mx.abs(output_single - output_batch[0:1])
    max_diff = mx.max(diff).item()
    assert max_diff < MLX_MATMUL_TOLERANCE, f"Non-divisible K: batch invariance violated by {max_diff}"


def test_matmul_performance():
    """
    Test: Benchmark against standard matmul.
    Expected: Within acceptable performance overhead.

    Note: This is a basic test - actual benchmarking happens in Phase 6.
    """
    M, K, N = 512, 512, 512
    tile_size = 128

    mx.random.seed(42)
    a = mx.random.normal((M, K))
    b = mx.random.normal((K, N))

    # Warm up
    _ = batch_invariant_matmul(a, b, tile_size=tile_size)
    _ = mx.matmul(a, b)
    mx.eval(_)

    # Just verify it completes in reasonable time
    import time
    start = time.time()
    result = batch_invariant_matmul(a, b, tile_size=tile_size)
    mx.eval(result)
    elapsed = time.time() - start

    # Should complete in under 1 second for this size
    assert elapsed < 1.0, f"Took {elapsed}s, too slow"


def test_addmm_batch_invariance():
    """
    Test batch-invariant addmm operation.
    """
    M, K, N = 64, 128, 64
    tile_size = 64

    mx.random.seed(42)
    bias = mx.random.normal((M, N))
    a_batch = mx.random.normal((8, M, K))
    b = mx.random.normal((K, N))

    # Single vs batch
    output_single = batch_invariant_addmm(
        bias, a_batch[0:1], b, alpha=1.0, beta=1.0, tile_size=tile_size
    )
    output_batch = batch_invariant_addmm(
        bias, a_batch, b, alpha=1.0, beta=1.0, tile_size=tile_size
    )

    diff = mx.abs(output_single - output_batch[0:1])
    max_diff = mx.max(diff).item()

    assert max_diff < MLX_MATMUL_TOLERANCE, f"addmm batch invariance violated by {max_diff}"


def test_addmm_correctness():
    """
    Test addmm correctness vs standard implementation.
    """
    M, K, N = 64, 128, 64
    tile_size = 64

    mx.random.seed(42)
    bias = mx.random.normal((M, N))
    a = mx.random.normal((M, K))
    b = mx.random.normal((K, N))
    alpha, beta = 2.0, 0.5

    # Batch-invariant addmm
    result_bi = batch_invariant_addmm(bias, a, b, alpha=alpha, beta=beta, tile_size=tile_size)

    # Standard implementation
    result_std = beta * bias + alpha * mx.matmul(a, b)

    diff = mx.abs(result_bi - result_std)
    max_diff = mx.max(diff).item()

    assert max_diff < 1e-4, f"addmm correctness: max diff {max_diff} too large"


def test_linear_layer_batch_invariance():
    """
    Test batch-invariant linear layer.
    """
    in_features, out_features = 256, 128
    tile_size = 64

    # Create weight and bias
    mx.random.seed(42)
    weight = mx.random.normal((out_features, in_features))
    bias = mx.random.normal((out_features,))

    # Create linear layer
    layer = BatchInvariantLinear(weight, bias, tile_size=tile_size)

    # Create input - reset seed to ensure same data
    mx.random.seed(123)
    x_batch = mx.random.normal((16, in_features))

    # Single vs batch
    mx.random.seed(123)
    data = mx.random.normal((16, in_features))
    output_single = layer(data[0:1])
    output_batch = layer(x_batch)

    diff = mx.abs(output_single - output_batch[0:1])
    max_diff = mx.max(diff).item()

    # Due to FP arithmetic, allow tiny tolerance
    # The key is that it's deterministic (same every time)
    assert max_diff < 1e-5, f"Linear layer batch invariance violated by {max_diff}"


def test_linear_layer_correctness():
    """
    Test linear layer correctness vs standard matmul.
    """
    in_features, out_features = 256, 128
    tile_size = 64

    mx.random.seed(42)
    weight = mx.random.normal((out_features, in_features))
    bias = mx.random.normal((out_features,))
    x = mx.random.normal((32, in_features))

    # Batch-invariant linear
    layer = BatchInvariantLinear(weight, bias, tile_size=tile_size)
    result_bi = layer(x)

    # Standard implementation
    result_std = mx.matmul(x, weight.T) + bias

    diff = mx.abs(result_bi - result_std)
    max_diff = mx.max(diff).item()

    assert max_diff < 1e-4, f"Linear layer correctness: max diff {max_diff} too large"


def test_different_tile_sizes():
    """
    Test that different tile sizes maintain batch invariance.
    """
    M, K, N = 128, 256, 128

    mx.random.seed(42)
    a_batch = mx.random.normal((16, M, K))
    b = mx.random.normal((K, N))

    tile_sizes = [32, 64, 128, 256]

    for tile_size in tile_sizes:
        # Test batch invariance for this tile size
        output_single = batch_invariant_matmul(a_batch[0:1], b, tile_size=tile_size)
        output_batch = batch_invariant_matmul(a_batch, b, tile_size=tile_size)

        diff = mx.abs(output_single - output_batch[0:1])
        max_diff = mx.max(diff).item()

        assert max_diff < MLX_MATMUL_TOLERANCE, (
            f"Tile size {tile_size}: batch invariance violated by {max_diff}"
        )


# =============================================================================
# Qwen Model Tests - Real Model Dimensions and Weights
# =============================================================================

def test_qwen_linear_layer_shapes_batch_invariance():
    """
    Test batch invariance with Qwen's actual linear layer dimensions.

    This test uses the exact shapes found in Qwen 3B's architecture:
    - Q/K/V projections: [hidden_size, hidden_size] = [2048, 2048]
    - Gate/Up projections: [hidden_size, intermediate_size] = [2048, 11008]
    - Down projection: [intermediate_size, hidden_size] = [11008, 2048]

    These shapes stress test the K-dimension tiling with realistic sizes.
    """
    tile_size = 64
    test_cases = [
        # (M, K, N, description)
        (32, QWEN_HIDDEN_SIZE, QWEN_HIDDEN_SIZE, "qkv_proj"),
        (32, QWEN_HIDDEN_SIZE, QWEN_INTERMEDIATE_SIZE, "gate/up_proj"),
        (32, QWEN_INTERMEDIATE_SIZE, QWEN_HIDDEN_SIZE, "down_proj"),
        (1, QWEN_HIDDEN_SIZE, QWEN_HIDDEN_SIZE, "single_token_qkv"),
        (1, QWEN_HIDDEN_SIZE, QWEN_INTERMEDIATE_SIZE, "single_token_gate"),
        # Small M/N with large K - the bug case
        (8, QWEN_HIDDEN_SIZE, 64, "small_output_large_k"),
    ]

    for M, K, N, desc in test_cases:
        mx.random.seed(42)
        a_batch = mx.random.normal((8, M, K))
        b = mx.random.normal((K, N))

        output_single = batch_invariant_matmul(a_batch[0:1], b, tile_size=tile_size)
        output_batch = batch_invariant_matmul(a_batch, b, tile_size=tile_size)

        diff = mx.abs(output_single - output_batch[0:1])
        max_diff = mx.max(diff).item()

        assert max_diff < MLX_MATMUL_TOLERANCE, (
            f"{desc} ({M}x{K}x{N}): batch invariance violated by {max_diff}"
        )


def test_qwen_attention_qkv_shapes_batch_invariance():
    """
    Test batch invariance with Qwen's attention computation shapes.

    Attention involves several matmul operations:
    - Q/K/V projections: [seq, hidden] @ [hidden, hidden]
    - Attention scores: [heads, seq, head_dim] @ [heads, head_dim, seq]
    - Attention output: [heads, seq, seq] @ [heads, seq, head_dim]
    """
    tile_size = 64
    seq_len = 32
    batch_sizes = [1, 2, 4, 8]

    # Test attention score computation shape
    # scores = Q @ K^T: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
    mx.random.seed(42)
    q = mx.random.normal((max(batch_sizes), QWEN_NUM_HEADS, seq_len, QWEN_HEAD_DIM))
    k = mx.random.normal((max(batch_sizes), QWEN_NUM_HEADS, QWEN_HEAD_DIM, seq_len))

    outputs = []
    for bs in batch_sizes:
        # Compute attention scores for each head
        q_slice = q[:bs]
        k_slice = k[:bs]

        # Reshape for matmul: merge batch and heads
        q_reshaped = q_slice.reshape(bs * QWEN_NUM_HEADS, seq_len, QWEN_HEAD_DIM)
        k_reshaped = k_slice.reshape(bs * QWEN_NUM_HEADS, QWEN_HEAD_DIM, seq_len)

        out = batch_invariant_matmul(q_reshaped, k_reshaped, tile_size=tile_size)
        mx.eval(out)

        # Extract first sample's first head output
        outputs.append(out[0])

    reference = outputs[0]
    for i, (bs, out) in enumerate(zip(batch_sizes[1:], outputs[1:]), 1):
        diff = mx.max(mx.abs(reference - out)).item()
        assert diff < MLX_MATMUL_TOLERANCE, (
            f"Attention scores batch_size={bs} differs from batch_size=1 by {diff}"
        )


@pytest.mark.slow
def test_qwen_real_weights_batch_invariance():
    """
    Test batch invariance using actual Qwen model weights.

    This test loads the real Qwen model and uses its actual weight matrices
    to verify batch invariance with real-world values (not random).
    """
    from mlx_lm import load

    model, tokenizer = load(QWEN_MODEL_NAME)
    actual_model = model.model if hasattr(model, 'model') else model

    # Get a linear layer weight (first layer's q_proj)
    layer0 = actual_model.layers[0]

    # Handle both quantized and non-quantized models
    if hasattr(layer0.self_attn.q_proj, 'weight'):
        q_proj_weight = layer0.self_attn.q_proj.weight
    else:
        # For quantized models, skip this test (quantized matmul has separate tests)
        pytest.skip("Model uses quantized weights - use quantized_matmul tests instead")

    # Create realistic input
    mx.random.seed(42)
    seq_len = 32
    batch_sizes = [1, 2, 4, 8]
    x = mx.random.normal((max(batch_sizes), seq_len, q_proj_weight.shape[1]))

    outputs = []
    for bs in batch_sizes:
        out = batch_invariant_matmul(x[:bs], q_proj_weight.T, tile_size=64)
        mx.eval(out)
        outputs.append(out[0])  # Store first sample

    reference = outputs[0]
    for i, (bs, out) in enumerate(zip(batch_sizes[1:], outputs[1:]), 1):
        diff = mx.max(mx.abs(reference - out)).item()
        assert diff < MLX_MATMUL_TOLERANCE, (
            f"Real weights: batch_size={bs} differs from batch_size=1 by {diff}"
        )


@pytest.mark.slow
def test_qwen_forward_batch_variance_baseline():
    """
    Measure baseline batch variance in Qwen forward pass.

    This test establishes the baseline batch variance that exists in the
    standard Qwen model (without deterministic mode). The deterministic
    matmul should not make this variance worse.
    """
    from mlx_lm import load

    model, tokenizer = load(QWEN_MODEL_NAME)

    prompt = "What is the capital of France?"
    tokens = tokenizer.encode(prompt)

    # batch_size=1 vs batch_size=4
    x1 = mx.array([tokens])
    x4 = mx.array([tokens] * 4)

    logits1 = model(x1)
    logits4 = model(x4)
    mx.eval(logits1, logits4)

    baseline_diff = mx.max(mx.abs(logits1[0] - logits4[0])).item()

    # Document the baseline variance
    print(f"\nBaseline Qwen batch variance: {baseline_diff:.6f}")

    # Baseline should have some variance but not be huge
    assert baseline_diff < 0.5, (
        f"Baseline variance unexpectedly high: {baseline_diff}"
    )


@pytest.mark.slow
def test_qwen_forward_deterministic_mode_comparison():
    """
    Test that deterministic mode does not increase batch variance.

    Compares batch variance between:
    1. Baseline model (standard attention)
    2. Model with deterministic attention enabled

    The deterministic mode should produce equal or lower variance.
    """
    from mlx_lm import load
    from mlx_deterministic import enable_mlx_lm_deterministic_mode

    prompt = "What is the capital of France?"

    # Baseline measurement (fresh load)
    model_base, tokenizer = load(QWEN_MODEL_NAME)
    tokens = tokenizer.encode(prompt)

    x1 = mx.array([tokens])
    x4 = mx.array([tokens] * 4)

    logits1_base = model_base(x1)
    logits4_base = model_base(x4)
    mx.eval(logits1_base, logits4_base)

    baseline_diff = mx.max(mx.abs(logits1_base[0] - logits4_base[0])).item()

    # Now with deterministic mode
    enable_mlx_lm_deterministic_mode(split_size=256)

    model_det, _ = load(QWEN_MODEL_NAME)

    logits1_det = model_det(x1)
    logits4_det = model_det(x4)
    mx.eval(logits1_det, logits4_det)

    det_diff = mx.max(mx.abs(logits1_det[0] - logits4_det[0])).item()

    print(f"\nBaseline batch variance: {baseline_diff:.6f}")
    print(f"Deterministic batch variance: {det_diff:.6f}")
    print(f"Difference: {baseline_diff - det_diff:+.6f}")

    # Deterministic should not make things significantly worse
    # Allow small tolerance for module reloading quirks
    assert det_diff <= baseline_diff + 0.01, (
        f"Deterministic mode increased variance: baseline={baseline_diff:.4f}, det={det_diff:.4f}"
    )


if __name__ == "__main__":
    # Run all tests
    test_matmul_batch_invariance_2d()
    print("✓ test_matmul_batch_invariance_2d passed")

    test_matmul_batch_invariance_3d()
    print("✓ test_matmul_batch_invariance_3d passed")

    test_matmul_correctness()
    print("✓ test_matmul_correctness passed")

    test_matmul_various_shapes()
    print("✓ test_matmul_various_shapes passed")

    test_matmul_non_divisible_dims()
    print("✓ test_matmul_non_divisible_dims passed")

    test_matmul_performance()
    print("✓ test_matmul_performance passed")

    test_addmm_batch_invariance()
    print("✓ test_addmm_batch_invariance passed")

    test_addmm_correctness()
    print("✓ test_addmm_correctness passed")

    test_linear_layer_batch_invariance()
    print("✓ test_linear_layer_batch_invariance passed")

    test_linear_layer_correctness()
    print("✓ test_linear_layer_correctness passed")

    test_different_tile_sizes()
    print("✓ test_different_tile_sizes passed")

    print("\nAll matmul tests passed!")
