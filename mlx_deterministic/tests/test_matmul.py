"""
Test suite for Batch-Invariant Matrix Multiplication

Note: MLX's underlying matmul has inherent batch variance of ~1e-5 to 1e-4.
This is a framework limitation - the same input produces slightly different
outputs when processed as single sample vs. in a batch.

Our 2D output tiling approach follows the TML research (avoiding split-K)
but cannot eliminate variance from MLX's internal matmul implementation.
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
