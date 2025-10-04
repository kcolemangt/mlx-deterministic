"""
Test suite for Batch-Invariant RMSNorm
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_deterministic.ops.rms_norm import BatchInvariantRMSNorm, rms_norm_batch_invariant


def test_batch_invariance_basic():
    """
    Test: Process batch_size=1 vs batch_size=100, slice first element.
    Expected: Outputs are bitwise identical.
    """
    dims = 512
    eps = 1e-6
    chunk_size = 64

    # Create deterministic input data
    mx.random.seed(42)
    data = mx.random.normal((100, dims))

    # Create model
    model = BatchInvariantRMSNorm(dims, eps=eps, chunk_size=chunk_size)

    # Process batch_size=1
    output_single = model(data[0:1])

    # Process full batch and slice
    output_batch = model(data)
    output_batch_sliced = output_batch[0:1]

    # Compare
    diff = mx.abs(output_single - output_batch_sliced)
    max_diff = mx.max(diff).item()

    assert max_diff == 0.0, f"Outputs differ by {max_diff}, expected exactly 0.0"


def test_batch_invariance_varied_sizes():
    """
    Test: Multiple batch sizes (1, 2, 4, 8, 16, 32, 64, 128).
    Expected: All produce identical results for same input data.
    """
    dims = 256
    eps = 1e-6
    chunk_size = 64

    # Create deterministic input data
    mx.random.seed(42)
    data = mx.random.normal((128, dims))

    # Create model
    model = BatchInvariantRMSNorm(dims, eps=eps, chunk_size=chunk_size)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    outputs = []

    for batch_size in batch_sizes:
        output = model(data[:batch_size])
        outputs.append(output[0])  # Store first element

    # Compare all outputs to the first one
    reference = outputs[0]
    for i, output in enumerate(outputs[1:], 1):
        diff = mx.abs(output - reference)
        max_diff = mx.max(diff).item()
        assert max_diff == 0.0, (
            f"Batch size {batch_sizes[i]} differs from batch size 1 by {max_diff}"
        )


def test_numerical_correctness():
    """
    Test: Output values are numerically reasonable.
    Expected: Variance ≈ 1 after normalization (mean may not be 0 for RMSNorm).
    """
    dims = 512
    eps = 1e-6

    # Create input
    mx.random.seed(42)
    x = mx.random.normal((32, dims))

    # Apply normalization
    model = BatchInvariantRMSNorm(dims, eps=eps)
    output = model(x)

    # Check variance is close to 1
    variance = mx.var(output, axis=-1)
    mean_variance = mx.mean(variance).item()

    # RMSNorm should produce variance close to 1
    assert 0.5 < mean_variance < 1.5, (
        f"Mean variance {mean_variance} is not close to 1"
    )


def test_vs_standard_rmsnorm():
    """
    Test: Compare against MLX standard RMSNorm for single item.
    Expected: Results are close (within floating point tolerance).

    Note: Exact match not expected due to different variance computation,
    but should be numerically similar.
    """
    dims = 512
    eps = 1e-6

    # Create single input
    mx.random.seed(42)
    x = mx.random.normal((1, dims))

    # Batch-invariant RMSNorm
    bi_model = BatchInvariantRMSNorm(dims, eps=eps, chunk_size=64)
    bi_output = bi_model(x)

    # Standard MLX RMSNorm
    std_model = nn.RMSNorm(dims, eps=eps)
    std_model.weight = bi_model.weight  # Use same weights
    std_output = std_model(x)

    # Compare
    diff = mx.abs(bi_output - std_output)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    # Should be reasonably close
    assert max_diff < 1e-3, f"Max difference {max_diff} too large"
    assert mean_diff < 1e-4, f"Mean difference {mean_diff} too large"


def test_gradient_computation():
    """
    Test: Verify gradients can be computed.
    Expected: Backward pass works without errors.
    """
    dims = 256

    # Create model and input
    model = BatchInvariantRMSNorm(dims)
    mx.random.seed(42)
    x = mx.random.normal((8, dims))

    # Define loss function
    def loss_fn(x):
        output = model(x)
        return mx.sum(output * output)

    # Compute gradients
    loss, grads = mx.value_and_grad(loss_fn)(x)

    # Check gradients exist and are finite
    assert not mx.isnan(loss).item(), "Loss is NaN"
    assert not mx.isnan(grads).any().item(), "Gradients contain NaN"
    assert not mx.isinf(grads).any().item(), "Gradients contain Inf"


def test_functional_interface():
    """
    Test the functional batch-invariant RMSNorm interface.
    """
    dims = 256
    eps = 1e-6
    chunk_size = 64

    # Create input and weight
    mx.random.seed(42)
    x = mx.random.normal((32, dims))
    weight = mx.ones((dims,))

    # Apply functional normalization
    output = rms_norm_batch_invariant(x, weight, eps=eps, chunk_size=chunk_size)

    # Check output shape
    assert output.shape == x.shape, "Output shape mismatch"

    # Check output is reasonable
    variance = mx.var(output, axis=-1)
    mean_variance = mx.mean(variance).item()
    assert 0.5 < mean_variance < 1.5, (
        f"Mean variance {mean_variance} is not close to 1"
    )


def test_different_chunk_sizes():
    """
    Test that different chunk sizes produce consistent batch-invariant results.
    """
    dims = 256
    eps = 1e-6

    # Create input
    mx.random.seed(42)
    data = mx.random.normal((128, dims))

    chunk_sizes = [32, 64, 128]
    outputs = []

    for chunk_size in chunk_sizes:
        model = BatchInvariantRMSNorm(dims, eps=eps, chunk_size=chunk_size)

        # Test batch invariance for this chunk size
        output_single = model(data[0:1])
        output_batch = model(data)

        diff = mx.abs(output_single - output_batch[0:1])
        max_diff = mx.max(diff).item()

        assert max_diff == 0.0, (
            f"Chunk size {chunk_size}: batch invariance violated by {max_diff}"
        )


def test_edge_case_batch_equals_chunk_size():
    """
    Test edge case where batch size equals chunk size.
    """
    dims = 256
    chunk_size = 64

    mx.random.seed(42)
    data = mx.random.normal((chunk_size, dims))

    model = BatchInvariantRMSNorm(dims, chunk_size=chunk_size)

    # Process full chunk
    output_full = model(data)

    # Process first element
    output_single = model(data[0:1])

    # Should be batch invariant
    diff = mx.abs(output_single - output_full[0:1])
    max_diff = mx.max(diff).item()

    assert max_diff == 0.0, f"Batch size == chunk size failed: diff = {max_diff}"


def test_edge_case_batch_smaller_than_chunk():
    """
    Test edge case where batch size is smaller than chunk size.
    """
    dims = 256
    chunk_size = 64

    mx.random.seed(42)
    data = mx.random.normal((32, dims))  # Smaller than chunk_size

    model = BatchInvariantRMSNorm(dims, chunk_size=chunk_size)

    # Process all
    output_all = model(data)

    # Process first element
    output_single = model(data[0:1])

    # Should be batch invariant
    diff = mx.abs(output_single - output_all[0:1])
    max_diff = mx.max(diff).item()

    assert max_diff == 0.0, f"Batch size < chunk size failed: diff = {max_diff}"


if __name__ == "__main__":
    # Run all tests
    test_batch_invariance_basic()
    print("✓ test_batch_invariance_basic passed")

    test_batch_invariance_varied_sizes()
    print("✓ test_batch_invariance_varied_sizes passed")

    test_numerical_correctness()
    print("✓ test_numerical_correctness passed")

    test_vs_standard_rmsnorm()
    print("✓ test_vs_standard_rmsnorm passed")

    test_gradient_computation()
    print("✓ test_gradient_computation passed")

    test_functional_interface()
    print("✓ test_functional_interface passed")

    test_different_chunk_sizes()
    print("✓ test_different_chunk_sizes passed")

    test_edge_case_batch_equals_chunk_size()
    print("✓ test_edge_case_batch_equals_chunk_size passed")

    test_edge_case_batch_smaller_than_chunk()
    print("✓ test_edge_case_batch_smaller_than_chunk passed")

    print("\nAll RMSNorm tests passed!")
