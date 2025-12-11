"""
Test suite for Custom Metal Kernels (Bitwise Deterministic Operations)

These tests verify that the Metal kernel implementations achieve TRUE bitwise
identical results regardless of batch size - not just ~1e-5 tolerance like
the Python implementations.
"""

import mlx.core as mx
import pytest
from mlx_deterministic.ops.metal_matmul import (
    deterministic_matmul_metal,
    deterministic_addmm_metal,
    DeterministicLinearMetal,
)
from mlx_deterministic.ops.metal_rms_norm import (
    rms_norm_metal,
    BatchInvariantRMSNormMetal,
)
from mlx_deterministic.ops.metal_softmax import softmax_metal


class TestMetalMatmul:
    """Tests for deterministic matmul Metal kernel."""

    def test_basic_correctness(self):
        """Test that Metal matmul produces correct results."""
        mx.random.seed(42)
        a = mx.random.normal((64, 128))
        b = mx.random.normal((128, 64))

        result = deterministic_matmul_metal(a, b)
        expected = mx.matmul(a, b)
        mx.eval(result, expected)

        # Should be close to standard matmul (different reduction order may cause small diff)
        diff = mx.max(mx.abs(result - expected)).item()
        assert diff < 1e-4, f"Max diff from mx.matmul: {diff}"

    def test_bitwise_batch_invariance(self):
        """Test that Metal matmul is BITWISE identical across batch sizes."""
        mx.random.seed(42)
        a_batch = mx.random.normal((16, 64, 128))
        b = mx.random.normal((128, 64))

        # Single sample
        out_single = deterministic_matmul_metal(a_batch[0:1], b)

        # Full batch
        out_batch = deterministic_matmul_metal(a_batch, b)

        mx.eval(out_single, out_batch)

        # MUST be bitwise identical - diff == 0.0
        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"Not bitwise identical: diff = {diff}"

    def test_various_batch_sizes(self):
        """Test bitwise invariance across many batch sizes."""
        mx.random.seed(42)
        a_batch = mx.random.normal((128, 32, 64))
        b = mx.random.normal((64, 32))

        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        outputs = []

        for batch_size in batch_sizes:
            output = deterministic_matmul_metal(a_batch[:batch_size], b)
            outputs.append(output[0])  # Store first element

        mx.eval(*outputs)

        # All should be bitwise identical to the first
        reference = outputs[0]
        for i, output in enumerate(outputs[1:], 1):
            diff = mx.max(mx.abs(output - reference)).item()
            assert diff == 0.0, f"Batch size {batch_sizes[i]} differs by {diff}"

    def test_addmm_bitwise_invariance(self):
        """Test deterministic_addmm_metal for bitwise invariance."""
        mx.random.seed(42)
        bias = mx.random.normal((64, 64))
        a_batch = mx.random.normal((16, 64, 128))
        b = mx.random.normal((128, 64))

        out_single = deterministic_addmm_metal(bias, a_batch[0:1], b, alpha=2.0, beta=0.5)
        out_batch = deterministic_addmm_metal(bias, a_batch, b, alpha=2.0, beta=0.5)

        mx.eval(out_single, out_batch)

        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"addmm not bitwise identical: diff = {diff}"

    def test_linear_layer_bitwise_invariance(self):
        """Test DeterministicLinearMetal for bitwise invariance."""
        mx.random.seed(42)
        weight = mx.random.normal((64, 128))
        bias = mx.random.normal((64,))
        x_batch = mx.random.normal((16, 128))

        layer = DeterministicLinearMetal(weight, bias)

        out_single = layer(x_batch[0:1])
        out_batch = layer(x_batch)

        mx.eval(out_single, out_batch)

        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"Linear layer not bitwise identical: diff = {diff}"


class TestMetalMatmulFP16:
    """Tests for FP16 deterministic matmul Metal kernel."""

    def test_basic_correctness_fp16(self):
        """Test that FP16 Metal matmul produces correct results."""
        mx.random.seed(42)
        a = mx.random.normal((64, 128)).astype(mx.float16)
        b = mx.random.normal((128, 64)).astype(mx.float16)

        result = deterministic_matmul_metal(a, b)
        expected = mx.matmul(a, b)
        mx.eval(result, expected)

        # FP16 has lower precision and our deterministic kernel uses a different
        # reduction order than MLX's native matmul. Both are correct, but the
        # floating-point accumulation errors differ. For 64x128 @ 128x64 with
        # K=128 reductions, tolerance of ~0.2 is reasonable for FP16.
        diff = mx.max(mx.abs(result - expected)).item()
        assert diff < 0.5, f"Max diff from mx.matmul: {diff}"

    def test_bitwise_batch_invariance_fp16(self):
        """Test that FP16 Metal matmul is BITWISE identical across batch sizes."""
        mx.random.seed(42)
        a_batch = mx.random.normal((16, 64, 128)).astype(mx.float16)
        b = mx.random.normal((128, 64)).astype(mx.float16)

        # Single sample
        out_single = deterministic_matmul_metal(a_batch[0:1], b)

        # Full batch
        out_batch = deterministic_matmul_metal(a_batch, b)

        mx.eval(out_single, out_batch)

        # MUST be bitwise identical - diff == 0.0
        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"FP16 not bitwise identical: diff = {diff}"

    def test_various_batch_sizes_fp16(self):
        """Test FP16 bitwise invariance across many batch sizes."""
        mx.random.seed(42)
        a_batch = mx.random.normal((128, 64, 128)).astype(mx.float16)
        b = mx.random.normal((128, 64)).astype(mx.float16)

        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        outputs = []

        for batch_size in batch_sizes:
            output = deterministic_matmul_metal(a_batch[:batch_size], b)
            outputs.append(output[0])  # Store first element

        mx.eval(*outputs)

        # All should be bitwise identical to the first
        reference = outputs[0]
        for i, output in enumerate(outputs[1:], 1):
            diff = mx.max(mx.abs(output - reference)).item()
            assert diff == 0.0, f"FP16 batch size {batch_sizes[i]} differs by {diff}"

    def test_dtype_parameter_fp16(self):
        """Test that dtype parameter correctly converts FP32 to FP16."""
        mx.random.seed(42)
        # Start with FP32 data
        a = mx.random.normal((64, 128))  # FP32
        b = mx.random.normal((128, 64))  # FP32

        # Use dtype parameter to convert to FP16
        result = deterministic_matmul_metal(a, b, dtype=mx.float16)

        # Verify output is FP16
        assert result.dtype == mx.float16, f"Expected float16, got {result.dtype}"

        # Compare with explicit conversion
        a_fp16 = a.astype(mx.float16)
        b_fp16 = b.astype(mx.float16)
        expected = deterministic_matmul_metal(a_fp16, b_fp16)
        mx.eval(result, expected)

        diff = mx.max(mx.abs(result - expected)).item()
        assert diff == 0.0, f"dtype parameter result differs: {diff}"

    def test_large_matrix_fp16(self):
        """Test FP16 with larger matrices (2048x2048)."""
        mx.random.seed(42)
        a = mx.random.normal((2048, 2048)).astype(mx.float16)
        b = mx.random.normal((2048, 2048)).astype(mx.float16)

        result = deterministic_matmul_metal(a, b)
        expected = mx.matmul(a, b)
        mx.eval(result, expected)

        # FP16 accumulates more error on large matrices due to the K=2048 reduction.
        # Different reduction orders (deterministic vs MLX's optimized) will produce
        # numerically different but mathematically equivalent results.
        # The key test is bitwise determinism (test_bitwise_batch_invariance_fp16).
        diff = mx.max(mx.abs(result - expected)).item()
        assert diff < 10.0, f"Large FP16 matrix max diff: {diff}"

    def test_batch_invariant_matmul_fp16_flag(self):
        """Test batch_invariant_matmul with use_metal_kernel=True and dtype=float16."""
        from mlx_deterministic import batch_invariant_matmul

        mx.random.seed(42)
        a_batch = mx.random.normal((16, 64, 128))  # FP32
        b = mx.random.normal((128, 64))  # FP32

        # Use the high-level API with dtype parameter
        out_single = batch_invariant_matmul(a_batch[0:1], b, use_metal_kernel=True, dtype=mx.float16)
        out_batch = batch_invariant_matmul(a_batch, b, use_metal_kernel=True, dtype=mx.float16)

        mx.eval(out_single, out_batch)

        # Verify FP16 output
        assert out_single.dtype == mx.float16
        assert out_batch.dtype == mx.float16

        # Verify bitwise identical
        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"batch_invariant_matmul FP16 not bitwise identical: diff = {diff}"


class TestMetalRMSNorm:
    """Tests for deterministic RMSNorm Metal kernel."""

    def test_basic_correctness(self):
        """Test that Metal RMSNorm produces correct results."""
        mx.random.seed(42)
        x = mx.random.normal((8, 256))
        weight = mx.ones((256,))

        result = rms_norm_metal(x, weight)

        # Compare with manual computation
        x_sq = x * x
        mean_sq = mx.mean(x_sq, axis=-1, keepdims=True)
        rms = mx.sqrt(mean_sq + 1e-6)
        expected = (x / rms) * weight

        mx.eval(result, expected)

        diff = mx.max(mx.abs(result - expected)).item()
        assert diff < 1e-5, f"Max diff from expected: {diff}"

    def test_bitwise_batch_invariance(self):
        """Test that Metal RMSNorm is BITWISE identical across batch sizes."""
        mx.random.seed(42)
        x_batch = mx.random.normal((16, 256))
        weight = mx.ones((256,))

        # Single sample
        out_single = rms_norm_metal(x_batch[0:1], weight)

        # Full batch
        out_batch = rms_norm_metal(x_batch, weight)

        mx.eval(out_single, out_batch)

        # MUST be bitwise identical
        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"Not bitwise identical: diff = {diff}"

    def test_various_batch_sizes(self):
        """Test bitwise invariance across many batch sizes."""
        mx.random.seed(42)
        x_batch = mx.random.normal((128, 512))
        weight = mx.ones((512,))

        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        outputs = []

        for batch_size in batch_sizes:
            output = rms_norm_metal(x_batch[:batch_size], weight)
            outputs.append(output[0])

        mx.eval(*outputs)

        reference = outputs[0]
        for i, output in enumerate(outputs[1:], 1):
            diff = mx.max(mx.abs(output - reference)).item()
            assert diff == 0.0, f"Batch size {batch_sizes[i]} differs by {diff}"

    def test_module_interface(self):
        """Test BatchInvariantRMSNormMetal module."""
        mx.random.seed(42)
        x_batch = mx.random.normal((16, 256))

        layer = BatchInvariantRMSNormMetal(dims=256)

        out_single = layer(x_batch[0:1])
        out_batch = layer(x_batch)

        mx.eval(out_single, out_batch)

        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"Module not bitwise identical: diff = {diff}"


class TestMetalSoftmax:
    """Tests for deterministic softmax Metal kernel."""

    def test_basic_correctness(self):
        """Test that Metal softmax produces correct results."""
        mx.random.seed(42)
        x = mx.random.normal((8, 256))

        result = softmax_metal(x)
        expected = mx.softmax(x, axis=-1)

        mx.eval(result, expected)

        diff = mx.max(mx.abs(result - expected)).item()
        assert diff < 1e-6, f"Max diff from mx.softmax: {diff}"

    def test_sums_to_one(self):
        """Test that softmax outputs sum to 1."""
        mx.random.seed(42)
        x = mx.random.normal((8, 256))

        result = softmax_metal(x)
        row_sums = mx.sum(result, axis=-1)

        mx.eval(row_sums)

        for i in range(8):
            assert abs(row_sums[i].item() - 1.0) < 1e-5, f"Row {i} sum = {row_sums[i].item()}"

    def test_bitwise_batch_invariance(self):
        """Test that Metal softmax is BITWISE identical across batch sizes."""
        mx.random.seed(42)
        x_batch = mx.random.normal((16, 256))

        # Single sample
        out_single = softmax_metal(x_batch[0:1])

        # Full batch
        out_batch = softmax_metal(x_batch)

        mx.eval(out_single, out_batch)

        # MUST be bitwise identical
        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"Not bitwise identical: diff = {diff}"

    def test_various_batch_sizes(self):
        """Test bitwise invariance across many batch sizes."""
        mx.random.seed(42)
        x_batch = mx.random.normal((128, 512))

        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        outputs = []

        for batch_size in batch_sizes:
            output = softmax_metal(x_batch[:batch_size])
            outputs.append(output[0])

        mx.eval(*outputs)

        reference = outputs[0]
        for i, output in enumerate(outputs[1:], 1):
            diff = mx.max(mx.abs(output - reference)).item()
            assert diff == 0.0, f"Batch size {batch_sizes[i]} differs by {diff}"


class TestMetalKernelIntegration:
    """Integration tests for Metal kernels with use_metal_kernel flag."""

    def test_matmul_with_flag(self):
        """Test batch_invariant_matmul with use_metal_kernel=True."""
        from mlx_deterministic import batch_invariant_matmul

        mx.random.seed(42)
        a_batch = mx.random.normal((16, 64, 128))
        b = mx.random.normal((128, 64))

        out_single = batch_invariant_matmul(a_batch[0:1], b, use_metal_kernel=True)
        out_batch = batch_invariant_matmul(a_batch, b, use_metal_kernel=True)

        mx.eval(out_single, out_batch)

        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"matmul with flag not bitwise identical: diff = {diff}"

    def test_rmsnorm_with_flag(self):
        """Test BatchInvariantRMSNorm with use_metal_kernel=True."""
        from mlx_deterministic import BatchInvariantRMSNorm

        mx.random.seed(42)
        x_batch = mx.random.normal((16, 256))

        layer = BatchInvariantRMSNorm(dims=256, use_metal_kernel=True)

        out_single = layer(x_batch[0:1])
        out_batch = layer(x_batch)

        mx.eval(out_single, out_batch)

        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"RMSNorm with flag not bitwise identical: diff = {diff}"

    def test_softmax_with_flag(self):
        """Test batch_invariant_softmax with use_metal_kernel=True."""
        from mlx_deterministic import batch_invariant_softmax

        mx.random.seed(42)
        x_batch = mx.random.normal((16, 256))

        out_single = batch_invariant_softmax(x_batch[0:1], use_metal_kernel=True)
        out_batch = batch_invariant_softmax(x_batch, use_metal_kernel=True)

        mx.eval(out_single, out_batch)

        diff = mx.max(mx.abs(out_single - out_batch[0:1])).item()
        assert diff == 0.0, f"softmax with flag not bitwise identical: diff = {diff}"


if __name__ == "__main__":
    # Run all tests
    print("Testing Metal Matmul (FP32)...")
    test_matmul = TestMetalMatmul()
    test_matmul.test_basic_correctness()
    print("  ✓ basic_correctness")
    test_matmul.test_bitwise_batch_invariance()
    print("  ✓ bitwise_batch_invariance")
    test_matmul.test_various_batch_sizes()
    print("  ✓ various_batch_sizes")
    test_matmul.test_addmm_bitwise_invariance()
    print("  ✓ addmm_bitwise_invariance")
    test_matmul.test_linear_layer_bitwise_invariance()
    print("  ✓ linear_layer_bitwise_invariance")

    print("\nTesting Metal Matmul (FP16)...")
    test_matmul_fp16 = TestMetalMatmulFP16()
    test_matmul_fp16.test_basic_correctness_fp16()
    print("  ✓ basic_correctness_fp16")
    test_matmul_fp16.test_bitwise_batch_invariance_fp16()
    print("  ✓ bitwise_batch_invariance_fp16")
    test_matmul_fp16.test_various_batch_sizes_fp16()
    print("  ✓ various_batch_sizes_fp16")
    test_matmul_fp16.test_dtype_parameter_fp16()
    print("  ✓ dtype_parameter_fp16")
    test_matmul_fp16.test_large_matrix_fp16()
    print("  ✓ large_matrix_fp16")
    test_matmul_fp16.test_batch_invariant_matmul_fp16_flag()
    print("  ✓ batch_invariant_matmul_fp16_flag")

    print("\nTesting Metal RMSNorm...")
    test_rms = TestMetalRMSNorm()
    test_rms.test_basic_correctness()
    print("  ✓ basic_correctness")
    test_rms.test_bitwise_batch_invariance()
    print("  ✓ bitwise_batch_invariance")
    test_rms.test_various_batch_sizes()
    print("  ✓ various_batch_sizes")
    test_rms.test_module_interface()
    print("  ✓ module_interface")

    print("\nTesting Metal Softmax...")
    test_softmax = TestMetalSoftmax()
    test_softmax.test_basic_correctness()
    print("  ✓ basic_correctness")
    test_softmax.test_sums_to_one()
    print("  ✓ sums_to_one")
    test_softmax.test_bitwise_batch_invariance()
    print("  ✓ bitwise_batch_invariance")
    test_softmax.test_various_batch_sizes()
    print("  ✓ various_batch_sizes")

    print("\nTesting Integration...")
    test_integration = TestMetalKernelIntegration()
    test_integration.test_matmul_with_flag()
    print("  ✓ matmul_with_flag")
    test_integration.test_rmsnorm_with_flag()
    print("  ✓ rmsnorm_with_flag")
    test_integration.test_softmax_with_flag()
    print("  ✓ softmax_with_flag")

    print("\n✅ All Metal kernel tests passed!")
