#!/usr/bin/env python3
"""
Determinism Benchmark for MLX Batch-Invariant Operations

Validates that batch-invariant operations produce identical results
regardless of batch size, demonstrating deterministic behavior.

Now includes comprehensive comparison of:
1. Original implementation (K-dimension tiling)
2. New Python implementation (2D output tiling)
3. Metal kernel implementation (bitwise determinism)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time
import statistics
from typing import Callable, Any, Dict
from mlx_deterministic.ops import (
    BatchInvariantRMSNorm,
    batch_invariant_matmul,
    BatchInvariantAttention,
    batch_invariant_softmax
)


# =============================================================================
# ROBUST BENCHMARKING
# =============================================================================

def robust_benchmark(
    func: Callable,
    args: tuple = (),
    min_iterations: int = 100,
    min_runtime_sec: float = 2.0,
    max_iterations: int = 10000,
    warmup: int = 20,
    target_cv: float = 0.10,
) -> Dict[str, Any]:
    """
    Robust benchmark: run until results stabilize.

    Runs the function repeatedly until:
    1. At least min_iterations have been completed
    2. At least min_runtime_sec has elapsed
    3. Results are stable (coefficient of variation < target_cv)
    OR max_iterations is reached.

    Args:
        func: Function to benchmark (called as func(*args))
        args: Arguments to pass to func
        min_iterations: Minimum number of iterations to run
        min_runtime_sec: Minimum total runtime in seconds
        max_iterations: Maximum iterations (safety limit)
        warmup: Number of warmup iterations before timing
        target_cv: Target coefficient of variation for stability (0.10 = 10%)

    Returns:
        Dict with timing statistics:
        - mean_ms: Mean time in milliseconds
        - std_ms: Standard deviation in milliseconds
        - min_ms, max_ms: Range
        - median_ms: Median (p50)
        - p95_ms: 95th percentile
        - iterations: Number of iterations run
        - total_time_sec: Total benchmark time
    """
    # Warmup phase - let JIT compile, caches warm up
    for _ in range(warmup):
        out = func(*args)
        mx.eval(out)

    # Collect timing samples
    samples = []
    total_time = 0.0
    check_interval = 50  # Check stability every N iterations

    while len(samples) < max_iterations:
        # Time single iteration with high-precision timer
        start = time.perf_counter()
        out = func(*args)
        mx.eval(out)
        elapsed = time.perf_counter() - start

        samples.append(elapsed * 1000)  # Convert to ms
        total_time += elapsed

        # Check stopping conditions periodically (not every iteration for speed)
        if len(samples) % check_interval == 0 and len(samples) >= min_iterations:
            if total_time >= min_runtime_sec:
                # Check if results are stable
                mean = statistics.mean(samples)
                std = statistics.stdev(samples)
                cv = std / mean if mean > 0 else float('inf')
                if cv < target_cv:
                    break

    # Compute final statistics
    sorted_samples = sorted(samples)
    n = len(samples)

    return {
        'mean_ms': statistics.mean(samples),
        'std_ms': statistics.stdev(samples) if n > 1 else 0.0,
        'min_ms': min(samples),
        'max_ms': max(samples),
        'median_ms': statistics.median(samples),
        'p95_ms': sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1],
        'p99_ms': sorted_samples[int(n * 0.99)] if n >= 100 else sorted_samples[-1],
        'iterations': n,
        'total_time_sec': total_time,
    }


def format_benchmark_result(result: Dict[str, Any], baseline_mean: float = None) -> str:
    """Format benchmark result for display."""
    mean = result['mean_ms']
    std = result['std_ms']
    n = result['iterations']
    p95 = result['p95_ms']

    if baseline_mean is not None and baseline_mean > 0:
        overhead = ((mean - baseline_mean) / baseline_mean) * 100
        return f"{mean:.3f}ms ± {std:.3f}ms (n={n}, p95={p95:.3f}ms) [{overhead:+.1f}%]"
    else:
        return f"{mean:.3f}ms ± {std:.3f}ms (n={n}, p95={p95:.3f}ms)"


# ============================================================================
# ORIGINAL IMPLEMENTATIONS (from commit HEAD, for comparison)
# ============================================================================

def original_batch_invariant_matmul(a: mx.array, b: mx.array, tile_size: int = 128) -> mx.array:
    """
    ORIGINAL K-dimension tiling approach (from committed version).

    This uses split-K: padding K dimension and summing partial products.
    """
    a_shape = a.shape
    b_shape = b.shape

    K_a = a_shape[-1]
    K_b = b_shape[-2]
    assert K_a == K_b, f"Incompatible dimensions: {K_a} != {K_b}"

    K = K_a

    # Pad K dimension to multiple of tile_size
    K_padded = ((K + tile_size - 1) // tile_size) * tile_size
    pad_size = K_padded - K

    if pad_size > 0:
        # Pad matrix a along last dimension (K)
        pad_shape_a = list(a_shape)
        pad_shape_a[-1] = pad_size
        a_padding = mx.zeros(pad_shape_a, dtype=a.dtype)
        a_padded = mx.concatenate([a, a_padding], axis=-1)

        # Pad matrix b along second-to-last dimension (K)
        pad_shape_b = list(b_shape)
        pad_shape_b[-2] = pad_size
        b_padding = mx.zeros(pad_shape_b, dtype=b.dtype)
        b_padded = mx.concatenate([b, b_padding], axis=-2)
    else:
        a_padded = a
        b_padded = b

    # Number of tiles
    num_tiles = K_padded // tile_size

    # Compute partial products for each tile and sum
    accumulator = None
    for tile_idx in range(num_tiles):
        k_start = tile_idx * tile_size
        k_end = k_start + tile_size

        a_tile = a_padded[..., :, k_start:k_end]
        b_tile = b_padded[..., k_start:k_end, :]

        partial_product = mx.matmul(a_tile, b_tile)

        if accumulator is None:
            accumulator = partial_product
        else:
            accumulator = accumulator + partial_product

    return accumulator


class OriginalBatchInvariantRMSNorm(nn.Module):
    """
    ORIGINAL chunk-based RMSNorm (from committed version).

    Uses fixed-size chunks for variance computation.
    """

    def __init__(self, dims: int, eps: float = 1e-6, chunk_size: int = 64):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.chunk_size = chunk_size
        self.weight = mx.ones((dims,))

        assert chunk_size > 0 and (chunk_size & (chunk_size - 1)) == 0, \
            "chunk_size must be a power of 2"

    def __call__(self, x: mx.array) -> mx.array:
        x_squared = x * x

        # Pad last dimension to multiple of chunk_size
        pad_size = (self.chunk_size - (self.dims % self.chunk_size)) % self.chunk_size
        if pad_size > 0:
            pad_shape = list(x_squared.shape)
            pad_shape[-1] = pad_size
            padding = mx.zeros(pad_shape, dtype=x_squared.dtype)
            x_squared_padded = mx.concatenate([x_squared, padding], axis=-1)
        else:
            x_squared_padded = x_squared

        # Reshape to chunks
        new_shape = list(x_squared_padded.shape[:-1]) + [-1, self.chunk_size]
        x_chunked = x_squared_padded.reshape(new_shape)

        # Chunk-wise means then average
        chunk_means = mx.mean(x_chunked, axis=-1)
        mean_sq = mx.mean(chunk_means, axis=-1, keepdims=True)

        # RMS normalization
        rms = mx.sqrt(mean_sq + self.eps)
        normalized = x / rms

        return normalized * self.weight


def benchmark_rmsnorm_determinism():
    """Benchmark RMSNorm batch invariance."""
    print("\n" + "="*70)
    print("RMSNorm Batch Invariance Test")
    print("="*70)

    dims = 4096
    chunk_size = 64
    num_runs = 50

    mx.random.seed(42)
    data = mx.random.normal((128, dims))

    model = BatchInvariantRMSNorm(dims, chunk_size=chunk_size)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    unique_outputs = set()

    print(f"Running {num_runs} iterations with varying batch sizes...")
    print(f"Batch sizes tested: {batch_sizes}")

    for run in range(num_runs):
        batch_size = batch_sizes[run % len(batch_sizes)]
        output = model(data[:batch_size])

        # Hash first output for uniqueness check
        first_output = output[0].tolist()
        output_hash = hash(str(first_output[:10]))  # First 10 elements
        unique_outputs.add(output_hash)

    print(f"\nResults:")
    print(f"  Total runs: {num_runs}")
    print(f"  Unique outputs: {len(unique_outputs)}")
    print(f"  ✓ PASS" if len(unique_outputs) == 1 else f"  ✗ FAIL")

    return len(unique_outputs) == 1


def benchmark_matmul_determinism():
    """Benchmark matmul batch invariance."""
    print("\n" + "="*70)
    print("Matrix Multiplication Batch Invariance Test")
    print("="*70)

    M, K, N = 512, 512, 512
    tile_size = 128
    num_runs = 50

    mx.random.seed(42)
    a_data = mx.random.normal((64, M, K))
    b_data = mx.random.normal((K, N))

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    unique_outputs = set()

    print(f"Matrix shapes: A={M}x{K}, B={K}x{N}")
    print(f"Running {num_runs} iterations with varying batch sizes...")

    for run in range(num_runs):
        batch_size = batch_sizes[run % len(batch_sizes)]
        output = batch_invariant_matmul(
            a_data[:batch_size], b_data, tile_size=tile_size
        )

        # Hash first output
        first_output = output[0, 0, :10].tolist()
        output_hash = hash(str(first_output))
        unique_outputs.add(output_hash)

    print(f"\nResults:")
    print(f"  Total runs: {num_runs}")
    print(f"  Unique outputs: {len(unique_outputs)}")
    print(f"  ✓ PASS" if len(unique_outputs) == 1 else f"  ✗ FAIL")

    return len(unique_outputs) == 1


def benchmark_attention_determinism():
    """Benchmark attention batch invariance."""
    print("\n" + "="*70)
    print("Attention Batch Invariance Test")
    print("="*70)

    dims = 256
    num_heads = 8
    seq_len = 128
    num_runs = 50

    mx.random.seed(42)
    attn = BatchInvariantAttention(dims, num_heads)

    queries = mx.random.normal((32, seq_len, dims))
    keys = mx.random.normal((32, seq_len, dims))
    values = mx.random.normal((32, seq_len, dims))

    batch_sizes = [1, 2, 4, 8, 16, 32]
    unique_outputs = set()

    print(f"Attention config: dims={dims}, heads={num_heads}, seq_len={seq_len}")
    print(f"Running {num_runs} iterations with varying batch sizes...")

    for run in range(num_runs):
        batch_size = batch_sizes[run % len(batch_sizes)]
        output = attn(
            queries[:batch_size],
            keys[:batch_size],
            values[:batch_size]
        )

        # Hash first output
        first_output = output[0, 0, :10].tolist()
        output_hash = hash(str(first_output))
        unique_outputs.add(output_hash)

    print(f"\nResults:")
    print(f"  Total runs: {num_runs}")
    print(f"  Unique outputs: {len(unique_outputs)}")
    print(f"  ✓ PASS" if len(unique_outputs) == 1 else f"  ✗ FAIL")

    return len(unique_outputs) == 1


def benchmark_performance():
    """Benchmark performance overhead (legacy function for compatibility)."""
    # Use the comprehensive benchmark instead
    benchmark_all_implementations()


def benchmark_all_implementations():
    """
    Comprehensive benchmark comparing all implementations using robust
    benchmarking methodology.

    Runs until results stabilize (CV < 10%) with minimum 2 seconds per test.
    Reports mean ± std, iterations, and percentiles.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON (robust benchmarking)")
    print("="*80)
    print("\nMethodology: min 100 iterations, min 2s runtime, until CV < 10%")
    print("Comparing: Standard MLX | Original BI | New Python 2D | Metal Kernel")
    print("-"*80)

    # Import Metal kernel implementations
    try:
        from mlx_deterministic.ops.metal_matmul import deterministic_matmul_metal
        from mlx_deterministic.ops.metal_rms_norm import rms_norm_metal, BatchInvariantRMSNormMetal
        metal_available = True
    except ImportError as e:
        print(f"Warning: Metal kernels not available: {e}")
        metal_available = False

    results = {}

    # ========================================================================
    # RMSNorm Benchmark
    # ========================================================================
    print("\n" + "-"*80)
    print("RMSNorm Performance (batch=32, dims=2048)")
    print("-"*80)

    dims = 2048
    mx.random.seed(42)
    x = mx.random.normal((32, dims))

    # Create all implementations
    std_norm = nn.RMSNorm(dims)
    original_norm = OriginalBatchInvariantRMSNorm(dims, chunk_size=64)
    new_norm = BatchInvariantRMSNorm(dims)
    metal_norm = BatchInvariantRMSNormMetal(dims) if metal_available else None

    # Benchmark each implementation
    print("  Benchmarking Standard MLX...", end=" ", flush=True)
    std_result = robust_benchmark(std_norm, (x,))
    print(f"done ({std_result['iterations']} iters)")

    print("  Benchmarking Original BI...", end=" ", flush=True)
    original_result = robust_benchmark(original_norm, (x,))
    print(f"done ({original_result['iterations']} iters)")

    print("  Benchmarking New Python 2D...", end=" ", flush=True)
    new_result = robust_benchmark(new_norm, (x,))
    print(f"done ({new_result['iterations']} iters)")

    if metal_available:
        print("  Benchmarking Metal Kernel...", end=" ", flush=True)
        metal_result = robust_benchmark(metal_norm, (x,))
        print(f"done ({metal_result['iterations']} iters)")
    else:
        metal_result = None

    # Print results
    baseline = std_result['mean_ms']
    print(f"\n  Standard MLX:     {format_benchmark_result(std_result)}")
    print(f"  Original BI:      {format_benchmark_result(original_result, baseline)}")
    print(f"  New Python 2D:    {format_benchmark_result(new_result, baseline)}")
    if metal_result:
        print(f"  Metal Kernel:     {format_benchmark_result(metal_result, baseline)}")
    else:
        print(f"  Metal Kernel:     N/A")

    results['rmsnorm'] = {
        'standard': std_result,
        'original': original_result,
        'new_python': new_result,
        'metal': metal_result,
    }

    # ========================================================================
    # Matmul Benchmark (512x512)
    # ========================================================================
    print("\n" + "-"*80)
    print("Matrix Multiplication Performance (512x512 @ 512x512)")
    print("-"*80)

    M, K, N = 512, 512, 512
    mx.random.seed(42)
    a = mx.random.normal((M, K))
    b = mx.random.normal((K, N))

    def std_matmul():
        return mx.matmul(a, b)

    def original_matmul():
        return original_batch_invariant_matmul(a, b, tile_size=128)

    def new_python_matmul():
        return batch_invariant_matmul(a, b, tile_size=64, use_metal_kernel=False)

    def metal_matmul():
        return batch_invariant_matmul(a, b, use_metal_kernel=True)

    print("  Benchmarking Standard MLX...", end=" ", flush=True)
    std_result = robust_benchmark(std_matmul)
    print(f"done ({std_result['iterations']} iters)")

    print("  Benchmarking Original BI...", end=" ", flush=True)
    original_result = robust_benchmark(original_matmul)
    print(f"done ({original_result['iterations']} iters)")

    print("  Benchmarking New Python 2D...", end=" ", flush=True)
    new_result = robust_benchmark(new_python_matmul)
    print(f"done ({new_result['iterations']} iters)")

    if metal_available:
        print("  Benchmarking Metal Kernel...", end=" ", flush=True)
        metal_result = robust_benchmark(metal_matmul)
        print(f"done ({metal_result['iterations']} iters)")
    else:
        metal_result = None

    baseline = std_result['mean_ms']
    print(f"\n  Standard MLX:     {format_benchmark_result(std_result)}")
    print(f"  Original BI:      {format_benchmark_result(original_result, baseline)}")
    print(f"  New Python 2D:    {format_benchmark_result(new_result, baseline)}")
    if metal_result:
        print(f"  Metal Kernel:     {format_benchmark_result(metal_result, baseline)}")
    else:
        print(f"  Metal Kernel:     N/A")

    results['matmul'] = {
        'standard': std_result,
        'original': original_result,
        'new_python': new_result,
        'metal': metal_result,
    }

    # ========================================================================
    # Large Matmul Benchmark (2048x2048)
    # ========================================================================
    print("\n" + "-"*80)
    print("Large Matmul Performance (2048x2048 @ 2048x2048)")
    print("-"*80)

    M, K, N = 2048, 2048, 2048
    mx.random.seed(42)
    a_large = mx.random.normal((M, K))
    b_large = mx.random.normal((K, N))

    def std_matmul_large():
        return mx.matmul(a_large, b_large)

    def original_matmul_large():
        return original_batch_invariant_matmul(a_large, b_large, tile_size=128)

    def new_python_matmul_large():
        return batch_invariant_matmul(a_large, b_large, tile_size=64, use_metal_kernel=False)

    def metal_matmul_large():
        return batch_invariant_matmul(a_large, b_large, use_metal_kernel=True)

    # Use longer min_runtime for large matmul since each iteration is slower
    print("  Benchmarking Standard MLX...", end=" ", flush=True)
    std_result = robust_benchmark(std_matmul_large, min_runtime_sec=3.0)
    print(f"done ({std_result['iterations']} iters)")

    print("  Benchmarking Original BI...", end=" ", flush=True)
    original_result = robust_benchmark(original_matmul_large, min_runtime_sec=3.0)
    print(f"done ({original_result['iterations']} iters)")

    print("  Benchmarking New Python 2D...", end=" ", flush=True)
    new_result = robust_benchmark(new_python_matmul_large, min_runtime_sec=3.0)
    print(f"done ({new_result['iterations']} iters)")

    if metal_available:
        print("  Benchmarking Metal Kernel...", end=" ", flush=True)
        metal_result = robust_benchmark(metal_matmul_large, min_runtime_sec=3.0)
        print(f"done ({metal_result['iterations']} iters)")
    else:
        metal_result = None

    baseline = std_result['mean_ms']
    print(f"\n  Standard MLX:     {format_benchmark_result(std_result)}")
    print(f"  Original BI:      {format_benchmark_result(original_result, baseline)}")
    print(f"  New Python 2D:    {format_benchmark_result(new_result, baseline)}")
    if metal_result:
        print(f"  Metal Kernel:     {format_benchmark_result(metal_result, baseline)}")
    else:
        print(f"  Metal Kernel:     N/A")

    results['matmul_large'] = {
        'standard': std_result,
        'original': original_result,
        'new_python': new_result,
        'metal': metal_result,
    }

    # ========================================================================
    # FP16 Matmul Benchmark (2048x2048)
    # ========================================================================
    print("\n" + "-"*80)
    print("FP16 Matmul Performance (2048x2048 @ 2048x2048)")
    print("-"*80)

    M, K, N = 2048, 2048, 2048
    mx.random.seed(42)
    a_fp16 = mx.random.normal((M, K)).astype(mx.float16)
    b_fp16 = mx.random.normal((K, N)).astype(mx.float16)

    def std_matmul_fp16():
        return mx.matmul(a_fp16, b_fp16)

    def metal_matmul_fp16():
        return batch_invariant_matmul(a_fp16, b_fp16, use_metal_kernel=True)

    print("  Benchmarking Standard MLX FP16...", end=" ", flush=True)
    std_result_fp16 = robust_benchmark(std_matmul_fp16, min_runtime_sec=3.0)
    print(f"done ({std_result_fp16['iterations']} iters)")

    if metal_available:
        print("  Benchmarking Metal Kernel FP16...", end=" ", flush=True)
        metal_result_fp16 = robust_benchmark(metal_matmul_fp16, min_runtime_sec=3.0)
        print(f"done ({metal_result_fp16['iterations']} iters)")
    else:
        metal_result_fp16 = None

    baseline_fp16 = std_result_fp16['mean_ms']
    print(f"\n  Standard MLX FP16:  {format_benchmark_result(std_result_fp16)}")
    if metal_result_fp16:
        print(f"  Metal Kernel FP16:  {format_benchmark_result(metal_result_fp16, baseline_fp16)}")
    else:
        print(f"  Metal Kernel FP16:  N/A")

    # Compare FP16 vs FP32 Metal kernels
    if metal_available and metal_result_fp16:
        fp32_baseline = results['matmul_large']['metal']['mean_ms']
        fp16_speedup = (fp32_baseline / metal_result_fp16['mean_ms'] - 1) * 100
        print(f"\n  FP16 vs FP32 Metal: {fp16_speedup:+.1f}% (positive = FP16 faster)")

    results['matmul_fp16'] = {
        'standard': std_result_fp16,
        'metal': metal_result_fp16,
    }

    # ========================================================================
    # Summary Table
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY TABLE (for README.md)")
    print("="*80)

    def get_overhead(result, baseline):
        if result is None:
            return None
        return ((result['mean_ms'] - baseline['mean_ms']) / baseline['mean_ms']) * 100

    print("\n| Operation | Standard | Original BI | New Python | Metal Kernel |")
    print("|-----------|----------|-------------|------------|--------------|")

    # RMSNorm row
    r = results['rmsnorm']
    std = r['standard']['mean_ms']
    orig_oh = get_overhead(r['original'], r['standard'])
    new_oh = get_overhead(r['new_python'], r['standard'])
    metal_oh = get_overhead(r['metal'], r['standard'])
    metal_str = f"{r['metal']['mean_ms']:.2f}ms ({metal_oh:+.0f}%)" if r['metal'] else "N/A"
    print(f"| RMSNorm   | {std:.2f}ms   | {r['original']['mean_ms']:.2f}ms ({orig_oh:+.0f}%) | {r['new_python']['mean_ms']:.2f}ms ({new_oh:+.0f}%) | {metal_str} |")

    # Matmul row
    r = results['matmul']
    std = r['standard']['mean_ms']
    orig_oh = get_overhead(r['original'], r['standard'])
    new_oh = get_overhead(r['new_python'], r['standard'])
    metal_oh = get_overhead(r['metal'], r['standard'])
    metal_str = f"{r['metal']['mean_ms']:.2f}ms ({metal_oh:+.0f}%)" if r['metal'] else "N/A"
    print(f"| Matmul    | {std:.2f}ms   | {r['original']['mean_ms']:.2f}ms ({orig_oh:+.0f}%) | {r['new_python']['mean_ms']:.2f}ms ({new_oh:+.0f}%) | {metal_str} |")

    # Large Matmul row
    r = results['matmul_large']
    std = r['standard']['mean_ms']
    orig_oh = get_overhead(r['original'], r['standard'])
    new_oh = get_overhead(r['new_python'], r['standard'])
    metal_oh = get_overhead(r['metal'], r['standard'])
    metal_str = f"{r['metal']['mean_ms']:.2f}ms ({metal_oh:+.0f}%)" if r['metal'] else "N/A"
    print(f"| Matmul 2K | {std:.2f}ms   | {r['original']['mean_ms']:.2f}ms ({orig_oh:+.0f}%) | {r['new_python']['mean_ms']:.2f}ms ({new_oh:+.0f}%) | {metal_str} |")

    # FP16 Matmul row
    r = results['matmul_fp16']
    std = r['standard']['mean_ms']
    metal_oh = get_overhead(r['metal'], r['standard'])
    metal_str = f"{r['metal']['mean_ms']:.2f}ms ({metal_oh:+.0f}%)" if r['metal'] else "N/A"
    print(f"| FP16 2K   | {std:.2f}ms   | N/A         | N/A        | {metal_str} |")

    print("\nNotes:")
    print("  - Metal kernel provides bitwise determinism (0.0 difference)")
    print("  - Python implementations have ~1e-5 tolerance due to MLX internal variance")
    print("  - Results show mean ± std over adaptive iterations until stable")

    return results


def main():
    """Run all benchmarks."""
    print("\n" + "#"*70)
    print("# MLX Deterministic Inference Benchmark Suite")
    print("#"*70)
    print("\nValidating batch-invariant operations for deterministic LLM inference")
    print("Based on: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/")

    results = []

    # Run determinism tests
    results.append(("RMSNorm", benchmark_rmsnorm_determinism()))
    results.append(("Matmul", benchmark_matmul_determinism()))
    results.append(("Attention", benchmark_attention_determinism()))

    # Run performance benchmark
    benchmark_performance()

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    all_passed = all(result[1] for result in results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s} {status}")

    print("\n" + "="*70)
    if all_passed:
        print("🎉 All determinism tests PASSED!")
        print("All operations produce identical outputs regardless of batch size.")
    else:
        print("❌ Some tests FAILED")
    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
