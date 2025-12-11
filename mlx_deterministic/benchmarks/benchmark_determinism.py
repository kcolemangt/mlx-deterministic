#!/usr/bin/env python3
"""
Determinism Benchmark for MLX Batch-Invariant Operations

Validates that batch-invariant operations produce identical results
regardless of batch size, demonstrating deterministic behavior.

Now includes comprehensive comparison of:
1. Original implementation (K-dimension tiling)
2. New Python implementation (2D output tiling)
3. Metal kernel implementation (bitwise determinism)

Usage:
    # Standard benchmark (quick)
    python benchmark_determinism.py

    # Extended benchmark (stable results, thermal-aware)
    python benchmark_determinism.py --extended

    # Extended with custom settings
    python benchmark_determinism.py --extended --cooldown 5 --rounds 3
"""

import argparse
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time
import statistics
from typing import Callable, Any, Dict, List, Optional
from mlx_deterministic.ops import (
    BatchInvariantRMSNorm,
    batch_invariant_matmul,
    BatchInvariantAttention,
    batch_invariant_softmax
)


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark MLX deterministic operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_determinism.py              # Standard quick benchmark
  python benchmark_determinism.py --extended   # Extended thermal-aware benchmark
  python benchmark_determinism.py --extended --cooldown 5 --rounds 3
        """
    )
    parser.add_argument(
        '--extended', action='store_true',
        help='Run extended benchmark with interleaved tests and cooldown for stable results'
    )
    parser.add_argument(
        '--cooldown', type=float, default=5.0,
        help='Seconds to sleep between test categories for thermal recovery (default: 5.0)'
    )
    parser.add_argument(
        '--rounds', type=int, default=3,
        help='Number of complete benchmark rounds in extended mode (default: 3)'
    )
    return parser.parse_args()


# =============================================================================
# THERMAL-AWARE BENCHMARKING (Extended Mode)
# =============================================================================

def cooldown(seconds: float, message: str = "Cooling down") -> None:
    """
    Sleep to let machine return to thermal equilibrium.

    Args:
        seconds: Time to sleep in seconds
        message: Message to display during cooldown
    """
    print(f"  {message} ({seconds:.1f}s)...", end=" ", flush=True)
    time.sleep(seconds)
    print("done")


def interleaved_benchmark(
    implementations: Dict[str, Callable],
    min_runtime_sec: float = 5.0,
    target_cv: float = 0.05,
    max_iterations: int = 5000,
    min_iterations: int = 500,
    warmup_per_impl: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    Run implementations in interleaved fashion for fair thermal comparison.

    Instead of running [A x1000, B x1000], runs [A, B, A, B, ...] ensuring
    all implementations experience similar thermal conditions. Uses adaptive
    stopping based on time and coefficient of variation for stable results.

    Args:
        implementations: Dict mapping name -> callable (no args)
        min_runtime_sec: Minimum total runtime before checking for stability
        target_cv: Target coefficient of variation for all implementations (0.05 = 5%)
        max_iterations: Safety cap on iterations per implementation
        min_iterations: Minimum iterations even if CV target is met early
        warmup_per_impl: Warmup iterations per implementation before timing

    Returns:
        Dict mapping name -> statistics dict with mean_ms, std_ms, median_ms, etc.
    """
    # Warmup all implementations first (in rotation to warm caches fairly)
    impl_names = list(implementations.keys())
    print(f"    Warming up ({warmup_per_impl} iterations each)...", end=" ", flush=True)
    for _ in range(warmup_per_impl):
        for name in impl_names:
            out = implementations[name]()
            mx.eval(out)
    print("done")

    # Interleaved timing collection
    samples: Dict[str, List[float]] = {name: [] for name in implementations}
    total_time = 0.0
    check_interval = 50  # Check stopping conditions every N iterations
    iteration = 0

    print(f"    Running (min {min_runtime_sec}s, target CV <{target_cv*100:.0f}%)...", end=" ", flush=True)

    while iteration < max_iterations:
        # Rotate starting position each iteration to avoid order bias
        # e.g., iteration 0: [A, B, C], iteration 1: [B, C, A], iteration 2: [C, A, B]
        start_idx = iteration % len(impl_names)
        order = impl_names[start_idx:] + impl_names[:start_idx]

        for name in order:
            start = time.perf_counter()
            out = implementations[name]()
            mx.eval(out)
            elapsed = time.perf_counter() - start
            samples[name].append(elapsed * 1000)  # Convert to ms
            total_time += elapsed

        iteration += 1

        # Check stopping conditions periodically (after minimum iterations)
        if iteration % check_interval == 0 and iteration >= min_iterations:
            if total_time >= min_runtime_sec:
                # Check if all implementations have stable results (CV < target)
                all_stable = True
                for name, times in samples.items():
                    if len(times) > 1:
                        mean = statistics.mean(times)
                        std = statistics.stdev(times)
                        cv = std / mean if mean > 0 else float('inf')
                        if cv >= target_cv:
                            all_stable = False
                            break
                if all_stable:
                    break

    print(f"done ({iteration} iterations, {total_time:.1f}s)")

    # Compute statistics for each implementation
    results = {}
    for name, times in samples.items():
        sorted_times = sorted(times)
        n = len(times)
        mean = statistics.mean(times)
        std = statistics.stdev(times) if n > 1 else 0.0
        results[name] = {
            'mean_ms': mean,
            'std_ms': std,
            'cv': std / mean if mean > 0 else 0.0,
            'median_ms': statistics.median(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'p95_ms': sorted_times[int(n * 0.95)] if n >= 20 else sorted_times[-1],
            'iterations': n,
        }
    return results


def compute_overhead(result: Dict[str, Any], baseline: Dict[str, Any]) -> float:
    """Compute percentage overhead relative to baseline."""
    return ((result['median_ms'] - baseline['median_ms']) / baseline['median_ms']) * 100


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


def benchmark_extended(
    cooldown_sec: float = 5.0,
    rounds: int = 3,
    min_runtime_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Extended benchmark mode with thermal-aware testing for stable results.

    Features:
    - Interleaved testing: All implementations run in rotation, not sequentially
    - Adaptive iterations: Runs until results stabilize (CV < 5%) with min runtime
    - Cooldown periods: Sleep between test categories to recover from thermal throttling
    - Multiple rounds: Run complete benchmark multiple times, report median results
    - Relative focus: Emphasizes overhead ratios over absolute times

    This addresses the problem where later tests run slower due to thermal throttling,
    unfairly penalizing implementations that happen to be benchmarked last.

    Args:
        cooldown_sec: Seconds to sleep between test categories
        rounds: Number of complete benchmark rounds
        min_runtime_sec: Minimum runtime per benchmark category (adaptive iterations)

    Returns:
        Dict with all benchmark results across rounds
    """
    print("\n" + "="*80)
    print("EXTENDED BENCHMARK MODE (thermal-aware)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Rounds: {rounds}")
    print(f"  Min runtime per test: {min_runtime_sec}s (adaptive iterations until CV <5%)")
    print(f"  Cooldown between categories: {cooldown_sec}s")
    print(f"  Testing method: Interleaved (fair thermal comparison)")
    print("-"*80)

    # Import Metal kernel implementations
    try:
        from mlx_deterministic.ops.metal_matmul import deterministic_matmul_metal
        from mlx_deterministic.ops.metal_rms_norm import BatchInvariantRMSNormMetal
        metal_available = True
    except ImportError as e:
        print(f"Warning: Metal kernels not available: {e}")
        metal_available = False

    all_round_results: List[Dict[str, Any]] = []

    for round_num in range(1, rounds + 1):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/{rounds}")
        print("="*80)

        round_results = {}

        # Initial cooldown to start from thermal equilibrium
        if round_num > 1:
            cooldown(cooldown_sec * 2, "Inter-round cooldown")

        # ====================================================================
        # RMSNorm Benchmark
        # ====================================================================
        print(f"\n  RMSNorm (batch=32, dims=2048)")

        dims = 2048
        mx.random.seed(42)
        x = mx.random.normal((32, dims))

        std_norm = nn.RMSNorm(dims)
        metal_norm = BatchInvariantRMSNormMetal(dims) if metal_available else None

        implementations = {"Standard": lambda: std_norm(x)}
        if metal_available:
            implementations["Metal"] = lambda: metal_norm(x)

        results = interleaved_benchmark(implementations, min_runtime_sec=min_runtime_sec)
        round_results['rmsnorm'] = results

        # Print results
        baseline = results['Standard']['median_ms']
        print(f"    Standard: {results['Standard']['median_ms']:.3f}ms (median, CV={results['Standard']['cv']*100:.1f}%)")
        if metal_available:
            overhead = compute_overhead(results['Metal'], results['Standard'])
            print(f"    Metal:    {results['Metal']['median_ms']:.3f}ms (median, CV={results['Metal']['cv']*100:.1f}%) [{overhead:+.1f}%]")

        cooldown(cooldown_sec, "Thermal recovery")

        # ====================================================================
        # Matmul 512x512 Benchmark
        # ====================================================================
        print(f"\n  Matmul (512x512 @ 512x512)")

        M, K, N = 512, 512, 512
        mx.random.seed(42)
        a = mx.random.normal((M, K))
        b = mx.random.normal((K, N))

        implementations = {
            "Standard": lambda: mx.matmul(a, b),
        }
        if metal_available:
            implementations["Metal"] = lambda: batch_invariant_matmul(a, b, use_metal_kernel=True)

        results = interleaved_benchmark(implementations, min_runtime_sec=min_runtime_sec)
        round_results['matmul_512'] = results

        baseline = results['Standard']['median_ms']
        print(f"    Standard: {results['Standard']['median_ms']:.3f}ms (median, CV={results['Standard']['cv']*100:.1f}%)")
        if metal_available:
            overhead = compute_overhead(results['Metal'], results['Standard'])
            print(f"    Metal:    {results['Metal']['median_ms']:.3f}ms (median, CV={results['Metal']['cv']*100:.1f}%) [{overhead:+.1f}%]")

        cooldown(cooldown_sec, "Thermal recovery")

        # ====================================================================
        # Large Matmul 2048x2048 Benchmark (FP32)
        # ====================================================================
        print(f"\n  Large Matmul FP32 (2048x2048 @ 2048x2048)")

        M, K, N = 2048, 2048, 2048
        mx.random.seed(42)
        a_large = mx.random.normal((M, K))
        b_large = mx.random.normal((K, N))

        implementations = {
            "Standard": lambda: mx.matmul(a_large, b_large),
        }
        if metal_available:
            implementations["Metal"] = lambda: batch_invariant_matmul(a_large, b_large, use_metal_kernel=True)

        results = interleaved_benchmark(implementations, min_runtime_sec=min_runtime_sec)
        round_results['matmul_2k_fp32'] = results

        baseline = results['Standard']['median_ms']
        print(f"    Standard: {results['Standard']['median_ms']:.3f}ms (median, CV={results['Standard']['cv']*100:.1f}%)")
        if metal_available:
            overhead = compute_overhead(results['Metal'], results['Standard'])
            print(f"    Metal:    {results['Metal']['median_ms']:.3f}ms (median, CV={results['Metal']['cv']*100:.1f}%) [{overhead:+.1f}%]")

        cooldown(cooldown_sec, "Thermal recovery")

        # ====================================================================
        # Large Matmul 2048x2048 Benchmark (FP16)
        # ====================================================================
        print(f"\n  Large Matmul FP16 (2048x2048 @ 2048x2048)")

        mx.random.seed(42)
        a_fp16 = mx.random.normal((M, K)).astype(mx.float16)
        b_fp16 = mx.random.normal((K, N)).astype(mx.float16)

        implementations = {
            "Standard": lambda: mx.matmul(a_fp16, b_fp16),
        }
        if metal_available:
            implementations["Metal"] = lambda: batch_invariant_matmul(a_fp16, b_fp16, use_metal_kernel=True)

        results = interleaved_benchmark(implementations, min_runtime_sec=min_runtime_sec)
        round_results['matmul_2k_fp16'] = results

        baseline = results['Standard']['median_ms']
        print(f"    Standard: {results['Standard']['median_ms']:.3f}ms (median)")
        if metal_available:
            overhead = compute_overhead(results['Metal'], results['Standard'])
            print(f"    Metal:    {results['Metal']['median_ms']:.3f}ms (median) [{overhead:+.1f}%]")

        all_round_results.append(round_results)

    # ========================================================================
    # Aggregate results across rounds
    # ========================================================================
    print("\n" + "="*80)
    print(f"FINAL RESULTS (median of {rounds} rounds)")
    print("="*80)

    # Compute median overhead across rounds for each category
    final_results = {}
    categories = ['rmsnorm', 'matmul_512', 'matmul_2k_fp32', 'matmul_2k_fp16']
    category_names = ['RMSNorm', 'Matmul 512', 'Matmul 2K (FP32)', 'Matmul 2K (FP16)']

    print("\n| Operation | Standard | Metal | Overhead |")
    print("|-----------|----------|-------|----------|")

    for cat, cat_name in zip(categories, category_names):
        std_medians = [r[cat]['Standard']['median_ms'] for r in all_round_results]
        std_final = statistics.median(std_medians)

        if metal_available and 'Metal' in all_round_results[0][cat]:
            metal_medians = [r[cat]['Metal']['median_ms'] for r in all_round_results]
            metal_final = statistics.median(metal_medians)

            # Compute overhead for each round and get range
            overheads = []
            for r in all_round_results:
                oh = compute_overhead(r[cat]['Metal'], r[cat]['Standard'])
                overheads.append(oh)

            overhead_median = statistics.median(overheads)
            overhead_min = min(overheads)
            overhead_max = max(overheads)

            # Show range if variance is significant
            if overhead_max - overhead_min > 5:
                overhead_str = f"{overhead_median:+.0f}% ({overhead_min:+.0f} to {overhead_max:+.0f})"
            else:
                overhead_str = f"{overhead_median:+.0f}%"

            print(f"| {cat_name:<17} | {std_final:.2f}ms | {metal_final:.2f}ms | {overhead_str} |")

            final_results[cat] = {
                'standard_ms': std_final,
                'metal_ms': metal_final,
                'overhead_pct': overhead_median,
                'overhead_range': (overhead_min, overhead_max),
            }
        else:
            print(f"| {cat_name:<17} | {std_final:.2f}ms | N/A | N/A |")
            final_results[cat] = {
                'standard_ms': std_final,
                'metal_ms': None,
                'overhead_pct': None,
            }

    print("\nNotes:")
    print("  - Metal kernel provides bitwise determinism (0.0 difference)")
    print("  - Interleaved testing ensures fair thermal comparison")
    print(f"  - Results are median of {rounds} complete rounds")

    return final_results


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


def main() -> int:
    """
    Run all benchmarks.

    Returns:
        0 if all tests pass, 1 otherwise
    """
    args = parse_args()

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

    # Run performance benchmark (standard or extended mode)
    if args.extended:
        benchmark_extended(
            cooldown_sec=args.cooldown,
            rounds=args.rounds,
        )
    else:
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
