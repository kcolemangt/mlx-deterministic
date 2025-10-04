#!/usr/bin/env python3
"""
Determinism Benchmark for MLX Batch-Invariant Operations

Validates that batch-invariant operations produce identical results
regardless of batch size, demonstrating deterministic behavior.
"""

import mlx.core as mx
import numpy as np
import time
from mlx_deterministic.ops import (
    BatchInvariantRMSNorm,
    batch_invariant_matmul,
    BatchInvariantAttention,
    batch_invariant_softmax
)


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
    """Benchmark performance overhead."""
    print("\n" + "="*70)
    print("Performance Benchmark")
    print("="*70)

    dims = 2048
    M, K, N = 512, 512, 512

    # RMSNorm performance
    print("\nRMSNorm Performance:")
    mx.random.seed(42)
    x = mx.random.normal((32, dims))

    import mlx.nn as nn
    std_norm = nn.RMSNorm(dims)
    bi_norm = BatchInvariantRMSNorm(dims)

    # Warmup
    _ = std_norm(x)
    _ = bi_norm(x)
    mx.eval(_)

    # Benchmark standard
    start = time.time()
    for _ in range(100):
        out = std_norm(x)
        mx.eval(out)
    std_time = time.time() - start

    # Benchmark batch-invariant
    start = time.time()
    for _ in range(100):
        out = bi_norm(x)
        mx.eval(out)
    bi_time = time.time() - start

    overhead = ((bi_time - std_time) / std_time) * 100
    print(f"  Standard RMSNorm:        {std_time*10:.2f}ms")
    print(f"  Batch-invariant RMSNorm: {bi_time*10:.2f}ms")
    print(f"  Overhead: {overhead:.1f}%")

    # Matmul performance
    print("\nMatrix Multiplication Performance:")
    mx.random.seed(42)
    a = mx.random.normal((M, K))
    b = mx.random.normal((K, N))

    # Warmup
    _ = mx.matmul(a, b)
    _ = batch_invariant_matmul(a, b)
    mx.eval(_)

    # Benchmark standard
    start = time.time()
    for _ in range(100):
        out = mx.matmul(a, b)
        mx.eval(out)
    std_time = time.time() - start

    # Benchmark batch-invariant
    start = time.time()
    for _ in range(100):
        out = batch_invariant_matmul(a, b, tile_size=128)
        mx.eval(out)
    bi_time = time.time() - start

    overhead = ((bi_time - std_time) / std_time) * 100
    print(f"  Standard matmul:        {std_time*10:.2f}ms")
    print(f"  Batch-invariant matmul: {bi_time*10:.2f}ms")
    print(f"  Overhead: {overhead:.1f}%")


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
