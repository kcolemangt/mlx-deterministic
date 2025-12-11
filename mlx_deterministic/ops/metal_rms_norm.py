"""
Deterministic RMSNorm using Custom Metal Kernel

This module implements bitwise-deterministic RMSNorm using MLX's custom Metal
kernel API. Unlike the Python approach which relies on mx.mean (with potential
batch variance), this kernel uses fixed tree reduction for guaranteed determinism.

Key design principles:
1. Each threadgroup handles exactly one sample (row)
2. Fixed tree reduction pattern within threadgroup
3. All threads participate in cooperative reduction
4. No cross-threadgroup communication needed

Performance optimizations (v2 - M4 Max):
- Process 4 elements per thread iteration (better instruction pipelining)
- Unrolled reduction for lower latency
- Fused multiply-add operations
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Any, Dict, Optional

# Metal kernel for deterministic RMSNorm (optimized for M4 Max)
# Each threadgroup processes one sample with tree reduction
# eps is passed as an input array to support configurable epsilon values
RMSNORM_KERNEL_SOURCE: str = """
// Each threadgroup handles one sample (one row of input)
uint sample_idx = threadgroup_position_in_grid.x;
uint local_id = thread_position_in_threadgroup.x;
uint num_threads = threads_per_threadgroup.x;

// Get dimensions
uint batch_size = x_shape[0];
uint dims = x_shape[x_ndim - 1];

// Bounds check
if (sample_idx >= batch_size) {
    return;
}

// Shared memory for reduction - must be power of 2 for tree reduction
threadgroup T shared_sum[256];

// Initialize shared memory to zero first
// This is critical because threads >= dims won't write anything
shared_sum[local_id] = T(0);
threadgroup_barrier(mem_flags::mem_threadgroup);

// Each thread computes partial sum of squares
// Process 4 elements at a time for better throughput
T local_sum = T(0);
uint base = sample_idx * dims;

// Main loop - process 4 elements per iteration
uint i = local_id;
for (; i + 3 * num_threads < dims; i += 4 * num_threads) {
    T v0 = x[base + i];
    T v1 = x[base + i + num_threads];
    T v2 = x[base + i + 2 * num_threads];
    T v3 = x[base + i + 3 * num_threads];
    local_sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
}

// Handle remaining elements
for (; i < dims; i += num_threads) {
    T val = x[base + i];
    local_sum += val * val;
}

shared_sum[local_id] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Tree reduction - DETERMINISTIC order with explicit unrolling
// Each step halves the active threads
if (num_threads >= 256) {
    if (local_id < 128) shared_sum[local_id] += shared_sum[local_id + 128];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (num_threads >= 128) {
    if (local_id < 64) shared_sum[local_id] += shared_sum[local_id + 64];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (num_threads >= 64) {
    if (local_id < 32) shared_sum[local_id] += shared_sum[local_id + 32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Last warp - no barrier needed within same SIMD group
if (local_id < 16) {
    shared_sum[local_id] += shared_sum[local_id + 16];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 8) {
    shared_sum[local_id] += shared_sum[local_id + 8];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 4) {
    shared_sum[local_id] += shared_sum[local_id + 4];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 2) {
    shared_sum[local_id] += shared_sum[local_id + 2];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 1) {
    shared_sum[local_id] += shared_sum[local_id + 1];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Now shared_sum[0] contains sum of all x^2 for this sample
// Compute RMS: sqrt(mean(x^2) + eps)
T mean_sq = shared_sum[0] / T(dims);
T rms = metal::sqrt(mean_sq + eps[0]);
T inv_rms = T(1) / rms;  // Pre-compute inverse for faster division

// Each thread normalizes its portion of the output
// Process 4 elements at a time
i = local_id;
for (; i + 3 * num_threads < dims; i += 4 * num_threads) {
    T v0 = x[base + i];
    T v1 = x[base + i + num_threads];
    T v2 = x[base + i + 2 * num_threads];
    T v3 = x[base + i + 3 * num_threads];
    out[base + i] = v0 * inv_rms * weight[i];
    out[base + i + num_threads] = v1 * inv_rms * weight[i + num_threads];
    out[base + i + 2 * num_threads] = v2 * inv_rms * weight[i + 2 * num_threads];
    out[base + i + 3 * num_threads] = v3 * inv_rms * weight[i + 3 * num_threads];
}

// Handle remaining elements
for (; i < dims; i += num_threads) {
    T val = x[base + i];
    out[base + i] = val * inv_rms * weight[i];
}
"""

# Kernel cache
_kernel_cache: Dict[str, Any] = {}


def _create_rmsnorm_kernel():
    """Create the Metal kernel for deterministic RMSNorm."""
    return mx.fast.metal_kernel(
        name="deterministic_rmsnorm",
        input_names=["x", "weight", "eps"],
        output_names=["out"],
        source=RMSNORM_KERNEL_SOURCE,
        ensure_row_contiguous=True,
        atomic_outputs=False
    )


def rms_norm_metal(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """
    Bitwise deterministic RMSNorm using custom Metal kernel.

    Args:
        x: Input tensor [..., dims]
        weight: Weight tensor [dims]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as x
    """
    original_shape = x.shape
    dims = x.shape[-1]

    # Flatten to 2D: [batch, dims]
    x_flat = x.reshape(-1, dims)
    batch_size = x_flat.shape[0]

    # Get or create kernel
    key = x.dtype
    if key not in _kernel_cache:
        _kernel_cache[key] = _create_rmsnorm_kernel()
    kernel = _kernel_cache[key]

    # Each threadgroup handles one sample
    # Use 256 threads per group (power of 2 for tree reduction)
    num_threads = min(256, dims)
    # Ensure power of 2
    num_threads = 1 << (num_threads - 1).bit_length() if num_threads > 1 else 1
    num_threads = min(num_threads, 256)

    # Grid specifies total threads, not threadgroup count
    # We want batch_size threadgroups, each with num_threads threads
    total_threads = batch_size * num_threads
    grid = (total_threads, 1, 1)
    threadgroup = (num_threads, 1, 1)

    # Execute kernel with eps passed as input array
    eps_array = mx.array([eps], dtype=x.dtype)
    outputs = kernel(
        inputs=[x_flat, weight, eps_array],
        template=[("T", x.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[x_flat.shape],
        output_dtypes=[x.dtype],
    )

    return outputs[0].reshape(original_shape)


class BatchInvariantRMSNormMetal(nn.Module):
    """
    Bitwise deterministic RMSNorm using Metal kernel.

    Drop-in replacement for nn.RMSNorm with guaranteed bitwise identical
    results regardless of batch size.

    Args:
        dims: Feature dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm_metal(x, self.weight, self.eps)
