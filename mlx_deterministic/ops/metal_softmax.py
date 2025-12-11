"""
Deterministic Softmax using Custom Metal Kernel

This module implements bitwise-deterministic softmax using MLX's custom Metal
kernel API. The kernel uses fixed tree reduction for both max and sum operations,
guaranteeing identical results regardless of batch size.

Key design principles:
1. Each threadgroup handles exactly one row (softmax along last axis)
2. Fixed tree reduction for max computation
3. Fixed tree reduction for sum of exponentials
4. No cross-threadgroup communication

Performance optimizations (v2 - M4 Max):
- Process 4 elements per thread iteration (better instruction pipelining)
- Unrolled reduction for lower latency
- Pre-computed inverse for faster division
"""

import mlx.core as mx
from typing import Any, Dict, Optional

# Metal kernel for deterministic softmax (optimized for M4 Max)
# Each threadgroup processes one row with tree reductions for max and sum
SOFTMAX_KERNEL_SOURCE: str = """
// Each threadgroup handles one row
uint row_idx = threadgroup_position_in_grid.x;
uint local_id = thread_position_in_threadgroup.x;
uint num_threads = threads_per_threadgroup.x;

// Get dimensions
uint batch_size = x_shape[0];
uint cols = x_shape[x_ndim - 1];

// Bounds check
if (row_idx >= batch_size) {
    return;
}

// Shared memory for reductions
threadgroup T shared_max[256];
threadgroup T shared_sum[256];

// Initialize shared memory
shared_max[local_id] = T(-1e38);  // Very negative for max
shared_sum[local_id] = T(0);
threadgroup_barrier(mem_flags::mem_threadgroup);

// Base address for this row
uint base = row_idx * cols;

// Phase 1: Find max along row (each thread finds local max)
// Process 4 elements at a time for better throughput
T local_max = T(-1e38);
uint i = local_id;
for (; i + 3 * num_threads < cols; i += 4 * num_threads) {
    T v0 = x[base + i];
    T v1 = x[base + i + num_threads];
    T v2 = x[base + i + 2 * num_threads];
    T v3 = x[base + i + 3 * num_threads];
    local_max = metal::max(local_max, metal::max(metal::max(v0, v1), metal::max(v2, v3)));
}
// Handle remaining elements
for (; i < cols; i += num_threads) {
    T val = x[base + i];
    local_max = metal::max(local_max, val);
}
shared_max[local_id] = local_max;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Tree reduction for max - DETERMINISTIC order with explicit unrolling
if (num_threads >= 256) {
    if (local_id < 128) shared_max[local_id] = metal::max(shared_max[local_id], shared_max[local_id + 128]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (num_threads >= 128) {
    if (local_id < 64) shared_max[local_id] = metal::max(shared_max[local_id], shared_max[local_id + 64]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (num_threads >= 64) {
    if (local_id < 32) shared_max[local_id] = metal::max(shared_max[local_id], shared_max[local_id + 32]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (local_id < 16) shared_max[local_id] = metal::max(shared_max[local_id], shared_max[local_id + 16]);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 8) shared_max[local_id] = metal::max(shared_max[local_id], shared_max[local_id + 8]);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 4) shared_max[local_id] = metal::max(shared_max[local_id], shared_max[local_id + 4]);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 2) shared_max[local_id] = metal::max(shared_max[local_id], shared_max[local_id + 2]);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 1) shared_max[local_id] = metal::max(shared_max[local_id], shared_max[local_id + 1]);
threadgroup_barrier(mem_flags::mem_threadgroup);

// Now shared_max[0] contains the row max
T row_max = shared_max[0];

// Phase 2: Compute sum of exp(x - max)
// Process 4 elements at a time
T local_sum = T(0);
i = local_id;
for (; i + 3 * num_threads < cols; i += 4 * num_threads) {
    T v0 = x[base + i];
    T v1 = x[base + i + num_threads];
    T v2 = x[base + i + 2 * num_threads];
    T v3 = x[base + i + 3 * num_threads];
    local_sum += metal::exp(v0 - row_max) + metal::exp(v1 - row_max) + metal::exp(v2 - row_max) + metal::exp(v3 - row_max);
}
// Handle remaining elements
for (; i < cols; i += num_threads) {
    T val = x[base + i];
    local_sum += metal::exp(val - row_max);
}
shared_sum[local_id] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Tree reduction for sum - DETERMINISTIC order with explicit unrolling
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
if (local_id < 16) shared_sum[local_id] += shared_sum[local_id + 16];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 8) shared_sum[local_id] += shared_sum[local_id + 8];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 4) shared_sum[local_id] += shared_sum[local_id + 4];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 2) shared_sum[local_id] += shared_sum[local_id + 2];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_id < 1) shared_sum[local_id] += shared_sum[local_id + 1];
threadgroup_barrier(mem_flags::mem_threadgroup);

// Now shared_sum[0] contains sum of exp(x - max)
T inv_sum = T(1) / shared_sum[0];  // Pre-compute inverse for faster division

// Phase 3: Compute final softmax values
// Process 4 elements at a time
i = local_id;
for (; i + 3 * num_threads < cols; i += 4 * num_threads) {
    T v0 = x[base + i];
    T v1 = x[base + i + num_threads];
    T v2 = x[base + i + 2 * num_threads];
    T v3 = x[base + i + 3 * num_threads];
    out[base + i] = metal::exp(v0 - row_max) * inv_sum;
    out[base + i + num_threads] = metal::exp(v1 - row_max) * inv_sum;
    out[base + i + 2 * num_threads] = metal::exp(v2 - row_max) * inv_sum;
    out[base + i + 3 * num_threads] = metal::exp(v3 - row_max) * inv_sum;
}
// Handle remaining elements
for (; i < cols; i += num_threads) {
    T val = x[base + i];
    out[base + i] = metal::exp(val - row_max) * inv_sum;
}
"""

# Kernel cache
_kernel_cache: Dict[str, Any] = {}


def _create_softmax_kernel():
    """Create the Metal kernel for deterministic softmax."""
    return mx.fast.metal_kernel(
        name="deterministic_softmax",
        input_names=["x"],
        output_names=["out"],
        source=SOFTMAX_KERNEL_SOURCE,
        ensure_row_contiguous=True,
        atomic_outputs=False
    )


def softmax_metal(
    x: mx.array,
    axis: int = -1,
) -> mx.array:
    """
    Bitwise deterministic softmax using custom Metal kernel.

    Args:
        x: Input tensor
        axis: Axis along which to compute softmax (must be -1)

    Returns:
        Softmax probabilities with same shape as x
    """
    if axis != -1:
        raise NotImplementedError("Only axis=-1 is currently supported for Metal softmax")

    original_shape = x.shape
    cols = x.shape[-1]

    # Flatten to 2D: [batch, cols]
    x_flat = x.reshape(-1, cols)
    batch_size = x_flat.shape[0]

    # Get or create kernel
    key = x.dtype
    if key not in _kernel_cache:
        _kernel_cache[key] = _create_softmax_kernel()
    kernel = _kernel_cache[key]

    # Each threadgroup handles one row
    # Use 256 threads per group (power of 2 for tree reduction)
    num_threads = min(256, cols)
    # Ensure power of 2
    num_threads = 1 << (num_threads - 1).bit_length() if num_threads > 1 else 1
    num_threads = min(num_threads, 256)

    # Grid specifies total threads
    total_threads = batch_size * num_threads
    grid = (total_threads, 1, 1)
    threadgroup = (num_threads, 1, 1)

    # Execute kernel
    outputs = kernel(
        inputs=[x_flat],
        template=[("T", x.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[x_flat.shape],
        output_dtypes=[x.dtype],
    )

    return outputs[0].reshape(original_shape)
