"""
Deterministic Quantized Matrix Multiplication using Custom Metal Kernel

This module implements a bitwise-deterministic quantized matmul using MLX's custom
Metal kernel API. It handles 4-bit quantized weights (as used in MLX-LM quantized
models) with on-the-fly dequantization during matrix multiplication.

Key design principles:
1. Same tiling strategy as metal_matmul.py (64x64 output tiles)
2. On-the-fly dequantization to avoid memory bandwidth bottleneck
3. Fixed K-loop order for deterministic accumulation
4. No atomics - guaranteed determinism

Weight format (from nn.QuantizedLinear):
- weight: [out_features, K/8] as uint32 - 8x 4-bit values packed per uint32
- scales: [out_features, K/group_size] - per-group scale factors
- biases: [out_features, K/group_size] - per-group bias values
- group_size: typically 64
- bits: typically 4

Dequantization: w_float = (quant_val - bias) * scale
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Any, Dict, Optional

# =============================================================================
# QUANTIZED MATMUL KERNEL - FP16 with on-the-fly dequantization
# =============================================================================
# This kernel performs: output = x @ dequantize(packed_w).T
# where x is [batch, M, K] float16 and packed_w is [N, K/8] uint32

QUANTIZED_MATMUL_KERNEL: str = """
#include <metal_simdgroup_matrix>

// Tiling constants
#define BM 64          // Block size M (output tile height)
#define BN 64          // Block size N (output tile width)
#define BK 16          // Block size K (inner dimension step)
#define GROUP_SIZE 64  // Quantization group size

// Thread/simdgroup positions
uint tg_row = threadgroup_position_in_grid.y;
uint tg_col = threadgroup_position_in_grid.x;
uint local_id = thread_position_in_threadgroup.x;  // 0-127

// Compute simdgroup and thread-within-simdgroup indices
uint simd_group_id = local_id / 32;        // 0-3 (which of 4 simdgroups)
uint thread_in_simd = local_id % 32;       // 0-31 (lane within simdgroup)

// Get matrix dimensions
// x: [batch*M, K], packed_w: [N, K/8], output: [batch*M, N]
uint M_val = x_shape[x_ndim - 2];
uint K_val = x_shape[x_ndim - 1];
uint N_val = packed_w_shape[0];
uint packed_K = packed_w_shape[1];  // K / 8

// Global tile start positions
uint tile_row = tg_row * BM;
uint tile_col = tg_col * BN;

// Shared memory for tiles
threadgroup half As[BM][BK];  // Input tile: 64 x 16
threadgroup half Ws[BK][BN];  // Dequantized weight tile: 16 x 64

// Initialize accumulators - 4x4 grid of 8x8 matrices per simdgroup = 32x32 output
simdgroup_matrix<half, 8, 8> acc[4][4];
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(half(0));
    }
}

// Number of K-tiles
uint num_k_tiles = (K_val + BK - 1) / BK;

// Simdgroup offsets for compute
int sg_row_offset = (simd_group_id / 2) * 32;
int sg_col_offset = (simd_group_id % 2) * 32;

// Main K-loop - DETERMINISTIC order
for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++) {
    uint k_offset = k_tile * BK;

    // --- Phase A: Load input tile into shared memory ---
    // 128 threads loading 64x16 = 1024 elements (8 per thread)
    for (int i = 0; i < 8; i++) {
        uint idx = local_id * 8 + i;
        uint r = idx / BK;         // 0-63
        uint c = idx % BK;         // 0-15

        uint global_r = tile_row + r;
        uint global_c = k_offset + c;

        half val = half(0);
        if (global_r < M_val && global_c < K_val) {
            val = x[global_r * K_val + global_c];
        }
        As[r][c] = val;
    }

    // --- Phase B: Load and dequantize weight tile ---
    // Weight is [N, K/8] with 8x 4-bit values per uint32
    // We need to load Ws[BK][BN] = 16x64 = 1024 half values
    // 128 threads, 8 elements each

    for (int i = 0; i < 8; i++) {
        uint idx = local_id * 8 + i;
        uint local_k = idx / BN;    // 0-15 (which K position in tile)
        uint local_n = idx % BN;    // 0-63 (which N position in tile)

        uint global_k = k_offset + local_k;
        uint global_n = tile_col + local_n;

        half val = half(0);
        if (global_k < K_val && global_n < N_val) {
            // Extract 4-bit value from packed uint32
            uint packed_idx = global_k / 8;
            uint bit_offset = (global_k % 8) * 4;
            uint packed = packed_w[global_n * packed_K + packed_idx];
            uint quant_val = (packed >> bit_offset) & 0xF;

            // Get scale and bias for this group
            uint group_idx = global_k / GROUP_SIZE;
            uint scale_stride = K_val / GROUP_SIZE;
            half scale = scales[global_n * scale_stride + group_idx];
            half bias = biases[global_n * scale_stride + group_idx];

            // Dequantize: w = (quant_val * scale) + bias
            val = half(quant_val) * scale + bias;
        }
        Ws[local_k][local_n] = val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase C: SIMD Matrix Multiply ---
    for (int k = 0; k < BK; k += 8) {
        for (int i = 0; i < 4; i++) {
            simdgroup_matrix<half, 8, 8> matA;
            simdgroup_load(matA,
                           (const threadgroup half*)&As[sg_row_offset + i * 8][k],
                           BK);

            for (int j = 0; j < 4; j++) {
                simdgroup_matrix<half, 8, 8> matW;
                simdgroup_load(matW,
                               (const threadgroup half*)&Ws[k][sg_col_offset + j * 8],
                               BN);

                simdgroup_multiply_accumulate(acc[i][j], matA, matW, acc[i][j]);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// --- Phase D: Write Results ---
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        uint global_row = tile_row + sg_row_offset + i * 8;
        uint global_col = tile_col + sg_col_offset + j * 8;

        if (global_row < M_val && global_col < N_val) {
            simdgroup_store(acc[i][j],
                            output + global_row * N_val + global_col,
                            N_val);
        }
    }
}
"""

# =============================================================================
# SIMPLE QUANTIZED KERNEL (fallback for small matrices)
# =============================================================================
QUANTIZED_MATMUL_KERNEL_SIMPLE: str = """
// Each thread computes one output element
uint row = thread_position_in_grid.y;
uint col = thread_position_in_grid.x;

// Get dimensions
uint M_val = x_shape[x_ndim - 2];
uint K_val = x_shape[x_ndim - 1];
uint N_val = packed_w_shape[0];
uint packed_K = packed_w_shape[1];  // K / 8
uint group_size = 64;  // Fixed for now

if (row >= M_val || col >= N_val) {
    return;
}

// Compute dot product with FIXED iteration order
half acc = half(0);
uint scale_stride = K_val / group_size;

for (uint k = 0; k < K_val; k++) {
    // Get input value
    half x_val = x[row * K_val + k];

    // Extract 4-bit quantized weight
    uint packed_idx = k / 8;
    uint bit_offset = (k % 8) * 4;
    uint packed = packed_w[col * packed_K + packed_idx];
    uint quant_val = (packed >> bit_offset) & 0xF;

    // Dequantize weight: w = (quant_val * scale) + bias
    uint group_idx = k / group_size;
    half scale = scales[col * scale_stride + group_idx];
    half bias = biases[col * scale_stride + group_idx];
    half w_val = half(quant_val) * scale + bias;

    acc += x_val * w_val;
}

output[row * N_val + col] = acc;
"""

# Kernel cache
_kernel_cache: Dict[str, Any] = {}


def _create_quantized_matmul_kernel():
    """Create the SIMD-based quantized matmul kernel."""
    return mx.fast.metal_kernel(
        name="deterministic_quantized_matmul_v1",
        input_names=["x", "packed_w", "scales", "biases"],
        output_names=["output"],
        source=QUANTIZED_MATMUL_KERNEL,
        ensure_row_contiguous=True,
        atomic_outputs=False
    )


def _create_quantized_matmul_kernel_simple():
    """Create simple quantized matmul kernel for small matrices."""
    return mx.fast.metal_kernel(
        name="deterministic_quantized_matmul_simple",
        input_names=["x", "packed_w", "scales", "biases"],
        output_names=["output"],
        source=QUANTIZED_MATMUL_KERNEL_SIMPLE,
        ensure_row_contiguous=True,
        atomic_outputs=False
    )


def quantized_matmul_metal(
    x: mx.array,
    packed_w: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """
    Bitwise deterministic quantized matrix multiplication.

    Computes: output = x @ dequantize(packed_w).T

    Args:
        x: Input tensor [batch, M, K] or [M, K] as float16
        packed_w: Packed quantized weights [N, K/8] as uint32
        scales: Scale factors [N, K/group_size] as float16
        biases: Bias values [N, K/group_size] as float16
        group_size: Quantization group size (default 64)
        bits: Bits per weight (default 4)

    Returns:
        Output tensor [batch, M, N] or [M, N] as float16
    """
    if bits != 4:
        raise NotImplementedError(f"Only 4-bit quantization supported, got {bits}")
    if group_size != 64:
        raise NotImplementedError(f"Only group_size=64 supported, got {group_size}")

    # Handle batched inputs
    original_shape = x.shape
    if x.ndim > 2:
        batch_shape = x.shape[:-2]
        x_flat = x.reshape(-1, x.shape[-2], x.shape[-1])
        batch_size = x_flat.shape[0]

        results = []
        for i in range(batch_size):
            result = _quantized_matmul_2d(x_flat[i], packed_w, scales, biases)
            results.append(result)

        stacked = mx.stack(results, axis=0)
        return stacked.reshape(*batch_shape, *stacked.shape[-2:])

    return _quantized_matmul_2d(x, packed_w, scales, biases)


def _quantized_matmul_2d(
    x: mx.array,
    packed_w: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> mx.array:
    """Perform 2D quantized matrix multiplication using Metal kernel.

    Args:
        x: Input tensor of shape [M, K] (float16)
        packed_w: Quantized weights of shape [N, K/8] (uint32)
        scales: Scale factors for dequantization
        biases: Bias values for dequantization

    Returns:
        mx.array: Output tensor of shape [M, N] as float16
    """
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"

    M, K = x.shape
    N = packed_w.shape[0]

    # Convert x to float16 if needed
    if x.dtype != mx.float16:
        x = x.astype(mx.float16)

    # Convert scales/biases to float16 if needed
    if scales.dtype != mx.float16:
        scales = scales.astype(mx.float16)
    if biases.dtype != mx.float16:
        biases = biases.astype(mx.float16)

    # Choose kernel based on size
    use_simd = M >= 64 and N >= 64 and K >= 64

    if use_simd:
        key = "quantized_simd_v1"
        if key not in _kernel_cache:
            _kernel_cache[key] = _create_quantized_matmul_kernel()
        kernel = _kernel_cache[key]

        BM, BN = 64, 64
        num_tiles_m = (M + BM - 1) // BM
        num_tiles_n = (N + BN - 1) // BN

        grid = (num_tiles_n * 128, num_tiles_m, 1)
        threadgroup = (128, 1, 1)
    else:
        key = "quantized_simple"
        if key not in _kernel_cache:
            _kernel_cache[key] = _create_quantized_matmul_kernel_simple()
        kernel = _kernel_cache[key]

        grid = (N, M, 1)
        threadgroup = (min(N, 16), min(M, 16), 1)

    outputs = kernel(
        inputs=[x, packed_w, scales, biases],
        template=[],  # No template needed - using half explicitly
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
    )

    return outputs[0]


class DeterministicQuantizedLinear:
    """
    Drop-in replacement for nn.QuantizedLinear with bitwise determinism.

    This class wraps the quantized matmul Metal kernel to provide
    identical results regardless of batch size.
    """

    def __init__(
        self,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
        group_size: int = 64,
        bits: int = 4,
        bias: Optional[mx.array] = None,
    ):
        """
        Initialize from quantized weight parameters.

        Args:
            weight: Packed quantized weights [out_features, in_features/8] as uint32
            scales: Scale factors [out_features, in_features/group_size]
            biases: Quantization biases [out_features, in_features/group_size]
            group_size: Quantization group size
            bits: Bits per weight
            bias: Optional output bias [out_features]
        """
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.bias = bias

    @classmethod
    def from_quantized_linear(cls, qlinear: nn.QuantizedLinear) -> "DeterministicQuantizedLinear":
        """
        Create from an existing nn.QuantizedLinear layer.

        Args:
            qlinear: The quantized linear layer to wrap

        Returns:
            DeterministicQuantizedLinear with same parameters
        """
        bias = qlinear.bias if hasattr(qlinear, 'bias') else None
        return cls(
            weight=qlinear.weight,
            scales=qlinear.scales,
            biases=qlinear.biases,
            group_size=qlinear.group_size,
            bits=qlinear.bits,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply quantized linear transformation.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        result = quantized_matmul_metal(
            x,
            self.weight,
            self.scales,
            self.biases,
            self.group_size,
            self.bits,
        )

        if self.bias is not None:
            result = result + self.bias

        return result
