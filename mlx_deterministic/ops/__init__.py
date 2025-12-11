"""
Batch-Invariant Operations for MLX

This module provides deterministic, batch-invariant implementations of core
operations for LLM inference on Apple Silicon.

Based on Thinking Machines Labs research:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
"""

from .rms_norm import BatchInvariantRMSNorm, rms_norm_batch_invariant
from .matmul import batch_invariant_matmul, batch_invariant_addmm, BatchInvariantLinear
from .attention import (
    batch_invariant_softmax,
    BatchInvariantAttention,
    create_causal_mask,
    flash_attention_fixed_split,
    scaled_dot_product_attention_deterministic,
)
from .kv_cache import DeterministicKVCache, RotatingDeterministicKVCache

# Metal kernel implementations for bitwise determinism
from .metal_matmul import deterministic_matmul_metal, deterministic_addmm_metal, DeterministicLinearMetal
from .metal_rms_norm import rms_norm_metal, BatchInvariantRMSNormMetal
from .metal_softmax import softmax_metal
from .metal_quantized_matmul import quantized_matmul_metal, DeterministicQuantizedLinear

__all__ = [
    # Python-based implementations (~1e-5 tolerance)
    "BatchInvariantRMSNorm",
    "rms_norm_batch_invariant",
    "batch_invariant_matmul",
    "batch_invariant_addmm",
    "BatchInvariantLinear",
    "batch_invariant_softmax",
    "BatchInvariantAttention",
    "create_causal_mask",
    "flash_attention_fixed_split",
    "scaled_dot_product_attention_deterministic",
    "DeterministicKVCache",
    "RotatingDeterministicKVCache",
    # Metal kernel implementations (bitwise determinism)
    "deterministic_matmul_metal",
    "deterministic_addmm_metal",
    "DeterministicLinearMetal",
    "rms_norm_metal",
    "BatchInvariantRMSNormMetal",
    "softmax_metal",
    "quantized_matmul_metal",
    "DeterministicQuantizedLinear",
]
