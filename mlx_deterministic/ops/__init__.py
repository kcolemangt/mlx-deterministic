"""
Batch-Invariant Operations for MLX

This module provides deterministic, batch-invariant implementations of core
operations for LLM inference on Apple Silicon.
"""

from .rms_norm import BatchInvariantRMSNorm, rms_norm_batch_invariant
from .matmul import batch_invariant_matmul, batch_invariant_addmm, BatchInvariantLinear
from .attention import (
    batch_invariant_softmax,
    BatchInvariantAttention,
    create_causal_mask
)

__all__ = [
    "BatchInvariantRMSNorm",
    "rms_norm_batch_invariant",
    "batch_invariant_matmul",
    "batch_invariant_addmm",
    "BatchInvariantLinear",
    "batch_invariant_softmax",
    "BatchInvariantAttention",
    "create_causal_mask",
]
