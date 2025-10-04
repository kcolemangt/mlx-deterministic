"""
MLX Deterministic Inference

Batch-invariant operations for deterministic LLM inference on Apple Silicon.

Based on research from Thinking Machines Labs:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
"""

from .ops import (
    BatchInvariantRMSNorm,
    rms_norm_batch_invariant,
    batch_invariant_matmul,
    batch_invariant_addmm,
    BatchInvariantLinear,
    batch_invariant_softmax,
    BatchInvariantAttention,
    create_causal_mask,
)

__version__ = "0.1.0"

__all__ = [
    "BatchInvariantRMSNorm",
    "rms_norm_batch_invariant",
    "batch_invariant_matmul",
    "batch_invariant_addmm",
    "BatchInvariantLinear",
    "batch_invariant_softmax",
    "BatchInvariantAttention",
    "create_causal_mask",
    "DeterministicConfig",
    "enable_deterministic_mode",
]


class DeterministicConfig:
    """
    Configuration for deterministic operations.

    Args:
        rms_norm_chunk_size: Chunk size for RMSNorm variance computation
        matmul_tile_size: Tile size for matrix multiplication
        softmax_chunk_size: Chunk size for softmax reduction
        attention_matmul_tile_size: Tile size for attention matmuls
    """

    def __init__(
        self,
        rms_norm_chunk_size: int = 64,
        matmul_tile_size: int = 128,
        softmax_chunk_size: int = 128,
        attention_matmul_tile_size: int = 128,
    ):
        self.rms_norm_chunk_size = rms_norm_chunk_size
        self.matmul_tile_size = matmul_tile_size
        self.softmax_chunk_size = softmax_chunk_size
        self.attention_matmul_tile_size = attention_matmul_tile_size


def enable_deterministic_mode(model, config: DeterministicConfig = None):
    """
    Enable deterministic mode for an MLX-LM model by replacing operations
    with batch-invariant versions.

    This function traverses the model and replaces:
    - RMSNorm layers with BatchInvariantRMSNorm
    - Linear layers can be wrapped to use batch-invariant matmul
    - Attention modules with BatchInvariantAttention

    Args:
        model: MLX model to convert
        config: Configuration for deterministic operations

    Returns:
        Modified model with deterministic operations

    Example:
        >>> from mlx_lm import load
        >>> from mlx_deterministic import enable_deterministic_mode
        >>> model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
        >>> model = enable_deterministic_mode(model)
    """
    if config is None:
        config = DeterministicConfig()

    import mlx.nn as nn

    def replace_modules(module, parent_name=""):
        """Recursively replace modules with deterministic versions."""
        # Get all attributes that are nn.Module instances
        for name in dir(module):
            if name.startswith('_'):
                continue

            try:
                child = getattr(module, name)
            except:
                continue

            # Skip if not a module
            if not isinstance(child, nn.Module):
                continue

            full_name = f"{parent_name}.{name}" if parent_name else name

            # Replace RMSNorm
            if isinstance(child, nn.RMSNorm):
                new_module = BatchInvariantRMSNorm(
                    dims=child.dims,
                    eps=child.eps,
                    chunk_size=config.rms_norm_chunk_size
                )
                # Copy weights
                new_module.weight = child.weight
                setattr(module, name, new_module)
                print(f"Replaced {full_name} with BatchInvariantRMSNorm")

            # Replace MultiHeadAttention
            elif isinstance(child, nn.MultiHeadAttention):
                new_module = BatchInvariantAttention(
                    dims=child.dims,
                    num_heads=child.num_heads,
                    query_input_dims=child.query_input_dims,
                    key_input_dims=child.key_input_dims,
                    value_dims=child.value_dims,
                    value_output_dims=child.value_output_dims,
                    bias=child.query_proj.bias is not None,
                    matmul_tile_size=config.attention_matmul_tile_size,
                    softmax_chunk_size=config.softmax_chunk_size,
                )
                # Copy weights
                new_module.query_proj.weight = child.query_proj.weight
                new_module.key_proj.weight = child.key_proj.weight
                new_module.value_proj.weight = child.value_proj.weight
                new_module.out_proj.weight = child.out_proj.weight
                if child.query_proj.bias is not None:
                    new_module.query_proj.bias = child.query_proj.bias
                    new_module.key_proj.bias = child.key_proj.bias
                    new_module.value_proj.bias = child.value_proj.bias
                    new_module.out_proj.bias = child.out_proj.bias
                setattr(module, name, new_module)
                print(f"Replaced {full_name} with BatchInvariantAttention")

            # Recursively process children
            elif hasattr(child, '__dict__'):
                replace_modules(child, full_name)

    replace_modules(model)
    print(f"\nDeterministic mode enabled with config:")
    print(f"  RMSNorm chunk size: {config.rms_norm_chunk_size}")
    print(f"  Matmul tile size: {config.matmul_tile_size}")
    print(f"  Softmax chunk size: {config.softmax_chunk_size}")
    print(f"  Attention matmul tile size: {config.attention_matmul_tile_size}")

    return model
