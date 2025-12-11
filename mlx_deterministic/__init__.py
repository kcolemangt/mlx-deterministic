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
    flash_attention_fixed_split,
    scaled_dot_product_attention_deterministic,
    DeterministicKVCache,
    RotatingDeterministicKVCache,
    # Metal kernel implementations (bitwise determinism)
    deterministic_matmul_metal,
    deterministic_addmm_metal,
    DeterministicLinearMetal,
    rms_norm_metal,
    BatchInvariantRMSNormMetal,
    softmax_metal,
)

# Store original function for restoration
_ORIGINAL_SDPA = None

__version__ = "0.3.0"  # Version bump for Metal kernel implementations

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
    "DeterministicKVCache",
    "RotatingDeterministicKVCache",
    "DeterministicConfig",
    "enable_deterministic_mode",
    "enable_mlx_lm_deterministic_mode",
    "disable_mlx_lm_deterministic_mode",
    "scaled_dot_product_attention_deterministic",
    # Metal kernel implementations (bitwise determinism)
    "deterministic_matmul_metal",
    "deterministic_addmm_metal",
    "DeterministicLinearMetal",
    "rms_norm_metal",
    "BatchInvariantRMSNormMetal",
    "softmax_metal",
    "quantized_matmul_metal",
    "DeterministicQuantizedLinear",
    "replace_quantized_linear_layers",
]


class DeterministicConfig:
    """
    Configuration for deterministic operations.

    Args:
        rms_norm_chunk_size: Chunk size for RMSNorm variance computation
        matmul_tile_size: Tile size for matrix multiplication
        softmax_chunk_size: Chunk size for softmax reduction
        attention_matmul_tile_size: Tile size for attention matmuls
        use_metal_kernels: If True, use custom Metal kernels for bitwise-identical
                          determinism. If False (default), use Python implementations
                          with ~1e-5 tolerance.
    """

    def __init__(
        self,
        rms_norm_chunk_size: int = 64,
        matmul_tile_size: int = 128,
        softmax_chunk_size: int = 128,
        attention_matmul_tile_size: int = 128,
        use_metal_kernels: bool = False,
    ):
        self.rms_norm_chunk_size = rms_norm_chunk_size
        self.matmul_tile_size = matmul_tile_size
        self.softmax_chunk_size = softmax_chunk_size
        self.attention_matmul_tile_size = attention_matmul_tile_size
        self.use_metal_kernels = use_metal_kernels


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

    visited = set()  # Track visited modules to prevent cycles
    replacement_count = 0

    def replace_modules(module, parent_name=""):
        """Recursively replace modules with deterministic versions."""
        nonlocal replacement_count

        # Prevent infinite recursion from circular references (e.g., module.state returns self)
        if id(module) in visited:
            return
        visited.add(id(module))

        # Get child module names using MLX's children() method
        if not hasattr(module, 'children'):
            return

        for name in module.children():
            try:
                child = getattr(module, name)
            except:
                continue

            full_name = f"{parent_name}.{name}" if parent_name else name

            # Handle lists of modules (e.g., transformer layers)
            if isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        item_name = f"{full_name}[{i}]"
                        # Recurse into each list item to replace its children
                        replace_modules(item, item_name)
                continue

            # Skip if not a module
            if not isinstance(child, nn.Module):
                continue

            # Replace RMSNorm
            if isinstance(child, nn.RMSNorm):
                # Infer dims from weight shape since MLX doesn't store it as attribute
                dims = child.weight.shape[0]
                if config.use_metal_kernels:
                    new_module = BatchInvariantRMSNormMetal(dims=dims, eps=child.eps)
                    new_module.weight = child.weight
                    setattr(module, name, new_module)
                    replacement_count += 1
                    print(f"Replaced {full_name} with BatchInvariantRMSNormMetal (bitwise deterministic)")
                else:
                    new_module = BatchInvariantRMSNorm(
                        dims=dims,
                        eps=child.eps,
                        chunk_size=config.rms_norm_chunk_size
                    )
                    # Copy weights
                    new_module.weight = child.weight
                    setattr(module, name, new_module)
                    replacement_count += 1
                    print(f"Replaced {full_name} with BatchInvariantRMSNorm")

            # Replace MultiHeadAttention
            elif isinstance(child, nn.MultiHeadAttention):
                # Infer dimensions from weight shapes since MLX doesn't store them as attributes
                dims = child.query_proj.weight.shape[0]  # Output dim of query projection
                query_input_dims = child.query_proj.weight.shape[1]  # Input dim
                key_input_dims = child.key_proj.weight.shape[1]
                value_dims = child.value_proj.weight.shape[0]
                value_output_dims = child.out_proj.weight.shape[0]
                has_bias = hasattr(child.query_proj, 'bias')

                new_module = BatchInvariantAttention(
                    dims=dims,
                    num_heads=child.num_heads,
                    query_input_dims=query_input_dims,
                    key_input_dims=key_input_dims,
                    value_dims=value_dims,
                    value_output_dims=value_output_dims,
                    bias=has_bias,
                    matmul_tile_size=config.attention_matmul_tile_size,
                    softmax_chunk_size=config.softmax_chunk_size,
                )
                # Copy weights
                new_module.query_proj.weight = child.query_proj.weight
                new_module.key_proj.weight = child.key_proj.weight
                new_module.value_proj.weight = child.value_proj.weight
                new_module.out_proj.weight = child.out_proj.weight
                if has_bias:
                    new_module.query_proj.bias = child.query_proj.bias
                    new_module.key_proj.bias = child.key_proj.bias
                    new_module.value_proj.bias = child.value_proj.bias
                    new_module.out_proj.bias = child.out_proj.bias
                setattr(module, name, new_module)
                replacement_count += 1
                print(f"Replaced {full_name} with BatchInvariantAttention")

            # Recursively process children
            elif hasattr(child, '__dict__'):
                replace_modules(child, full_name)

    replace_modules(model)
    print(f"\nDeterministic mode enabled ({replacement_count} modules replaced)")
    print(f"Config:")
    print(f"  RMSNorm chunk size: {config.rms_norm_chunk_size}")
    print(f"  Matmul tile size: {config.matmul_tile_size}")
    print(f"  Softmax chunk size: {config.softmax_chunk_size}")
    print(f"  Attention matmul tile size: {config.attention_matmul_tile_size}")

    return model


def enable_mlx_lm_deterministic_mode(split_size: int = 256):
    """
    Enable batch-invariant attention for all MLX-LM models.

    This monkey-patches mlx_lm.models.base.scaled_dot_product_attention
    to use our deterministic implementation. This is the recommended way
    to enable determinism for real MLX-LM models (Qwen, Llama, Mistral, etc.)
    which use custom Attention classes that don't inherit from nn.MultiHeadAttention.

    IMPORTANT: Call this BEFORE importing any model modules (before load()).
    Each model module (qwen2, llama, etc.) imports scaled_dot_product_attention
    at module load time, so patching must happen before that import.

    Args:
        split_size: KV split size for flash attention (default: 256).
                   Smaller values = more deterministic but slower.
                   Larger values = faster but may have more numerical variance.

    Example:
        >>> from mlx_deterministic import enable_mlx_lm_deterministic_mode
        >>> enable_mlx_lm_deterministic_mode()  # MUST be called BEFORE load()
        >>>
        >>> from mlx_lm import load, generate
        >>> model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")
        >>>
        >>> # Now all inference is batch-invariant
        >>> response = generate(model, tokenizer, "Hello", max_tokens=50)
    """
    global _ORIGINAL_SDPA

    try:
        import mlx_lm.models.base as mlx_base
    except ImportError:
        raise ImportError(
            "mlx-lm is required for enable_mlx_lm_deterministic_mode(). "
            "Install with: pip install mlx-lm"
        )

    # Store original for potential restoration
    if _ORIGINAL_SDPA is None:
        _ORIGINAL_SDPA = mlx_base.scaled_dot_product_attention

    # Create wrapper with configured split_size
    def patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        return scaled_dot_product_attention_deterministic(
            queries, keys, values, cache, scale, mask, sinks,
            split_size=split_size
        )

    # Apply patch to base module
    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Also patch any already-imported model modules
    # These modules do `from .base import scaled_dot_product_attention`
    # which creates a local binding that doesn't update when we patch base
    import sys
    model_modules = [
        'mlx_lm.models.qwen2',
        'mlx_lm.models.qwen3',
        'mlx_lm.models.llama',
        'mlx_lm.models.mistral',
        'mlx_lm.models.phi3',
        'mlx_lm.models.gemma',
        'mlx_lm.models.gemma2',
        'mlx_lm.models.starcoder2',
        'mlx_lm.models.cohere',
        'mlx_lm.models.dbrx',
        'mlx_lm.models.minicpm',
        'mlx_lm.models.internlm2',
        'mlx_lm.models.deepseek',
        'mlx_lm.models.deepseek_v2',
        'mlx_lm.models.deepseek_v3',
        'mlx_lm.models.olmo',
        'mlx_lm.models.openelm',
        'mlx_lm.models.gpt2',
        'mlx_lm.models.gpt_bigcode',
        'mlx_lm.models.gpt_neox',
        'mlx_lm.models.phi',
        'mlx_lm.models.phimoe',
        'mlx_lm.models.mamba',
        'mlx_lm.models.exaone',
        'mlx_lm.models.plamo',
        'mlx_lm.models.recurrentgemma',
    ]

    patched_count = 0
    for module_name in model_modules:
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, 'scaled_dot_product_attention'):
                module.scaled_dot_product_attention = patched_sdpa
                patched_count += 1

    print(f"MLX-LM deterministic attention enabled (split_size={split_size})")
    if patched_count > 0:
        print(f"  Patched {patched_count} already-imported model modules")


def disable_mlx_lm_deterministic_mode():
    """
    Restore original MLX-LM attention (disable deterministic mode).

    This restores the original mx.fast.scaled_dot_product_attention behavior.
    """
    global _ORIGINAL_SDPA

    if _ORIGINAL_SDPA is None:
        print("MLX-LM deterministic mode was not enabled")
        return

    try:
        import mlx_lm.models.base as mlx_base
        mlx_base.scaled_dot_product_attention = _ORIGINAL_SDPA
        print("MLX-LM deterministic mode disabled (original attention restored)")
    except ImportError:
        pass


def replace_quantized_linear_layers(model, verbose: bool = False):
    """
    Replace all QuantizedLinear layers in a model with DeterministicQuantizedLinear.

    This enables bitwise-identical outputs regardless of batch size for quantized models.
    Also patches QuantizedEmbedding.as_linear for models with tied embeddings.

    Args:
        model: MLX-LM model to patch
        verbose: If True, print each replacement

    Returns:
        The modified model (also modifies in-place)

    Example:
        >>> from mlx_lm import load
        >>> from mlx_deterministic import replace_quantized_linear_layers
        >>>
        >>> model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")
        >>> replace_quantized_linear_layers(model)
    """
    import mlx.nn as nn
    from .ops.metal_quantized_matmul import DeterministicQuantizedLinear, quantized_matmul_metal

    replaced_count = 0
    visited = set()

    def replace_in_module(module, parent_name=""):
        nonlocal replaced_count

        if id(module) in visited:
            return
        visited.add(id(module))

        if not hasattr(module, 'children'):
            return

        for name in module.children():
            try:
                child = getattr(module, name)
            except:
                continue

            full_name = f"{parent_name}.{name}" if parent_name else name

            # Handle lists (e.g., layers)
            if isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        replace_in_module(item, f"{full_name}[{i}]")
                continue

            if not isinstance(child, nn.Module):
                continue

            # Replace QuantizedLinear
            if isinstance(child, nn.QuantizedLinear):
                det_linear = DeterministicQuantizedLinear.from_quantized_linear(child)
                setattr(module, name, det_linear)
                replaced_count += 1
                if verbose:
                    print(f"Replaced {full_name} with DeterministicQuantizedLinear")
            # Patch QuantizedEmbedding.as_linear for tied embeddings
            elif isinstance(child, nn.QuantizedEmbedding):
                # Create deterministic as_linear method
                def make_det_as_linear(embed):
                    def det_as_linear(x):
                        # The embedding weight is [vocab_size, embed_dim]
                        # as_linear computes: x @ W.T (transpose=True)
                        # Our kernel already handles this since it's x @ packed_w where packed_w is [N, K/8]
                        # For transposed: we need x @ W.T which is x @ [embed_dim, vocab_size]
                        # The weight is [vocab_size, embed_dim/8], so transposed is [embed_dim, vocab_size/8]?
                        # Actually simpler: just use our kernel which expects W as [out_features, in_features/8]
                        # For tied embedding as lm_head: out_features=vocab_size, in_features=embed_dim
                        return quantized_matmul_metal(
                            x,
                            embed.weight,
                            embed.scales,
                            embed.biases if hasattr(embed, 'biases') and embed.biases is not None else embed.get("biases"),
                            embed.group_size,
                            embed.bits,
                        )
                    return det_as_linear

                # Monkey-patch the as_linear method
                import types
                child.as_linear = types.MethodType(lambda self, x, fn=make_det_as_linear(child): fn(x), child)
                if verbose:
                    print(f"Patched {full_name}.as_linear with deterministic version")
            else:
                # Recurse into children
                replace_in_module(child, full_name)

    replace_in_module(model)
    print(f"Replaced {replaced_count} QuantizedLinear layers with DeterministicQuantizedLinear")
    return model
