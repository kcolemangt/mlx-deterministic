# MLX Batch Determinism: Historical Context and Current Status

This document provides context on MLX's batch invariance behavior and why this library exists.

## Background

Standard LLM inference can produce **different outputs** for the same prompt when processed with different batch sizes, even with `temperature=0`. This non-determinism stems from floating-point operations using different reduction patterns depending on batch size and GPU thread scheduling.

## MLX Version History

### Pre-0.24.0: Known Batch Variance Issues

MLX had documented batch-dependent matrix multiplication issues:

- **Issue [#1958](https://github.com/ml-explore/mlx/issues/1958)** (March 2025): Matrix multiplication with batched arrays produced different results when using non-FP32 data types (bfloat16, float16).
- Root cause: `gemv` and `gemm` operations used different accumulation precision levels.

### 0.24.0+: Partial Fix (March 2025)

- **[PR #1962](https://github.com/ml-explore/mlx/pull/1962)**: "Use same accumulation precision in gemv as gemm"
- This resolved the major source of batch variance for many operations.

### Current Behavior (0.30.0)

Testing shows mixed results depending on model architecture:

| Model | Non-deterministic Mode | Notes |
|-------|----------------------|-------|
| Qwen2.5-3B-4bit | Deterministic | No variance detected |
| Qwen2.5-7B-4bit | Deterministic | No variance detected |
| Qwen3-0.6B-4bit | **Non-deterministic** | Max diff ~0.75 |
| Qwen3-4B-4bit | Deterministic | No variance detected |

**Key insight**: The Qwen3-0.6B model exhibits significant batch variance (~0.75 logit difference) while larger models may not. This suggests variance behavior depends on:
- Model architecture (attention patterns, layer dimensions)
- Quantization implementation details
- How operations align with MLX's internal tiling patterns

## Why This Library Still Matters

Even with MLX 0.24+ improvements:

1. **Not all models are deterministic** - As shown above, Qwen3-0.6B still has variance
2. **Guarantees across MLX versions** - Ensures consistent behavior regardless of MLX version
3. **Explicit determinism** - Production systems benefit from explicit guarantees rather than relying on implicit behavior
4. **Testing and validation** - The `--no-determinism` flag helps verify when determinism matters

## References

- [The Hidden Problem With MLX: Why Your Apple Silicon LLM Isn't Reproducible](https://adityakarnam.com/mlx-non-determinism-apple-silicon/)
- [MLX Issue #1958: Matrix Multiplication Bug](https://github.com/ml-explore/mlx/issues/1958)
- [MLX PR #1962: Fix accumulation precision](https://github.com/ml-explore/mlx/pull/1962)
- [MLX Releases](https://github.com/ml-explore/mlx/releases)

## Testing Determinism

Use the `determinism_check.py` script to verify behavior:

```bash
# Test with deterministic Metal kernels (recommended)
python examples/determinism_check.py --metal --verbose --quick

# Test without deterministic modifications (baseline)
python examples/determinism_check.py --no-determinism --verbose --quick

# Test a specific model
python examples/determinism_check.py --metal --model mlx-community/Qwen3-0.6B-4bit

# Full test with all batch sizes
python examples/determinism_check.py --metal --verbose --batch-sizes "1,2,4,8,16,32"
```
