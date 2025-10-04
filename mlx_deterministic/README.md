# MLX Deterministic Inference

Batch-invariant operations for **deterministic LLM inference** on Apple Silicon using MLX.

Based on research from [Thinking Machines Labs](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) and implementations in [SGLang](https://lmsys.org/blog/2025-09-22-sglang-deterministic/).

## 🎯 Problem

Standard LLM inference implementations produce **different outputs** when the same prompt is processed with different batch sizes, even with `temperature=0`. This nondeterminism stems from floating-point operations using different reduction patterns depending on batch size.

## ✨ Solution

This library provides **batch-invariant** implementations of core operations:
- **RMSNorm**: Fixed-chunk variance computation
- **Matrix Multiplication**: Fixed-tile reduction
- **Attention**: Deterministic softmax and attention scores
- **Softmax**: Fixed-chunk max/sum reductions

All operations produce **bitwise-identical results** regardless of batch size.

## 📦 Installation

```bash
pip install mlx mlx-lm numpy pytest
```

## 🚀 Quick Start

### Using Batch-Invariant Operations

```python
import mlx.core as mx
from mlx_deterministic.ops import (
    BatchInvariantRMSNorm,
    batch_invariant_matmul,
    BatchInvariantAttention
)

# RMSNorm
norm = BatchInvariantRMSNorm(dims=512, chunk_size=64)
x = mx.random.normal((batch_size, 512))
output = norm(x)  # Deterministic regardless of batch_size

# Matrix Multiplication
a = mx.random.normal((batch_size, 512, 256))
b = mx.random.normal((256, 512))
result = batch_invariant_matmul(a, b, tile_size=128)

# Attention
attn = BatchInvariantAttention(dims=512, num_heads=8)
output = attn(queries, keys, values)
```

### Configuration

```python
from mlx_deterministic import DeterministicConfig

config = DeterministicConfig(
    rms_norm_chunk_size=64,      # RMSNorm chunk size
    matmul_tile_size=128,          # Matmul tile size
    softmax_chunk_size=128,        # Softmax reduction chunk size
    attention_matmul_tile_size=128 # Attention matmul tile size
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
python -m pytest mlx_deterministic/tests/ -v

# Individual components
python -m pytest mlx_deterministic/tests/test_rms_norm.py -v
python -m pytest mlx_deterministic/tests/test_matmul.py -v
python -m pytest mlx_deterministic/tests/test_attention.py -v
```

## 📊 Benchmarks

Run the determinism validation benchmark:

```bash
PYTHONPATH=. python mlx_deterministic/benchmarks/benchmark_determinism.py
```

Example output:
```
🎉 All determinism tests PASSED!
All operations produce identical outputs regardless of batch size.

RMSNorm              ✓ PASS (50/50 runs identical)
Matmul               ✓ PASS (50/50 runs identical)
Attention            ✓ PASS (50/50 runs identical)

Performance:
  RMSNorm overhead:  ~53%
  Matmul overhead:   ~35%
```

## 🏗️ Architecture

### Batch-Invariant RMSNorm

RMSNorm computes: `output = x / sqrt(mean(x²) + ε) * weight`

Standard implementation varies based on batch size due to different reduction patterns when computing `mean(x²)`.

**Solution**: Use fixed-size chunks for reduction:
1. Pad feature dimension to multiple of `chunk_size`
2. Reshape into chunks: `(batch, features) -> (batch, num_chunks, chunk_size)`
3. Compute mean within each chunk
4. Average chunk means (fixed pattern regardless of batch size)

### Batch-Invariant Matrix Multiplication

Standard matmul: `C = A @ B` where `A: [M, K]`, `B: [K, N]`

Different batch sizes can trigger different BLAS routines with varying reduction orders.

**Solution**: Fixed-tile reduction over K dimension:
1. Pad K dimension to multiple of `tile_size`
2. Split into fixed-size tiles: `K = tile_0 + tile_1 + ... + tile_n`
3. Compute partial products per tile
4. Sum tiles in fixed order

### Batch-Invariant Attention

Attention softmax: `softmax(QK^T / sqrt(d))` has two reduction points:
- Max reduction for numerical stability
- Sum reduction for normalization

**Solution**: Chunked reductions for both operations:
1. Pad sequence dimension to `chunk_size`
2. Compute chunk-wise max, then global max
3. Compute exp(x - max)
4. Compute chunk-wise sum, then global sum
5. Divide by global sum

## 📈 Performance

Performance overhead compared to standard MLX operations (M2 Ultra):

| Operation | Overhead | Notes |
|-----------|----------|-------|
| RMSNorm   | ~50-60% | Fixed chunk processing |
| Matmul    | ~30-40% | Fixed tile reduction |
| Attention | ~40-50% | Chunked softmax |

**Key Insight**: The overhead is the cost of determinism. For applications requiring reproducibility (testing, debugging, compliance), this is acceptable.

## 🔬 Technical Details

### Why Batch Size Affects Results

Modern accelerators use different optimization strategies based on input size:
- Small batches: Different SIMD lane usage
- Large batches: Different thread block configurations
- Varying batch sizes: Different reduction tree structures

Floating-point addition is **not associative**: `(a + b) + c ≠ a + (b + c)` in FP32/FP16.

### How We Achieve Determinism

1. **Fixed Reduction Patterns**: Always use same reduction tree structure
2. **Fixed Chunk Sizes**: Independent of batch size
3. **Explicit Padding**: Make all dimensions compatible with chunk size
4. **Sequential Processing**: Process tiles/chunks in deterministic order

### Validation

All operations tested for:
- **Batch Invariance**: Single vs batched processing produces identical results
- **Numerical Correctness**: Results within tolerance of standard implementations
- **Multiple Batch Sizes**: [1, 2, 4, 8, 16, 32, 64, 128]
- **Gradient Computation**: Backward pass works correctly

## 📚 References

- [Thinking Machines Labs: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [TML batch_invariant_ops GitHub](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [SGLang Deterministic Inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)

## 🧑‍💻 Implementation Status

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| RMSNorm | ✅ Complete | 9/9 passing | Chunk-based variance |
| Matmul | ✅ Complete | 11/11 passing | Tile-based reduction |
| Attention | ✅ Complete | 10/10 passing | Chunked softmax |
| Softmax | ✅ Complete | 3/3 passing | Fixed reductions |
| Integration | 🚧 Partial | 4/5 passing | Manual usage works |

## 🛠️ Development

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install mlx mlx-lm numpy pytest

# Run tests
python -m pytest mlx_deterministic/tests/ -v

# Run benchmarks
PYTHONPATH=. python mlx_deterministic/benchmarks/benchmark_determinism.py
```

## 📄 License

This implementation is based on open research from Thinking Machines Labs and SGLang project.

## 🙏 Acknowledgments

- **Thinking Machines Labs** for the original research and PyTorch implementation
- **SGLang team** for production validation and additional implementations
- **Apple MLX team** for the excellent MLX framework

## 🔮 Future Work

- [ ] Integration with MLX-LM for drop-in model replacement
- [ ] Support for more attention variants (FlashAttention, xFormers)
- [ ] Quantized operations (4-bit, 8-bit)
- [ ] Automatic model conversion utilities
- [ ] Extended benchmarks with real models (Llama, Qwen, Mistral)
- [ ] KV cache determinism for multi-turn conversations

## 📧 Contributing

This is research code demonstrating batch-invariant operations for MLX. Contributions welcome!
