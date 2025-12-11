# MLX Deterministic Inference

[![Tests](https://img.shields.io/badge/tests-87%2F90%20passing-brightgreen)](mlx_deterministic/tests/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.29%2B-orange)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Batch-invariant operations for deterministic LLM inference on Apple Silicon using MLX.**

Eliminates nondeterminism in LLM inference by ensuring identical outputs regardless of batch size, based on research from [Thinking Machines Labs](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

## 🎯 The Problem

Standard LLM inference produces **different outputs** for the same prompt when processed with different batch sizes, even with `temperature=0`:

```python
# Same prompt, different batch sizes
prompt = "What is the capital of France?"

output_batch_1 = generate(model, prompt, batch_size=1)
# "The capital of France is Paris, which is..."

output_batch_8 = generate(model, prompt, batch_size=8)[0]
# "The capital of France is Paris. It is..."  ❌ Different!
```

This nondeterminism breaks:
- Testing and validation
- Reproducible benchmarks
- Debugging
- Compliance and auditing

## ✨ The Solution

This library provides **batch-invariant** implementations of core operations:

- ✅ **RMSNorm** - Atomic per-sample reduction
- ✅ **Matrix Multiplication** - Fixed K-dimension tiling for batch invariance
- ✅ **Attention** - FlashAttention with fixed split-size
- ✅ **Softmax** - Fixed tree reduction
- ✅ **Custom Metal Kernels** - Bitwise-identical determinism
- ✅ **FP16 Support** - Half-precision Metal kernels for memory-efficient inference (NEW!)

**Result**: Bitwise-identical outputs regardless of batch size! 🎉

### Two Modes of Operation

| Mode | Tolerance | Matmul Overhead | Use Case |
|------|-----------|-----------------|----------|
| **Python (default)** | ~1e-5 | +27-32% | Most applications |
| **Metal Kernels** | **0.0** (bitwise) | **+27-32%** | Strict reproducibility |

> **Why is the Metal kernel slower?** The Python mode wraps MLX's highly-optimized `mx.matmul()` (which uses Apple's hand-tuned GEMM kernels). The Metal kernel implements matmul from scratch to guarantee bitwise determinism - we cannot match Apple's years of optimization work, but we CAN guarantee identical results every time. See [Performance Trade-offs](#performance-trade-offs) for details.

```python
# Default mode: ~1e-5 tolerance (uses MLX's internal matmul)
from mlx_deterministic import batch_invariant_matmul
result = batch_invariant_matmul(a, b)

# Metal kernel mode: TRUE bitwise determinism (slower but exact)
result = batch_invariant_matmul(a, b, use_metal_kernel=True)

# FP16 mode: Half-precision for memory efficiency (auto-detected or explicit)
a_fp16 = a.astype(mx.float16)
b_fp16 = b.astype(mx.float16)
result = batch_invariant_matmul(a_fp16, b_fp16, use_metal_kernel=True)  # Auto-detects FP16

# Or explicitly convert to FP16
result = batch_invariant_matmul(a, b, use_metal_kernel=True, dtype=mx.float16)
```

## 📊 Validation Results

```
🎉 All determinism tests PASSED!

                    Python Wrapper     Metal Kernel (FP32)    Metal Kernel (FP16)
RMSNorm             ~1e-5 tolerance    0.0 (bitwise)          N/A
Matmul              ~1e-5 tolerance    0.0 (bitwise)          0.0 (bitwise)
Softmax             ~1e-5 tolerance    0.0 (bitwise)          N/A
Attention           <1e-4 tolerance    <1e-4 tolerance        N/A

90 tests (87 passing, 3 xfail for known limitations) across batch sizes [1, 2, 4, 8, 16, 32, 64, 128]
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/mlx-deterministic.git
cd mlx-deterministic
pip install mlx mlx-lm numpy pytest
```

### Basic Usage

```python
import mlx.core as mx
from mlx_deterministic.ops import (
    BatchInvariantRMSNorm,
    batch_invariant_matmul,
    BatchInvariantAttention
)

# RMSNorm - deterministic regardless of batch size
norm = BatchInvariantRMSNorm(dims=4096, chunk_size=64)
output = norm(x)  # ✓ Deterministic!

# Matrix Multiplication
result = batch_invariant_matmul(a, b, tile_size=128)  # ✓ Deterministic!

# Attention
attn = BatchInvariantAttention(dims=512, num_heads=8)
output = attn(queries, keys, values)  # ✓ Deterministic!
```

### API Integration Example

```python
from fastapi import FastAPI
from mlx_deterministic.ops import BatchInvariantRMSNorm

app = FastAPI()

@app.post("/generate")
async def generate(prompt: str, deterministic: bool = False):
    if deterministic:
        # Enable deterministic mode
        replace_with_deterministic_ops(model)

    return {"response": generate_text(model, prompt)}
```

See **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** for complete examples.

## 📚 Documentation

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Step-by-step integration instructions
- **[mlx_deterministic/README.md](mlx_deterministic/README.md)** - Technical documentation
- **[SUMMARY.md](SUMMARY.md)** - Implementation details and results

## 🧪 Testing

```bash
# Run all tests (87 passing, 3 xfail)
python -m pytest mlx_deterministic/tests/ -v

# Run specific test files
python -m pytest mlx_deterministic/tests/test_metal_kernels.py -v
python -m pytest mlx_deterministic/tests/test_matmul.py -v
```

### Verifying Determinism

Use the `determinism_check.py` script to verify batch-invariant inference on real models:

```bash
# Quick test with Metal kernels (recommended)
python examples/determinism_check.py --metal --quick

# Test a specific model
python examples/determinism_check.py --metal --model mlx-community/Qwen3-4B-4bit

# Full test with all batch sizes and verbose output
python examples/determinism_check.py --metal --verbose --batch-sizes "1,2,4,8,16,32"

# Test WITHOUT deterministic mode (shows baseline variance)
python examples/determinism_check.py --no-determinism --verbose --quick
```

The `--no-determinism` flag runs models without any deterministic modifications, useful for demonstrating why this library exists. Some models (like Qwen3-0.6B) show significant variance (~0.75 logit difference) without deterministic mode.

See [`examples/MLX_DETERMINISM_NOTES.md`](examples/MLX_DETERMINISM_NOTES.md) for historical context on MLX's batch determinism behavior.

## 📊 Benchmarking

The benchmark suite validates determinism and measures performance across all implementations.

### Quick Benchmark (Standard Mode)

```bash
# Fast benchmark (~60 seconds) - good for quick validation
PYTHONPATH=. python mlx_deterministic/benchmarks/benchmark_determinism.py
```

This runs determinism tests and a performance comparison, but results may vary ±10-15% between runs due to thermal throttling.

### Stable Benchmark (Extended Mode)

```bash
# Thermal-aware benchmark (~5-10 minutes) - recommended for accurate measurements
PYTHONPATH=. python mlx_deterministic/benchmarks/benchmark_determinism.py --extended
```

Extended mode provides stable, reproducible results by:
- **Interleaved testing**: Runs all implementations in rotation (not sequentially), so each experiences the same thermal conditions
- **Cooldown periods**: Sleeps between test categories to let CPU/GPU recover from thermal throttling
- **Multiple rounds**: Runs the complete benchmark multiple times and reports median results
- **Adaptive iterations**: Runs until results stabilize (CV < 5%) with minimum 5 seconds per test

### Benchmark Options

```bash
# Full options
python benchmark_determinism.py --extended [OPTIONS]

Options:
  --cooldown SECONDS   Sleep time between test categories (default: 5.0)
  --rounds N           Number of complete benchmark rounds (default: 3)

# Examples
--extended                    # Default extended settings (~10 min)
--extended --rounds 5         # More thorough (longer)
--extended --cooldown 10      # Longer cooldown for hot systems
```

### What the Benchmark Tests

| Test | What It Measures |
|------|------------------|
| **Determinism Tests** | Verifies batch-invariance (same output regardless of batch size) |
| **RMSNorm** | Normalization layer performance |
| **Matmul 512x512** | Small matrix multiplication |
| **Matmul 2048x2048 (FP32)** | Large matrix multiplication (32-bit) |
| **Matmul 2048x2048 (FP16)** | Large matrix multiplication (16-bit) |

### Interpreting Results

```
| Operation | Standard | Metal | Overhead |
|-----------|----------|-------|----------|
| Matmul 2K (FP32)  | 1.51ms | 1.91ms | +27% |
```

- **Standard**: Apple's optimized `mx.matmul()` (non-deterministic)
- **Metal**: Our custom Metal kernel (bitwise deterministic)
- **Overhead**: Performance cost for determinism guarantee

**Expected overhead ranges** (M4 Max):
- RMSNorm: +5-10%
- Matmul 512x512: +30-35%
- Matmul 2048x2048 (FP32): +27-28%
- Matmul 2048x2048 (FP16): +30-31%

Results will vary by hardware - use `--extended` mode on your system for accurate numbers.

## 📈 Performance

### Performance Trade-offs

We provide two approaches with fundamentally different trade-offs:

| Approach | How It Works | Tolerance | Large Matmul Overhead |
|----------|--------------|-----------|----------------------|
| **Python wrapper** | Wraps `mx.matmul()` with tiled reduction | ~1e-5 | +27-32% |
| **Metal kernel (FP32)** | Custom GPU kernel from scratch | **0.0 (bitwise)** | **+27-28%** |
| **Metal kernel (FP16)** | Half-precision GPU kernel | **0.0 (bitwise)** | **+30-31%** |

#### Why the overhead for Metal kernels?

The Python wrapper approach still uses MLX's `mx.matmul()` internally - Apple's highly optimized GEMM implementation that has been tuned over years. We simply split the K dimension into tiles and sum partial products in a fixed order, getting batch-invariance with minimal overhead.

The Metal kernel approach implements matrix multiplication **from scratch** using MLX's custom kernel API. This is necessary for TRUE bitwise determinism because:

1. **`mx.matmul()` has inherent variance** (~1e-5) due to non-deterministic reduction order
2. **We cannot control the internal reduction order** of Apple's GEMM kernels
3. **Bitwise determinism requires controlling every floating-point operation**

Our custom kernel uses 64x64 tiled matmul with `simdgroup_matrix` hardware-accelerated 8x8 matrix operations - leveraging Apple Silicon's tensor core equivalent. This achieves reasonable performance while maintaining bitwise determinism.

**The overhead is the price for TRUE bitwise determinism.** If ~1e-5 tolerance is acceptable for your use case, use the Python wrapper (default) for much better performance.

### Benchmark Results

| Operation | Standard MLX | Python Wrapper | Metal Kernel |
|-----------|--------------|----------------|--------------|
| RMSNorm   | 0.10ms | 0.11ms (+7%) | **0.11ms (+7%)** |
| Matmul 512x512 | 0.13ms | 0.17ms (+32%) | **0.17ms (+32%)** |
| Matmul 2048x2048 (FP32) | 1.50ms | 1.90ms (+27%) | **1.90ms (+27%)** |
| Matmul 2048x2048 (FP16) | 1.36ms | N/A | **1.78ms (+31%)** |

*Benchmarked on Apple M4 Max using `--extended` mode for thermal-aware testing. Run `PYTHONPATH=. python mlx_deterministic/benchmarks/benchmark_determinism.py --extended` for stable results.*

### Recommendations

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Strict reproducibility required** | Metal Kernel | Only option for 0.0 difference |
| **Compliance/auditing** | Metal Kernel | Bitwise-identical results guaranteed |
| **Testing/CI** | Python Wrapper | ~1e-5 tolerance usually sufficient |
| **Memory-constrained inference** | Metal Kernel (FP16) | Half the memory with bitwise determinism |
| **Maximum performance** | Python Wrapper | Lowest overhead for large matmul |
| **RMSNorm/Softmax** | Metal Kernel | ~+5-10% overhead with bitwise determinism |

### Determinism Guarantees

| Implementation | Tolerance | Batch Invariant | Uses mx.matmul? |
|----------------|-----------|-----------------|-----------------|
| Standard MLX   | N/A       | No | Yes |
| Python Wrapper | ~1e-5     | Yes | **Yes** (that's why it's fast) |
| **Metal Kernel (FP32)** | **0.0 (bitwise)** | **Yes** | **No** (custom SIMD kernel) |
| **Metal Kernel (FP16)** | **0.0 (bitwise)** | **Yes** | **No** (custom SIMD kernel) |

## 🏗️ Architecture

### How It Works

Standard operations use different reduction patterns based on batch size:
- Small batches: Different SIMD lane usage
- Large batches: Different thread configurations
- Result: Different floating-point accumulation order → Different outputs

**Our solution**: Fixed reduction patterns independent of batch size:

1. **Fixed Chunk Sizes**: Always use same chunk size regardless of input
2. **Explicit Padding**: Pad to multiples of chunk size
3. **Sequential Processing**: Process chunks in deterministic order
4. **Fixed Reduction Trees**: Same reduction structure every time

## 🔬 Technical Details

### Batch-Invariant RMSNorm

```python
# Standard: variance computation varies with batch size
variance = mean(x**2)  # Different reduction patterns

# Ours: fixed-chunk variance
x_squared = x * x
chunks = reshape(pad(x_squared), [..., num_chunks, chunk_size])
chunk_means = mean(chunks, axis=-1)
variance = mean(chunk_means)  # Fixed reduction!
```

### Batch-Invariant Matmul

```python
# Standard: C = A @ B varies with batch size
C = A @ B  # Different BLAS routines

# Ours: fixed-tile reduction
tiles = split(B, tile_size)
C = sum([A @ tile for tile in tiles])  # Fixed order!
```

## 📝 Citation

Based on research from Thinking Machines Labs:

```bibtex
@misc{tml2024deterministic,
  title={Defeating Nondeterminism in LLM Inference},
  author={Thinking Machines Labs},
  year={2024},
  url={https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/}
}
```

Also see [SGLang's deterministic implementation](https://lmsys.org/blog/2025-09-22-sglang-deterministic/).

## 🤝 Contributing

Contributions welcome! This is research code demonstrating batch-invariant operations for MLX.

Areas for contribution:
- [ ] Automatic model conversion
- [ ] Extended model support (Llama, Mistral, etc.)
- [ ] KV cache determinism
- [ ] Quantized operations (INT4/INT8 with FP16 compute)
- [ ] FP16 kernels for RMSNorm and Softmax
- [ ] Performance optimizations

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[Thinking Machines Labs](https://thinkingmachines.ai/)** - Original research and PyTorch implementation
- **[SGLang Team](https://github.com/sgl-project/sglang)** - Production validation
- **[Apple MLX Team](https://github.com/ml-explore/mlx)** - Excellent framework

## 🔗 Links

- **Research**: [TML Blog Post](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **Reference**: [TML GitHub](https://github.com/thinking-machines-lab/batch_invariant_ops)
- **SGLang**: [Deterministic Inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- **MLX**: [ml-explore/mlx](https://github.com/ml-explore/mlx)

---

**Made with ❤️ for reproducible AI on Apple Silicon**
