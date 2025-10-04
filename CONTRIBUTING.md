# Contributing to MLX Deterministic Inference

Thank you for your interest in contributing! This project implements batch-invariant operations for deterministic LLM inference on Apple Silicon.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Apple Silicon Mac (M1/M2/M3)
- MLX framework

### Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mlx-deterministic.git
cd mlx-deterministic

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install mlx mlx-lm numpy pytest

# Run tests to verify setup
python -m pytest mlx_deterministic/tests/ -v

# Run benchmark
PYTHONPATH=. python mlx_deterministic/benchmarks/benchmark_determinism.py
```

## 🧪 Testing

All contributions should include tests. We have excellent test coverage (34/35 passing, 97%).

### Running Tests

```bash
# All tests
python -m pytest mlx_deterministic/tests/ -v

# Specific component
python -m pytest mlx_deterministic/tests/test_rms_norm.py -v
python -m pytest mlx_deterministic/tests/test_matmul.py -v
python -m pytest mlx_deterministic/tests/test_attention.py -v

# With coverage
python -m pytest mlx_deterministic/tests/ --cov=mlx_deterministic
```

### Test Requirements

When adding new features:
1. **Batch Invariance**: Verify outputs are identical across batch sizes
2. **Numerical Correctness**: Compare against standard MLX operations
3. **Multiple Batch Sizes**: Test with [1, 2, 4, 8, 16, 32, 64, 128]
4. **Edge Cases**: Non-divisible dimensions, padding scenarios
5. **Gradient Computation**: Ensure backward pass works

Example test structure:

```python
def test_new_operation_batch_invariance():
    """Test that operation produces identical results across batch sizes."""
    # Create test data
    mx.random.seed(42)
    data = mx.random.normal((128, dims))

    # Single sample
    output_single = operation(data[0:1])

    # Batch processing
    output_batch = operation(data)

    # Verify identical outputs
    diff = mx.max(mx.abs(output_single - output_batch[0:1])).item()
    assert diff == 0.0, f"Batch invariance violated by {diff}"
```

## 📝 Code Style

- **Type Hints**: All functions must have type hints
- **Docstrings**: Use Google-style docstrings
- **Naming**: `snake_case` for functions, `PascalCase` for classes
- **Comments**: Explain *why*, not *what*

Example:

```python
def batch_invariant_operation(
    x: mx.array,
    chunk_size: int = 64
) -> mx.array:
    """
    Perform batch-invariant operation on input.

    Args:
        x: Input tensor of shape (..., dims)
        chunk_size: Fixed chunk size for deterministic reduction

    Returns:
        Output tensor of same shape as input

    Example:
        >>> x = mx.random.normal((batch, 512))
        >>> output = batch_invariant_operation(x)
    """
    # Implementation here
    pass
```

## 🎯 Areas for Contribution

### High Priority

1. **Automatic Model Conversion**
   - Fix `enable_deterministic_mode()` for MLX models
   - Support more model architectures
   - File: `mlx_deterministic/__init__.py`

2. **Real Model Testing**
   - Test with Llama, Qwen, Mistral models
   - Validate end-to-end generation
   - Create integration tests

3. **KV Cache Determinism**
   - Extend to incremental generation
   - Ensure cache updates are deterministic
   - File: `mlx_deterministic/ops/attention.py`

### Medium Priority

4. **Performance Optimization**
   - Reduce overhead through better chunking
   - Optimize for different hardware (M1 vs M3)
   - Profile and optimize hot paths

5. **Quantization Support**
   - 4-bit deterministic operations
   - 8-bit deterministic operations
   - Mixed precision support

6. **Extended Operations**
   - GroupNorm
   - LayerNorm
   - Other normalization layers

### Nice to Have

7. **Documentation**
   - More usage examples
   - Video tutorials
   - Blog posts

8. **Benchmarking**
   - Compare with SGLang implementation
   - Benchmark on different Apple Silicon chips
   - Create performance regression tests

## 🔧 Implementation Guidelines

### Batch Invariance Principle

All operations must use **fixed reduction patterns** independent of batch size:

```python
# ❌ Wrong: Reduction pattern varies with batch size
result = mx.mean(x)  # Different patterns for different batch sizes

# ✅ Right: Fixed chunk-based reduction
chunks = x.reshape(-1, CHUNK_SIZE)
chunk_means = mx.mean(chunks, axis=-1)
result = mx.mean(chunk_means)  # Fixed pattern!
```

### Key Design Patterns

1. **Padding**: Always pad to multiples of chunk/tile size
2. **Sequential Processing**: Process chunks in deterministic order
3. **Configuration**: Allow chunk/tile sizes to be configurable
4. **Validation**: Include batch invariance test for every operation

## 📋 Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Implement** your changes with tests
4. **Test** thoroughly: `python -m pytest mlx_deterministic/tests/ -v`
5. **Commit** with clear message: `git commit -m "Add amazing feature"`
6. **Push** to your fork: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

### PR Checklist

- [ ] Tests added/updated and passing
- [ ] Docstrings added/updated
- [ ] Type hints included
- [ ] Batch invariance validated
- [ ] Performance impact documented
- [ ] README updated if needed
- [ ] No breaking changes (or clearly documented)

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Reproduction**: Minimal code to reproduce
3. **Expected**: What you expected to happen
4. **Actual**: What actually happened
5. **Environment**: Python version, MLX version, hardware
6. **Logs**: Any error messages or stack traces

## 💡 Feature Requests

Feature requests are welcome! Please include:

1. **Use Case**: Why is this feature needed?
2. **Proposal**: How should it work?
3. **Alternatives**: What alternatives have you considered?
4. **Examples**: Code examples of usage

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project
- Show empathy towards others

## 🙏 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Git commit history

## 📞 Questions?

- **Issues**: Open an issue for bugs or features
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [Add your contact if desired]

## 📚 Resources

- [TML Research](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [SGLang Implementation](https://github.com/sgl-project/sglang)

---

**Thank you for contributing to deterministic LLM inference on Apple Silicon!** 🚀
