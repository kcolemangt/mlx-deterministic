"""
Integration tests for deterministic inference
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_deterministic import (
    enable_deterministic_mode,
    DeterministicConfig,
    BatchInvariantRMSNorm,
    BatchInvariantAttention
)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""

    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiHeadAttention(dims, num_heads)
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)
        self.mlp = nn.Linear(dims, dims)

    def __call__(self, x):
        # Self-attention with residual
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x


def test_enable_deterministic_mode_replaces_modules():
    """
    Test that enable_deterministic_mode replaces modules correctly.
    """
    dims = 128
    num_heads = 4

    # Create simple model
    model = SimpleTransformerBlock(dims, num_heads)

    # Enable deterministic mode
    config = DeterministicConfig(
        rms_norm_chunk_size=64,
        softmax_chunk_size=64,
        attention_matmul_tile_size=64
    )
    model = enable_deterministic_mode(model, config)

    # Check that modules were replaced
    assert isinstance(model.norm1, BatchInvariantRMSNorm), \
        "norm1 should be BatchInvariantRMSNorm"
    assert isinstance(model.norm2, BatchInvariantRMSNorm), \
        "norm2 should be BatchInvariantRMSNorm"
    assert isinstance(model.attention, BatchInvariantAttention), \
        "attention should be BatchInvariantAttention"


def test_deterministic_model_batch_invariance():
    """
    Test that a model with deterministic mode enabled produces
    batch-invariant results.
    """
    dims = 128
    num_heads = 4
    seq_len = 16

    # Create and convert model
    mx.random.seed(42)
    model = SimpleTransformerBlock(dims, num_heads)
    model = enable_deterministic_mode(model)

    # Create input
    mx.random.seed(123)
    x_batch = mx.random.normal((8, seq_len, dims))

    # Single vs batch
    output_single = model(x_batch[0:1])
    output_batch = model(x_batch)

    diff = mx.abs(output_single - output_batch[0:1])
    max_diff = mx.max(diff).item()

    # Allow small FP tolerance
    assert max_diff < 1e-3, f"Model batch invariance violated by {max_diff}"


def test_deterministic_model_consistency():
    """
    Test that deterministic model produces consistent results across runs.
    """
    dims = 128
    num_heads = 4
    seq_len = 16

    # Create model
    mx.random.seed(42)
    model = SimpleTransformerBlock(dims, num_heads)
    model = enable_deterministic_mode(model)

    # Create input
    mx.random.seed(123)
    x = mx.random.normal((4, seq_len, dims))

    # Run multiple times
    outputs = []
    for _ in range(3):
        output = model(x)
        outputs.append(output)

    # All outputs should be identical
    for i in range(1, len(outputs)):
        diff = mx.abs(outputs[0] - outputs[i])
        max_diff = mx.max(diff).item()
        assert max_diff == 0.0, f"Run {i}: outputs differ by {max_diff}"


def test_config_parameters():
    """
    Test that DeterministicConfig parameters are applied correctly.
    """
    config = DeterministicConfig(
        rms_norm_chunk_size=32,
        matmul_tile_size=64,
        softmax_chunk_size=128,
        attention_matmul_tile_size=256
    )

    assert config.rms_norm_chunk_size == 32
    assert config.matmul_tile_size == 64
    assert config.softmax_chunk_size == 128
    assert config.attention_matmul_tile_size == 256


def test_deterministic_forward_pass():
    """
    Test a complete forward pass through a deterministic model.
    """
    dims = 256
    num_heads = 8
    seq_len = 32
    batch_size = 4

    # Create model
    mx.random.seed(42)
    model = SimpleTransformerBlock(dims, num_heads)
    model = enable_deterministic_mode(model)

    # Create input
    mx.random.seed(123)
    x = mx.random.normal((batch_size, seq_len, dims))

    # Forward pass
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, dims), \
        f"Output shape {output.shape} doesn't match expected"

    # Check output is not NaN or Inf
    assert not mx.isnan(output).any().item(), "Output contains NaN"
    assert not mx.isinf(output).any().item(), "Output contains Inf"


if __name__ == "__main__":
    test_enable_deterministic_mode_replaces_modules()
    print("✓ test_enable_deterministic_mode_replaces_modules passed")

    test_deterministic_model_batch_invariance()
    print("✓ test_deterministic_model_batch_invariance passed")

    test_deterministic_model_consistency()
    print("✓ test_deterministic_model_consistency passed")

    test_config_parameters()
    print("✓ test_config_parameters passed")

    test_deterministic_forward_pass()
    print("✓ test_deterministic_forward_pass passed")

    print("\nAll integration tests passed!")
