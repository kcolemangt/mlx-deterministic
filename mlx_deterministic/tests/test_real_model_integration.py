#!/usr/bin/env python3
"""
Real Model Integration Tests

Tests batch invariance on actual MLX-LM models with real tokenized inputs.
These tests expose issues that synthetic random data tests miss.

The existing tests pass because they:
1. Use random synthetic data (mx.random.normal(...))
2. Test isolated operations (single RMSNorm, single matmul)
3. Never test through multiple stacked layers with real input patterns

The bug only manifests when:
1. Using real tokenized text (specific embedding patterns)
2. Passing through 36+ transformer layers where ~2e-5 differences amplify
3. Comparing outputs at different batch sizes
"""
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm import load
from mlx_deterministic.ops import BatchInvariantRMSNorm


# =============================================================================
# Test Configuration
# =============================================================================

MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Prompts known to FAIL with BatchInvariantRMSNorm (variance gets WORSE)
FAILING_PROMPTS = [
    "What is the capital of France?",  # README example!
    "Nice to meet you.",
    "Please help me.",
    "What color is the sky?",
]

# Prompts that happen to pass (for comparison)
PASSING_PROMPTS = [
    "Hello",
    "Hi",
    "Once upon a time",
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_and_tokenizer():
    """Load the test model."""
    return load(MODEL_NAME)


def measure_batch_variance(model, tokenizer, prompt):
    """
    Measure max logit difference between batch_size=1 and batch_size=4.

    This is the core metric: if batch-invariant, this should be 0.
    """
    tokens = tokenizer.encode(prompt)
    x1 = mx.array([tokens])
    x4 = mx.array([tokens] * 4)

    out1 = model(x1)
    out4 = model(x4)

    logits1 = out1[0, -1, :]
    logits4 = out4[0, -1, :]
    mx.eval(logits1, logits4)

    return mx.max(mx.abs(logits1 - logits4)).item()


def enable_deterministic_mode_fixed(model):
    """
    Fixed traversal that properly replaces all RMSNorm modules.

    Note: The library's enable_deterministic_mode() uses dir() which
    fails to traverse into list attributes like model.layers.
    This version uses children() and handles lists correctly.
    """
    replaced = 0
    visited = set()
    actual_model = model.model if hasattr(model, 'model') else model

    def replace_in_module(module):
        nonlocal replaced
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
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        replace_in_module(item)
                continue
            if not isinstance(child, nn.Module):
                continue
            if isinstance(child, nn.RMSNorm) and not isinstance(child, BatchInvariantRMSNorm):
                new_norm = BatchInvariantRMSNorm(
                    dims=child.weight.shape[0],
                    eps=child.eps,
                    chunk_size=64
                )
                new_norm.weight = child.weight
                setattr(module, name, new_norm)
                replaced += 1
            else:
                replace_in_module(child)

    replace_in_module(actual_model)
    return replaced


# =============================================================================
# Test: Baseline Behavior
# =============================================================================

class TestBaselineBehavior:
    """Establish baseline behavior of unmodified model."""

    def test_baseline_has_small_variance(self):
        """
        Baseline (unmodified) model should have small batch variance.
        This establishes the baseline we're trying to improve upon.
        """
        model, tokenizer = get_model_and_tokenizer()

        for prompt in FAILING_PROMPTS:
            variance = measure_batch_variance(model, tokenizer, prompt)
            # Baseline variance should be small (typically 0.03-0.07)
            assert variance < 0.1, (
                f"Baseline variance for '{prompt}' is {variance}, expected < 0.1"
            )


# =============================================================================
# Test: BatchInvariantRMSNorm Regression
# =============================================================================

class TestBatchInvariantRMSNormRegression:
    """
    Tests that demonstrate BatchInvariantRMSNorm makes things WORSE.

    These tests are expected to FAIL, demonstrating the bug.
    """

    @pytest.mark.xfail(reason="Known limitation: BatchInvariantRMSNorm chunked variance differs from standard computation")
    def test_batch_invariant_rmsnorm_should_not_increase_variance(self):
        """
        After applying BatchInvariantRMSNorm, variance should decrease or stay same.

        THIS TEST FAILS - demonstrating the bug.
        """
        # Get baseline measurements
        model, tokenizer = get_model_and_tokenizer()
        baseline_variances = {}
        for prompt in FAILING_PROMPTS:
            baseline_variances[prompt] = measure_batch_variance(model, tokenizer, prompt)

        # Reload and apply BatchInvariantRMSNorm
        model, tokenizer = get_model_and_tokenizer()
        replaced = enable_deterministic_mode_fixed(model)
        assert replaced == 73, f"Expected 73 modules replaced, got {replaced}"

        # Check that variance didn't get worse
        for prompt in FAILING_PROMPTS:
            new_variance = measure_batch_variance(model, tokenizer, prompt)
            baseline = baseline_variances[prompt]

            assert new_variance <= baseline, (
                f"'{prompt}': BatchInvariantRMSNorm made variance WORSE!\n"
                f"  Baseline: {baseline:.4f}\n"
                f"  After:    {new_variance:.4f}\n"
                f"  Change:   {new_variance - baseline:+.4f}"
            )

    @pytest.mark.xfail(reason="Known limitation: BatchInvariantRMSNorm chunked variance differs from standard computation")
    def test_batch_invariant_rmsnorm_achieves_zero_variance(self):
        """
        BatchInvariantRMSNorm should achieve zero (or near-zero) batch variance.

        THIS TEST FAILS - demonstrating the bug.
        """
        model, tokenizer = get_model_and_tokenizer()
        replaced = enable_deterministic_mode_fixed(model)
        assert replaced == 73, f"Expected 73 modules replaced, got {replaced}"

        for prompt in FAILING_PROMPTS:
            variance = measure_batch_variance(model, tokenizer, prompt)
            assert variance < 1e-5, (
                f"'{prompt}': Expected ~0 variance, got {variance:.4f}"
            )

    @pytest.mark.xfail(reason="Known limitation: BatchInvariantRMSNorm chunked variance differs from standard computation")
    def test_readme_example_prompt(self):
        """
        The README uses "What is the capital of France?" as an example.
        This prompt should work if the library works as advertised.

        THIS TEST FAILS - demonstrating that the README example doesn't work.
        """
        model, tokenizer = get_model_and_tokenizer()

        # Get baseline
        baseline = measure_batch_variance(model, tokenizer, "What is the capital of France?")

        # Apply fix and measure again
        model, tokenizer = get_model_and_tokenizer()
        enable_deterministic_mode_fixed(model)
        after_fix = measure_batch_variance(model, tokenizer, "What is the capital of France?")

        # Should be better, not worse
        assert after_fix <= baseline, (
            f"README example 'What is the capital of France?' got WORSE!\n"
            f"  Baseline: {baseline:.4f}\n"
            f"  After:    {after_fix:.4f}"
        )

        # Should be near zero
        assert after_fix < 1e-5, (
            f"README example should have ~0 variance, got {after_fix:.4f}"
        )


# =============================================================================
# Test: Numerical Equivalence
# =============================================================================

class TestRMSNormNumericalEquivalence:
    """Tests to understand the numerical differences."""

    def test_single_layer_difference_with_real_embeddings(self):
        """
        Measure how much BatchInvariantRMSNorm differs from nn.RMSNorm
        on a single layer with real token embeddings.
        """
        model, tokenizer = get_model_and_tokenizer()
        actual_model = model.model if hasattr(model, 'model') else model

        prompt = "What is the capital of France?"
        tokens = tokenizer.encode(prompt)
        x = mx.array([tokens])

        # Get real embeddings
        embeddings = actual_model.embed_tokens(x)
        mx.eval(embeddings)

        # Compare outputs
        dims = embeddings.shape[-1]
        standard_norm = nn.RMSNorm(dims)
        batch_inv_norm = BatchInvariantRMSNorm(dims=dims, eps=1e-6, chunk_size=64)
        batch_inv_norm.weight = standard_norm.weight

        out_std = standard_norm(embeddings)
        out_bi = batch_inv_norm(embeddings)
        mx.eval(out_std, out_bi)

        diff = mx.max(mx.abs(out_std - out_bi)).item()

        # Document the single-layer difference
        # This difference (~0.04) amplifies through 36 layers
        # Note: The chunked variance computation produces different results
        # than standard mean(x^2), which causes this divergence
        print(f"\nSingle layer BatchInvariantRMSNorm vs nn.RMSNorm diff: {diff}")

        # This test documents that BatchInvariantRMSNorm produces noticeably
        # different outputs than standard RMSNorm, even on a single layer
        assert diff < 0.1, f"Single layer diff {diff} unexpectedly large"


# =============================================================================
# Test: Library's Original enable_deterministic_mode
# =============================================================================

class TestOriginalTraversal:
    """Tests for the library's original enable_deterministic_mode function."""

    def test_original_enable_deterministic_mode_replaces_all_modules(self):
        """
        The library's enable_deterministic_mode() should replace all RMSNorm modules.

        Previous bugs (now fixed):
        1. Used dir() which caused infinite recursion on some models
        2. Failed to traverse into list attributes like model.layers

        This test verifies all modules are correctly replaced.
        """
        from mlx_deterministic import enable_deterministic_mode
        import sys

        model, tokenizer = get_model_and_tokenizer()

        # Count total RMSNorm modules using proper traversal
        def count_rmsnorm(module, visited=None):
            if visited is None:
                visited = set()
            if id(module) in visited:
                return 0
            visited.add(id(module))

            count = 1 if isinstance(module, nn.RMSNorm) else 0

            if hasattr(module, 'children'):
                for name in module.children():
                    child = getattr(module, name, None)
                    if isinstance(child, list):
                        for item in child:
                            if isinstance(item, nn.Module):
                                count += count_rmsnorm(item, visited)
                    elif isinstance(child, nn.Module):
                        count += count_rmsnorm(child, visited)
            return count

        actual_model = model.model if hasattr(model, 'model') else model
        total_rmsnorm = count_rmsnorm(actual_model)

        # The original enable_deterministic_mode either:
        # 1. Causes infinite recursion (RecursionError)
        # 2. Replaces 0 modules due to traversal bug
        try:
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(500)  # Lower limit to catch recursion faster

            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                enable_deterministic_mode(model, verbose=True)
                output = sys.stdout.getvalue()
                replaced_count = output.count("Replaced")
            finally:
                sys.stdout = old_stdout
                sys.setrecursionlimit(old_limit)

            # Verify all modules were replaced (bug is now fixed)
            assert replaced_count == total_rmsnorm, (
                f"enable_deterministic_mode() should replace all {total_rmsnorm} modules, "
                f"but only replaced {replaced_count}"
            )

        except RecursionError:
            pytest.fail(
                "enable_deterministic_mode() caused infinite recursion - "
                "the traversal logic needs to handle circular references"
            )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
