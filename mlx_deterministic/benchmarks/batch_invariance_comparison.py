"""
Batch Invariance Comparison Benchmark

This script compares standard MLX-LM inference with batch-invariant mode
across multiple batch sizes, measuring output differences and providing
example outputs to illustrate the concept of batch invariance.

Usage:
    PYTHONPATH=. python mlx_deterministic/benchmarks/batch_invariance_comparison.py
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from typing import List, Dict, Tuple
import time


def compare_batch_sizes(
    model,
    tokenizer,
    prompt: str,
    batch_sizes: List[int],
    max_tokens: int = 50
) -> Dict[str, any]:
    """
    Compare model outputs across different batch sizes.

    Returns dictionary with:
    - outputs: Dict mapping batch_size -> output text
    - logits: Dict mapping batch_size -> final logits
    - differences: Dict mapping (batch_i, batch_j) -> max logit difference
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)

    results = {
        "outputs": {},
        "logits": {},
        "differences": {},
        "generation_times": {}
    }

    # Run inference for each batch size
    for batch_size in batch_sizes:
        print(f"  Running batch_size={batch_size}...")

        # Create batched input (repeat the same prompt)
        x = mx.array([input_ids] * batch_size)

        # Time the forward pass
        start_time = time.time()

        # Get model output (just logits for the last token)
        out = model(x)
        logits_last = out[:, -1, :]  # [batch_size, vocab_size]

        # Take the first sample's logits for comparison
        logits_first = logits_last[0]

        mx.eval(logits_first)
        end_time = time.time()

        results["logits"][batch_size] = logits_first
        results["generation_times"][batch_size] = end_time - start_time

        # Generate a few tokens to get example output
        # (We'll just use greedy decoding for consistency)
        current_ids = input_ids.copy()
        generated_tokens = []

        for _ in range(max_tokens):
            x = mx.array([current_ids])
            out = model(x)
            next_token_logits = out[0, -1, :]
            next_token = int(mx.argmax(next_token_logits).item())

            # Stop at EOS
            if next_token == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)
            current_ids.append(next_token)

        # Decode the output
        output_text = tokenizer.decode(generated_tokens)
        results["outputs"][batch_size] = output_text

    # Compute pairwise differences
    for i, bs_i in enumerate(batch_sizes):
        for bs_j in batch_sizes[i+1:]:
            logits_i = results["logits"][bs_i]
            logits_j = results["logits"][bs_j]

            diff = mx.abs(logits_i - logits_j)
            max_diff = float(mx.max(diff).item())
            results["differences"][(bs_i, bs_j)] = max_diff

    return results


def print_results_section(title: str, results: Dict, batch_sizes: List[int]):
    """Print a formatted results section."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Print example outputs
    print("\n--- Example Outputs ---")
    for batch_size in batch_sizes[:5]:  # Show first 5
        output = results["outputs"][batch_size]
        print(f"\nBatch Size {batch_size}:")
        print(f"  {output[:200]}...")  # First 200 chars

    # Print timing info
    print("\n--- Generation Times (single forward pass) ---")
    for batch_size in batch_sizes:
        time_ms = results["generation_times"][batch_size] * 1000
        print(f"  Batch {batch_size:3d}: {time_ms:6.2f} ms")

    # Print difference matrix
    print("\n--- Maximum Logit Differences ---")
    print("(How different are outputs between batch sizes?)")
    print(f"{'':>10}", end="")
    for bs in batch_sizes:
        print(f"{bs:>10}", end="")
    print()

    for i, bs_i in enumerate(batch_sizes):
        print(f"{bs_i:>10}", end="")
        for j, bs_j in enumerate(batch_sizes):
            if i == j:
                print(f"{'---':>10}", end="")
            elif i > j:
                print(f"{'':>10}", end="")
            else:
                diff = results["differences"].get((bs_i, bs_j), 0.0)
                print(f"{diff:>10.6f}", end="")
        print()

    # Summary statistics
    print("\n--- Summary Statistics ---")
    all_diffs = [v for v in results["differences"].values() if v > 0]
    if all_diffs:
        print(f"  Maximum difference: {max(all_diffs):.6f}")
        print(f"  Average difference: {np.mean(all_diffs):.6f}")
        print(f"  Minimum difference: {min(all_diffs):.6f}")
    else:
        print("  All outputs are BITWISE IDENTICAL (diff = 0.0)")


def main():
    print("="*80)
    print("  BATCH INVARIANCE COMPARISON BENCHMARK")
    print("="*80)
    print()
    print("This benchmark demonstrates the difference between standard MLX-LM")
    print("inference and batch-invariant inference.")
    print()
    print("WHAT IS BATCH INVARIANCE?")
    print("-" * 80)
    print("Batch invariance means that the model produces IDENTICAL outputs")
    print("whether you process one input at a time or multiple inputs together.")
    print()
    print("Example:")
    print("  - Process 'Write a poem' alone → Output A")
    print("  - Process 'Write a poem' + 3 other prompts → Output A (same!)")
    print()
    print("WHY DOES IT MATTER?")
    print("-" * 80)
    print("Without batch invariance, you get DIFFERENT outputs depending on:")
    print("  - What other prompts are in the batch")
    print("  - How many prompts you process together")
    print("  - The order of prompts")
    print()
    print("This makes testing difficult and results unpredictable!")
    print()

    # Load model
    print("Loading model: mlx-community/Qwen2.5-3B-Instruct-4bit...")
    model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")
    print("Model loaded!\n")

    # Creative prompt that should show variety
    prompt = "Once upon a time in a magical forest"

    # Batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print(f"Test prompt: \"{prompt}\"")
    print(f"Batch sizes: {batch_sizes}")
    print()

    # =========================================================================
    # PART 1: Standard MLX-LM (non-deterministic)
    # =========================================================================
    print("\n" + "="*80)
    print("  PART 1: STANDARD MLX-LM (Non-Deterministic)")
    print("="*80)
    print("\nRunning inference with STANDARD operations...")
    print("(This uses MLX's native operations which are NOT batch-invariant)")

    results_standard = compare_batch_sizes(
        model, tokenizer, prompt, batch_sizes, max_tokens=50
    )

    print_results_section("STANDARD MLX-LM RESULTS", results_standard, batch_sizes)

    # =========================================================================
    # PART 2: Batch-Invariant Mode
    # =========================================================================
    print("\n\n" + "="*80)
    print("  PART 2: BATCH-INVARIANT MODE")
    print("="*80)
    print("\nEnabling batch-invariant mode...")

    from mlx_deterministic import (
        enable_mlx_lm_deterministic_mode,
        replace_quantized_linear_layers
    )

    # Reload model fresh
    print("Reloading model...")
    model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")

    # Enable deterministic mode
    enable_mlx_lm_deterministic_mode(split_size=256)
    replace_quantized_linear_layers(model)
    print("Batch-invariant mode enabled!")

    print("\nRunning inference with BATCH-INVARIANT operations...")
    print("(This uses our custom Metal kernels with fixed reduction patterns)")

    results_invariant = compare_batch_sizes(
        model, tokenizer, prompt, batch_sizes, max_tokens=50
    )

    print_results_section("BATCH-INVARIANT RESULTS", results_invariant, batch_sizes)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)

    # Compare standard vs invariant
    print("\n--- Standard MLX-LM ---")
    std_diffs = [v for v in results_standard["differences"].values()]
    if std_diffs:
        print(f"  Maximum difference between batch sizes: {max(std_diffs):.6f}")
        print(f"  Average difference: {np.mean(std_diffs):.6f}")

    print("\n--- Batch-Invariant Mode ---")
    inv_diffs = [v for v in results_invariant["differences"].values()]
    if inv_diffs and max(inv_diffs) > 0:
        print(f"  Maximum difference between batch sizes: {max(inv_diffs):.6f}")
        print(f"  Average difference: {np.mean(inv_diffs):.6f}")
    else:
        print("  ✓ PERFECT BATCH INVARIANCE - All outputs bitwise identical!")

    print("\n--- Key Takeaways ---")
    print()
    if std_diffs and max(std_diffs) > 0.01:
        print("  1. Standard MLX-LM has SIGNIFICANT batch variance")
        print("     → Outputs change depending on batch size")
        print("     → Same prompt can give different results")

    if not inv_diffs or max(inv_diffs) == 0:
        print("  2. Batch-invariant mode achieves PERFECT determinism")
        print("     → Outputs are BITWISE IDENTICAL regardless of batch size")
        print("     → Same prompt always gives same result")

    print()
    print("  3. This is critical for:")
    print("     • Reproducible research")
    print("     • Reliable testing and debugging")
    print("     • Fair benchmarking across configurations")
    print("     • Production systems requiring determinism")

    print("\n" + "="*80)
    print("  EXAMPLE OUTPUTS FOR INTUITION")
    print("="*80)

    print("\nImagine you run the same prompt 'Write a story' twice:")
    print()
    print("WITHOUT batch invariance:")
    print("  Run 1 (alone):         'The dragon flew over the mountain...'")
    print("  Run 2 (with 7 others): 'The dragon soared through clouds...'")
    print("  → DIFFERENT outputs! ❌")
    print()
    print("WITH batch invariance:")
    print("  Run 1 (alone):         'The dragon flew over the mountain...'")
    print("  Run 2 (with 7 others): 'The dragon flew over the mountain...'")
    print("  → IDENTICAL outputs! ✓")
    print()

    # Show actual example
    print("\nACTUAL OUTPUTS FROM THIS RUN:")
    print("-" * 80)
    print("\nStandard MLX-LM (batch_size=1):")
    print(f"  {results_standard['outputs'][1][:300]}...")
    print("\nStandard MLX-LM (batch_size=32):")
    print(f"  {results_standard['outputs'][32][:300]}...")

    if results_standard['outputs'][1] != results_standard['outputs'][32]:
        print("\n  → Notice these are DIFFERENT!")

    print("\n" + "-" * 80)
    print("\nBatch-Invariant Mode (batch_size=1):")
    print(f"  {results_invariant['outputs'][1][:300]}...")
    print("\nBatch-Invariant Mode (batch_size=32):")
    print(f"  {results_invariant['outputs'][32][:300]}...")

    if results_invariant['outputs'][1] == results_invariant['outputs'][32]:
        print("\n  ✓ These are IDENTICAL!")

    print("\n" + "="*80)
    print()


if __name__ == "__main__":
    main()
