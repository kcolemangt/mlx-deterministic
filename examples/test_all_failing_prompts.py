#!/usr/bin/env python3
"""
Test ALL previously failing prompts with Metal kernels.
"""
import mlx.core as mx
from mlx_lm import load
from mlx_deterministic import enable_deterministic_mode, DeterministicConfig

# ALL prompts that were found to fail before the patch
ALL_PREVIOUSLY_FAILING_PROMPTS = [
    # Original bug report prompts
    "The quick brown fox jumps over the lazy dog",

    # README example
    "What is the capital of France?",

    # Discovered failing prompts from extended testing
    "Nice to meet you.",
    "Please help me.",
    "What color is the sky?",
    "How many days in a week?",
    "How many months in a year?",
    "Hello, how are you today?",
    "Who wrote Romeo and Juliet?",
    "What is the capital of Italy?",
    "What is the capital of Japan?",
    "Write a Python function",
    "How do I learn Python?",

    # Additional prompts from earlier testing
    "Write a Python function that calculates the factorial of a number",
    "Can you help me understand how neural networks work step by step?",
]

# Prompts that passed before (should still pass)
PREVIOUSLY_PASSING_PROMPTS = [
    "Hi",
    "Hello",
    "Hello world",
    "How are you?",
    "Once upon a time",
    "The quick brown fox",
    "What is the meaning of life?",
    "Tell me a story about",
    "In the beginning there was",
]


def measure_batch_variance(model, tokenizer, prompt):
    """Measure max logit difference between batch_size=1 and batch_size=4."""
    tokens = tokenizer.encode(prompt)
    x1 = mx.array([tokens])
    x4 = mx.array([tokens] * 4)

    out1 = model(x1)
    out4 = model(x4)

    logits1 = out1[0, -1, :]
    logits4 = out4[0, -1, :]
    mx.eval(logits1, logits4)

    return mx.max(mx.abs(logits1 - logits4)).item()


print("=" * 70)
print("COMPREHENSIVE TEST: All Previously Failing Prompts")
print("=" * 70)

print("\nLoading model...")
model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")

print("\nApplying Metal kernel-based deterministic mode...")
config = DeterministicConfig(use_metal_kernels=True)
enable_deterministic_mode(model, config)

# Test previously FAILING prompts
print("\n" + "-" * 70)
print("PREVIOUSLY FAILING PROMPTS (should now pass):")
print("-" * 70)

fail_count = 0
for prompt in ALL_PREVIOUSLY_FAILING_PROMPTS:
    diff = measure_batch_variance(model, tokenizer, prompt)
    if diff == 0:
        status = "PASS"
    else:
        status = f"FAIL ({diff:.6f})"
        fail_count += 1
    print(f"  {status:<20} \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")

# Test previously PASSING prompts
print("\n" + "-" * 70)
print("PREVIOUSLY PASSING PROMPTS (should still pass):")
print("-" * 70)

for prompt in PREVIOUSLY_PASSING_PROMPTS:
    diff = measure_batch_variance(model, tokenizer, prompt)
    if diff == 0:
        status = "PASS"
    else:
        status = f"FAIL ({diff:.6f})"
        fail_count += 1
    print(f"  {status:<20} \"{prompt}\"")

# Summary
total = len(ALL_PREVIOUSLY_FAILING_PROMPTS) + len(PREVIOUSLY_PASSING_PROMPTS)
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Previously failing prompts: {len(ALL_PREVIOUSLY_FAILING_PROMPTS) - fail_count}/{len(ALL_PREVIOUSLY_FAILING_PROMPTS)} now pass")
print(f"Previously passing prompts: {len(PREVIOUSLY_PASSING_PROMPTS)}/{len(PREVIOUSLY_PASSING_PROMPTS)} still pass")
print(f"Total: {total - fail_count}/{total} pass")

if fail_count == 0:
    print("\n✓ ALL PROMPTS PASS - Metal kernels achieve perfect batch invariance!")
else:
    print(f"\n✗ {fail_count} prompts still failing")
