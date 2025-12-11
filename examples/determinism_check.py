#!/usr/bin/env python3
"""
Deterministic Inference Sanity Check

A user-friendly tool to verify that MLX deterministic inference produces
identical outputs regardless of batch size.

Usage:
    python examples/determinism_check.py
    python examples/determinism_check.py --input my_queries.json
    python examples/determinism_check.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
    python examples/determinism_check.py --max-tokens 100 --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

from mlx_deterministic import (
    DeterministicConfig,
    enable_deterministic_mode,
    enable_mlx_lm_deterministic_mode,
    replace_quantized_linear_layers,
)


def load_queries(input_path: str) -> list[dict[str, str]]:
    """Load queries from a JSON file."""
    with open(input_path, "r") as f:
        queries = json.load(f)

    # Validate structure
    for i, q in enumerate(queries):
        if "query" not in q:
            raise ValueError(f"Query {i} missing 'query' field")
        if "name" not in q:
            q["name"] = f"Query {i + 1}"

    return queries


def format_prompt(tokenizer: Any, query: str) -> str:
    """Format query using the model's chat template if available."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": query}]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        # Fallback for models without chat template
        return f"User: {query}\nAssistant:"


def check_batch_invariance(
    model: Any,
    tokenizer: Any,
    query: str,
    batch_sizes: list[int],
) -> tuple[bool, float, dict[int, float]]:
    """
    Check if model produces identical outputs across different batch sizes.

    Returns:
        (is_deterministic, max_diff, diffs_by_batch_size)
    """
    prompt = format_prompt(tokenizer, query)
    tokens = tokenizer.encode(prompt)

    # Get reference logits at batch_size=1
    x1 = mx.array([tokens])
    out1 = model(x1)
    ref_logits = out1[0, -1, :]
    mx.eval(ref_logits)

    diffs: dict[int, float] = {1: 0.0}
    max_diff = 0.0

    # Compare against other batch sizes
    for batch_size in batch_sizes:
        if batch_size == 1:
            continue

        x_batch = mx.array([tokens] * batch_size)
        out_batch = model(x_batch)
        batch_logits = out_batch[0, -1, :]  # First sample, last token
        mx.eval(batch_logits)

        diff = mx.max(mx.abs(ref_logits - batch_logits)).item()
        diffs[batch_size] = diff
        max_diff = max(max_diff, diff)

    is_deterministic = max_diff == 0.0
    return is_deterministic, max_diff, diffs


def generate_text(
    model: Any,
    tokenizer: Any,
    query: str,
    max_tokens: int = 50,
) -> str:
    """Generate text using greedy decoding for reproducibility."""
    prompt = format_prompt(tokenizer, query)
    current_ids = tokenizer.encode(prompt)
    generated_tokens = []

    for _ in range(max_tokens):
        x = mx.array([current_ids])
        out = model(x)
        next_token_logits = out[0, -1, :]
        next_token = int(mx.argmax(next_token_logits).item())

        # Stop at EOS or end-of-turn tokens
        if next_token == tokenizer.eos_token_id:
            break
        # Qwen uses <|im_end|> as end-of-turn
        if hasattr(tokenizer, "im_end_id") and next_token == tokenizer.im_end_id:
            break

        generated_tokens.append(next_token)
        current_ids.append(next_token)

    return tokenizer.decode(generated_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify deterministic inference across batch sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/determinism_check.py
    python examples/determinism_check.py --input my_queries.json
    python examples/determinism_check.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
    python examples/determinism_check.py --max-tokens 100 --verbose
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to JSON file with queries (default: queries.json in script dir)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="Model to use (default: mlx-community/Qwen2.5-3B-Instruct-4bit)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate for sanity check (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output including per-batch-size differences",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes to test (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick mode: test only 1 prompt with batch sizes [1, 4], 15 tokens",
    )
    parser.add_argument(
        "--metal",
        action="store_true",
        help="Use Metal kernels (bitwise determinism) instead of Python wrapper",
    )

    args = parser.parse_args()

    # Quick mode overrides - minimal for fast iteration
    if args.quick:
        args.batch_sizes = "1,4"
        args.max_tokens = 15

    # Determine input file path
    if args.input:
        input_path = args.input
    else:
        script_dir = Path(__file__).parent
        input_path = script_dir / "queries.json"

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    if 1 not in batch_sizes:
        batch_sizes = [1] + batch_sizes
    batch_sizes = sorted(set(batch_sizes))

    # Print header
    print("=" * 70)
    print("  DETERMINISTIC INFERENCE SANITY CHECK")
    print("=" * 70)
    print()

    # Load queries
    print(f"Loading queries from: {input_path}")
    try:
        queries = load_queries(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        print("Create a queries.json file or specify --input path")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_path}: {e}")
        return 1

    # Quick mode: limit to first 1 query for fast iteration
    if args.quick and len(queries) > 1:
        queries = queries[:1]
        print(f"Quick mode: using 1 query")
    else:
        print(f"Found {len(queries)} queries")
    print()

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    print("Model loaded!")
    print()

    # Enable deterministic mode
    if args.metal:
        print("Enabling deterministic mode (Metal kernels)...")
        config = DeterministicConfig(use_metal_kernels=True)
        enable_deterministic_mode(model, config)
        mode_str = "Metal kernels (bitwise)"
    else:
        print("Enabling deterministic mode (Python wrapper)...")
        enable_mlx_lm_deterministic_mode(split_size=256)
        replace_quantized_linear_layers(model)
        mode_str = "Python wrapper (split_size=256)"
    print("Deterministic mode enabled!")
    print()

    # Print configuration
    print("-" * 70)
    print(f"Model:       {args.model}")
    print(f"Mode:        {mode_str}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Queries:     {len(queries)} from {Path(input_path).name}")
    print("-" * 70)
    print()

    # Test each query
    results: list[dict[str, Any]] = []
    pass_count = 0

    for i, q in enumerate(queries):
        query = q["query"]
        name = q["name"]

        print("-" * 70)
        print(f"[{i + 1}/{len(queries)}] {name}")
        print(f'Query: "{query}"')
        print("-" * 70)

        # Check batch invariance
        is_det, max_diff, diffs = check_batch_invariance(
            model, tokenizer, query, batch_sizes
        )

        if is_det:
            print("Batch invariance: PASS (all batch sizes produce identical logits)")
            pass_count += 1
        else:
            print(f"Batch invariance: FAIL (max diff: {max_diff:.6e})")

        # Show per-batch differences in verbose mode
        if args.verbose:
            print("  Per-batch-size differences:")
            for bs in batch_sizes:
                diff = diffs.get(bs, 0.0)
                status = "identical" if diff == 0.0 else f"{diff:.6e}"
                print(f"    batch_size={bs:3d}: {status}")

        # Generate text for sanity check
        print()
        print("Generated response:")
        response = generate_text(model, tokenizer, query, args.max_tokens)
        # Indent the response
        for line in response.split("\n"):
            print(f"  {line}")
        print()

        results.append(
            {
                "name": name,
                "query": query,
                "is_deterministic": is_det,
                "max_diff": max_diff,
                "response_preview": response[:200],
            }
        )

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print(f"Determinism: {pass_count}/{len(queries)} queries PASS")
    print()

    if pass_count == len(queries):
        print(f"All outputs are BITWISE IDENTICAL across batch sizes {batch_sizes}")
        print()
        print("The model is operating deterministically.")
    else:
        print("Some queries showed variance across batch sizes:")
        for r in results:
            if not r["is_deterministic"]:
                print(f'  - "{r["name"]}": max diff = {r["max_diff"]:.6e}')
        print()
        print("Check that deterministic mode is properly enabled.")

    print()
    print("=" * 70)

    return 0 if pass_count == len(queries) else 1


if __name__ == "__main__":
    sys.exit(main())
