#!/usr/bin/env python
"""
Utility to inspect the contents of PyTorch data files.

This script displays the first few examples from .pt files containing
(codes, text) pairs, useful for debugging data preprocessing pipelines.

Usage:
  python utils/inspect_pt_file.py <path_to_pt_file> [--num 10]
"""
import argparse
import torch
from pathlib import Path
from typing import List, Tuple


def inspect_pt_file(pt_path: Path, num_examples: int = 5):
    """Read and print the first N examples from a PT file."""
    if not pt_path.exists():
        print(f"❌ File not found: {pt_path}")
        return

    print(f"Reading: {pt_path}")
    try:
        pairs: List[Tuple[List[str], str]] = torch.load(pt_path)
        print(f"Total pairs: {len(pairs)}\n")

        for i, (codes, text) in enumerate(pairs[:num_examples]):
            print(f"--- Pair {i+1} ---")
            print(f"Codes ({len(codes)}): {codes}")
            print(f"Note: {text}\n")

    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect PT file contents")
    parser.add_argument("pt_file", type=Path, help="Path to the PT file")
    parser.add_argument(
        "--num", type=int, default=5, help="Number of examples to show (default: 5)"
    )
    args = parser.parse_args()
    inspect_pt_file(args.pt_file, args.num)


if __name__ == "__main__":
    main()
