#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Remove explicit ICD-9 code mentions from clinical notes to prevent label leakage.

This utility cleans (codes, text) pairs by removing any explicit mentions of the
assigned ICD-9 codes from the note text. This prevents models from simply
memorising code patterns rather than learning clinical reasoning.

The cleaning process matches various code formats (dotted/undotted, case variants)
and removes them using token boundary detection.

Usage:
  python utils/clean_synth_notes.py --input_pt input.pt --output_pt output.pt

Programmatic usage:
  from utils.clean_synth_notes import clean_pairs_in_memory
  cleaned = clean_pairs_in_memory(pairs, quiet=True)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, List, Tuple

import torch


# Tolerant extraction
def _extract_codes_text(item: Any) -> Tuple[List[str], str]:
    if isinstance(item, (tuple, list)):
        if len(item) < 2:
            raise ValueError(f"Pair-like item too short: {type(item)}")
        codes, text = item[0], item[1]
    elif isinstance(item, dict):
        if "codes" in item and "text" in item:
            codes, text = item["codes"], item["text"]
        elif "ICD9_CODE" in item and "TEXT" in item:
            raw = item["ICD9_CODE"]
            if isinstance(raw, str):
                try:
                    codes = json.loads(raw)
                except Exception:
                    codes = [c.strip() for c in raw.split(",") if c.strip()]
            else:
                codes = raw
            text = item["TEXT"]
        else:
            raise ValueError(f"Unsupported dict keys: {list(item.keys())}")
    else:
        raise ValueError(f"Unsupported sample type: {type(item)}")

    if not isinstance(text, str):
        text = str(text or "")
    if not isinstance(codes, list):
        try:
            codes = list(codes)
        except Exception:
            codes = [str(codes)]
    codes = [str(c).strip() for c in codes if str(c).strip()]
    return codes, text


# Code matching
def _code_variants(code: str) -> List[str]:
    out = set()
    c = (code or "").strip()
    if not c:
        return []
    out.update({c, c.upper(), c.lower()})
    undotted = c.replace(".", "")
    out.update({undotted, undotted.upper(), undotted.lower()})
    if undotted.isdigit() and len(undotted) > 3:
        dotted = undotted[:3] + "." + undotted[3:]
        out.update({dotted, dotted.upper(), dotted.lower()})
    return list(out)


def _build_code_regex(codes: List[str]) -> re.Pattern | None:
    variants = set()
    for code in codes:
        for v in _code_variants(code):
            if v:
                variants.add(re.escape(v))
    if not variants:
        return None

    pattern = (
        r"(?:(?<=^)|(?<=[^A-Za-z0-9_]))(?:"
        + "|".join(sorted(variants, key=len, reverse=True))
        + r")(?=$|[^A-Za-z0-9_])"
    )
    try:
        return re.compile(pattern)
    except re.error as e:
        print(f"Regex compile failed: {e}", file=sys.stderr)
        return None


def _strip_codes_globally(text: str, code_re: re.Pattern | None) -> str:
    if not text or not code_re:
        return text or ""

    return code_re.sub("", text)


# Public API
def clean_pairs_in_memory(
    pairs: List[Tuple[List[str], str]], quiet: bool = True
) -> List[Tuple[List[str], str]]:
    cleaned: List[Tuple[List[str], str]] = []
    n_deleted_tokens = 0
    for i, (codes, text) in enumerate(pairs):
        code_re = _build_code_regex(codes)
        before_len = len(text or "")
        out_text = _strip_codes_globally(text or "", code_re)
        n_deleted_tokens += int(before_len != len(out_text))
        cleaned.append((codes, out_text))
    if not quiet:
        print(
            f"[clean_pairs_in_memory] cleaned={len(cleaned)} modified_samples≈{n_deleted_tokens}",
            file=sys.stderr,
        )
    return cleaned


# CLI
def _main():
    ap = argparse.ArgumentParser(
        description="Remove any ICD-9 code mentions from notes."
    )
    ap.add_argument(
        "--input_pt",
        required=True,
        help="Input .pt (list of (codes, text) pairs or dicts)",
    )
    ap.add_argument("--output_pt", help="Output .pt (default: overwrite if --inplace)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input file")
    ap.add_argument("--quiet", action="store_true", help="Less verbose logging")
    args = ap.parse_args()

    data = torch.load(args.input_pt)
    if not isinstance(data, (list, tuple)):
        print(f"ERROR: Expected a list/tuple; got {type(data)}", file=sys.stderr)
        sys.exit(1)

    pairs: List[Tuple[List[str], str]] = []
    for item in data:
        codes, text = _extract_codes_text(item)
        pairs.append((codes, text))

    cleaned = clean_pairs_in_memory(pairs, quiet=args.quiet)

    out_path = (
        args.input_pt
        if args.inplace
        else (args.output_pt or (args.input_pt + ".cleaned.pt"))
    )
    torch.save(cleaned, out_path)
    if not args.quiet:
        print(f"Saved cleaned pairs → {out_path}")


if __name__ == "__main__":
    _main()
