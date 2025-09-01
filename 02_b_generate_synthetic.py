#!/usr/bin/env python
"""
Generate synthetic discharge notes using trained differentially private models.

This script loads a trained generative model and produces synthetic clinical notes
conditioned on ICD-9 codes from the real training set. The generated data maintains
the same format as real data whilst providing privacy guarantees.

The script verifies model integrity via cryptographic checksums before generation.
"""
from __future__ import annotations
import argparse, logging, os, json, hashlib
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
from tqdm.auto import tqdm

SEP, NOTE = "<|codes|>", "<|note|>"


def _sha256_of_dir(dirpath: Path, patterns=(".bin", ".safetensors")) -> str | None:
    files = []
    for p in sorted(dirpath.glob("*")):
        if p.is_file() and p.suffix in patterns:
            files.append(p)
    if not files:
        return None
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(str(p.stat().st_size).encode("utf-8"))
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def _verify_digest(model_dir: Path):
    dig_fp = model_dir / "digests.json"
    if not dig_fp.exists():
        raise RuntimeError(f"Missing digest file: {dig_fp}")
    dig = json.loads(dig_fp.read_text())
    # Prefer merged snapshot
    merged_dir = model_dir / "merged_fp16"
    lora_dir = model_dir / "lora"
    if merged_dir.exists():
        calc = _sha256_of_dir(merged_dir)
        exp = (dig.get("merged_fp16") or {}).get("hexdigest")
        if not calc or not exp or calc != exp:
            raise RuntimeError(
                f"Digest mismatch for merged_fp16: expected {exp}, got {calc}"
            )
        logging.info("✅ Verified merged_fp16 digest")
    if lora_dir.exists():
        calc = _sha256_of_dir(lora_dir)
        exp = (dig.get("lora") or {}).get("hexdigest")
        if not calc or not exp or calc != exp:
            raise RuntimeError(f"Digest mismatch for lora: expected {exp}, got {calc}")
        logging.info("✅ Verified lora digest")

    if not merged_dir.exists() and not lora_dir.exists():
        raise RuntimeError("Neither merged_fp16/ nor lora/ found under best_model/")


def load_model(model_dir: Path, local_rank: int):
    _verify_digest(model_dir)
    tok = LlamaTokenizerFast.from_pretrained(model_dir / "tokenizer")

    merged = model_dir / "merged_fp16"
    if merged.exists():
        model = LlamaForCausalLM.from_pretrained(
            merged,
            torch_dtype=torch.float16,
            device_map={"": f"cuda:{local_rank}"},
            attn_implementation="flash_attention_2",
        ).eval()
        return tok, model

    base_dir = model_dir / "base_clean"
    lora_dir = model_dir / "lora"
    base = LlamaForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{local_rank}"},
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, lora_dir, is_trainable=False).eval()
    return tok, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help=".../best_model (from 02_a)")
    p.add_argument(
        "--real_train_pairs_pt",
        required=True,
        help="data/real/real_train_codes_notes.pt",
    )
    p.add_argument("--gen_batch", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--out_pt",
        required=True,
        help="e.g., data/synthetic/eps6/synth_train_codes_notes.pt",
    )
    args = p.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    if local_rank == 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
        )
        logging.info("World size = %d · device = cuda:%d", world_size, local_rank)

    # Load the real train pairs
    all_pairs: List[Tuple[List[str], str]] = torch.load(args.real_train_pairs_pt)
    all_codes: List[List[str]] = [codes for codes, _ in all_pairs]
    total = len(all_codes)

    obj = [total]
    dist.broadcast_object_list(obj, src=0)
    total = obj[0]

    my_indices = list(range(local_rank, total, world_size))
    my_codes = [all_codes[i] for i in my_indices]

    tok, model = load_model(Path(args.model_dir), local_rank)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    rows: List[Tuple[int, List[str], str]] = []
    torch.manual_seed(args.seed + local_rank)
    torch.cuda.manual_seed(args.seed + local_rank)

    for i in tqdm(range(0, len(my_codes), args.gen_batch), desc=f"Rank{local_rank}"):
        batch_codes = my_codes[i : i + args.gen_batch]
        batch_idx = my_indices[i : i + args.gen_batch]
        prompts = [f"{SEP} {' '.join(c)}\n{NOTE}" for c in batch_codes]
        inp = tok(prompts, return_tensors="pt", padding=True).to(f"cuda:{local_rank}")

        with torch.no_grad():
            outs = model.generate(**inp, generation_config=gen_cfg)

        for j, out_ids in enumerate(outs):
            prompt_len = (inp["attention_mask"][j] == 1).sum().item()
            txt = tok.decode(out_ids[prompt_len:], skip_special_tokens=True).strip()
            rows.append((batch_idx[j], batch_codes[j], txt))

    tmp = Path(args.out_pt).with_suffix(f".rank{local_rank}.pt")
    torch.save(rows, tmp)

    dist.barrier()
    if local_rank == 0:
        merged: List[Tuple[int, List[str], str]] = []
        for r in range(world_size):
            shard = torch.load(Path(args.out_pt).with_suffix(f".rank{r}.pt"))
            merged.extend(shard)
        merged.sort(key=lambda x: x[0])
        final = [(codes, text) for _, codes, text in merged]
        torch.save(final, args.out_pt)
        for r in range(world_size):
            Path(args.out_pt).with_suffix(f".rank{r}.pt").unlink()
        logging.info("Merged %d pairs → %s", len(final), args.out_pt)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
