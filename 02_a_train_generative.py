#!/usr/bin/env python
"""
Train differentially private generative models using LoRA fine-tuning.

This script implements DP-SGD training for generative language models that learn
to produce synthetic clinical notes conditioned on ICD-9 codes. The trained models
serve as privacy-preserving data generators for downstream classification tasks.

Outputs:
  - best_model/tokenizer/ (tokeniser configuration)
  - best_model/lora/ (LoRA adapter weights)
  - best_model/base_clean/ (base model without adapters)
  - best_model/merged_fp16/ (merged model in half precision)
  - best_model/digests.json (cryptographic checksums)
"""

from __future__ import annotations
import os, argparse, random, math, logging, sys, copy, json, hashlib
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample.grad_sample_module import GradSampleModule

from peft import get_peft_model, LoraConfig, PeftModel
from transformers import LlamaTokenizerFast, LlamaForCausalLM
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SPECIAL_TOKENS = {"additional_special_tokens": ["<|codes|>", "<|note|>"]}
SEP, NOTE = "<|codes|>", "<|note|>"
LOGFMT = "%(asctime)s — %(levelname)s — %(message)s"


def seed_every(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(model_name: str):
    tok = LlamaTokenizerFast.from_pretrained(model_name, token=True)
    tok.add_special_tokens(SPECIAL_TOKENS)
    tok.pad_token = tok.eos_token
    return tok


def unwrap(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if isinstance(m, GradSampleModule):
        m = m._module
    return m


def _sha256_of_dir(dirpath: Path, patterns=(".bin", ".safetensors")) -> dict:
    files = []
    for p in sorted(dirpath.glob("*")):
        if p.is_file() and p.suffix in patterns:
            files.append(p)
    if not files:
        return {"algo": "sha256", "files": [], "hexdigest": None}
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(str(p.stat().st_size).encode("utf-8"))
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return {
        "algo": "sha256",
        "files": [str(p.name) for p in files],
        "hexdigest": h.hexdigest(),
    }


def save_exact_checkpoint(peft_model: PeftModel, tok, out_dir: Path):
    tok_dir = out_dir / "tokenizer"
    lora_dir = out_dir / "lora"
    base_dir = out_dir / "base_clean"
    merged_dir = out_dir / "merged_fp16"
    for d in (tok_dir, lora_dir, base_dir, merged_dir):
        d.mkdir(parents=True, exist_ok=True)

    tok.save_pretrained(tok_dir)

    peft_model.save_pretrained(lora_dir)

    pure_base = peft_model.get_base_model()
    if hasattr(pure_base, "tie_weights"):
        pure_base.tie_weights()
    pure_base.config.pad_token_id = tok.pad_token_id
    pure_base.config.eos_token_id = tok.eos_token_id
    pure_base.save_pretrained(base_dir, safe_serialization=False)

    logging.info("Creating merged snapshot (FP16)…")
    cpu_model = copy.deepcopy(peft_model).to("cpu")
    merged = cpu_model.merge_and_unload()
    if hasattr(merged, "tie_weights"):
        merged.tie_weights()
    merged.config.pad_token_id = tok.pad_token_id
    merged.config.eos_token_id = tok.eos_token_id
    merged.save_pretrained(merged_dir, safe_serialization=False)

    digests = {
        "algo": "sha256",
        "merged_fp16": _sha256_of_dir(merged_dir),
        "lora": _sha256_of_dir(lora_dir),
    }
    with (out_dir / "digests.json").open("w") as f:
        json.dump(digests, f, indent=2)
    logging.info(
        "Saved tokenizer/, lora/, base_clean/, merged_fp16/ and digests.json → %s",
        out_dir,
    )


def build_lora_model(model_name: str, tokenizer, grad_ckpt: bool, device: torch.device):
    base = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )
    base.resize_token_embeddings(len(tokenizer))
    if grad_ckpt:
        base.gradient_checkpointing_enable()
        base.config.use_cache = False

    lcfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base, lcfg).to(device)

    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n

    # Initialise special tokens with EOS embedding
    with torch.no_grad():
        eos_id = tokenizer.eos_token_id
        sp_ids = tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["additional_special_tokens"]
        )
        emb = model.get_input_embeddings().weight
        lm_head = (model.get_output_embeddings() or model.lm_head).weight
        for sid in sp_ids:
            emb[sid] = emb[eos_id]
            lm_head[sid] = lm_head[eos_id]

    return model


class CodeNoteDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[str], str]], tokenizer, max_len: int):
        prefixes, texts = [], []
        for codes, txt in pairs:
            codes_str = " ".join(codes or [])
            pref = f"{SEP} {codes_str}\n{NOTE}"
            prefixes.append(pref)
            texts.append(pref + (txt or "") + tokenizer.eos_token)

        self.enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        pref_ids = tokenizer(
            prefixes, padding=False, truncation=True, max_length=max_len
        )["input_ids"]
        self.pref_lens = [len(ids) for ids in pref_ids]

    def __len__(self):
        return self.enc.input_ids.size(0)

    def __getitem__(self, idx):
        out = {k: v[idx] for k, v in self.enc.items()}
        labels = out["input_ids"].clone()
        labels[out["attention_mask"] == 0] = -100
        cut = min(self.pref_lens[idx], labels.size(0))
        labels[:cut] = -100
        out["labels"] = labels
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--model_name", required=True)
    p.add_argument("--train_pairs_pt", required=True)
    p.add_argument("--val_pairs_pt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epsilon", type=float, default=6.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=2)  # per-rank
    p.add_argument(
        "--accum_steps", type=int, default=1
    )  # microbatches per optimizer step
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    best_dir = Path(args.out_dir, "best_model")
    best_dir.mkdir(parents=True, exist_ok=True)

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=6000))
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    seed_every(args.seed + local_rank)

    if local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format=LOGFMT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(Path(args.out_dir) / "train.log"),
            ],
            force=True,
        )
        logging.info("Args %s", vars(args))

    tok = get_tokenizer(args.model_name)

    train_pairs: List[Tuple[List[str], str]] = torch.load(args.train_pairs_pt)
    val_pairs: List[Tuple[List[str], str]] = torch.load(args.val_pairs_pt)

    train_ds = CodeNoteDataset(train_pairs, tok, args.max_seq_len)
    val_ds = CodeNoteDataset(val_pairs, tok, args.max_seq_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=not distributed,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False
    )

    if local_rank == 0:
        logging.info("Train batches=%d · Val batches=%d", len(train_dl), len(val_dl))

    model = build_lora_model(args.model_name, tok, args.grad_ckpt, device)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    priv_engine = PrivacyEngine(secure_mode=False, accountant="rdp")

    model, opt, train_dl = priv_engine.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=train_dl,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        poisson_sampling=False,
    )

    if distributed:
        model = DPDDP(model)

    best_val, no_imp = float("inf"), 0
    if local_rank == 0:
        mlog = open(Path(args.out_dir) / "metrics.csv", "w")
        print("epoch,split,loss,ppl,eps", file=mlog, flush=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        t_loss = t_tok = 0
        pbar = tqdm(
            train_dl,
            desc=f"Ep{epoch}/{args.epochs} [train]",
            disable=(local_rank != 0),
            leave=False,
            file=sys.stdout,
        )

        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()

            # DP microbatching
            opt.step()
            opt.zero_grad(set_to_none=True)

            n_tok = batch["labels"].ne(-100).sum().item()
            t_loss += loss.item() * n_tok
            t_tok += n_tok
            if local_rank == 0:
                eps_now = priv_engine.get_epsilon(delta=args.delta)
                pbar.set_postfix(
                    {
                        "loss": f"{t_loss/t_tok:.4f}",
                        "ppl": f"{math.exp(t_loss/t_tok):.1f}",
                        "ε": f"{eps_now:.2f}",
                    }
                )
        pbar.close()

        model.eval()
        v_loss = v_tok = 0
        pbar = tqdm(
            val_dl,
            desc=f"Ep{epoch}/{args.epochs} [val]  ",
            disable=(local_rank != 0),
            leave=False,
            file=sys.stdout,
        )
        with torch.no_grad():
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                n_tok = batch["labels"].ne(-100).sum().item()
                v_loss += out.loss.item() * n_tok
                v_tok += n_tok
                if local_rank == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{v_loss/v_tok:.4f}",
                            "ppl": f"{math.exp(v_loss/v_tok):.1f}",
                        }
                    )
        pbar.close()
        val_ppl = math.exp(v_loss / v_tok)

        # Early stopping and checkpointing
        stop_flag = torch.zeros(1, device=device, dtype=torch.uint8)
        if local_rank == 0:
            eps_now = priv_engine.get_epsilon(delta=args.delta)
            print(
                f"{epoch},train,{t_loss/t_tok:.6f},{math.exp(t_loss/t_tok):.4f},{eps_now:.3f}",
                file=mlog,
                flush=True,
            )
            print(
                f"{epoch},val,{v_loss/v_tok:.6f},{val_ppl:.4f},{eps_now:.3f}",
                file=mlog,
                flush=True,
            )
            logging.info(
                "Ep%02d summary ▸ train_ppl %.2f ▸ val_ppl %.2f ▸ ε %.2f",
                epoch,
                math.exp(t_loss / t_tok),
                val_ppl,
                eps_now,
            )

            if val_ppl < best_val:
                best_val, no_imp = val_ppl, 0
                real = unwrap(model)
                save_exact_checkpoint(real, tok, best_dir)
            else:
                no_imp += 1
                if no_imp >= args.patience:
                    logging.info("Early stop at epoch %d", epoch)
                    stop_flag[0] = 1

        if distributed:
            torch.distributed.broadcast(stop_flag, src=0)
        if stop_flag.item():
            break

    if local_rank == 0:
        mlog.close()
        logging.info("Training finished; artifacts in %s", args.out_dir)

    if distributed:
        torch.distributed.destroy_process_group()
        logging.info("Distributed process group destroyed.")


if __name__ == "__main__":
    main()
