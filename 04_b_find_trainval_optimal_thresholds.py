#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute optimal classification thresholds using training and validation data.

This script calibrates per-label decision thresholds by maximising F1 scores
on the combined training and validation sets. This provides an alternative
to thresholds computed during training for improved test performance.

Outputs:
  - optimal_thresholds_trainval.npy (numpy array of thresholds)
  - optimal_thresholds_trainval.csv (human-readable threshold table)
"""

import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from transformers import AutoConfig, LlamaTokenizerFast, LlamaForSequenceClassification
from peft import PeftModel
from tqdm.auto import tqdm

SPECIAL_TOKENS = {"additional_special_tokens": ["<|codes|>", "<|note|>"]}


# Data
class PairsDataset(Dataset):
    def __init__(self, pairs, tokenizer, code2idx, max_len=512):
        self.pairs = pairs
        self.tok = tokenizer
        self.code2idx = code2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        codes, text = self.pairs[i]
        enc = self.tok(
            text or "",
            padding=False,
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
        )
        y = torch.zeros(len(self.code2idx), dtype=torch.float32)
        for c in codes:
            j = self.code2idx.get(c)
            if j is not None:
                y[j] = 1.0
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": y,
        }


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        ids = pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
        )
        ams = pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        )
        ys = torch.stack([b["labels"] for b in batch])
        return {"input_ids": ids, "attention_mask": ams, "labels": ys}

    return collate


# Threshold search
def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    n = y_true.shape[1]
    thr = np.full(n, 0.5)
    for i in range(n):
        if y_true[:, i].sum() == 0:
            continue
        best_t, best = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.02):
            pred = (y_prob[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best:
                best, best_t = f1, t
        thr[i] = best_t
    return thr


# Digest helpers
def compute_trainable_digest(model: torch.nn.Module) -> str:
    """Compute SHA-256 hash over trainable parameters (LoRA adapters and classification head)."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        name = k.lower()
        if ("lora_" not in name) and ("score" not in name):
            continue
        t = v.detach().to(torch.float32).cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def verify_digest(ckpt_dir: Path, model: torch.nn.Module):
    meta_fp = ckpt_dir / "state_digest.json"
    if not meta_fp.exists():
        raise RuntimeError(f"Missing digest file: {meta_fp}")
    meta = json.loads(meta_fp.read_text())
    expected = meta.get("only_trainable_digest")
    if not expected:
        raise RuntimeError("Digest file missing 'only_trainable_digest'")
    calc = compute_trainable_digest(model)
    if calc != expected:
        raise RuntimeError(f"Digest mismatch: expected {expected}, got {calc}")
    print("✅ Digest verified")


# Model loading
def load_model_for_calibration(ckpt_dir: Path, device: torch.device):
    """Load trained model for threshold calibration with integrity verification."""
    base_name = (ckpt_dir / "base_model_name.txt").read_text().strip()
    tokenizer = LlamaTokenizerFast.from_pretrained(ckpt_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    code2idx = json.loads((ckpt_dir / "code2idx.json").read_text())

    cfg = AutoConfig.from_pretrained(
        base_name,
        num_labels=len(code2idx),
        problem_type="multi_label_classification",
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    base = LlamaForSequenceClassification.from_pretrained(
        base_name, config=cfg, torch_dtype=torch.float32
    )
    base.resize_token_embeddings(len(tokenizer))

    head = ckpt_dir / "head.pt"
    if head.exists():
        state = torch.load(head, map_location="cpu")
        base.load_state_dict(state, strict=False)

    model = PeftModel.from_pretrained(base, ckpt_dir)
    model = model.float().eval().to(device)

    verify_digest(ckpt_dir, model)

    return model, tokenizer, code2idx


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", type=Path, required=True, help="Path to .../best_model directory"
    )
    ap.add_argument("--train_pairs", type=Path, required=True)
    ap.add_argument("--val_pairs", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    out_npy = args.ckpt / "optimal_thresholds_trainval.npy"
    out_csv = args.ckpt / "optimal_thresholds_trainval.csv"
    if out_npy.exists():
        print(f"✅ Found existing thresholds: {out_npy}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, code2idx = load_model_for_calibration(args.ckpt, device)

    if device.type == "cuda":
        model.half()

    train_pairs = torch.load(args.train_pairs)
    val_pairs = torch.load(args.val_pairs)
    all_pairs = train_pairs + val_pairs

    dl = DataLoader(
        PairsDataset(all_pairs, tokenizer, code2idx, args.max_len),
        batch_size=args.batch,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=4,
        pin_memory=True,
    )

    probs, labels = [], []
    amp_enabled = device.type == "cuda"
    autocast_ctx = torch.amp.autocast if amp_enabled else torch.cpu.amp.autocast
    autocast_args = ("cuda",) if amp_enabled else ("cpu",)
    with torch.inference_mode(), autocast_ctx(*autocast_args, dtype=torch.float16):
        for batch in tqdm(dl, desc="Calibrating thresholds (train+val)"):
            ids = batch["input_ids"].to(device, non_blocking=True)
            ams = batch["attention_mask"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            out = model(input_ids=ids, attention_mask=ams, labels=labs)
            p = torch.sigmoid(out.logits).detach().cpu().numpy()
            probs.append(p)
            labels.append(labs.cpu().numpy())

    y_prob = np.vstack(probs)
    y_true = np.vstack(labels)
    thr = find_optimal_thresholds(y_true, y_prob)

    np.save(out_npy, thr)
    idx2code = {v: k for k, v in code2idx.items()}
    with open(out_csv, "w") as f:
        f.write("code,threshold\n")
        for i, t in enumerate(thr):
            f.write(f"{idx2code[i]},{t:.4f}\n")

    print(f"✅ Saved train+val thresholds → {out_npy}")
    print(f"📝 CSV: {out_csv}")


if __name__ == "__main__":
    main()
