#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive evaluation of trained clinical code classification models.

This script evaluates classifier performance across multiple dimensions:
- Utility metrics (micro/macro F1, AUPRC, Jaccard, Hamming loss)
- Computational performance (latency, throughput, memory usage)
- Privacy analysis via membership inference attacks
- Formal differential privacy accounting (reports ε at best validation performance)

Results are saved as structured JSON reports with concise summaries printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    hamming_loss,
    jaccard_score,
)
from transformers import AutoConfig, LlamaTokenizerFast, LlamaForSequenceClassification
from peft import PeftModel
from tqdm.auto import tqdm

# Constants
SPECIAL_TOKENS = {"additional_special_tokens": ["<|codes|>", "<|note|>"]}
LOGFMT = "%(asctime)s | %(levelname)-8s | %(message)s"


# Logging
def setup_logging(verbosity: int):
    """Configure logging level: 0=warning, 1=info, 2=debug"""
    level = (
        logging.WARNING
        if verbosity <= 0
        else logging.INFO if verbosity == 1 else logging.DEBUG
    )
    logging.basicConfig(level=level, format=LOGFMT, datefmt="%Y-%m-%d %H:%M:%S")
    if level > logging.DEBUG:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Digest helpers
def _sha256_trainables(model: torch.nn.Module) -> str:
    """Compute SHA-256 hash over trainable parameters (LoRA adapters + classification head)."""
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        name = k.lower()
        if ("lora_" not in name) and ("score" not in name):
            continue
        t = v.detach().to(torch.float32).cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def verify_digest(ckpt_dir: Path, model: torch.nn.Module) -> Optional[bool]:
    """Verify model integrity by comparing saved and computed parameter hashes."""
    meta_fp = ckpt_dir / "state_digest.json"
    if not meta_fp.exists():
        logging.warning(
            "No state_digest.json found in checkpoint; skipping digest verification."
        )
        return None
    meta = json.loads(meta_fp.read_text())
    expected = meta.get("only_trainable_digest")
    if not expected:
        logging.warning(
            "Digest file missing 'only_trainable_digest'; skipping digest verification."
        )
        return None
    calc = _sha256_trainables(model)
    if calc != expected:
        raise RuntimeError(f"Digest mismatch: expected {expected}, got {calc}")
    print("✅ Digest verified")
    return True


# Data
class PairsDataset(Dataset):
    def __init__(self, pairs, tokenizer, code2idx, max_len=512, return_idx=False):
        self.pairs = pairs
        self.tok = tokenizer
        self.code2idx = code2idx
        self.max_len = max_len
        self.return_idx = return_idx

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
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": y,
        }
        if self.return_idx:
            item["idx"] = i
        return item


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
        out = {"input_ids": ids, "attention_mask": ams, "labels": ys}
        if "idx" in batch[0]:
            out["idx"] = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
        return out

    return collate


# Metrics
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict:
    m = {}
    m["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    m["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m["micro_prec"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    m["macro_prec"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["micro_rec"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    m["macro_rec"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m["hamming"] = hamming_loss(y_true, y_pred)
    m["jaccard_micro"] = jaccard_score(y_true, y_pred, average="micro", zero_division=0)
    m["jaccard_macro"] = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        if y_true.sum() > 0:
            m["avg_prec_micro"] = average_precision_score(
                y_true, y_prob, average="micro"
            )
            m["avg_prec_macro"] = average_precision_score(
                y_true, y_prob, average="macro"
            )
            valid = [
                i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1
            ]
            m["roc_auc"] = (
                roc_auc_score(y_true[:, valid], y_prob[:, valid], average="macro")
                if valid
                else 0.0
            )
        else:
            m["avg_prec_micro"] = m["avg_prec_macro"] = m["roc_auc"] = 0.0
    except Exception as e:
        logging.debug(f"avg_prec/roc_auc calc error: {e}")
        m["avg_prec_micro"] = m["avg_prec_macro"] = m["roc_auc"] = 0.0

    freqs = y_true.sum(axis=0)
    if len(freqs) > 4:
        thr = np.percentile(freqs, 25)
        rare_idx = np.where(freqs <= thr)[0]
        if len(rare_idx):
            per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            m["rare_f1"] = float(np.mean(per_f1[rare_idx]))
        else:
            m["rare_f1"] = 0.0
    else:
        m["rare_f1"] = 0.0
    return m


# MIA features
def per_sample_features(logits: torch.Tensor, labels: torch.Tensor):
    with torch.no_grad():
        logits = logits.float()
        labels = labels.float()
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(
            dim=1
        )
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        maxc = probs.max(dim=1).values
        ent = -(probs * probs.log() + (1 - probs) * (1 - probs).log()).sum(
            dim=1
        ) / probs.size(1)
        top2 = torch.topk(probs, k=min(2, probs.size(1)), dim=1).values
        margin = (top2[:, 0] - top2[:, 1]) if top2.size(1) == 2 else top2[:, 0]
        l2 = torch.norm(logits, p=2, dim=1)
    return (
        bce.cpu().numpy(),
        maxc.cpu().numpy(),
        ent.cpu().numpy(),
        margin.cpu().numpy(),
        l2.cpu().numpy(),
    )


def auc_from_scores(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    larger_means_more_member: bool = True,
) -> float:
    y = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    s = np.concatenate([member_scores, nonmember_scores])
    if not larger_means_more_member:
        s = -s
    return float(roc_auc_score(y, s))


# Model loading
def load_classifier_matching_training(ckpt_dir: Path, device: torch.device):
    """Load classifier model matching the training configuration for integrity verification."""
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
    model = model.float()
    model.eval().to(device)

    _ = verify_digest(ckpt_dir, model)
    return model, tokenizer, code2idx


# Utils
def bytes_in_dir(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


@dataclass
class DPInfo:
    dp: Optional[bool] = None
    epsilon_target: Optional[float] = None
    delta: Optional[float] = None
    epsilon_spent_at_best: Optional[float] = None
    best_epoch: Optional[int] = None
    source: Optional[str] = (
        None  # "training_summary.json" | "metrics.csv" | "privacy_tracking.json" | None
    )


def _epsilon_from_training_summary(ckpt_dir: Path) -> Optional[DPInfo]:
    """Read ε at best μAP epoch from training_summary.json."""
    parent = ckpt_dir.parent
    summ_path = parent / "training_summary.json"
    if not summ_path.exists():
        return None
    try:
        summ = json.loads(summ_path.read_text())
    except Exception as e:
        logging.warning(f"Could not parse training_summary.json: {e}")
        return None

    info = DPInfo(source="training_summary.json")
    info.dp = bool(summ.get("dp")) if "dp" in summ else None
    info.epsilon_target = summ.get("epsilon_target", None)
    info.delta = summ.get("delta", None)

    best_epoch_keys = [
        "best_epoch_muAP",
        "best_epoch_muap",
        "best_epoch",
        "best_muap_epoch",
    ]
    best_eps_keys = ["epsilon_at_best", "eps_at_best", "epsilon_best", "best_epsilon"]

    for k in best_epoch_keys:
        if k in summ:
            info.best_epoch = summ[k]
            break

    for k in best_eps_keys:
        if k in summ:
            info.epsilon_spent_at_best = summ[k]
            break

    if (
        info.epsilon_spent_at_best is None
        and isinstance(summ.get("history"), list)
        and info.best_epoch is not None
    ):
        try:
            for rec in summ["history"]:
                if int(rec.get("epoch", -1)) == int(info.best_epoch) and (
                    "epsilon" in rec
                ):
                    info.epsilon_spent_at_best = rec["epsilon"]
                    break
        except Exception:
            pass

    return info


def _epsilon_from_metrics_csv(ckpt_dir: Path) -> Optional[DPInfo]:
    """Fallback: parse metrics.csv to find μAP-best row epsilon."""
    parent = ckpt_dir.parent
    csv_path = parent / "metrics.csv"
    if not csv_path.exists():
        return None
    try:
        import csv

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            best_ap, best_row = -1.0, None
            for row in reader:
                ap_micro = row.get("avg_prec_micro")
                if ap_micro is None or ap_micro == "":
                    continue
                ap_micro = float(ap_micro)
                if ap_micro > best_ap:
                    best_ap, best_row = ap_micro, row
            if best_row is None:
                return None
            info = DPInfo(source="metrics.csv")
            info.best_epoch = int(float(best_row.get("epoch", -1)))
            info.epsilon_spent_at_best = float(best_row.get("epsilon", "nan"))
            return info
    except Exception as e:
        logging.warning(f"Failed reading metrics.csv for ε_at_best: {e}")
        return None


def _epsilon_from_privacy_tracking(ckpt_dir: Path) -> Optional[DPInfo]:
    """Last-resort fallback: read last ε from privacy_tracking.json."""
    parent = ckpt_dir.parent
    priv_path = parent / "privacy_tracking.json"
    if not priv_path.exists():
        return None
    try:
        eps = None
        last_epoch = None
        with open(priv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                eps = obj.get("epsilon", eps)
                last_epoch = obj.get("epoch", last_epoch)
        if eps is None:
            return None
        info = DPInfo(source="privacy_tracking.json")
        info.epsilon_spent_at_best = float(eps)
        info.best_epoch = int(last_epoch) if last_epoch is not None else None
        return info
    except Exception as e:
        logging.warning(f"Failed reading privacy_tracking.json: {e}")
        return None


def gather_dp_info(ckpt_dir: Path) -> DPInfo:
    """Gather DP info with fallback chain for ε at best μAP epoch."""
    info = _epsilon_from_training_summary(ckpt_dir)
    if info and (info.epsilon_spent_at_best is not None):
        return info

    if info is None:
        info = _epsilon_from_metrics_csv(ckpt_dir)
        if info and (info.epsilon_spent_at_best is not None):
            logging.info("ε_at_best recovered from metrics.csv (μAP-best row).")
            return info

    fallback = _epsilon_from_privacy_tracking(ckpt_dir)
    if fallback:
        logging.warning(
            "Falling back to last ε from privacy_tracking.json (may differ from μAP-best epoch)."
        )
        return fallback

    return DPInfo(source=None)


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", type=Path, required=True, help="Path to .../best_model directory"
    )
    ap.add_argument("--pairs", type=Path, required=True, help="Eval split pairs (.pt)")
    ap.add_argument(
        "--thresholds_path",
        type=Path,
        required=True,
        help="Path to thresholds .npy to use",
    )
    ap.add_argument("--out_dir", type=Path, required=True)

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--verbosity", type=int, default=1, help="0=warn, 1=info, 2=debug")

    # MIA
    ap.add_argument(
        "--mi_member_pairs", type=Path, help="Member pairs (used in training)"
    )
    ap.add_argument(
        "--mi_nonmember_pairs",
        type=Path,
        help="Non-member pairs (not used in training)",
    )
    ap.add_argument("--mi_batch", type=int, default=48)

    args = ap.parse_args()
    setup_logging(args.verbosity)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, code2idx = load_classifier_matching_training(args.ckpt, device)
    if device.type == "cuda":
        model.half()

    idx2code = {v: k for k, v in code2idx.items()}

    # Load eval data
    eval_pairs = torch.load(args.pairs)
    dl = DataLoader(
        PairsDataset(eval_pairs, tokenizer, code2idx, args.max_len),
        batch_size=args.batch,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=4,
        pin_memory=True,
    )

    # Thresholds
    thresholds = np.load(args.thresholds_path)
    if thresholds.shape[0] != len(code2idx):
        raise ValueError(
            f"Thresholds length {thresholds.shape[0]} != num labels {len(code2idx)}"
        )
    logging.info(f"Using thresholds from: {args.thresholds_path}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.time()
    total_tokens = 0

    y_probs, y_true = [], []
    it = tqdm(dl, desc="Eval utility", unit="batch")
    amp_enabled = device.type == "cuda"
    autocast_ctx = torch.amp.autocast if amp_enabled else torch.cpu.amp.autocast
    autocast_args = ("cuda",) if amp_enabled else ("cpu",)
    with torch.inference_mode(), autocast_ctx(*autocast_args, dtype=torch.float16):
        for batch in it:
            ids = batch["input_ids"].to(device, non_blocking=True)
            ams = batch["attention_mask"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            out = model(input_ids=ids, attention_mask=ams, labels=labs)
            probs = torch.sigmoid(out.logits).detach().cpu().numpy()
            y_probs.append(probs)
            y_true.append(labs.cpu().numpy())
            total_tokens += ams.sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_mem_mb = (
        (torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )
    tokens_per_sec = total_tokens / max(1e-9, elapsed)
    ms_per_example = 1000.0 * elapsed / max(1, len(eval_pairs))
    logging.info(
        f"Throughput: {tokens_per_sec:.1f} tok/s | {ms_per_example:.2f} ms/example | peak_mem={peak_mem_mb:.1f} MB"
    )

    y_prob = np.vstack(y_probs)
    y_true_np = np.vstack(y_true)
    y_pred = (y_prob > thresholds.reshape(1, -1)).astype(int)

    metrics = compute_metrics(y_true_np, y_prob, y_pred)
    logging.info(
        f"Utility micro_f1={metrics['micro_f1']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"μAP={metrics.get('avg_prec_micro', 0.0):.4f} macroAUPRC={metrics.get('avg_prec_macro', 0.0):.4f}"
    )

    per_class = {}
    per_f1 = f1_score(y_true_np, y_pred, average=None, zero_division=0)
    per_p = precision_score(y_true_np, y_pred, average=None, zero_division=0)
    per_r = recall_score(y_true_np, y_pred, average=None, zero_division=0)
    for i in range(len(code2idx)):
        per_class[idx2code[i]] = {
            "f1": float(per_f1[i]),
            "prec": float(per_p[i]),
            "rec": float(per_r[i]),
        }

    adapter_bytes = bytes_in_dir(args.ckpt)
    model_size_mb = adapter_bytes / (1024**2)

    dp_info = gather_dp_info(args.ckpt)
    if dp_info.source == "privacy_tracking.json":
        logging.warning(
            "Reported ε is last logged value, not guaranteed to match μAP-best epoch."
        )
    elif dp_info.source is None:
        logging.info("No DP accounting files found alongside the checkpoint.")

    # Membership Inference (optional)
    mi_result = None
    if args.mi_member_pairs and args.mi_nonmember_pairs:
        logging.info(
            "Membership inference evaluation (per-feature + logistic ensemble)…"
        )
        mem_pairs = torch.load(args.mi_member_pairs)
        non_pairs = torch.load(args.mi_nonmember_pairs)

        N_member, N_non = len(mem_pairs), len(non_pairs)
        if N_member != N_non:
            rng = np.random.default_rng(42)
            if N_member > N_non:
                idx = rng.permutation(N_member)[:N_non]
                mem_pairs = [mem_pairs[i] for i in idx]
                logging.info(
                    f"Resampled members: {N_member} → {len(mem_pairs)} to match nonmembers {N_non}."
                )
            else:
                idx = rng.integers(0, N_member, size=N_non)
                mem_pairs = [mem_pairs[i] for i in idx]
                logging.info(
                    f"Upsampled members: {N_member} → {len(mem_pairs)} to match nonmembers {N_non}."
                )

        collate = make_collate_fn(tokenizer)
        dl_mem = DataLoader(
            PairsDataset(mem_pairs, tokenizer, code2idx, args.max_len),
            batch_size=args.mi_batch,
            shuffle=False,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True,
        )
        dl_non = DataLoader(
            PairsDataset(non_pairs, tokenizer, code2idx, args.max_len),
            batch_size=args.mi_batch,
            shuffle=False,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True,
        )

        amp_enabled = device.type == "cuda"
        autocast_ctx = torch.amp.autocast if amp_enabled else torch.cpu.amp.autocast
        autocast_args = ("cuda",) if amp_enabled else ("cpu",)

        def collect(dl, desc):
            feats = {"bce": [], "maxc": [], "ent": [], "margin": [], "l2": []}
            with torch.inference_mode(), autocast_ctx(
                *autocast_args, dtype=torch.float16
            ):
                for batch in tqdm(dl, desc=desc, unit="batch", leave=False):
                    ids = batch["input_ids"].to(device, non_blocking=True)
                    ams = batch["attention_mask"].to(device, non_blocking=True)
                    labs = batch["labels"].to(device, non_blocking=True)
                    out = model(input_ids=ids, attention_mask=ams, labels=labs)
                    bce, maxc, ent, margin, l2 = per_sample_features(out.logits, labs)
                    feats["bce"].append(bce)
                    feats["maxc"].append(maxc)
                    feats["ent"].append(ent)
                    feats["margin"].append(margin)
                    feats["l2"].append(l2)
            return {k: np.concatenate(v, axis=0) for k, v in feats.items()}

        feats_mem = collect(dl_mem, "MI collect (members)")
        feats_non = collect(dl_non, "MI collect (nonmembers)")

        res = {}
        res["auc_loss"] = auc_from_scores(-feats_mem["bce"], -feats_non["bce"], True)
        res["auc_conf"] = auc_from_scores(feats_mem["maxc"], feats_non["maxc"], True)
        res["auc_entropy"] = auc_from_scores(-feats_mem["ent"], -feats_non["ent"], True)
        res["auc_margin"] = auc_from_scores(
            feats_mem["margin"], feats_non["margin"], True
        )
        res["auc_l2"] = auc_from_scores(feats_mem["l2"], feats_non["l2"], True)

        X_mem = np.stack(
            [
                -feats_mem["bce"],
                feats_mem["maxc"],
                -feats_mem["ent"],
                feats_mem["margin"],
                feats_mem["l2"],
            ],
            axis=1,
        )
        X_non = np.stack(
            [
                -feats_non["bce"],
                feats_non["maxc"],
                -feats_non["ent"],
                feats_non["margin"],
                feats_non["l2"],
            ],
            axis=1,
        )
        y_mem = np.ones(X_mem.shape[0], dtype=np.int32)
        y_non = np.zeros(X_non.shape[0], dtype=np.int32)
        X = np.concatenate([X_mem, X_non], axis=0)
        y = np.concatenate([y_mem, y_non], axis=0)

        rng = np.random.default_rng(123)
        idx = rng.permutation(len(y))
        split = int(0.7 * len(y))
        tr, te = idx[:split], idx[split:]

        clf = LogisticRegression(max_iter=1000, n_jobs=None)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        auc_ens = float(roc_auc_score(y[te], proba))

        res["auc_ensemble"] = auc_ens
        res["n_member"] = int(len(mem_pairs))
        res["n_nonmember"] = int(len(non_pairs))
        res["train_size"] = int(len(tr))
        res["test_size"] = int(len(te))
        mi_result = res

    # Write JSON report
    out = {
        "ckpt": str(args.ckpt),
        "pairs": str(args.pairs),
        "thresholds_path": str(args.thresholds_path),
        "n_examples": len(eval_pairs),
        "n_labels": len(code2idx),
        "metrics": metrics,
        "per_class": per_class,
        "perf": {
            "tokens_per_sec": tokens_per_sec,
            "ms_per_example": ms_per_example,
            "peak_mem_mb": peak_mem_mb,
            "adapter_size_mb": model_size_mb,
        },
        "dp": {
            "dp": dp_info.dp,
            "epsilon_target": dp_info.epsilon_target,
            "delta": dp_info.delta,
            "epsilon_spent_at_best": dp_info.epsilon_spent_at_best,
            "best_epoch": dp_info.best_epoch,
            "source": dp_info.source,
        },
        "mi": mi_result,
    }

    out_fp = args.out_dir / "eval_metrics.json"
    with open(out_fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_fp}")

    print(
        f"Utility — μF1={metrics['micro_f1']:.4f} | μAP={metrics.get('avg_prec_micro', 0.0):.4f} | "
        f"Hamm={metrics['hamming']:.4f}"
    )
    if dp_info.epsilon_spent_at_best is not None:
        print(
            f"DP — ε@best={dp_info.epsilon_spent_at_best} (epoch={dp_info.best_epoch}, source={dp_info.source})"
        )
    elif dp_info.source:
        print(f"DP — ε unavailable (source={dp_info.source})")
    else:
        print("DP — no DP metadata found")

    if mi_result is not None:
        print(
            "MI — ensemble AUC={:.3f} | loss={:.3f} conf={:.3f} ent={:.3f} margin={:.3f} L2={:.3f}".format(
                mi_result["auc_ensemble"],
                mi_result["auc_loss"],
                mi_result["auc_conf"],
                mi_result["auc_entropy"],
                mi_result["auc_margin"],
                mi_result["auc_l2"],
            )
        )


if __name__ == "__main__":
    main()
