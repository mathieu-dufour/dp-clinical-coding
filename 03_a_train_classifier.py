#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train clinical code classification models using LoRA fine-tuning.

This script trains multi-label ICD-9 code classifiers on clinical notes using
Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters. Supports both
standard and differentially private training regimes.

Features:
- Dual validation sets for early stopping and threshold optimisation
- Optional differential privacy via DP-SGD
- Knowledge distillation support for student-teacher training
- Automatic threshold calibration for optimal F1 scores
"""
import os, sys, time, json, logging, argparse, random, hashlib, math
from pathlib import Path
from datetime import timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch import amp

from transformers import (
    LlamaTokenizerFast,
    LlamaForSequenceClassification,
    AutoConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, PeftModel

try:
    from opacus import PrivacyEngine
    from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
    from opacus.grad_sample.grad_sample_module import GradSampleModule

    OPACUS_AVAILABLE = True
except Exception:
    OPACUS_AVAILABLE = False

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    hamming_loss,
    jaccard_score,
)

LOGFMT = "%(asctime)s — %(levelname)s — %(message)s"
SPECIAL_TOKENS = {"additional_special_tokens": ["<|codes|>", "<|note|>"]}


# Utilities
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    if OPACUS_AVAILABLE and isinstance(model, GradSampleModule):
        model = model._module
    return model


def compute_param_digest(model: torch.nn.Module, only_trainable: bool = True) -> str:
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        name = k.lower()
        if only_trainable and ("lora_" not in name) and ("score" not in name):
            continue
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class MultiLabelDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[List[str], str]],
        code2idx: Dict[str, int],
        tokenizer: LlamaTokenizerFast,
        max_len: int = 512,
    ):
        self.pairs = pairs
        self.code2idx = code2idx
        self.tok = tokenizer
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
        labels = torch.zeros(len(self.code2idx), dtype=torch.float32)
        for c in codes:
            j = self.code2idx.get(c)
            if j is not None:
                labels[j] = 1.0
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": labels,
            "row_idx": i,
        }


def make_collate_fn(tokenizer: LlamaTokenizerFast):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        ids = pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
        )
        ams = pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        )
        labels = torch.stack([b["labels"] for b in batch])
        row_idx = torch.tensor([b["row_idx"] for b in batch], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": ams,
            "labels": labels,
            "row_idx": row_idx,
        }

    return collate


def calculate_pos_weight(
    pairs: List[Tuple[List[str], str]], code2idx: Dict[str, int]
) -> torch.Tensor:
    label_counts = torch.zeros(len(code2idx))
    N = len(pairs)
    for codes, _ in pairs:
        for c in codes:
            j = code2idx.get(c)
            if j is not None:
                label_counts[j] += 1
    pos_weight = (N - label_counts) / (label_counts + 1.0)
    return torch.clamp(pos_weight, min=1.0, max=5.0)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, prefix=""
) -> Dict:
    m = {}
    m[f"{prefix}micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    m[f"{prefix}macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m[f"{prefix}weighted_f1"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    m[f"{prefix}micro_prec"] = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    m[f"{prefix}macro_prec"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    m[f"{prefix}micro_rec"] = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    m[f"{prefix}macro_rec"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    m[f"{prefix}hamming"] = hamming_loss(y_true, y_pred)
    m[f"{prefix}jaccard_micro"] = jaccard_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    m[f"{prefix}jaccard_macro"] = jaccard_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    try:
        if y_true.sum() > 0:
            m[f"{prefix}avg_prec_micro"] = average_precision_score(
                y_true, y_prob, average="micro"
            )  # μAP
            m[f"{prefix}avg_prec_macro"] = average_precision_score(
                y_true, y_prob, average="macro"
            )
            valid = [
                i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1
            ]
            m[f"{prefix}roc_auc"] = (
                roc_auc_score(y_true[:, valid], y_prob[:, valid], average="macro")
                if valid
                else 0.0
            )
        else:
            m[f"{prefix}avg_prec_micro"] = m[f"{prefix}avg_prec_macro"] = m[
                f"{prefix}roc_auc"
            ] = 0.0
    except Exception:
        m[f"{prefix}avg_prec_micro"] = m[f"{prefix}avg_prec_macro"] = m[
            f"{prefix}roc_auc"
        ] = 0.0
    return m


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


def get_soft_batch(
    soft_store_cpu: torch.Tensor,
    idxs: torch.Tensor,
    saved_kind: str,
    device: torch.device,
):
    if soft_store_cpu is None:
        return None
    soft = soft_store_cpu[idxs.cpu().numpy()].to(device, dtype=torch.float32)
    if saved_kind == "logits":
        return soft
    elif saved_kind == "probs":
        soft = torch.clamp(soft, 1e-7, 1 - 1e-7)
        return torch.logit(soft)
    else:
        raise ValueError(f"Unknown kind: {saved_kind}")


# Main
def main():
    ap = argparse.ArgumentParser()
    # Model / LoRA
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--use_gradient_checkpointing", action="store_true")

    # Data
    ap.add_argument("--train_pairs", type=Path, required=True)
    ap.add_argument(
        "--val_pairs",
        type=Path,
        required=True,
        help="Validation set used for early stopping",
    )
    ap.add_argument(
        "--threshold_val_pairs",
        type=Path,
        help="Validation set used for threshold tuning (defaults to --val_pairs)",
    )
    ap.add_argument("--max_seq_len", type=int, default=512)

    # Training
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1.5e-3)  # 1B default
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=8)

    # LR scheduler
    ap.add_argument(
        "--scheduler", type=str, default="none", choices=["none", "cosine", "linear"]
    )
    ap.add_argument("--warmup_ratio", type=float, default=0.1)

    # Privacy
    ap.add_argument("--with_dp", action="store_true")
    ap.add_argument("--epsilon", type=float, default=4.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Experiment / misc
    ap.add_argument(
        "--pipeline_type",
        type=str,
        choices=["DP-Synthetic", "DP-Distil", "DP-Small", "LoRA-No-DP"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_flash_attention", action="store_true")

    # Distillation (student)
    ap.add_argument("--distil_train", type=Path)
    ap.add_argument("--distil_val", type=Path)
    ap.add_argument("--distil_alpha", type=float, default=0.0)
    ap.add_argument("--distil_source", choices=["logits", "probs"], default=None)

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOGFMT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.out_dir / "train.log"),
        ],
        force=True,
    )

    if args.with_dp and not OPACUS_AVAILABLE:
        logging.error("DP requested but Opacus not available.")
        sys.exit(1)

    seed_everything(args.seed)

    # DDP
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=7200))
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load data
    train_pairs = torch.load(args.train_pairs)
    val_pairs = torch.load(args.val_pairs)
    thr_pairs = (
        torch.load(args.threshold_val_pairs) if args.threshold_val_pairs else val_pairs
    )

    all_codes = sorted(set(c for codes, _ in train_pairs for c in codes))
    code2idx = {c: i for i, c in enumerate(all_codes)}
    logging.info(
        f"Codes={len(code2idx)} Train={len(train_pairs)} Val(ES)={len(val_pairs)} Val(THR)={len(thr_pairs)}"
    )

    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name, token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    train_ds = MultiLabelDataset(train_pairs, code2idx, tokenizer, args.max_seq_len)
    val_ds = MultiLabelDataset(val_pairs, code2idx, tokenizer, args.max_seq_len)
    thr_ds = MultiLabelDataset(thr_pairs, code2idx, tokenizer, args.max_seq_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    collate = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(not distributed),
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Model
    is_dp = args.with_dp
    dtype = torch.float32 if is_dp else torch.float16
    attn_impl = (
        "sdpa"
        if is_dp
        else ("flash_attention_2" if args.use_flash_attention else "sdpa")
    )

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(code2idx),
        problem_type="multi_label_classification",
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    base_kwargs = dict(config=config, torch_dtype=dtype)
    if not is_dp:
        base_kwargs["attn_implementation"] = attn_impl

    base_model = LlamaForSequenceClassification.from_pretrained(
        args.model_name, **base_kwargs
    )
    base_model.resize_token_embeddings(len(tokenizer))

    if args.use_gradient_checkpointing and not is_dp:
        base_model.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing ENABLED (non-DP).")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg).to(device)

    for name, p in model.named_parameters():
        want_grad = ("lora_" in name) or ("score" in name)
        p.requires_grad = want_grad
        if want_grad and not is_dp:
            p.data = p.data.float()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parameters — total={total_params:,} trainable={trainable_params:,}")

    # Optimizer
    if is_dp:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0,
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=0.01,
        )

    use_amp = (not is_dp) and (device.type == "cuda")
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    pos_weight = calculate_pos_weight(train_pairs, code2idx).to(device)

    privacy_engine = None
    if is_dp:
        privacy_engine = PrivacyEngine(secure_mode=False, accountant="rdp")
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
            poisson_sampling=False,
        )
        logging.info(
            f"DP init: ε={args.epsilon}, δ={args.delta}, clip={args.max_grad_norm}"
        )

    if distributed:
        model = (
            DPDDP(model) if is_dp else torch.nn.parallel.DistributedDataParallel(model)
        )

    # LR scheduler
    scheduler = None
    if args.scheduler != "none":
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(max(1, args.warmup_ratio * total_steps))
        if args.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif args.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        logging.info(
            f"LR scheduler: {args.scheduler} | warmup_steps={warmup_steps} total_steps={total_steps}"
        )

    # Distillation blobs
    def _load_soft(path: Optional[Path]):
        if not path:
            return None, None
        blob = torch.load(path, map_location="cpu")
        return blob["data"], blob["meta"].get("saved", "logits")

    soft_train, saved_train = _load_soft(args.distil_train)
    soft_val, saved_val = _load_soft(args.distil_val)
    if args.distil_source:
        saved_train = saved_val = args.distil_source

    # Metrics logging
    metrics_file = None
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        metrics_file = open(args.out_dir / "metrics.csv", "w")
        metrics_file.write(
            ",".join(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "micro_f1",
                    "macro_f1",
                    "weighted_f1",
                    "micro_prec",
                    "macro_prec",
                    "micro_rec",
                    "macro_rec",
                    "hamming",
                    "jaccard_micro",
                    "jaccard_macro",
                    "avg_prec_micro",
                    "avg_prec_macro",
                    "roc_auc",
                    "epsilon",
                    "time_s",
                    "peak_mem_MB",
                    "avg_batch_ms",
                    "lr",
                ]
            )
            + "\n"
        )
        metrics_file.flush()

    # Train
    best_micro_ap = -1.0
    best_epoch = -1
    epsilon_at_best = None
    patience_ctr = 0
    start_time = time.time()

    logging.info("Start training")
    import tqdm

    for epoch in range(args.epochs):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        n_batches = 0
        peak_mem_mb = 0.0
        t0 = time.time()
        for batch in tqdm.tqdm(
            train_loader, total=len(train_loader), desc=f"Train e{epoch}"
        ):
            ids = batch["input_ids"].to(device, non_blocking=True)
            ams = batch["attention_mask"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            idxs = batch["row_idx"].to(device, non_blocking=True)

            soft_batch = (
                get_soft_batch(soft_train, idxs, saved_train, device)
                if soft_train is not None
                else None
            )

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with amp.autocast("cuda"):
                    out = model(input_ids=ids, attention_mask=ams, labels=labs)
                    logits = out.logits
                    bce = F.binary_cross_entropy_with_logits(
                        logits, labs, pos_weight=pos_weight
                    )
                    if soft_batch is not None:
                        mse = F.mse_loss(logits, soft_batch)
                        loss = (1.0 - float(args.distil_alpha)) * bce + float(
                            args.distil_alpha
                        ) * mse
                    else:
                        loss = bce
                scaler.scale(loss).step(optimizer)
                scaler.update()
            else:
                out = model(input_ids=ids, attention_mask=ams, labels=labs)
                logits = out.logits
                bce = F.binary_cross_entropy_with_logits(
                    logits, labs, pos_weight=pos_weight
                )
                if soft_batch is not None:
                    mse = F.mse_loss(logits, soft_batch)
                    loss = (1.0 - float(args.distil_alpha)) * bce + float(
                        args.distil_alpha
                    ) * mse
                else:
                    loss = bce
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            n_batches += 1
            if device.type == "cuda":
                peak_mem_mb = max(
                    peak_mem_mb, torch.cuda.max_memory_allocated(device) / (1024**2)
                )
                torch.cuda.reset_peak_memory_stats(device)

        train_loss /= max(1, n_batches)
        epoch_time = time.time() - t0

        model.eval()
        y_probs, y_true = [], []
        with torch.inference_mode():
            for batch in tqdm.tqdm(
                val_loader, total=len(val_loader), desc=f"Val e{epoch}"
            ):
                ids = batch["input_ids"].to(device, non_blocking=True)
                ams = batch["attention_mask"].to(device, non_blocking=True)
                labs = batch["labels"].to(device, non_blocking=True)
                out = model(input_ids=ids, attention_mask=ams, labels=labs)
                logits = out.logits
                y_probs.append(torch.sigmoid(logits).cpu().numpy())
                y_true.append(labs.cpu().numpy())
        y_prob = np.vstack(y_probs)
        y_true_np = np.vstack(y_true)

        val_loss = float(
            F.binary_cross_entropy(
                torch.tensor(y_prob, dtype=torch.float32),
                torch.tensor(y_true_np, dtype=torch.float32),
            ).item()
        )
        val_metrics = compute_metrics(y_true_np, y_prob, (y_prob > 0.5).astype(int))
        micro_ap = float(val_metrics["avg_prec_micro"])

        eps_spent = None
        if is_dp and hasattr(privacy_engine, "get_epsilon"):
            try:
                eps_spent = float(privacy_engine.get_epsilon(delta=args.delta))
            except Exception:
                eps_spent = None

        if metrics_file is not None:
            ms_per_batch = 1000.0 * epoch_time / max(1, len(train_loader))
            lr_now = optimizer.param_groups[0]["lr"]
            metrics_file.write(
                ",".join(
                    [
                        str(epoch),
                        f"{train_loss:.6f}",
                        f"{val_loss:.6f}",
                        f"{val_metrics['micro_f1']:.6f}",
                        f"{val_metrics['macro_f1']:.6f}",
                        f"{val_metrics['weighted_f1']:.6f}",
                        f"{val_metrics['micro_prec']:.6f}",
                        f"{val_metrics['macro_prec']:.6f}",
                        f"{val_metrics['micro_rec']:.6f}",
                        f"{val_metrics['macro_rec']:.6f}",
                        f"{val_metrics['hamming']:.6f}",
                        f"{val_metrics['jaccard_micro']:.6f}",
                        f"{val_metrics['jaccard_macro']:.6f}",
                        f"{micro_ap:.6f}",
                        f"{val_metrics['avg_prec_macro']:.6f}",
                        f"{val_metrics['roc_auc']:.6f}",
                        f"{(eps_spent if eps_spent is not None else 0.0):.6f}",
                        f"{epoch_time:.3f}",
                        f"{peak_mem_mb:.1f}",
                        f"{ms_per_batch:.2f}",
                        f"{lr_now:.6g}",
                    ]
                )
                + "\n"
            )
            metrics_file.flush()

        improved = micro_ap > best_micro_ap
        if improved:
            best_micro_ap = micro_ap
            best_epoch = epoch
            epsilon_at_best = eps_spent
            patience_ctr = 0

            best_dir = args.out_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)

            digest = compute_param_digest(unwrap_model(model), only_trainable=True)
            with (best_dir / "state_digest.json").open("w") as f:
                json.dump({"only_trainable_digest": digest}, f)

            (best_dir / "base_model_name.txt").write_text(args.model_name)
            with (best_dir / "code2idx.json").open("w") as f:
                json.dump(code2idx, f, indent=2)

            torch.save(unwrap_model(model).state_dict(), best_dir / "head.pt")

            thr_loader = DataLoader(
                thr_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            y_probs_thr, y_true_thr = [], []
            with torch.inference_mode():
                for batch in tqdm.tqdm(
                    thr_loader,
                    total=len(thr_loader),
                    desc="Calibrate thresholds (real val)",
                ):
                    ids = batch["input_ids"].to(device, non_blocking=True)
                    ams = batch["attention_mask"].to(device, non_blocking=True)
                    labs = batch["labels"].to(device, non_blocking=True)
                    out = model(input_ids=ids, attention_mask=ams, labels=labs)
                    logits = out.logits
                    y_probs_thr.append(torch.sigmoid(logits).cpu().numpy())
                    y_true_thr.append(labs.cpu().numpy())
            y_prob_thr = np.vstack(y_probs_thr)
            y_true_thr_np = np.vstack(y_true_thr)
            thresholds = find_optimal_thresholds(y_true_thr_np, y_prob_thr)

            thr_path = best_dir / "optimal_thresholds.npy"
            np.save(thr_path, thresholds)
            with (best_dir / "optimal_thresholds.csv").open("w") as f:
                f.write("code,threshold\n")
                inv = {v: k for k, v in code2idx.items()}
                for i, t in enumerate(thresholds):
                    f.write(f"{inv[i]},{t:.4f}\n")

            logging.info(f"Saved best_model and thresholds → {best_dir}")

        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            logging.info(
                f"Early stopping at epoch={epoch} (best μAP at epoch={best_epoch})"
            )
            break

    total_train_time = time.time() - start_time
    avg_epoch_time = total_train_time / max(1, best_epoch + 1)

    # Write training_summary.json
    summary = {
        "pipeline_type": args.pipeline_type,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_epoch_1based": (best_epoch + 1 if best_epoch >= 0 else None),
        "best_val_micro_ap": best_micro_ap,
        "epsilon_at_best": epsilon_at_best,
        "delta": (args.delta if args.with_dp else None),
        "with_dp": bool(args.with_dp),
        "n_total_params": int(total_params),
        "n_trainable_params": int(trainable_params),
        "total_train_time_s": float(total_train_time),
        "avg_epoch_time_s": float(avg_epoch_time),
        "thresholds_path": str(args.out_dir / "best_model" / "optimal_thresholds.npy"),
    }
    with (args.out_dir / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if metrics_file is not None:
        metrics_file.close()


if __name__ == "__main__":
    main()
