#!/usr/bin/env python
"""
Export teacher model outputs for knowledge distillation.

This script loads a trained teacher classifier and exports its predictions
(logits or probabilities) on synthetic data. These outputs are used to train
student models via knowledge distillation whilst preserving privacy guarantees.

The script verifies model integrity before export and saves outputs in
memory-efficient float16 format.
"""
import argparse, json, torch, tqdm, hashlib
from pathlib import Path
from typing import List, Tuple
from transformers import AutoConfig, LlamaTokenizerFast, LlamaForSequenceClassification
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

SPECIAL_TOKENS = {"additional_special_tokens": ["<|codes|>", "<|note|>"]}


def compute_trainable_digest(model: torch.nn.Module) -> str:
    """Compute SHA-256 over trainable params (LoRA + score head)."""
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


class TextOnlyDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        _, text = self.pairs[i]
        enc = self.tok(
            text or "",
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "idx": i,
        }


def collate_fn(pad_id):
    def f(batch):
        ids = pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
        )
        ams = pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        )
        idx = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ams, "idx": idx}

    return f


def load_teacher_matching_training(
    ckpt_dir: Path, num_labels: int, device: torch.device
):
    """Load teacher model matching training-time dtypes for digest verification."""
    base_name = (ckpt_dir / "base_model_name.txt").read_text().strip()

    tok = LlamaTokenizerFast.from_pretrained(ckpt_dir)
    tok.pad_token = tok.eos_token
    tok.add_special_tokens(SPECIAL_TOKENS)

    cfg = AutoConfig.from_pretrained(
        base_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        pad_token_id=tok.pad_token_id,
        use_cache=False,
    )

    base = LlamaForSequenceClassification.from_pretrained(
        base_name, config=cfg, torch_dtype=torch.float32
    )
    base.resize_token_embeddings(len(tok))

    head = ckpt_dir / "head.pt"
    if head.exists():
        state = torch.load(head, map_location="cpu")
        base.load_state_dict(state, strict=False)

    model = PeftModel.from_pretrained(base, ckpt_dir)
    model = model.float()
    model.eval().to(device)
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", type=Path, required=True)
    ap.add_argument("--pairs_pt", type=Path, required=True)
    ap.add_argument("--out_path", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--save_logits", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = torch.load(args.pairs_pt)
    code2idx = json.loads((args.teacher_ckpt / "code2idx.json").read_text())

    tok, teacher = load_teacher_matching_training(
        args.teacher_ckpt, num_labels=len(code2idx), device=device
    )
    verify_digest(args.teacher_ckpt, teacher)

    # Dataset / Loader
    ds = TextOnlyDataset(pairs, tok, args.max_seq_len)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tok.pad_token_id),
        num_workers=4,
        pin_memory=True,
    )

    N, C = len(ds), len(code2idx)
    store = torch.empty((N, C), dtype=torch.float16, device="cpu")

    use_amp = device.type == "cuda"
    autocast_ctx = torch.amp.autocast if use_amp else torch.cpu.amp.autocast
    autocast_args = ("cuda",) if use_amp else ("cpu",)

    with torch.inference_mode(), autocast_ctx(*autocast_args, dtype=torch.float16):
        pbar = tqdm.tqdm(dl, total=len(dl), desc="Export teacher")
        for batch in pbar:

            for k in ("input_ids", "attention_mask"):
                batch[k] = batch[k].to(device, non_blocking=True)
            _idx = batch.pop("idx")
            logits = teacher(**batch).logits.detach()
            buff = logits if args.save_logits else torch.sigmoid(logits)

            store[_idx] = buff.half().cpu()

    meta = {
        "source_pairs": str(args.pairs_pt),
        "teacher_ckpt": str(args.teacher_ckpt),
        "saved": "logits" if args.save_logits else "probs",
        "dtype": "float16",
        "max_seq_len": args.max_seq_len,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": store, "meta": meta}, args.out_path)
    print(f"Saved {tuple(store.shape)} to {args.out_path}")


if __name__ == "__main__":
    main()
