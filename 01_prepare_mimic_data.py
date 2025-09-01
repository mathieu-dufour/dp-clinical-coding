#!/usr/bin/env python
# 01_prepare_mimic_data.py
"""
Prepare MIMIC-III discharge summary data for top-50 ICD-9 multi-label classification.

This script processes raw MIMIC-III data to create training, validation, and test sets
for clinical code prediction whilst ensuring no data leakage.

Outputs:
  - data/real/real_train_codes_notes.pt
  - data/real/real_val_codes_notes.pt
  - data/real/real_test_codes_notes.pt
  - data/real/prep_log.csv (dataset statistics)

Key processing steps:
- Filters to discharge summary notes only
- Retains admissions with ≥1 of the top-50 most frequent ICD-9 codes
- Creates deterministic 80/10/10 train/validation/test split by admission
- Removes explicit ICD-9 code mentions from note text to prevent label leakage
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


from utils.clean_synth_notes import clean_pairs_in_memory

LOGFMT = "%(asctime)s — %(levelname)s — %(message)s"


# I/O helpers
def load_tables(note_path: Path, diag_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    notes = pd.read_csv(
        note_path,
        usecols=["ROW_ID", "SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
        dtype={
            "ROW_ID": "Int64",
            "SUBJECT_ID": "Int64",
            "HADM_ID": "Int64",
            "CATEGORY": "string",
            "TEXT": "string",
        },
        low_memory=False,
    )
    diag = pd.read_csv(
        diag_path,
        usecols=["HADM_ID", "ICD9_CODE"],
        dtype={"HADM_ID": "Int64", "ICD9_CODE": "string"},
        low_memory=False,
    )
    return notes, diag


def write_prep_log(out_dir: Path, **counts) -> None:
    df = pd.DataFrame([counts])
    fp = out_dir / "prep_log.csv"
    df.to_csv(fp, index=False)
    logging.info("prep_log.csv written → %s", fp)


# Transform helpers
def filter_discharge(notes: pd.DataFrame) -> pd.DataFrame:
    df = notes[notes["CATEGORY"] == "Discharge summary"].copy()
    df = df.dropna(subset=["HADM_ID"])
    df["HADM_ID"] = df["HADM_ID"].astype("Int64")
    return df


def top_k_codes(diag: pd.DataFrame, k: int) -> List[str]:
    return diag["ICD9_CODE"].value_counts().nlargest(k).index.tolist()


def hadm_to_codes(diag: pd.DataFrame, keep_codes: List[str]) -> Dict[int, List[str]]:
    d = diag[diag["ICD9_CODE"].isin(keep_codes)].copy()
    d = d.dropna(subset=["HADM_ID"])
    d["HADM_ID"] = d["HADM_ID"].astype("Int64")
    return (
        d.groupby("HADM_ID")["ICD9_CODE"]
        .apply(lambda s: sorted(set(s.dropna().tolist())))
        .to_dict()
    )


def split_admissions(
    all_hadm: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ids, valtest_ids = train_test_split(
        all_hadm, test_size=0.20, random_state=seed, shuffle=True
    )
    rng = np.random.RandomState(seed)
    rng.shuffle(valtest_ids)
    half = len(valtest_ids) // 2
    val_ids = valtest_ids[:half]
    test_ids = valtest_ids[half:]
    if len(val_ids) != len(test_ids):
        val_ids = valtest_ids[: half + 1]
        test_ids = valtest_ids[half + 1 :]
    return (
        np.array(train_ids, dtype=int),
        np.array(val_ids, dtype=int),
        np.array(test_ids, dtype=int),
    )


def build_pairs(
    discharge: pd.DataFrame,
    hadm_ids: np.ndarray,
    hadm2codes: Dict[int, List[str]],
) -> List[Tuple[List[str], str]]:
    subset = discharge[discharge["HADM_ID"].astype(int).isin(hadm_ids)].copy()
    out: List[Tuple[List[str], str]] = []
    for _, row in subset.iterrows():
        hadm = int(row["HADM_ID"])
        codes = hadm2codes.get(hadm, [])
        text = row["TEXT"] if isinstance(row["TEXT"], str) else ""
        out.append((codes, text))
    return out


# Main
def main():
    ap = argparse.ArgumentParser(
        description="Prepare (codes, text) PT files for top-50 ICD-9."
    )
    ap.add_argument("--note_events_csv", type=Path, default=Path("data/NOTEEVENTS.csv"))
    ap.add_argument(
        "--diagnoses_csv", type=Path, default=Path("data/DIAGNOSES_ICD.csv")
    )
    ap.add_argument("--out_dir", type=Path, default=Path("data/real"))
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOGFMT)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading tables…")
    notes, diag = load_tables(args.note_events_csv, args.diagnoses_csv)

    logging.info("Filtering to discharge summaries…")
    discharge = filter_discharge(notes)
    n_discharge = len(discharge)
    logging.info("Discharge summaries: %d rows", n_discharge)

    logging.info("Selecting top-%d ICD-9 codes…", args.top_k)
    keep_codes = top_k_codes(diag, args.top_k)
    hadm2codes = hadm_to_codes(diag, keep_codes)

    logging.info("Keeping only admissions with >=1 top-%d code…", args.top_k)
    keep_hadm = np.array(sorted(int(h) for h in hadm2codes.keys()))
    discharge_topk = discharge[discharge["HADM_ID"].astype(int).isin(keep_hadm)].copy()
    n_discharge_topk = len(discharge_topk)
    logging.info(
        "Discharge summaries with top-%d codes: %d", args.top_k, n_discharge_topk
    )

    all_hadm = discharge_topk["HADM_ID"].astype(int).unique()
    logging.info("Eligible admissions: %d", len(all_hadm))

    logging.info("Splitting admissions 80/10/10 with equal-sized val/test…")
    train_ids, val_ids, test_ids = split_admissions(all_hadm, seed=args.seed)
    logging.info(
        "Split admissions — train=%d, val=%d, test=%d",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    logging.info("Building (codes, text) pairs…")
    train_pairs = build_pairs(discharge_topk, train_ids, hadm2codes)
    val_pairs = build_pairs(discharge_topk, val_ids, hadm2codes)
    test_pairs = build_pairs(discharge_topk, test_ids, hadm2codes)

    logging.info("Cleaning notes by removing any ICD-9 code mentions…")
    train_pairs = clean_pairs_in_memory(train_pairs, quiet=True)
    val_pairs = clean_pairs_in_memory(val_pairs, quiet=True)
    test_pairs = clean_pairs_in_memory(test_pairs, quiet=True)

    out_train = args.out_dir / "real_train_codes_notes.pt"
    out_val = args.out_dir / "real_val_codes_notes.pt"
    out_test = args.out_dir / "real_test_codes_notes.pt"
    torch.save(train_pairs, out_train)
    torch.save(val_pairs, out_val)
    torch.save(test_pairs, out_test)
    logging.info("Saved PT files to %s", args.out_dir)

    write_prep_log(
        args.out_dir,
        top_k=args.top_k,
        n_discharge=int(n_discharge),
        n_discharge_topk=int(n_discharge_topk),
        n_train=len(train_pairs),
        n_val=len(val_pairs),
        n_test=len(test_pairs),
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
