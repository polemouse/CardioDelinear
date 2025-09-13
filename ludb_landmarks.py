#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal LUDB viewer (single record/lead) using TorchECG benchmark dataset
and your project's Plotly helper (plot.py).

Goal: Display LUDB record 2, Lead II (lead index=1) with its ground-truth labels.
No training, no special cases. Copy & run.

Environment: macOS (MPS) / Linux / CPU all fine.

Usage:
  python ludb_quick_view.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# --- EDIT THESE PATHS IF NEEDED ---
CFG_DIR = "/Users/steven/Project/CardioDelinear/torch_ecg-0.0.31/benchmarks/train_unet_ludb"  # contains cfg.py & dataset.py
DB_DIR  = "/Users/steven/Project/CardioDelinear/data/ludb"  # folder where 1.dat/1.hea/1.ii ... live
REC_ID = 30          # LUDB record id (1-based)
LEAD_IDX = 1        # 0=Lead I, 1=Lead II, 2=Lead III, ...
OUT_HTML = "ludb_quick_view.html"

# --- Import benchmark cfg & dataset (so labels match training preprocessing) ---
sys.path.insert(0, CFG_DIR)
from cfg import TrainCfg as TrainCfg  # TorchECG benchmark config (dict-like)
from dataset import LUDB as LUDataset  # TorchECG benchmark dataset class
from torch_ecg.cfg import CFG as TCFG  # runtime CFG wrapper

# --- Import your Plotly helper ---
from plot import plot_interactive_ecg  # function: plot_interactive_ecg(sig, fs, spans, peaks=None, title, out_html)

# ---------- small helpers (pure NumPy, no side effects) ----------
def onehot_or_ids_to_ids(arr: np.ndarray) -> np.ndarray:
    """Convert labels to 1D integer IDs per sample.
    Accepts (T,C) one-hot or (T,) integer; also handles (C,T) by transposing if needed.
    """
    if arr.ndim == 2:
        # Decide if (T,C) or (C,T) by comparing with time length
        if arr.shape[0] < arr.shape[1]:  # likely (C,T)
            arr = arr.T                   # -> (T,C)
        return np.argmax(arr, axis=-1).astype(int)
    return arr.astype(int)

def runs_of_class(ids: np.ndarray, c: int) -> List[Tuple[int, int]]:
    """Return [(start,end), ...] contiguous spans where ids==c. end is exclusive."""
    spans: List[Tuple[int, int]] = []
    in_run = False
    s = 0
    for i, v in enumerate(ids.tolist()):
        if not in_run and v == c:
            in_run, s = True, i
        elif in_run and v != c:
            spans.append((s, i)); in_run = False
    if in_run:
        spans.append((s, len(ids)))
    return spans

def case_ins_get(d: dict, key: str, default=None):
    """Case-insensitive dict get (for class_map keys like 'p','N','t','i')."""
    for k, v in d.items():
        if str(k).lower() == str(key).lower():
            return v
    return default

# ---------- main logic ----------
# 1) Build cfg object and override db_dir & lazy
cfg = TCFG(TrainCfg)
cfg.db_dir = DB_DIR
cfg.lazy = True               # read full-length records; we will not sample windows here
fs = int(cfg.get("fs", 500))

# 2) Load dataset and fetch the whole record (signals + labels)
ds = LUDataset(cfg, training=False, lazy=True)
rec_idx = REC_ID - 1
signals, labels = ds.fdr[rec_idx]   # signals: (n_leads, T), labels: (n_leads, T, C) or (n_leads, T)

# 3) Pick Lead II (index=1); ensure length alignment between signal and labels
sig = signals[LEAD_IDX]
lab = labels[LEAD_IDX]
T = sig.shape[-1]
# If labels length differs slightly, trim/pad to match signal length
if lab.ndim == 2:
    # align along time axis
    if lab.shape[0] != T and lab.shape[1] == T:
        lab = lab.T  # make it (T,C)
    if lab.shape[0] > T:
        lab = lab[:T]
    elif lab.shape[0] < T:
        pad = np.zeros((T - lab.shape[0], lab.shape[1]), dtype=lab.dtype)
        lab = np.concatenate([lab, pad], axis=0)
else:  # (T,) integer ids
    if lab.shape[0] > T:
        lab = lab[:T]
    elif lab.shape[0] < T:
        pad = np.zeros((T - lab.shape[0],), dtype=lab.dtype)
        lab = np.concatenate([lab, pad], axis=0)

# 4) Convert labels to per-sample IDs, then to P/QRS/T spans using cfg.class_map
ids = onehot_or_ids_to_ids(lab)   # (T,)
cm = cfg.get("class_map", {"p":1, "N":2, "t":3, "i":0})
idx_p   = int(case_ins_get(cm, "p", 1))   # P wave
idx_qrs = int(case_ins_get(cm, "N", 2))   # QRS is stored as 'N' in LUDB config
idx_t   = int(case_ins_get(cm, "t", 3))   # T wave

spans: Dict[str, List[Tuple[int, int]]] = {
    "P":   runs_of_class(ids, idx_p),
    "QRS": runs_of_class(ids, idx_qrs),
    "T":   runs_of_class(ids, idx_t),
}

# 5) Plot with your Plotly helper (interactive HTML)
title = f"LUDB rec {REC_ID} | Lead II | fs={fs} Hz | length={T/fs:.1f}s"
plot_interactive_ecg(sig, fs, spans=spans, peaks=None, title=title, out_html=OUT_HTML)

print(f"Saved interactive HTML to: {OUT_HTML}")
