#!/usr/bin/env python3
# main.py — TorchECG WFDB Landmark Export (no NeuroKit)
# Python 3.10+ recommended

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import wfdb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --- import torch_ecg (fail loudly if missing) ---
from torch_ecg.models.unets.ecg_unet import ECG_UNet
from torch_ecg.cfg import CFG
from copy import deepcopy
from torch_ecg.model_configs import ECG_UNET_VANILLA_CONFIG

config = deepcopy(ECG_UNET_VANILLA_CONFIG)
config.in_channels = 1  # Single lead
config.classes = ['P', 'QRS', 'T', 'background']  # Adjust for landmarks
config.n_leads = 1


# ----------------------
# CLI
# ----------------------
def get_args():
    p = argparse.ArgumentParser(description="TorchECG WFDB Landmark Export")
    p.add_argument("record", type=str,
                   help="WFDB record stem (no extension), e.g. data/ludb/1 or data/qtdb/sele0609")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to .pth weights (optional). If empty, uses random weights.")
    p.add_argument("--lead", type=str, default="II",
                   help="Preferred lead name (e.g., II). Falls back to first if not found.")
    p.add_argument("--window", type=int, default=20000,
                   help="Sliding inference window (samples) for long signals")
    p.add_argument("--overlap", type=int, default=1000,
                   help="Window overlap (samples)")
    p.add_argument("--plot_seconds", type=float, default=10.0,
                   help="Seconds to plot from start (0 = full trace)")
    p.add_argument("--outdir", type=str, default="outputs/ann",
                   help="Folder to write .ann")
    return p.parse_args()


# ----------------------
# Model builder
# ----------------------
def build_model(in_ch: int = 1, n_classes: int = 4, base_filters: int = 16) -> nn.Module:
    m = ECG_UNet(in_channels=in_ch, num_classes=n_classes, base_filters=base_filters)
    m.eval()
    return m


# ----------------------
# Lead selection & inference
# ----------------------
def pick_lead_index(sig_names: List[str], preferred: str = "II") -> int:
    return sig_names.index(preferred) if preferred in sig_names else 0

@torch.no_grad()
def infer_windowed(x: np.ndarray, model: nn.Module, device: torch.device,
                   win: int, overlap: int) -> np.ndarray:
    """Return per-sample class indices [T] by sliding window (handles long signals)."""
    T = len(x)
    if T <= win:
        X = torch.from_numpy(x[None, None, :]).float().to(device)
        logits = model(X)
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy()
        return pred

    step = max(1, win - overlap)
    n_steps = math.ceil((T - overlap) / step)
    vote = np.zeros((4, T), dtype=np.float32)  # 4 classes: BG/P/QRS/T

    for i in range(n_steps):
        s = i * step
        e = min(s + win, T)
        chunk = x[s:e]
        X = torch.from_numpy(chunk[None, None, :]).float().to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # [C, L]
        vote[:, s:e] += probs

    pred = vote.argmax(axis=0).astype(np.int64)
    return pred


# ----------------------
# Segmentation → segments → landmarks
# ----------------------
CLASS_MAP: Dict[int, str] = {0: "BG", 1: "P", 2: "QRS", 3: "T"}

@dataclass
class Segment:
    label: str
    start: int
    end: int

@dataclass
class BeatLM:
    Pon: Optional[int]; Ppk: Optional[int]; Poff: Optional[int]
    QRSon: Optional[int]; Rpk: Optional[int]; QRSoff: Optional[int]
    Ton: Optional[int]; Tpk: Optional[int]; Toff: Optional[int]

def segments_from_labels(labels: np.ndarray) -> List[Segment]:
    segs: List[Segment] = []
    N = len(labels); i = 0
    while i < N:
        c = int(labels[i]); lab = CLASS_MAP.get(c, "BG")
        if lab != "BG":
            j = i + 1
            while j < N and int(labels[j]) == c:
                j += 1
            segs.append(Segment(lab, i, j - 1))
            i = j
        else:
            i += 1
    return segs

def enforce_min_durations(segs: List[Segment], fs: int) -> List[Segment]:
    min_len = {"P": int(0.04 * fs), "QRS": int(0.06 * fs), "T": int(0.08 * fs)}
    return [s for s in segs if (s.end - s.start + 1) >= min_len.get(s.label, 1)]

def pick_peak(x: np.ndarray, s: int, e: int) -> int:
    if e <= s: return s
    sub = x[s:e+1]
    return int(s + int(np.argmax(np.abs(sub))))

def group_to_beats(segs: List[Segment], x: np.ndarray, fs: int) -> List[BeatLM]:
    """Anchor each 'beat' on QRS and attach nearest preceding P (≤300 ms) and following T (≤600 ms)."""
    beats: List[BeatLM] = []
    P = [s for s in segs if s.label == "P"]
    Q = [s for s in segs if s.label == "QRS"]
    T = [s for s in segs if s.label == "T"]

    p_idx = 0
    t_idx = 0
    for q in Q:
        # preceding P within 300 ms
        Pon = Ppk = Poff = None
        max_p_gap = int(0.30 * fs)
        while p_idx < len(P) and P[p_idx].end <= q.start:
            p_idx += 1
        candP = P[p_idx-1] if p_idx - 1 >= 0 and p_idx - 1 < len(P) else None
        if candP and (q.start - candP.end) <= max_p_gap:
            Pon, Poff = candP.start, candP.end
            Ppk = pick_peak(x, Pon, Poff)

        # following T within 600 ms
        Ton = Tpk = Toff = None
        max_t_gap = int(0.60 * fs)
        while t_idx < len(T) and T[t_idx].end < q.end:
            t_idx += 1
        candT = T[t_idx] if t_idx < len(T) else None
        if candT and (candT.start - q.end) <= max_t_gap:
            Ton, Toff = candT.start, candT.end
            Tpk = pick_peak(x, Ton, Toff)

        Rpk = pick_peak(x, q.start, q.end)
        beats.append(BeatLM(Pon, Ppk, Poff, q.start, Rpk, q.end, Ton, Tpk, Toff))
    return beats


# ----------------------
# WFDB .ann writer & plotting
# ----------------------
def write_ann(beats: List[BeatLM], fs: int, record_stem: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    samples: List[int] = []
    symbols: List[str] = []
    notes: List[str] = []

    def push(idx: Optional[int], tag: str):
        if idx is None: return
        samples.append(int(idx))
        symbols.append('"')  # generic annotation symbol
        notes.append(tag)

    for b in beats:
        push(b.Pon, "Pon");   push(b.Ppk, "Ppk");   push(b.Poff, "Poff")
        push(b.QRSon, "QRSon"); push(b.Rpk, "Rpk"); push(b.QRSoff, "QRSoff")
        push(b.Ton, "Ton");   push(b.Tpk, "Tpk");   push(b.Toff, "Toff")

    samples_np = np.array(samples, dtype=int)
    stem = os.path.basename(record_stem)
    wfdb.wrann(
        record_name=stem,
        extension="ann",
        sample=samples_np,
        symbol=symbols,
        aux_note=notes,
        fs=fs,
        write_dir=outdir,
    )
    return os.path.join(outdir, stem + ".ann")

def plot_with_ann(x: np.ndarray, fs: int, ann_stem: str, plot_seconds: float, title: str, lead_label: str):
    ann = wfdb.rdann(ann_stem, "ann")
    max_samp = len(x) if plot_seconds <= 0 else min(len(x), int(plot_seconds * fs))

    plt.figure(figsize=(12, 4))
    plt.plot(x[:max_samp], label=f"ECG ({lead_label})")
    for s, lab in zip(ann.sample, getattr(ann, "aux_note", [""] * len(ann.sample))):
        if s < max_samp:
            plt.axvline(s, linestyle="--", alpha=0.4)
            if lab:
                try:
                    plt.text(s, x[s], lab, rotation=90, va="bottom", fontsize=7)
                except Exception:
                    pass
    plt.title(title)
    plt.xlabel("Samples"); plt.ylabel("Normalized amplitude")
    plt.legend(loc="upper right"); plt.tight_layout(); plt.show()


# ----------------------
# Main
# ----------------------
def main():
    args = get_args()

    # Load WFDB record
    rec = wfdb.rdrecord(args.record)
    fs = int(rec.fs)
    lead_idx = pick_lead_index(rec.sig_name, preferred=args.lead)
    sig = rec.p_signal[:, lead_idx].astype("float32")

    # Normalize
    x = (sig - sig.mean()) / (sig.std() + 1e-6)

    # Build model & (optional) load weights
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = build_model(in_ch=1, n_classes=4, base_filters=16).to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[INFO] Loaded checkpoint: {args.checkpoint}")
    else:
        if args.checkpoint:
            print(f"[WARN] Checkpoint not found: {args.checkpoint} — using random weights (demo).")
        else:
            print("[INFO] No checkpoint passed — using random weights (demo).")

    # Inference
    pred = infer_windowed(x, model, device, win=args.window, overlap=args.overlap)

    # (Optional DEV HACK) — if you want to prove I/O before training, uncomment:
    # step = fs
    # for s in range(0, len(pred), step):
    #     e = min(s + int(0.1 * fs), len(pred))
    #     pred[s:e] = 2  # force some QRS (class 2) — REMOVE after training

    # Segments → clean → beats
    segs = segments_from_labels(pred)
    segs = enforce_min_durations(segs, fs)
    beats = group_to_beats(segs, x, fs)

    print(f"[DEBUG] segments={len(segs)} beats={len(beats)}")

    # Write .ann
    ann_path = write_ann(beats, fs, args.record, args.outdir)
    print(f"[INFO] Wrote annotations → {ann_path}")

    # Plot
    stem = os.path.basename(args.record)
    plot_with_ann(
        x, fs,
        ann_stem=os.path.join(args.outdir, stem),
        plot_seconds=args.plot_seconds,
        title=f"{stem} — predicted landmarks",
        lead_label=rec.sig_name[lead_idx],
    )


if __name__ == "__main__":
    main()
