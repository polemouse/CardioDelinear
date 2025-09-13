#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict LUDB landmarks with the exact training config from TorchECG 0.0.31.

What this script does:
1) Imports the training cfg.py you used (so the model architecture matches training).
2) Builds ECG_UNET with that cfg and loads your checkpoint safely.
3) Reads LUDB locally via WFDB, runs inference on a chosen record/time window.
4) Converts per-sample predictions (bg/P/QRS/T) to spans and peaks (fiducials).
5) Plots waveform + colored spans + peak markers.

"""


import os
from pyexpat import model
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest import signals
from xml.etree.ElementPath import find

from cfg import TrainCfg as TrainCfg #local copied cfg
from dataset import LUDB as LUDataset  # TorchECG benchmark dataset class
from cfg import CFG as TCFG  # runtime CFG wrapper

from ludb_landmarks import DB_DIR, REC_ID
import plotly.graph_objects as go

import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------- Setting ----------
CALCULATE_PEAKS = False

# ---------- User paths (EDIT THESE IF NEEDED) ----------
#CKPT_PATH = "/Users/steven/.cache/torch_ecg/saved_models/BestModel_ECG_UNET_LUDB_epoch95_09-09_11-37_metric_0.93.pth.tar"
#CKPT_PATH = "/Users/steven/.cache/torch_ecg/saved_models/BestModel_ECG_UNET_LUDB_epoch179_09-10_13-57_metric_0.95.pth.tar"
CKPT_PATH = "/Users/steven/.cache/torch_ecg/saved_models/BestModel_ECG_UNET_LUDB_epoch120_09-12_02-12_metric_0.92.pth.tar"

CFG_DIR   = "/Users/steven/Project/CardioDelinear/torch_ecg-0.0.31/benchmarks/train_unet_ludb"  # contains cfg.py
LOCAL_LUDB_DIR = "/Users/steven/Project/CardioDelinear/data/ludb/data"  # where 1.hea/1.dat etc live

# ---------- Inference settings ----------
RECORD_ID = 30          # which LUDB record to visualize
TARGET_LEAD = 1        # prefer lead II (index 1)
START_SECONDS = 0      # start time
WINDOW_SECONDS = 10      # seconds to visualize; will be adjusted to cfg.input_len if present
#USE_SINGLE_LEAD = True  # whether to use a single lead for inference, overwrtien by cfg 
# ---------- Optional dependency for reading .hea/.dat ----------
try:
    import wfdb  # pip install wfdb
except Exception:
    wfdb = None


# ----------------- Utilities -----------------
def get_device() -> torch.device:
    """Prefer Apple MPS; fallback to CUDA or CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def add_python_path(dirpath: str) -> None:
    """Prepend a directory to sys.path so we can import cfg.py from it."""
    p = str(Path(dirpath).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def import_training_cfg() -> object:
    """
    Import the training config module at CFG_DIR/cfg.py and return the *instance* used in training.
    We prioritize typical instance names and explicitly avoid returning the CFG class itself.
    """
    import importlib
    add_python_path(CFG_DIR)
    mod = importlib.import_module("cfg")  # the cfg.py you trained with

    # Priority: common instance names in TorchECG benchmarks
    candidate_names = ["TrainCfg", "ModelCfg", "train_cfg", "model_cfg", "config", "cfg"]
    for name in candidate_names:
        obj = getattr(mod, name, None)
        # Skip if it's the CFG *class* (we only want an instance / dict-like)
        if obj is not None and not (isinstance(obj, type) and getattr(obj, "__name__", "") == "CFG"):
            return obj

    # Fallback: search any attribute that is an instance of torch_ecg.cfg.CFG
    try:
        from torch_ecg.cfg import CFG as _CFG
        for k, v in mod.__dict__.items():
            if isinstance(v, _CFG):
                return v
    except Exception:
        pass

    raise RuntimeError(
        "Could not find a config *instance* in cfg.py. "
        "Expected one of TrainCfg/ModelCfg/train_cfg/model_cfg/config/cfg."
    )



def to_cfg_dict(cfg_obj: object) -> dict:
    """
    Convert TorchECG CFG (likely a 'CFG' object) or module to a plain dict-like object.
    We only access attributes we care about; missing ones will be handled with defaults.
    """
    # Many TorchECG CFGs behave like objects with attributes. We'll use getattr with defaults.
    class _CfgView(dict):
        def __getattr__(self, k):
            return self.get(k, None)
    out = _CfgView()
    for key in dir(cfg_obj):
        if key.startswith("_"):
            continue
        try:
            val = getattr(cfg_obj, key)
        except Exception:
            continue
        # Keep simple types and nested dict-like objects
        out[key] = val
    return out


def safe_load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    Safely load a state_dict from a checkpoint.
    - Allowlist TorchECG CFG so torch>=2.6 weights_only path can unpickle safely.
    - Fall back to weights_only=False for older checkpoints (only if you trust the file).
    - Strip only known prefixes ('module.', 'model.', 'net.') when they actually exist.
    """
    from torch import serialization as ts
    try:
        # Allowlist TorchECG's CFG for weights_only=True safe unpickling
        from torch_ecg.cfg import CFG
        if hasattr(ts, "add_safe_globals"):
            ts.add_safe_globals([CFG])
    except Exception:
        pass

    # 1) Try safe (weights_only) path first; if it fails, fall back.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")  # torch>=2.6 defaults to weights_only=True
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 2) Locate a state_dict inside the checkpoint
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        if state_dict is None and ckpt:
            # Maybe it's already a plain dict of tensors
            vals = list(ckpt.values())
            if vals and all(isinstance(v, torch.Tensor) for v in vals):
                state_dict = ckpt
    if state_dict is None:
        raise RuntimeError("No state_dict found in checkpoint.")

    # 3) Strip ONLY when the prefix actually exists
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    return cleaned


def build_model_from_cfg(cfg_obj: object, fallback_in_channels=12, fallback_classes=("bg","P","QRS","T")) -> torch.nn.Module:
    """
    Build ECG_UNET exactly like training when possible.

    - Only pass the config if we have an *instance* (not the CFG class).
    - Also sync n_leads/in_channels and classes from cfg when available.
    - Fall back to manual kwargs if constructor doesn't accept a config.
    """
    import inspect
    from collections.abc import Mapping
    from torch_ecg.models import ECG_UNET

    # Convert to a dict-view for convenience (reads attributes safely)
    cfg = to_cfg_dict(cfg_obj) if not isinstance(cfg_obj, type) else {}

    sig = inspect.signature(ECG_UNET)
    params = sig.parameters
    kwargs = {}

    # Pass the whole config only if we did NOT receive a class type
    is_cfg_instance = not isinstance(cfg_obj, type)

    if is_cfg_instance:
        if "config" in params:
            kwargs["config"] = cfg_obj
        elif "cfg" in params:
            kwargs["cfg"] = cfg_obj

    # Sync input channels
    n_leads = cfg.get("n_leads") or cfg.get("in_channels") or fallback_in_channels
    if "in_channels" in params:
        kwargs["in_channels"] = int(n_leads)
    elif "n_leads" in params:
        kwargs["n_leads"] = int(n_leads)

    # Sync classes
    classes = None
    if "classes" in cfg and isinstance(cfg["classes"], (list, tuple)):
        classes = list(cfg["classes"])
    if classes is None:
        classes = list(fallback_classes)

    if "classes" in params:
        kwargs["classes"] = classes
    elif "out_channels" in params:
        kwargs["out_channels"] = len(classes)
    elif "n_classes" in params:
        kwargs["n_classes"] = len(classes)

    print("ECG_UNET init kwargs:", kwargs)
    model = ECG_UNET(**kwargs)
    return model


def read_ludb_local(rec_id: int) -> Tuple[np.ndarray, int]:
    """
    Read LUDB record via TorchECG benchmark dataset (no WFDB used here).
    Returns:
        signals: (n_leads, n_samples), float
        fs: sampling rate in Hz
    """
    # Build a fresh cfg for the dataset.
    # NOTE: DB_DIR must be the parent folder that contains the `data/` subfolder.
    cfg = TCFG(TrainCfg)
    cfg.db_dir = DB_DIR      # e.g. "/Users/steven/.../data/ludb" (NOT ".../data/ludb/data")
    cfg.lazy = True          # load full-length records; no window sampling

    # Create dataset reader and fetch the record (1-based rec_id -> 0-based index)
    ds = LUDataset(cfg, training=False, lazy=True)
    if not (1 <= rec_id <= len(ds.records)):
        raise ValueError(f"rec_id {rec_id} out of range 1..{len(ds.records)}")
    signals, labels = ds.fdr[rec_id - 1]   # signals: (n_leads, T)

    # fs: prefer reader.fs if present, otherwise fall back to cfg.fs
    fs = int(getattr(ds.reader, "fs", cfg.get("fs", 500)))

    # Ensure dtype and layout are exactly (n_leads, T)
    signals = signals.astype(float)

    return signals, fs



def ensure_length(sig_2d: np.ndarray, want_len: int) -> np.ndarray:
    """
    Center-crop or zero-pad along time axis to exact length.
    C++ analogy: make the array exactly 'want_len' in the time dimension.
    """
    n_leads, n = sig_2d.shape
    if n == want_len:
        return sig_2d
    if n > want_len:
        s = (n - want_len) // 2
        return sig_2d[:, s:s+want_len]
    out = np.zeros((n_leads, want_len), dtype=sig_2d.dtype)
    s = (want_len - n) // 2
    out[:, s:s+n] = sig_2d
    return out


def zscore(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Per-lead z-score normalization."""
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return (x - m) / (s + eps)


def logits_to_pred_classes(logits: torch.Tensor, n_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert model output to per-sample class indices.
    Supports both layouts:
      - (B, C, T): argmax over dim=1
      - (B, T, C): argmax over dim=2
    Returns (T,) for batch size 1.
    """
    # Unwrap common containers
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if isinstance(logits, dict):
        for v in logits.values():
            if torch.is_tensor(v):
                logits = v
                break
    if not torch.is_tensor(logits):
        raise TypeError("Model output is not a Tensor.")

    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits (1,*,*), got {tuple(logits.shape)}")

    B, D1, D2 = logits.shape  # could be (1, C, T) or (1, T, C)

    # Heuristic to detect layout:
    # - If we know n_classes, prefer the axis that equals n_classes.
    # - Otherwise, if the last dim is small (<= 32), assume (B, T, C).
    if n_classes is not None:
        if D1 == n_classes and D2 != n_classes:
            # (B, C, T)
            pred = torch.argmax(logits, dim=1).squeeze(0)  # (T,)
            return pred.detach().cpu().numpy()
        if D2 == n_classes and D1 != n_classes:
            # (B, T, C)
            pred = torch.argmax(logits, dim=2).squeeze(0)  # (T,)
            return pred.detach().cpu().numpy()

    # Fallback heuristic
    if D2 <= 32 and D1 > D2:
        # treat as (B, T, C)
        pred = torch.argmax(logits, dim=2).squeeze(0)
    else:
        # treat as (B, C, T)
        pred = torch.argmax(logits, dim=1).squeeze(0)

    return pred.detach().cpu().numpy()



def resize_labels_nearest(pred: np.ndarray, target_len: int) -> np.ndarray:
    """Nearest-neighbor resize for 1D label sequence (keeps integers)."""
    if pred.size == target_len:
        return pred
    src = np.arange(pred.size, dtype=float)
    tgt = np.linspace(0, pred.size - 1, num=target_len)
    idx = np.clip(np.round(tgt).astype(int), 0, pred.size - 1)
    return pred[idx]


def runs_of_class(pred: np.ndarray, cls: int) -> List[Tuple[int, int]]:
    """Return [(start, end), ...] contiguous runs where pred[t] == cls (end exclusive)."""
    idx = np.where(pred == cls)[0]
    if idx.size == 0:
        return []
    brk = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[brk + 1]]
    ends   = np.r_[idx[brk], idx[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def pick_peak(signal_1d: np.ndarray, start: int, end: int, mode: str = "pos") -> int:
    """Pick a peak index within [start, end) (max or abs-max)."""
    seg = signal_1d[start:end]
    if seg.size == 0:
        return start
    off = int(np.argmax(np.abs(seg))) if mode == "abs" else int(np.argmax(seg))
    return start + off


def plot_with_annotations(sig_1d: np.ndarray, fs: int,
                          spans: Dict[str, List[Tuple[int,int]]],
                          peaks: Dict[str, List[int]],
                          title: str):
    """Plot waveform, colored spans, and vertical peak markers."""
    t = np.arange(sig_1d.size) / float(fs)
    plt.figure(figsize=(12, 5))
    plt.plot(t, sig_1d, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(title)

    colors = {"P": (0.2, 0.6, 1.0, 0.2), "QRS": (1.0, 0.2, 0.2, 0.2), "T": (0.2, 1.0, 0.2, 0.2)}
    for name, segs in spans.items():
        col = colors.get(name, (0.5, 0.5, 0.5, 0.15))
        for s, e in segs:
            plt.axvspan(s / fs, e / fs, color=col)

    ymax = np.nanmax(sig_1d) if sig_1d.size else 1.0
    for name, idxs in peaks.items():
        for i in idxs:
            plt.axvline(i / fs, linestyle="--", linewidth=0.8)
            plt.text(i / fs, ymax * 0.9, name, ha="center", va="top", fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_interactive_ecg(sig, fs, spans, peaks=None, title="ECG", out_html="ecg_interactive.html"):
    """
    Create an interactive ECG plot with P/QRS/T shaded spans and optional peak markers.

    Parameters
    ----------
    sig : 1D np.ndarray
        Single-lead signal.
    fs : int
        Sampling rate (Hz).
    spans : dict
        {"P":[(s,e),...], "QRS":[(s,e),...], "T":[(s,e),...]} in sample indices.
    peaks : dict or None
        e.g., {"R":[idx1, idx2, ...]} in sample indices.
    title : str
        Figure title.
    out_html : str
        Output HTML file path.
    """
    t = np.arange(sig.size) / float(fs)

    # --- fixed colors ---
    class_colors = {"P":"#FF7F0E", "QRS":"#D62728", "T":"#2CA02C"}  # orange/red/green

    fig = go.Figure()

    # 1) signal line (Scattergl is fast for large arrays)
    fig.add_trace(go.Scattergl(
        x=t, y=sig, name="Signal", mode="lines",
        line=dict(width=1.2, color="#1f77b4"),
        hovertemplate="t=%{x:.3f}s<br>amp=%{y:.4f}<extra>Signal</extra>"
    ))

    # 2) add shaded spans as shapes, and add invisible traces for legend toggling
    #    (click legend to hide/show a class)
    for cls in ("P","QRS","T"):
        col = class_colors[cls]
        segs = spans.get(cls, [])
        # An invisible trace only for legend entry and toggling
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color=col, width=12),
            name=cls, showlegend=True
        ))
        # Add shapes for each segment
        for s,e in segs:
            fig.add_shape(
                type="rect",
                x0=s/fs, x1=e/fs,
                y0=min(sig), y1=max(sig),
                fillcolor=col, opacity=0.20, line_width=0, layer="below"
            )

    # 3) optional peak markers
    if peaks:
        for name, idxs in peaks.items():
            if not idxs: 
                continue
            tt = np.array(idxs)/fs
            yy = np.interp(tt, t, sig)
            fig.add_trace(go.Scatter(
                x=tt, y=yy, mode="markers",
                name=f"{name} peaks", marker=dict(symbol="x", size=8, color="#000000"),
                hovertemplate=f"{name} @ %{{x:.3f}}s<br>amp=%{{y:.4f}}<extra></extra>"
            ))

    fig.update_layout(
        title={"text": title, "y": 0.98, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (a.u.)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=110, b=50),
    )

    # Better initial view
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(fixedrange=False)

    fig.write_html(out_html, include_plotlyjs="cdn", auto_open=True)  # opens in browser

def majority_filter_labels(pred: np.ndarray, k: int = 11) -> np.ndarray:
    """
    Simple majority filter on label sequence to remove pepper noise.
    k should be odd; O(T*k) but T=2000 so it's fine.
    """
    assert k % 2 == 1, "k must be odd"
    out = pred.copy()
    r = k // 2
    max_cls = int(pred.max())
    for i in range(pred.size):
        s = max(0, i - r); e = min(pred.size, i + r + 1)
        window = pred[s:e]
        # bincount is fast for small integer labels
        counts = np.bincount(window, minlength=max_cls + 1)
        # tie-break: keep current label
        winner = int(np.argmax(counts))
        out[i] = winner
    return out


def enforce_min_durations(pred: np.ndarray, idx_bg: int, idx_p: int, idx_qrs: int, idx_t: int,
                          fs: int,
                          min_ms_p: int = 40, min_ms_qrs: int = 60, min_ms_t: int = 80) -> np.ndarray:
    """
    Remove segments shorter than physiologic minima by reassigning them to background.
    You can later make this smarter (merge into neighbor majority).
    """
    out = pred.copy()
    mins = {
        idx_p:   int(round(min_ms_p   * fs / 1000.0)),
        idx_qrs: int(round(min_ms_qrs * fs / 1000.0)),
        idx_t:   int(round(min_ms_t   * fs / 1000.0)),
    }
    for cls, min_len in mins.items():
        for s, e in runs_of_class(out, cls):
            if e - s < min_len:
                out[s:e] = idx_bg
    return out

def main():
    device = get_device()
    print("Using device:", device)

    # 1) Import training cfg and build model accordingly
    cfg_obj = import_training_cfg()
    cfg = to_cfg_dict(cfg_obj)
    print("cfg keys (subset):", sorted([k for k in cfg.keys() if k in {"n_leads","in_channels","classes","fs","input_len"}]))

    model = build_model_from_cfg(cfg_obj).to(device)

    # 2) Load checkpoint weights
    state = safe_load_state_dict(CKPT_PATH)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 3) Read LUDB locally
    signals, fs = read_ludb_local(RECORD_ID)

    # 4) Determine effective settings from cfg
    n_leads = int(cfg.get("n_leads") or cfg.get("in_channels") or signals.shape[0])
    # prefer cfg.fs if present (resample here if mismatch — omitted for brevity)
    fs_eff = int(cfg.get("fs") or fs)
    want_len = int(cfg.get("input_len") or (WINDOW_SECONDS * fs_eff))

     # ---------- 5) Slice window EXACTLY as training expects ----------
    # Use fs_eff (from cfg) and want_len (= cfg.input_len) to align with training.
    b = max(0, int(START_SECONDS * fs_eff))   # left index in samples
    e = b + want_len                          # exclusive right index

    # Left-aligned crop+pad to exact length
    win = signals[:, b:min(e, signals.shape[1])]            # (n_leads, <=want_len)
    if win.shape[1] < want_len:
        pad = np.zeros((win.shape[0], want_len - win.shape[1]), dtype=win.dtype)
        win = np.concatenate([win, pad], axis=1)            # (n_leads, want_len)

    # (Optional) light band-pass to mimic training preprocessing
    win_bp = win.copy()
    try:
        from scipy.signal import butter, filtfilt
        bpf_b, bpf_a = butter(2, [0.5/(fs_eff/2.0), 40/(fs_eff/2.0)], btype="band")
        win_bp = np.vstack([filtfilt(bpf_b, bpf_a, lead) for lead in win_bp])
    except Exception:
        pass  # skip if scipy not installed

    # ---------- Single-lead vs multi-lead input ----------
    # Decide single-lead from your flag OR cfg
    USE_SINGLE_LEAD = cfg.get("use_single_lead", False)

    # Choose a target lead (e.g., Lead II = index 1) safely
    lead_idx = TARGET_LEAD if TARGET_LEAD < win_bp.shape[0] else 0

    if USE_SINGLE_LEAD:
        # Single-lead model expects 1 input channel
        # 1) pick the lead you want
        sig = win_bp[lead_idx]  # (T,)
        # 2) normalize THIS lead only (per-lead z-score)
        sig_norm = zscore(sig, axis=None)  # (T,)
        # 3) shape to (1, T) then to (1, 1, T)
        x_np = sig_norm[None, :]                          # (1, T)
        x = torch.from_numpy(x_np).float().unsqueeze(0)   # (1, 1, T)
    else:
        # Multi-lead model expects n_leads input channels
        # 1) keep all leads
        # 2) normalize per-lead along time axis
        win_norm = zscore(win_bp, axis=1)                 # (n_leads, T)
        # 3) shape to (1, n_leads, T)
        x = torch.from_numpy(win_norm).float().unsqueeze(0)  # (1, n_leads, T)
        # Still keep a reference lead for plotting
        sig = win_bp[lead_idx]  # (T,)

    # Move to device
    x = x.to(device)
    
    model.eval()  # Set model to evaluation mode
    torch.set_grad_enabled(False)
    
    with torch.no_grad():
        logits = model(x)  # TorchECG UNet may output (1, T, C) or (1, C, T)
    print("logits shape:", tuple(logits.shape))

    num_classes = len(cfg.get("classes")) if isinstance(cfg.get("classes"), (list, tuple)) else 4
    pred = logits_to_pred_classes(logits, n_classes=num_classes)  # auto-detect (B,T,C) vs (B,C,T)
    # pred length already equals want_len; no resize needed.

    # Debug: class histogram
    uni, cnt = np.unique(pred, return_counts=True)
    print("pred unique:", dict(zip(uni.tolist(), cnt.tolist())))


    # ---------- 6) Map cfg.classes (channel order) -> indices, then smooth & dur constraints ----------
    # Important: indices come from the ORDER in cfg.classes, e.g. ['p','N','t','i']
    names = [str(c).lower() for c in (cfg.get("classes") or [])]
    def find(name: str, default: int) -> int:
        return names.index(name) if name in names else default

    idx_bg  = find('i', 0)   # background (inactive)
    idx_p   = find('p', 1)   # P wave
    idx_qrs = find('n', 2)   # 'N' (normal) == QRS
    idx_t   = find('t', 3)   # T wave
    print("class indices -> bg:", idx_bg, "P:", idx_p, "QRS:", idx_qrs, "T:", idx_t)

    # Temporal smoothing to remove pepper noise
    pred_smooth = majority_filter_labels(pred, k=11)  # increase to 15/21 if still noisy

    # Enforce physiologic minimum durations (ms); too-short segments -> background
    pred_smooth = enforce_min_durations(
        pred_smooth,
        idx_bg=idx_bg, idx_p=idx_p, idx_qrs=idx_qrs, idx_t=idx_t,
        fs=fs_eff,
        min_ms_p=40, min_ms_qrs=70, min_ms_t=80,  # tweak as needed
    )

    # Recompute spans on the smoothed labels
    spans = {
        "P":   runs_of_class(pred_smooth, idx_p),
        "QRS": runs_of_class(pred_smooth, idx_qrs),
        "T":   runs_of_class(pred_smooth, idx_t),
    }


    # 7) Spans and peaks
    spans = {}
    if idx_p   is not None: spans["P"]   = runs_of_class(pred, idx_p)
    if idx_qrs is not None: spans["QRS"] = runs_of_class(pred, idx_qrs)
    if idx_t   is not None: spans["T"]   = runs_of_class(pred, idx_t)

    peaks = {"R": [], "P": [], "T": []}
    for (s, e) in spans.get("QRS", []):
        peaks["R"].append(pick_peak(sig, s, e, mode="pos"))
    for (s, e) in spans.get("P", []):
        peaks["P"].append(pick_peak(sig, s, e, mode="abs"))
    for (s, e) in spans.get("T", []):
        peaks["T"].append(pick_peak(sig, s, e, mode="abs"))

    # Print quick stats
    uni, cnt = np.unique(pred, return_counts=True)
    print("pred unique:", dict(zip(uni.tolist(), cnt.tolist())))
    print("QRS spans (first 6):", [ (round(s/fs_eff,3), round(e/fs_eff,3)) for (s,e) in spans.get("QRS", []) ][:6])

    # --- Map class indices from training cfg, then build spans ---

    # Your cfg shows:
    #   classes: ['p','N','t','i']
    #   class_map or mask_class_map: {'p':1, 'N':2, 't':3, 'i':0}
    # We'll read that mapping case-insensitively and get the indices we need.
    class_map = cfg.get("mask_class_map") or cfg.get("class_map") or {}
    cm = {str(k).lower(): int(v) for k, v in class_map.items()} if isinstance(class_map, dict) else {}

    # Indices for fiducials:
    # 'p' -> P wave class
    # 'N' (normal) -> QRS class
    # 't' -> T wave class
    # 'i' -> background (inactive)
    idx_p   = cm.get("p", 1)   # default to 1 if missing
    idx_qrs = cm.get("n", 2)   # note: 'N' in cfg, we use lower 'n' here
    idx_t   = cm.get("t", 3)
    idx_bg  = cm.get("i", 0)

    # Build contiguous spans (start, end) for P/QRS/T using predicted labels
    spans = {
        "P":   runs_of_class(pred, idx_p),
        "QRS": runs_of_class(pred, idx_qrs),
        "T":   runs_of_class(pred, idx_t),
    }


    # 8) Plot
    title = f"LUDB rec {RECORD_ID} lead {lead_idx+1} | {START_SECONDS}-{START_SECONDS+WINDOW_SECONDS}s \n"
    #plot_with_annotations(sig, fs_eff, spans=spans, peaks=peaks, title=title)
    if CALCULATE_PEAKS:
        plot_interactive_ecg(sig, fs_eff, spans=spans, peaks=peaks, title=title,out_html="ludb_interactive.html")
    else:
        plot_interactive_ecg(sig, fs_eff, spans=spans, peaks=None, title=title, out_html="ludb_interactive.html")

if __name__ == "__main__":
    main()
