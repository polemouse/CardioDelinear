import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple

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