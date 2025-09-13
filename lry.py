"""
LRY file reader and annotation synchronizer

This module implements a `LRYRecord` class that can load a custom .lry
multichannel biosignal file and two annotation .txt files (R_on and R_off),
mirroring the ergonomics of a typical dataset class (e.g., LUDB-like).

Target environments:
- macOS on Apple Silicon (Metal acceleration where applicable)
- Ubuntu with NVIDIA GPU (prefer CUDA if used by downstream code)

No GPU dependencies are required here; we only use NumPy (numpy) and Pandas (pandas).

Key points from the LRY spec (per your description):
- Header is 1 KiB (1024 bytes).
- First 16 bytes: date/time string like "HH:MM.SS"; only seconds are trusted.
  Example: "17:30.16" => start_sec = 17*3600 + 30*60 + 16 = 63016.
- Byte 96-97 (little-endian, unsigned): sample period value "v" where period_ms = v/10.
  Thus fs (Hz) = 1000 / period_ms = 10000 / v.
  Examples: v=10 -> 1 ms -> 1 kHz; v=1 -> 0.1 ms -> 10 kHz; v=100 -> 10 ms -> 100 Hz.
- Byte 103: species code (0 mouse, 1 rat, 2 rabbit, 3 dog/cat/monkey, 4 cattle/horse).
- Byte 115: number of channels (n_channels).
- After 1 KiB, raw samples are interleaved int16 (little-endian) across channels:
  CH1_s0, CH2_s0, ..., CHn_s0, CH1_s1, CH2_s1, ...
- Scale: 200 raw units == 1 mV => 1 LSB == 1/200 mV == 5 µV.

Annotations (two .txt files):
- Each file contains two whitespace-separated columns per line:
  1) time_stamp (seconds, float, may include decimals; absolute time)
  2) gap_to_next_R_edge (ms, integer/float)
- R_on.txt contains only rising-edge events; R_off.txt contains only falling-edge events.
- The .lry start time is an integer second ("... .000"); annotations include decimals.
  Synchronization is done in absolute seconds by aligning annotation timestamps to
  the signal time base (start_sec + sample_index/fs).

This file exposes one public class:
- LRYRecord

Typical usage:

    from lry import LRYRecord

    rec = LRYRecord(
        path_lry="/path/to/file.lry",
        path_r_on="/path/to/R_on.txt",
        path_r_off="/path/to/R_off.txt",
    )
    rec.read()                  # read header + waveform into memory
    ann_df = rec.read_annotations()  # pandas DataFrame with time_s, edge, gap_ms, sample_index

    # Plot (requires your plot.py). You can pass a callable that consumes (t, y, ann_df, ...),
    # or we will try to auto-detect a function from plot.py.
    rec.plot(lead=0, start=None, duration=10.0)  # 10 seconds from file start

Design choices for robustness:
- We avoid assuming a specific function signature from plot.py. `plot()` will try several
  common function names. If none is found, it raises with a helpful message.
- Annotation lines are parsed with pandas and converted to absolute sample indices
  via `to_sample_index()` using rounding to the nearest integer sample.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from email import header
from typing import Callable, Dict, List, Optional, Tuple
import os
import io
import struct
import numpy as np
import pandas as pd

# Optional: set default file paths here so you can quickly switch inside code.
# These can be overridden by passing explicit paths to LRYRecord.__init__.
DEFAULT_LRY_PATH: Optional[str] = None
DEFAULT_R_ON_PATH: Optional[str] = None
DEFAULT_R_OFF_PATH: Optional[str] = None


# Map species code to readable string.
SPECIES_MAP: Dict[int, str] = {
    0: "mouse",
    1: "rat",
    2: "rabbit",
    3: "dog_cat_monkey",
    4: "cattle_horse",
}


@dataclass
class LRYRecord:
    """Container for an LRY recording and its annotations.

    Attributes
    ----------
    path_lry : str
        Path to the .lry data file.
    path_r_on : Optional[str]
        Path to the rising-edge annotation file (.txt). If None, skip.
    path_r_off : Optional[str]
        Path to the falling-edge annotation file (.txt). If None, skip.

    After calling `read()`, these fields are populated:
    fs : float
        Sampling rate in Hz.
    n_channels : int
        Number of channels.
    species_code : int
        Species code from header.
    species : str
        Human-readable species category.
    start_sec : int
        Absolute start time in seconds since 00:00:00 (integer seconds).
    signals : np.ndarray
        Shape (n_channels, n_samples), dtype float32, units mV.
    t_abs : np.ndarray
        Shape (n_samples,), absolute time in seconds (float64): start_sec + i/fs.
    leads : List[str]
        Channel names, defaults to ["CH1", ..., f"CH{n}"]
    """

    path_lry: str = field(default_factory=lambda: DEFAULT_LRY_PATH or "")
    path_r_on: Optional[str] = field(default_factory=lambda: DEFAULT_R_ON_PATH)
    path_r_off: Optional[str] = field(default_factory=lambda: DEFAULT_R_OFF_PATH)

    # Populated after read()
    fs: float = 0.0
    n_channels: int = 0
    species_code: int = -1
    species: str = "unknown"
    start_sec: int = 0
    signals: Optional[np.ndarray] = None
    t_abs: Optional[np.ndarray] = None
    leads: List[str] = field(default_factory=list)

    # Constants
    HEADER_SIZE: int = 1024  # bytes
    SCALE_PER_UNIT_MV: float = 1.0 / 200.0  # mV per raw unit

    def _read_header(self, f: io.BufferedReader) -> None:
        """Read and parse the 1 KiB header.

        Parameters
        ----------
        f : io.BufferedReader
            Opened binary file positioned at offset 0.

        Notes
        -----
        - First 16 bytes: ASCII date/time (we only trust seconds). Examples:
          b"17:30.16\x00..." -> parse HH:MM.SS. The integer seconds are used as start_sec.
        - Byte 96-97: sample period code (little-endian unsigned short).
        - Byte 103: species code (unsigned byte).
        - Byte 115: number of channels (unsigned byte).
        """
        header = f.read(self.HEADER_SIZE)
        if len(header) < self.HEADER_SIZE:
            raise ValueError("LRY header truncated: expected 1024 bytes")

        # First 16 bytes as ASCII, strip nulls and spaces
        dt_raw = header[0:16].decode(errors="ignore").strip("\x00 \t\r\n")

        # --- Parse time-of-day ONLY (ignore date), robust to both "HH:MM:SS" and "HH:MM.SS"
        # Prefer the last occurrence to avoid picking "DD/MM/YY" pieces by mistake.
        import re
        hh_i = mm_i = ss_i = 0  # defaults

        m = re.search(r'(\d{2}):(\d{2}):(\d{2})', dt_raw)  # matches "HH:MM:SS"
        if not m:
            m = re.search(r'(\d{2}):(\d{2})[.:](\d{2})', dt_raw)  # fallback "HH:MM.SS"
        if m:
            # Extract HH, MM, SS as integers
            hh_i, mm_i, ss_i = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        else:
            # As a final fallback, try the last 8 chars (may look like HH:MM:SS)
            last8 = dt_raw[-8:]
        try:
            h, mi, s = last8.split(":")
            hh_i, mm_i, ss_i = int(h), int(mi), int(s)
        except Exception:
            # keep zeros; we will log for debugging
            pass

        self.start_sec = hh_i * 3600 + mm_i * 60 + ss_i
        print(f"[LRY] raw_dt='{dt_raw}' -> start_sec={self.start_sec} (={hh_i:02d}:{mm_i:02d}:{ss_i:02d})")

        # Sample period code at bytes 96-97 (0-based indexing)
        v = struct.unpack_from("<H", header, 96)[0]  # unsigned short, little-endian
        if v == 0:
            raise ValueError("Invalid sample period code v=0 in header")
        period_ms = v / 10.0  # each unit is 0.1 ms
        self.fs = 1000.0 / period_ms  # Hz

        # Species code (byte 103)
        self.species_code = header[103]
        self.species = SPECIES_MAP.get(self.species_code, f"unknown_{self.species_code}")

        # Channel count (byte 115)
        self.n_channels = header[115]
        if self.n_channels <= 0:
            raise ValueError(f"Invalid n_channels={self.n_channels} in header")

        # Initialize channel names
        self.leads = [f"CH{i+1}" for i in range(self.n_channels)]

    def _read_data(self, f: io.BufferedReader) -> None:
        """Read interleaved int16 samples and convert to mV.

        Parameters
        ----------
        f : io.BufferedReader
            File opened in binary mode; file position should be after header.
        """
        # Determine data size
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        data_bytes = file_size - self.HEADER_SIZE
        if data_bytes <= 0:
            raise ValueError("No data after header in LRY file")

        if data_bytes % 2 != 0:
            raise ValueError("Data length is not a multiple of 2 bytes (int16)")

        total_samples = data_bytes // 2
        if total_samples % self.n_channels != 0:
            raise ValueError(
                f"Interleaved sample count {total_samples} not divisible by n_channels={self.n_channels}"
            )

        n_frames = total_samples // self.n_channels  # number of time points

        # Read raw int16 little-endian
        f.seek(self.HEADER_SIZE, os.SEEK_SET)
        raw = np.frombuffer(f.read(data_bytes), dtype=np.int16)
        # Reshape to (n_frames, n_channels), then transpose to (n_channels, n_frames)
        arr = raw.reshape((n_frames, self.n_channels)).astype(np.float32).T

        # Convert to mV
        arr *= self.SCALE_PER_UNIT_MV

        self.signals = arr  # (n_channels, n_frames), float32 mV
        # Build absolute time axis (float64 for precision)
        t = self.start_sec + np.arange(n_frames, dtype=np.float64) / self.fs
        self.t_abs = t

    def read(self) -> "LRYRecord":
        """Load header and waveform into memory.

        Returns
        -------
        LRYRecord
            Self (for chaining).
        """
        if not self.path_lry:
            raise ValueError("path_lry is empty; set path_lry or DEFAULT_LRY_PATH")
        with open(self.path_lry, "rb") as f:
            self._read_header(f)
            self._read_data(f)
        return self

    def _read_one_annotation(self, path: str, edge_label: str) -> pd.DataFrame:
        """Read a single annotation file as a DataFrame.

        Parameters
        ----------
        path : str
            File path to the annotation .txt.
        edge_label : str
            Label to attach to rows ("on" or "off").

        Returns
        -------
        pd.DataFrame
            Columns: [time_s(float), gap_ms(float), edge(str), sample_index(int)]
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Annotation file not found: {path}")

        # Use pandas to parse whitespace-delimited columns. We ignore empty/comment lines.
        df = pd.read_csv(
            path,
            sep='\s+',
            header=None,
            comment="#",
            names=["time_s", "gap_ms"],
            dtype={"time_s": float, "gap_ms": float},
        )
        df["edge"] = edge_label

        # Convert to sample indices aligned to absolute time base.
        # Annotations are absolute seconds (with decimals), while start_sec is an integer.
        # Sample index is rounded to nearest sample.
        if self.fs <= 0 or self.t_abs is None:
            raise RuntimeError("Call read() before read_annotations() to know fs/time base")

        # Vectorized conversion: sample_index = round((time_s - start_sec) * fs)
        sample_index = np.rint((df["time_s"].values - float(self.start_sec)) * float(self.fs)).astype(int)
        # Clip to valid range
        max_idx = len(self.t_abs) - 1
        sample_index = np.clip(sample_index, 0, max_idx)
        df["sample_index"] = sample_index

        return df

    def read_annotations(self) -> pd.DataFrame:
        """Read R_on and R_off annotation files and merge them chronologically.

        Returns
        -------
        pd.DataFrame
            Columns: [time_s, gap_ms, edge, sample_index], sorted by time_s.
        """
        dfs: List[pd.DataFrame] = []
        if self.path_r_on:
            dfs.append(self._read_one_annotation(self.path_r_on, "on"))
        if self.path_r_off:
            dfs.append(self._read_one_annotation(self.path_r_off, "off"))
        if not dfs:
            # No annotation files provided
            return pd.DataFrame(columns=["time_s", "gap_ms", "edge", "sample_index"]).astype(
                {"time_s": float, "gap_ms": float, "edge": str, "sample_index": int}
            )
        df = pd.concat(dfs, ignore_index=True)
        df.sort_values("time_s", inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
        return df

    def to_sample_index(self, time_s: float) -> int:
        """Map an absolute-second timestamp to a sample index.

        Parameters
        ----------
        time_s : float
            Absolute time in seconds.

        Returns
        -------
        int
            Nearest sample index within [0, n_samples-1].
        """
        if self.fs <= 0 or self.t_abs is None:
            raise RuntimeError("Call read() first to initialize fs/time base")
        idx = int(round((time_s - float(self.start_sec)) * float(self.fs)))
        return int(np.clip(idx, 0, len(self.t_abs) - 1))

    def build_edge_mask(self, window_ms: float = 4.0) -> Dict[str, np.ndarray]:
        """Build sparse 0/1 masks for R_on and R_off edges over the sample axis.

        Parameters
        ----------
        window_ms : float
            Width of the 1-valued window centered at each edge, in milliseconds.
            For example, 4 ms at 2 kHz covers ~8 samples total.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys "on" and/or "off" present if corresponding files were provided.
            Each value is a 1D array of shape (n_samples,), dtype uint8.
        """
        if self.signals is None or self.t_abs is None:
            raise RuntimeError("Call read() first to have signals/time base")

        n = self.signals.shape[1]
        masks: Dict[str, np.ndarray] = {}
        df = self.read_annotations()
        if df.empty:
            return masks

        half_w = max(0, int(round((window_ms / 1000.0) * self.fs / 2.0)))
        for edge_label in ["on", "off"]:
            sub = df[df["edge"] == edge_label]
            if sub.empty:
                continue
            m = np.zeros(n, dtype=np.uint8)
            for idx in sub["sample_index"].astype(int).tolist():
                lo = max(0, idx - half_w)
                hi = min(n, idx + half_w + 1)
                m[lo:hi] = 1
            masks[edge_label] = m
        return masks

    def plot(
        self,
        lead: int = 0,
        start: Optional[float] = None,
        duration: Optional[float] = None,
        show_edges: bool = True,
        plotter: Optional[Callable[..., None]] = None,
        **kwargs,
    ) -> None:
        """Plot one channel with annotations using your plot.py (Plotly-based).

        Parameters
        ----------
        lead : int
            Channel index to plot (0-based).
        start : Optional[float]
            Start time in seconds (absolute). If None, use file start.
        duration : Optional[float]
            Duration in seconds. If None, plot the full record.
        show_edges : bool
            Whether to include R_on/R_off markers in the plotter call when possible.
        plotter : Optional[Callable]
            If provided, we call `plotter(t, y, ann_df=..., fs=self.fs, lead_name=..., **kwargs)`.
            If None, we try to import `plot` and call one of common function names.
        kwargs : dict
            Extra keyword arguments forwarded to the plotter.

        Notes
        -----
        - We do *not* hardcode a plot signature; instead we prepare generic inputs:
          t (np.ndarray), y (np.ndarray), ann_df (pd.DataFrame), and metadata (fs, lead_name).
        - The function tries multiple fallbacks to integrate with your existing plot.py.
        """
        if self.signals is None or self.t_abs is None:
            raise RuntimeError("Call read() first to have signals/time base")
        if not (0 <= lead < self.n_channels):
            raise IndexError(f"lead index {lead} out of range [0, {self.n_channels-1}]")

        t = self.t_abs
        y = self.signals[lead]
        lead_name = self.leads[lead] if self.leads else f"CH{lead+1}"

        # Compute slicing window in sample indices
        if start is None:
            i0 = 0
        else:
            i0 = self.to_sample_index(start)
        if duration is None:
            i1 = len(t)
        else:
            i1 = min(len(t), i0 + int(round(duration * self.fs)))

    

        # Keep only annotations in [t0_abs, t1_abs]
        #ann_df = ann_df[(ann_df["time_s"] >= t0_abs) & (ann_df["time_s"] <= t1_abs)].copy()

        t_seg = t[i0:i1]
        y_seg = y[i0:i1]

        ann_df = self.read_annotations() if show_edges else pd.DataFrame()
        spans_all: List[float] = []
        spans_on: List[float] = []
        spans_off: List[float] = []
        if not ann_df.empty:
            # Clip annotations to the plotted window for cleaner visuals
            t_lo = t_seg[0]
            t_hi = t_seg[-1] if len(t_seg) > 0 else t[0]
            ann_df = ann_df[(ann_df["time_s"] >= t_lo) & (ann_df["time_s"] <= t_hi)].copy()
            spans_on = ann_df[ann_df["edge"] == "on"]["time_s"].astype(float).tolist()
            spans_off = ann_df[ann_df["edge"] == "off"]["time_s"].astype(float).tolist()
            spans_all = sorted(spans_on + spans_off)

        # If user passed a custom plotter, use it
        if plotter is not None:
            plotter(t_seg, y_seg, ann_df=ann_df, fs=self.fs, lead_name=lead_name, **kwargs)
            return

        # Otherwise try to import plot.py and find a reasonable function
        try:
            import importlib
            plot_mod = importlib.import_module("plot")
        except Exception as e:
            raise RuntimeError(
                "Could not import plot.py. Provide a `plotter` callable or ensure plot.py is in PYTHONPATH"
            ) from e

        # Candidate function names to try, from most to least specific.
        candidates = [
            "plot_interative_ecg",  # project-provided function name (as given)
            "plot_interactive_ecg", # common spelling fallback
            "plot_waveform_with_annotations",
            "plot_lry",
            "plot_waveform",
            "plot_series_with_events",
        ]

        func = None
        for name in candidates:
            if hasattr(plot_mod, name):
                func = getattr(plot_mod, name)
                break
        if func is None:
            raise RuntimeError(
                "plot.py loaded but no suitable function found. Expected one of: " + ", ".join(candidates)
            )
        # Special-case: your project plotter expects (sig, fs, spans, peaks=None, title=..., out_html=...)
        if func.__name__ in ("plot_interactive_ecg", "plot_interative_ecg"):
            # Build spans dict in **sample indices relative to the current segment**
            spans_dict = {"P": [], "QRS": [], "T": []}
            peaks_dict = {}
            if not ann_df.empty:
                # On/Off pairs become QRS spans; on-times also serve as R peaks if needed
                t0 = t_seg[0]
                
                on_times = ann_df[ann_df["edge"] == "on"]["time_s"].astype(float).tolist()
                off_times = ann_df[ann_df["edge"] == "off"]["time_s"].astype(float).tolist()
                print(f"[LRY] t0={t0}, on_times={on_times}, off_times={off_times}")
                # Pair on/off by time order within the window
                oi, fi = 0, 0
                r_idx_list = []
                while oi < len(on_times) and fi < len(off_times):
                    on_t = on_times[oi]
                    off_t = off_times[fi]
                    if off_t < on_t:
                        fi += 1
                        continue
                    on_i = int(round((on_t - t0) * self.fs))
                    off_i = int(round((off_t - t0) * self.fs))
                    if 0 <= on_i < len(y_seg) and 0 < off_i <= len(y_seg) and off_i > on_i:
                        spans_dict["QRS"].append((on_i, off_i))
                        r_idx_list.append(on_i)
                    oi += 1
                    fi += 1
                if r_idx_list:
                    peaks_dict["R"] = r_idx_list
            # Call your plotter with exact signature to avoid arg-collision
            func(y_seg, self.fs, spans_dict, peaks=peaks_dict or None, title=lead_name)
            return

        # Generic fallback: inspect signature and best-effort map annotations
        import inspect
        params = list(inspect.signature(func).parameters.values())
        param_names = [p.name for p in params]
        call_kwargs = {}
        if "fs" in param_names:
            call_kwargs["fs"] = self.fs
        if "lead_name" in param_names:
            call_kwargs["lead_name"] = lead_name
        if "ann_df" in param_names:
            call_kwargs["ann_df"] = ann_df
        if "spans" in param_names:
            call_kwargs.setdefault("spans", spans_all)
        if "spans_on" in param_names:
            call_kwargs.setdefault("spans_on", spans_on)
        if "spans_off" in param_names:
            call_kwargs.setdefault("spans_off", spans_off)
        for k, v in list(kwargs.items()):
            if k in param_names:
                call_kwargs[k] = v
        required_pos = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect._empty]
        args = [t_seg, y_seg]
        if len(required_pos) >= 3:
            third_name = required_pos[2].name
            if third_name in ("spans", "events", "markers"):
                args.append(spans_all)
            else:
                args.append(spans_all)
        if len(required_pos) >= 4:
            fourth_name = required_pos[3].name
            if fourth_name == "fs":
                args.append(self.fs)
            else:
                args.append(lead_name if fourth_name in ("lead", "lead_name", "channel") else self.fs)
        try:
            func(*args, **call_kwargs)
            return
        except Exception as e:
            raise RuntimeError(
                f"Failed to call {func.__name__} with inspected signature {param_names}. "
                f"Args provided lengths: {len(args)}; kwargs keys: {list(call_kwargs.keys())}. "
                f"Error: {type(e).__name__}: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to call {func.__name__} correctly. Error: {type(e).__name__}: {e}")
