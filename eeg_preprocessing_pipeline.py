# -*- coding: utf-8 -*-
"""
EEG Pre-processing Pipeline for BrainVision (32-electrode) Data
================================================================
Experiment: Interoception / Breath-Hold study
Segments  : LB (baseline), Intero, BH 1-4, Recov (recovery)
Output    : Spectral band-power Excel files at window, segment, and region levels

This script is organised in VSCode-style cells (#%%).
Cell 1 contains ALL imports, configuration, and function definitions.
Subsequent cells follow an interactive inspect-first workflow:

  Cell 1 — Imports, config, ALL function definitions
  Cell 2 — Load raw + list annotations + segment inspection  (BREAKPOINT)
  Cell 3 — Visualize RAW signal segment by segment
  Cell 4 — Preprocessing (Steps 1-7)
  Cell 5 — Visualize CLEANED signal segment by segment
  Cell 6 — Spectral power computation (all segments)
  Cell 7 — Aggregation & Excel export

Author : auto-generated pipeline
Date   : 2026-02-09

EXECUTION SCHEDULE (per subject):
=================================
1. Cell 1  - Run once per session (loads functions)
2. Cell 2  - Edit SUJETO and EEG_FILE, then run
3. Cells 3a-3e - Inspect raw data, mark BAD sections
4. Cell 4  - Run preprocessing (takes 2-5 min for ICA)
5. Cells 5a-5d - Inspect cleaned data, mark residual artifacts
6. Cell 6  - Review summary, save cleaned data
7. Cell 7  - Compute spectral power
8. Cell 8  - Export per-subject Excel
9. Cell 10 - Generate topomaps (optional)

After all subjects:
10. Cell 9 - Aggregate all subjects into global Excel
"""

#%%
# =============================================================================
# CELL 1 — IMPORTS · CONFIGURATION · FUNCTION DEFINITIONS
# =============================================================================
# Nothing is *executed* here other than defining constants, functions, and
# performing the minimal side-effect-free imports.
# =============================================================================

# ---------------------------------------------------------------------------
# 1.1  IMPORTS
# ---------------------------------------------------------------------------
import os
import sys
import json
import copy
import logging
import warnings
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import mne
from mne.preprocessing import ICA

# Optional fast-channel-detection library
try:
    from pyprep.find_noisy_channels import NoisyChannels
    HAS_PYPREP = True
except ImportError:
    HAS_PYPREP = False

# ---------------------------------------------------------------------------
# 1.2  CONFIGURATION CONSTANTS
# ---------------------------------------------------------------------------

# --- Paths ----------------------------------------------------------------
BASE_DIR   = Path(r"C:\Users\marag\Desktop\eeg_bh")
INPUT_DIR  = BASE_DIR / "participantes_eeg_bh"
OUTPUT_DIR = BASE_DIR / "output"
FIF_DIR    = OUTPUT_DIR / "fif_steps"
QC_DIR     = OUTPUT_DIR / "qc"
EXCEL_DIR  = OUTPUT_DIR / "excels"

# --- Subject list ---------------------------------------------------------
# Set to None for auto-discovery, or e.g. [3, 4, 5] for a manual subset.
SUBJECT_LIST: Optional[List[int]] = None

# --- Channel configuration -------------------------------------------------
EXPECTED_CHANNELS: List[str] = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9",
    "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8",
    "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8",
    "Fp2",
]
EOG_CHANNEL: str = "TP9"            # mastoid used as EOG proxy
ACQUISITION_REFERENCE: str = "FCz"  # nominal (actual varies per subject)

# EEG channels for spectral output (all expected minus EOG)
EEG_CHANNELS: List[str] = [ch for ch in EXPECTED_CHANNELS if ch != EOG_CHANNEL]

# --- Region mapping for spatial aggregation --------------------------------
REGION_MAP: Dict[str, List[str]] = {
    "frontal":   ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
                  "FC1", "FC2", "FC5", "FC6", "FT9", "FT10"],
    "central":   ["C3", "C4", "Cz"],
    "parietal":  ["CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8"],
    "occipital": ["O1", "O2", "Oz"],
    "temporal":  ["T7", "T8"],
}

# --- Filtering parameters --------------------------------------------------
NOTCH_FREQ: float      = 50.0       # Hz  (line noise)
NOTCH_HARMONICS: bool  = True       # also notch at 100 Hz
BANDPASS_LOW: float    = 1.0        # Hz  (high-pass)
BANDPASS_HIGH: float   = 40.0       # Hz  (low-pass)

# --- ICA parameters --------------------------------------------------------
ICA_METHOD: str                      = "fastica"
ICA_N_COMPONENTS: Optional[float]    = 0.999999  # variance-based (avoids unstable mixing matrix)
ICA_RANDOM_STATE: int                = 42
ICA_MAX_ITER: int                    = 1000

# --- Bad-channel detection --------------------------------------------------
USE_PYPREP: bool           = False
BAD_CHAN_ZSCORE_THRESH: float = 4.0   # robust z-score on variance
FLATLINE_THRESH_UV: float  = 0.5     # uV - channels with std < this

# --- Artifact annotation ---------------------------------------------------
# Peak-to-peak threshold per 1-s window (uV). Increase if too many segments
# are flagged as artifacts (200 is conservative; 300-400 is more lenient).
# Set to 0 or very high (e.g., 9999) to disable automatic artifact detection.
ARTIFACT_PTP_THRESH_UV: float = 500.0   # Conservative: rely more on manual marking
AUTO_ARTIFACT_DETECTION: bool = True    # Set to False to skip automatic detection

# --- Spectral analysis -----------------------------------------------------
WELCH_N_FFT: int      = 2048
WELCH_N_OVERLAP: int   = 1024
PSD_FMIN: float        = 1.0
PSD_FMAX: float        = 40.0

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# --- Windowing / epoching ---------------------------------------------------
WINDOW_SEC: float       = 30.0     # window length in seconds
LB_MAX_SEC: float       = 300.0    # LB: keep first 5 min max
RECOV_MAX_SEC: float    = 300.0    # Recov: keep first 5 min max
BH_TRIM_END_SEC: float  = 10.0     # BH: drop last 10 s

# --- BAD window rejection ---------------------------------------------------
# A window is rejected only if the fraction of BAD time exceeds this threshold.
# 0.0 = reject if ANY overlap (old behaviour, very aggressive)
# 0.25 = reject if >25% of the window is BAD (recommended)
# 1.0 = never reject (not recommended)
BAD_OVERLAP_THRESHOLD: float = 0.25

# --- Baseline correction ---------------------------------------------------
# We do NOT apply ERP-style baseline correction.  For spectral analysis of
# long windows, baseline subtraction would distort power estimates.
# Optional linear detrend per window can be enabled below.
APPLY_DETREND: bool = False

# --- Segment-label expectations (soft; used only for warnings) -------------
# Intero: 1 segment (2 markers), BH: 4 segments (8 markers)
EXPECTED_MARKER_COUNTS: Dict[str, int] = {
    "LB": 2, "Recov": 2, "Intero": 2, "BH": 8,
}

# --- File saving flags ------------------------------------------------------
# Set to False to reduce disk usage (only final cleaned data will be saved)
SAVE_INTERMEDIATE_FIF: bool = False   # Save .fif after each preprocessing step
SAVE_EPOCHS_FIF: bool       = False   # Save per-segment Epochs .fif
OVERWRITE_FIF: bool         = True

# --- Label normalisation map ------------------------------------------------
_LABEL_NORMALIZE: Dict[str, str] = {
    "lb":       "LB",
    "intero":   "Intero",
    "bh":       "BH",
    "recov":    "Recov",
    "recovery": "Recov",
}


# ---------------------------------------------------------------------------
# 1.3  LOGGER (object creation only - handlers added at runtime in Cell 2)
# ---------------------------------------------------------------------------
logger = logging.getLogger("eeg_bh_pipeline")
logger.propagate = False  # prevent duplicate messages from root logger


def setup_logging(log_dir: Path = QC_DIR, console_level: int = logging.WARNING) -> None:
    """
    Configure file (DEBUG) + optional console logging.

    By default console only shows WARNING+, so cell output stays clean.
    Detailed logs go to the file for later review.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocess_log.txt"

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console: only warnings by default (keep output clean)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File: full debug
    fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.debug(f"Logging initialised  ->  {log_file}")


# ---------------------------------------------------------------------------
# 1.4  DIRECTORY HELPERS
# ---------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    for d in (OUTPUT_DIR, FIF_DIR, QC_DIR, EXCEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def subject_fif_dir(subj_id: Union[int, str]) -> Path:
    d = FIF_DIR / f"subject_{subj_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def subject_qc_dir(subj_id: Union[int, str]) -> Path:
    d = QC_DIR / f"subject_{subj_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1.5  SUBJECT DISCOVERY
# ---------------------------------------------------------------------------

def discover_subjects(input_dir: Path = INPUT_DIR) -> List[int]:
    """Return sorted list of integer subject IDs found as *eeg.vhdr files."""
    ids: List[int] = []
    for f in sorted(input_dir.glob("*eeg.vhdr")):
        num_str = f.stem.replace("eeg", "")
        if "_" in num_str:
            continue
        try:
            ids.append(int(num_str))
        except ValueError:
            pass
    return sorted(ids)


# ---------------------------------------------------------------------------
# 1.6  VHDR REFERENCE FIXER
# ---------------------------------------------------------------------------

def fix_vhdr_references(vhdr_path: Path, subj_id: Union[int, str]) -> Path:
    """
    If DataFile= / MarkerFile= inside the .vhdr do not point to files that
    actually exist on disk, create a corrected copy (*_fixed.vhdr) alongside
    the original and return its path.  Otherwise return the original path.
    """
    expected_eeg  = f"{subj_id}eeg.eeg"
    expected_vmrk = f"{subj_id}eeg.vmrk"
    vhdr_dir = vhdr_path.parent
    text = vhdr_path.read_text(encoding="utf-8")
    needs_fix = False
    new_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("datafile="):
            ref = stripped.split("=", 1)[1].strip()
            if not (vhdr_dir / ref).exists():
                logger.warning(
                    f"S{subj_id} vhdr: DataFile={ref} not found -> {expected_eeg}"
                )
                new_lines.append(f"DataFile={expected_eeg}")
                needs_fix = True
                continue
        if stripped.lower().startswith("markerfile="):
            ref = stripped.split("=", 1)[1].strip()
            if not (vhdr_dir / ref).exists():
                logger.warning(
                    f"S{subj_id} vhdr: MarkerFile={ref} not found -> {expected_vmrk}"
                )
                new_lines.append(f"MarkerFile={expected_vmrk}")
                needs_fix = True
                continue
        new_lines.append(line)

    if needs_fix:
        fixed_path = vhdr_dir / f"{subj_id}eeg_fixed.vhdr"
        fixed_path.write_text("\n".join(new_lines), encoding="utf-8")
        logger.info(f"S{subj_id}: corrected vhdr -> {fixed_path.name}")
        return fixed_path
    return vhdr_path


# ---------------------------------------------------------------------------
# 1.7  PARSE RECORDING REFERENCE FROM .VHDR
# ---------------------------------------------------------------------------

def parse_vhdr_reference(vhdr_path: Path) -> Optional[str]:
    """Extract 'Reference Channel Name = ...' from a BrainVision header.

    Skips comment lines (starting with ';') and lines containing angle
    brackets '<' / '>' which belong to the channel-info description header.
    """
    try:
        for line in vhdr_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            # Skip comments and description-header lines
            if stripped.startswith(";") or "<" in stripped or ">" in stripped:
                continue
            low = stripped.lower()
            if "reference channel name" in low and "=" in low:
                ref = stripped.split("=", 1)[1].strip()
                if ref:
                    return ref
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 1.8  ANNOTATION / MARKER PARSING
# ---------------------------------------------------------------------------

def list_raw_annotations(raw: mne.io.BaseRaw, subj_id: Union[int, str]) -> str:
    """
    Print and return a simple chronological list of ALL annotations in the file.
    This helps the user see exactly what markers exist before pairing.
    """
    lines: List[str] = [
        "",
        "=" * 80,
        f"  RAW ANNOTATIONS  (SUBJECT {subj_id})  --  in chronological order",
        "=" * 80,
        f"  {'#':<4}  {'Onset (s)':>12}  {'Sample':>10}  Description",
        "-" * 80,
    ]
    sfreq = raw.info["sfreq"]
    for i, ann in enumerate(raw.annotations, start=1):
        onset = float(ann["onset"])
        sample = int(onset * sfreq)
        desc = ann["description"]
        lines.append(f"  {i:<4}  {onset:>12.2f}  {sample:>10}  {desc}")
    lines.append("=" * 80)
    report = "\n".join(lines)
    print(report)
    return report


def parse_annotations(raw: mne.io.BaseRaw) -> Dict[str, List[float]]:
    """
    Return ``{canonical_label: [onset_sec, ...]}`` from raw.annotations.

    Handles descriptions such as ``"Comment/LB"``, ``"Comment/Recovery"``,
    or plain ``"LB"``.  Unknown labels are silently skipped.
    """
    label_onsets: Dict[str, List[float]] = defaultdict(list)
    for ann in raw.annotations:
        desc = ann["description"].strip()
        # Strip BrainVision Type/ prefix
        if "/" in desc:
            desc = desc.split("/", 1)[-1].strip()
        canonical = _LABEL_NORMALIZE.get(desc.lower())
        if canonical is not None:
            label_onsets[canonical].append(float(ann["onset"]))

    for lab in label_onsets:
        label_onsets[lab].sort()

    return dict(label_onsets)


def reconstruct_segments(
    label_onsets: Dict[str, List[float]],
) -> Dict[str, List[Tuple[float, Optional[float]]]]:
    """
    Pair consecutive same-label markers into ``(start, end)`` intervals.

    If a label has an odd number of markers the final unpaired marker is
    stored as ``(onset, None)`` so that inspection can flag it.
    """
    segments: Dict[str, List[Tuple[float, Optional[float]]]] = {}
    for label in ("LB", "Intero", "BH", "Recov"):
        onsets = label_onsets.get(label, [])
        pairs: List[Tuple[float, Optional[float]]] = []
        i = 0
        while i + 1 < len(onsets):
            pairs.append((onsets[i], onsets[i + 1]))
            i += 2
        if i < len(onsets):
            pairs.append((onsets[i], None))
        segments[label] = pairs
    return segments


# ---------------------------------------------------------------------------
# 1.9  SEGMENT INSPECTION  (human-readable console report)
# ---------------------------------------------------------------------------

def inspect_segments(
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    sfreq: float,
) -> str:
    """Print and return a formatted inspection table for one subject."""
    lines: List[str] = [
        "",
        "=" * 80,
        f"  SEGMENT DURATION INSPECTION  (SUBJECT {subj_id})",
        "=" * 80,
    ]
    warns: List[str] = []

    display_order = [
        ("LB",     "LB",     1),
        ("Intero", "Intero", 1),
        ("BH",     "BH",     4),
        ("Recov",  "Recov",  1),
    ]

    for label, prefix, expected_n in display_order:
        intervals = segments.get(label, [])
        if not intervals:
            lines.append(f"  {prefix:<14s}:  NOT FOUND (no markers)")
            warns.append(f"{label}: no markers found")
            continue

        n_markers = sum(2 if e is not None else 1 for _, e in intervals)
        exp_mk = EXPECTED_MARKER_COUNTS.get(label)
        if exp_mk and n_markers != exp_mk:
            warns.append(f"{label}: expected {exp_mk} markers, found {n_markers}")

        for idx, (start, end) in enumerate(intervals):
            seg_label = f"{prefix} {idx+1}" if expected_n > 1 else prefix
            if end is None:
                lines.append(
                    f"  {seg_label:<14s}:  UNPAIRED MARKER  |  "
                    f"Onset: {start:10.2f} s  |  "
                    f"Sample: {int(start * sfreq)}"
                )
                warns.append(f"{seg_label}: unpaired (odd) marker")
            else:
                dur = end - start
                s0, s1 = int(start * sfreq), int(end * sfreq)
                flag = ""
                if dur <= 0:
                    flag = "  *** NEGATIVE/ZERO ***"
                    warns.append(f"{seg_label}: duration {dur:.2f} s")
                elif dur < 2.0:
                    flag = "  *** VERY SHORT ***"
                    warns.append(f"{seg_label}: very short ({dur:.2f} s)")
                lines.append(
                    f"  {seg_label:<14s}:  {dur:8.2f} s  |  "
                    f"Start: {start:10.2f} s  |  End: {end:10.2f} s  |  "
                    f"Samples: {s0} - {s1}{flag}"
                )

    if warns:
        lines.append("-" * 80)
        lines.append("  WARNINGS:")
        for w in warns:
            lines.append(f"    >> {w}")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    return report


def save_segments_json(
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    out_dir: Path = QC_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"subject_{subj_id}_segments.json"
    payload = {
        lab: [{"start": s, "end": e} for s, e in ivs]
        for lab, ivs in segments.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_segments_json(
    subj_id: Union[int, str],
    out_dir: Path = QC_DIR,
) -> Dict[str, List[Tuple[float, Optional[float]]]]:
    path = out_dir / f"subject_{subj_id}_segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        lab: [(iv["start"], iv["end"]) for iv in ivs]
        for lab, ivs in data.items()
    }


# ---------------------------------------------------------------------------
# 1.10  PREPROCESSING  STEP FUNCTIONS  (Steps 0-7)
# ---------------------------------------------------------------------------

def save_step(
    raw: mne.io.Raw, subj_id: Union[int, str], step_name: str,
) -> Optional[Path]:
    """Save intermediate .fif if the flag is enabled and write manifests."""
    if not SAVE_INTERMEDIATE_FIF:
        return None
    fdir = subject_fif_dir(subj_id)
    fname = f"{step_name}_raw.fif"
    path = fdir / fname
    raw.save(str(path), overwrite=OVERWRITE_FIF, verbose=False)
    logger.debug(f"S{subj_id}: saved {path.name}")

    # --- per-subject manifest (JSON) -------------------------------------
    manifest_path = fdir / "manifest.json"
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {}
    except Exception:
        manifest = {}

    manifest[step_name] = {
        "filename": fname,
        "path": str(path),
        "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_channels": len(raw.ch_names),
        "sfreq": float(raw.info.get("sfreq", np.nan)),
    }
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"S{subj_id}: failed to write manifest.json: {exc}")

    # --- global CSV manifest in QC folder --------------------------------
    try:
        QC_DIR.mkdir(parents=True, exist_ok=True)
        csvp = QC_DIR / "fif_manifest.csv"
        header = not csvp.exists()
        row = {
            "subject": subj_id,
            "step": step_name,
            "filename": fname,
            "path": str(path),
            "saved_at": manifest[step_name]["saved_at"],
            "n_channels": manifest[step_name]["n_channels"],
            "sfreq": manifest[step_name]["sfreq"],
        }
        pd.DataFrame([row]).to_csv(csvp, mode="a", header=header, index=False)
    except Exception as exc:
        logger.debug(f"S{subj_id}: could not append global manifest CSV: {exc}")

    return path


# ---- Step 0: Load --------------------------------------------------------

def step0_load_raw(
    subj_id: Union[int, str], input_dir: Path = INPUT_DIR,
) -> mne.io.Raw:
    """Load BrainVision .vhdr with preload=True."""
    vhdr = input_dir / f"{subj_id}eeg.vhdr"
    if not vhdr.exists():
        raise FileNotFoundError(f"VHDR not found: {vhdr}")
    vhdr = fix_vhdr_references(vhdr, subj_id)
    logger.info(f"S{subj_id} | Step 0  Load  {vhdr.name}")
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    logger.info(
        f"S{subj_id} | {len(raw.ch_names)} ch, "
        f"sfreq={raw.info['sfreq']:.0f} Hz, "
        f"dur={raw.times[-1]:.1f} s"
    )
    return raw


# ---- Step 1: Channel setup -----------------------------------------------

def step1_setup_channels(
    raw: mne.io.Raw, subj_id: Union[int, str],
) -> mne.io.Raw:
    """
    Select the expected 31 channels, set TP9 -> eog, rest -> eeg.

    If a channel from the expected list is missing because it served as the
    on-line recording reference, it is re-added as a flat (zero) channel via
    ``mne.add_reference_channels`` so that it can be reconstructed when the
    average reference is applied later.
    """
    logger.info(f"S{subj_id} | Step 1  Channel setup")

    # Detect actual recording reference from the header
    orig_vhdr = INPUT_DIR / f"{subj_id}eeg.vhdr"
    ref_name = parse_vhdr_reference(orig_vhdr)
    logger.info(f"S{subj_id} | Recording reference (header): {ref_name}")

    # Re-add the reference channel if it is in our expected list
    for ch_name in EXPECTED_CHANNELS:
        if ch_name not in raw.ch_names:
            if ref_name and ch_name.lower() == ref_name.lower():
                logger.info(
                    f"S{subj_id} | Adding reference '{ch_name}' back as flat channel"
                )
                raw = mne.add_reference_channels(raw, ch_name)
            else:
                logger.warning(f"S{subj_id} | Channel '{ch_name}' not in data")

    # Keep only expected channels that exist
    pick = [ch for ch in EXPECTED_CHANNELS if ch in raw.ch_names]
    raw.pick(pick)

    # Assign types
    mapping = {ch: ("eog" if ch == EOG_CHANNEL else "eeg") for ch in raw.ch_names}
    raw.set_channel_types(mapping)

    n_eeg = sum(1 for v in mapping.values() if v == "eeg")
    n_eog = sum(1 for v in mapping.values() if v == "eog")
    logger.info(f"S{subj_id} | Step 1 done  {n_eeg} EEG + {n_eog} EOG")
    return raw


# ---- Step 2: Montage -----------------------------------------------------

def step2_set_montage(
    raw: mne.io.Raw, subj_id: Union[int, str],
) -> mne.io.Raw:
    logger.info(f"S{subj_id} | Step 2  Montage (standard_1020)")
    montage = mne.channels.make_standard_montage("standard_1020")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw.set_montage(montage, on_missing="warn", verbose=False)
    logger.info(f"S{subj_id} | Step 2 done")
    return raw


# ---- Step 3: Filtering ---------------------------------------------------

def step3_filter(
    raw: mne.io.Raw, subj_id: Union[int, str],
) -> mne.io.Raw:
    logger.info(f"S{subj_id} | Step 3  Filtering")
    freqs = [NOTCH_FREQ]
    if NOTCH_HARMONICS:
        freqs.append(NOTCH_FREQ * 2)
    raw.notch_filter(freqs, verbose=False)
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, verbose=False)
    logger.info(
        f"S{subj_id} | Notch {freqs} Hz, BP {BANDPASS_LOW}-{BANDPASS_HIGH} Hz"
    )
    return raw


# ---- Step 4: Average re-reference ----------------------------------------

def step4_rereference(
    raw: mne.io.Raw, subj_id: Union[int, str],
) -> mne.io.Raw:
    """
    Re-reference all EEG channels to the common average.
    TP9 (typed eog) is automatically excluded from the average computation
    by MNE, but remains in the data for ICA EOG detection.
    """
    logger.info(f"S{subj_id} | Step 4  Average reference")
    raw, _ = mne.set_eeg_reference(raw, ref_channels="average", verbose=False)
    logger.info(f"S{subj_id} | Step 4 done")
    return raw


# ---- Step 5: Bad channels ------------------------------------------------

def _robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad = max(mad, 1e-12)
    return (x - med) / (mad * 1.4826)


def step5_bad_channels(
    raw: mne.io.Raw, subj_id: Union[int, str],
) -> Tuple[mne.io.Raw, List[str]]:
    """Detect + interpolate bad EEG channels (variance / flatline heuristic)."""
    logger.info(f"S{subj_id} | Step 5  Bad-channel detection")

    eeg_idx = mne.pick_types(raw.info, eeg=True, eog=False)
    eeg_names = [raw.ch_names[i] for i in eeg_idx]
    data = raw.get_data(picks=eeg_idx)
    bads: List[str] = []

    if USE_PYPREP and HAS_PYPREP:
        try:
            nd = NoisyChannels(raw, do_detrend=True, random_state=ICA_RANDOM_STATE)
            nd.find_all_bads()
            bads = list(nd.get_bads())
            logger.info(f"S{subj_id} | pyprep bads: {bads}")
        except Exception as exc:
            logger.warning(f"S{subj_id} | pyprep failed ({exc}), using heuristic")

    if not bads:
        ch_std = np.std(data, axis=1)
        ch_var = np.var(data, axis=1)
        for i, name in enumerate(eeg_names):
            if ch_std[i] * 1e6 < FLATLINE_THRESH_UV:
                bads.append(name)
                logger.info(f"S{subj_id} | Flatline: {name}  std={ch_std[i]*1e6:.4f} uV")
        zsc = _robust_zscore(ch_var)
        for i, name in enumerate(eeg_names):
            if abs(zsc[i]) > BAD_CHAN_ZSCORE_THRESH and name not in bads:
                bads.append(name)
                logger.info(f"S{subj_id} | Variance outlier: {name}  z={zsc[i]:.2f}")

    raw.info["bads"] = bads
    logger.info(f"S{subj_id} | Bad channels: {bads if bads else 'none'}")

    (subject_qc_dir(subj_id) / "bad_channels.json").write_text(
        json.dumps({"bad_channels": bads}, indent=2), encoding="utf-8",
    )

    if bads:
        logger.info(f"S{subj_id} | Interpolating {len(bads)} channel(s)")
        raw.interpolate_bads(reset_bads=True, verbose=False)

    logger.info(f"S{subj_id} | Step 5 done")
    return raw, bads


# ---- Step 6: Artifact annotation -----------------------------------------

def step6_annotate_artifacts(
    raw: mne.io.Raw, subj_id: Union[int, str],
) -> mne.io.Raw:
    """Mark 1-s windows whose peak-to-peak exceeds threshold as BAD_artifact.
    
    If AUTO_ARTIFACT_DETECTION is False, this step is skipped (rely on manual marking).
    """
    logger.info(f"S{subj_id} | Step 6  Artifact annotation")

    if not AUTO_ARTIFACT_DETECTION:
        logger.info(f"S{subj_id} | Auto artifact detection disabled, skipping")
        return raw

    eeg_idx = mne.pick_types(raw.info, eeg=True, eog=False)
    sfreq = raw.info["sfreq"]
    win_samp = int(1.0 * sfreq)
    data = raw.get_data(picks=eeg_idx)
    n_samp = data.shape[1]
    thresh_v = ARTIFACT_PTP_THRESH_UV * 1e-6

    onsets, durs = [], []
    for s in range(0, n_samp - win_samp, win_samp):
        seg = data[:, s : s + win_samp]
        ptp = seg.max(axis=1) - seg.min(axis=1)
        if ptp.max() > thresh_v:
            onsets.append(s / sfreq)
            durs.append(1.0)

    if onsets:
        art = mne.Annotations(
            onset=onsets, duration=durs,
            description=["BAD_artifact"] * len(onsets),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + art)
        logger.info(f"S{subj_id} | {len(onsets)} s flagged as BAD_artifact")
    else:
        logger.info(f"S{subj_id} | No gross artefacts detected")

    logger.info(f"S{subj_id} | Step 6 done")
    return raw


# ---- Step 7: ICA ---------------------------------------------------------

def step7_ica(
    raw: mne.io.Raw, subj_id: Union[int, str],
) -> mne.io.Raw:
    """
    Fit ICA, auto-detect EOG components via TP9, and apply.

    Decision logic:
    * Save auto-suggestion to ``ica_suggest.json``.
    * If ``ica_exclude.json`` exists (manual override), use it.
    * Otherwise use the suggestion and write a template for future manual edits.
    """
    logger.info(f"S{subj_id} | Step 7  ICA")
    qdir = subject_qc_dir(subj_id)

    rank_dict = mne.compute_rank(raw, rank="info", verbose=False)
    eeg_rank = rank_dict.get("eeg", None)
    n_comp = ICA_N_COMPONENTS
    # If n_comp is None, fall back to variance-based (0.999999) to avoid unstable mixing matrix
    if n_comp is None:
        n_comp = 0.999999
    logger.info(f"S{subj_id} | EEG rank={eeg_rank}, n_components={n_comp}")

    ica = ICA(
        n_components=n_comp, method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE, max_iter=ICA_MAX_ITER,
    )
    ica.fit(raw, picks="eeg", reject_by_annotation=True, verbose=False)
    logger.info(f"S{subj_id} | ICA fitted ({ica.n_components_} components)")

    ica.save(str(qdir / "ica_solution-ica.fif"), overwrite=True, verbose=False)

    # EOG component detection
    eog_idx: List[int] = []
    eog_scores = np.array([])
    if EOG_CHANNEL in raw.ch_names:
        try:
            eog_idx, eog_scores = ica.find_bads_eog(
                raw, ch_name=EOG_CHANNEL, verbose=False,
            )
            logger.info(f"S{subj_id} | Auto EOG components: {eog_idx}")
        except Exception as exc:
            logger.warning(f"S{subj_id} | EOG detection error: {exc}")
    else:
        logger.warning(f"S{subj_id} | EOG channel '{EOG_CHANNEL}' absent")

    suggestion = {
        "auto_eog_indices": [int(x) for x in eog_idx],
        "eog_scores": eog_scores.tolist() if eog_scores.size else [],
        "n_components": int(ica.n_components_),
        "method": ICA_METHOD,
    }
    (qdir / "ica_suggest.json").write_text(
        json.dumps(suggestion, indent=2), encoding="utf-8",
    )

    # Save topography plots (non-interactive)
    try:
        prev_backend = plt.get_backend()
        plt.switch_backend("Agg")
        figs = ica.plot_components(show=False)
        if not isinstance(figs, list):
            figs = [figs]
        for i, fig in enumerate(figs):
            fig.savefig(str(qdir / f"ica_topomap_{i}.png"), dpi=120)
            plt.close(fig)
        plt.switch_backend(prev_backend)
    except Exception as exc:
        logger.warning(f"S{subj_id} | ICA plot failed: {exc}")

    # Decide which components to exclude
    override_path = qdir / "ica_exclude.json"
    if override_path.exists():
        try:
            override = json.loads(override_path.read_text(encoding="utf-8"))
            exclude = [int(x) for x in override.get("exclude", eog_idx)]
            logger.info(f"S{subj_id} | Manual ICA override -> {exclude}")
        except Exception:
            exclude = [int(x) for x in eog_idx]
    else:
        exclude = [int(x) for x in eog_idx]
        (qdir / "ica_exclude.json").write_text(
            json.dumps({
                "exclude": exclude,
                "_comment": "Edit 'exclude' list to override ICA component removal.",
            }, indent=2),
            encoding="utf-8",
        )

    ica.exclude = exclude
    raw = ica.apply(raw, verbose=False)
    logger.info(f"S{subj_id} | Step 7 done - removed {len(exclude)} component(s)")
    return raw


# ---------------------------------------------------------------------------
# 1.11  PREPROCESSING DRIVER  (Steps 1-7 only, takes already-loaded raw)
# ---------------------------------------------------------------------------

def run_preprocessing_steps(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
) -> Tuple[mne.io.Raw, List[str], Dict[str, Any]]:
    """
    Run Steps 1-7 on an already-loaded raw object.

    Returns (raw_clean, bad_channels, ica_info).
    Raises on failure (caller should handle).
    """
    # Step 1 - channels
    print(f"  [Step 1] Setting up channels...")
    raw = step1_setup_channels(raw, subj_id)
    save_step(raw, subj_id, "step1_channels")

    # Step 2 - montage
    print(f"  [Step 2] Applying montage (standard_1020)...")
    raw = step2_set_montage(raw, subj_id)
    save_step(raw, subj_id, "step2_montage")

    # Step 3 - filtering
    print(f"  [Step 3] Filtering (notch + bandpass)...")
    raw = step3_filter(raw, subj_id)
    save_step(raw, subj_id, "step3_filtered")

    # Step 4 - average reference
    print(f"  [Step 4] Applying average reference...")
    raw = step4_rereference(raw, subj_id)
    save_step(raw, subj_id, "step4_refavg")

    # Step 5 - bad channels
    print(f"  [Step 5] Detecting & interpolating bad channels...")
    raw, bads = step5_bad_channels(raw, subj_id)
    if bads:
        print(f"           Bad channels: {bads}")
    else:
        print(f"           No bad channels detected")
    save_step(raw, subj_id, "step5_interpolated")

    # Step 6 - artefact annotation
    print(f"  [Step 6] Annotating gross artifacts...")
    n_bad_before = sum(1 for a in raw.annotations if "BAD" in a["description"])
    raw = step6_annotate_artifacts(raw, subj_id)
    n_bad_after = sum(1 for a in raw.annotations if "BAD" in a["description"])
    print(f"           {n_bad_after - n_bad_before} s flagged as BAD_artifact")
    save_step(raw, subj_id, "step6_annotated")

    # Step 7 - ICA
    print(f"  [Step 7] Running ICA (this may take a few minutes)...")
    raw = step7_ica(raw, subj_id)
    save_step(raw, subj_id, "step7_ica")

    # Gather ICA QC info
    ica_info: Dict[str, Any] = {"method": ICA_METHOD, "excluded": []}
    suggest_p = subject_qc_dir(subj_id) / "ica_suggest.json"
    if suggest_p.exists():
        ica_info.update(json.loads(suggest_p.read_text(encoding="utf-8")))
    excl_p = subject_qc_dir(subj_id) / "ica_exclude.json"
    if excl_p.exists():
        ica_info["excluded"] = json.loads(
            excl_p.read_text(encoding="utf-8")
        ).get("exclude", [])

    print(f"\n  Preprocessing complete.")
    print(f"  ICA components excluded: {ica_info.get('excluded', [])}")
    return raw, bads, ica_info


# ---------------------------------------------------------------------------
# 1.12  VISUALIZATION FUNCTIONS
# ---------------------------------------------------------------------------

def _iter_valid_segments(
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> List[Tuple[str, int, float, float]]:
    """Yield (seg_type, seg_idx_1based, start, end) for all valid intervals."""
    # Only BH has multiple segments (4)
    order = [
        ("LB", 1), ("Intero", 1), ("BH", 4), ("Recov", 1),
    ]
    result = []
    for label, expected_n in order:
        for idx, (start, end) in enumerate(segments.get(label, [])):
            if end is None or end <= start:
                continue
            seg_name = f"{label} {idx+1}" if expected_n > 1 else label
            result.append((seg_name, idx + 1, start, end))
    return result


def get_segments_by_type(
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    seg_type: str,
    apply_time_rules: bool = False,
) -> List[Tuple[str, int, float, float]]:
    """
    Return list of (seg_name, seg_idx_1based, start, end) for a single segment type.

    seg_type : one of "LB", "Intero", "BH", "Recov"
    apply_time_rules : if True, apply LB/Recov 300s truncation and BH 10s trim
    """
    # Only BH has multiple segments (4)
    expected_n = {"LB": 1, "Intero": 1, "BH": 4, "Recov": 1}.get(seg_type, 1)
    result = []
    for idx, (start, end) in enumerate(segments.get(seg_type, [])):
        if end is None or end <= start:
            continue
        # Apply time rules if requested
        if apply_time_rules:
            start, end = apply_segment_rules(seg_type, start, end)
        seg_name = f"{seg_type} {idx+1}" if expected_n > 1 else seg_type
        result.append((seg_name, idx + 1, start, end))
    return result


# ---------------------------------------------------------------------------
# 1.12b  SAVE FINAL CLEANED DATA
# ---------------------------------------------------------------------------

CLEAN_DATA_DIR: Path = OUTPUT_DIR / "clean_data"


def save_clean_data(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    overwrite: bool = True,
) -> Path:
    """
    Save the final cleaned (post-ICA) Raw object to a dedicated folder.

    This allows reusing the cleaned data for future analyses (e.g., connectivity,
    time-frequency) without re-running the preprocessing pipeline.

    Returns the path to the saved .fif file.
    """
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DATA_DIR / f"subject_{subj_id}_clean_raw.fif"
    raw.save(str(out_path), overwrite=overwrite, verbose=False)
    logger.info(f"S{subj_id}: Saved cleaned data -> {out_path}")
    return out_path


def load_clean_data(subj_id: Union[int, str]) -> Optional[mne.io.Raw]:
    """
    Load previously saved cleaned data for a subject.

    Returns None if the file does not exist.
    """
    fif_path = CLEAN_DATA_DIR / f"subject_{subj_id}_clean_raw.fif"
    if not fif_path.exists():
        logger.warning(f"S{subj_id}: Clean data not found at {fif_path}")
        return None
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    logger.info(f"S{subj_id}: Loaded cleaned data from {fif_path}")
    return raw


# ---------------------------------------------------------------------------
# 1.12c  MANUAL BAD SECTION MARKING
# ---------------------------------------------------------------------------

def mark_bad_sections(
    raw: mne.io.Raw,
    bad_intervals: List[Tuple[float, float]],
    description: str = "BAD_manual",
) -> None:
    """
    Mark time intervals as BAD in the Raw object's annotations.

    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object to modify (in-place).
    bad_intervals : list of (start, end) tuples
        Time intervals in seconds to mark as BAD.
        Example: [(100, 130), (200, 210)] marks 100-130s and 200-210s as BAD.
    description : str
        Annotation description (default "BAD_manual").

    Example usage in a cell:
        mark_bad_sections(RAW, [(100, 130), (200, 210)])
    """
    if not bad_intervals:
        return

    onsets = []
    durations = []
    for start, end in bad_intervals:
        if end <= start:
            print(f"  Warning: Invalid interval ({start}, {end}) - skipped")
            continue
        onsets.append(float(start))
        durations.append(float(end - start))

    if onsets:
        new_annot = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=[description] * len(onsets),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + new_annot)
        print(f"  Marked {len(onsets)} BAD sections:")
        for s, d in zip(onsets, durations):
            print(f"    - {s:.1f}s to {s+d:.1f}s ({d:.1f}s)")


def show_bad_annotations(
    raw: mne.io.Raw,
    seg_start: Optional[float] = None,
    seg_end: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Show all BAD annotations on a Raw object, optionally filtered to a segment.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object to inspect.
    seg_start, seg_end : float, optional
        If provided, only show annotations that overlap with [seg_start, seg_end].
    
    Returns
    -------
    List of dicts with onset, duration, description for each BAD annotation.
    """
    if raw is None:
        print("!! Raw object is None.")
        return []
    
    bad_anns = []
    for ann in raw.annotations:
        if not ann["description"].upper().startswith("BAD"):
            continue
        onset = ann["onset"]
        duration = ann["duration"]
        end = onset + duration
        
        # Filter by segment if specified
        if seg_start is not None and seg_end is not None:
            if end < seg_start or onset > seg_end:
                continue
        
        bad_anns.append({
            "onset": onset,
            "duration": duration,
            "end": end,
            "description": ann["description"],
        })
    
    if bad_anns:
        print(f"  BAD annotations ({len(bad_anns)}):")
        for ann in bad_anns:
            print(f"    - {ann['onset']:.1f}s – {ann['end']:.1f}s  "
                  f"({ann['duration']:.1f}s)  [{ann['description']}]")
    else:
        print("  No BAD annotations.")
    
    return bad_anns


def inspect_and_mark_segment(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    seg_type: str,
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    *,
    tag: str = "raw",
    interactive: bool = True,
    apply_time_rules: bool = False,
) -> None:
    """
    Inspect a segment type and allow marking BAD sections in the interactive viewer.
    
    This is the main function for segment inspection. It:
    1. Shows an overview plot
    2. Opens the interactive viewer (if interactive=True)
    3. Any BAD annotations made in the viewer are automatically applied to raw
    4. Shows a summary of BAD annotations after closing the viewer
    
    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object to inspect and modify.
    subj_id : subject identifier
    seg_type : "LB", "Intero", "BH", or "Recov"
    segments : dict of segment intervals
    tag : "raw" or "clean"
    interactive : if True, open the interactive viewer
    apply_time_rules : if True, apply LB/Recov truncation and BH trimming
    """
    if raw is None:
        print(f"!! Raw object is None.")
        return
    
    qc_dir = subject_qc_dir(subj_id)
    segs = get_segments_by_type(segments, seg_type, apply_time_rules=apply_time_rules)
    
    if not segs:
        print(f"  No {seg_type} segments found.")
        return
    
    max_sec = ""
    if seg_type == "LB" and apply_time_rules:
        max_sec = f"  [first {LB_MAX_SEC:.0f}s]"
    elif seg_type == "Recov" and apply_time_rules:
        max_sec = f"  [first {RECOV_MAX_SEC:.0f}s]"
    
    print("=" * 70)
    print(f"  Subject {subj_id} -- {tag.upper()}: {seg_type}{max_sec}")
    print("=" * 70)
    
    for seg_name, seg_idx, start, end in segs:
        print(f"\n--- {seg_name} ---  ({end - start:.1f} s)  [absolute: {start:.1f}s - {end:.1f}s]")
        plot_segment_overview(raw, subj_id, f"{tag}_{seg_name}", start, end, save_dir=qc_dir)
        
        if interactive:
            launch_interactive_viewer(raw, subj_id, seg_name, start, end, tag=tag)
    
    # Show BAD annotations summary for this segment type
    print(f"\n  BAD annotations for {seg_type}:")
    for seg_name, seg_idx, start, end in segs:
        show_bad_annotations(raw, start, end)
    
    print(f"\n>>> {seg_type} inspection done.")


def print_preprocessing_summary(
    raw_clean: mne.io.Raw,
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    bad_channels: List[str],
    ica_info: Dict[str, Any],
) -> pd.DataFrame:
    """
    Print and return a summary of preprocessing results.
    
    Returns a DataFrame with segment-level summary.
    """
    print("=" * 70)
    print(f"  Subject {subj_id} -- PREPROCESSING SUMMARY")
    print("=" * 70)

    summary_rows = []
    for seg_type in ["LB", "Intero", "BH", "Recov"]:
        segs = get_segments_by_type(segments, seg_type)
        for seg_name, seg_idx, start, end in segs:
            orig_dur = end - start
            trunc_start, trunc_end = apply_segment_rules(seg_type, start, end)
            final_dur = trunc_end - trunc_start

            # Count BAD annotations within this segment
            bad_time = 0.0
            for ann in raw_clean.annotations:
                if ann["description"].upper().startswith("BAD"):
                    ann_start = ann["onset"]
                    ann_end = ann_start + ann["duration"]
                    overlap_start = max(ann_start, trunc_start)
                    overlap_end = min(ann_end, trunc_end)
                    if overlap_end > overlap_start:
                        bad_time += overlap_end - overlap_start

            summary_rows.append({
                "Segment": seg_name,
                "Original (s)": f"{orig_dur:.1f}",
                "Final (s)": f"{final_dur:.1f}",
                "BAD time (s)": f"{bad_time:.1f}",
                "Usable (s)": f"{max(0, final_dur - bad_time):.1f}",
                "Inspected": "Yes" if SEGMENTS_INSPECTED.get(seg_type, False) else "No",
            })

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_string(index=False))

    print(f"\n  Bad channels (auto + manual): {bad_channels}")
    print(f"  ICA components excluded: {ica_info.get('excluded', [])}")

    return df_summary


def save_and_report_clean(
    raw_clean: mne.io.Raw,
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    bad_channels: List[str],
    ica_info: Dict[str, Any],
) -> None:
    """
    Print summary, save cleaned data, and save summary CSV.
    """
    if raw_clean is None:
        print("!! Cleaned data not available. Cannot save.")
        return

    df_summary = print_preprocessing_summary(
        raw_clean, subj_id, segments, bad_channels, ica_info
    )

    # Save cleaned data
    print("\n" + "-" * 70)
    clean_path = save_clean_data(raw_clean, subj_id)
    print(f"  Saved -> {clean_path}")
    print(f"  Channels: {len(raw_clean.ch_names)}")
    print(f"  Duration: {raw_clean.times[-1]:.1f} s")
    print(f"  Sfreq:    {raw_clean.info['sfreq']:.0f} Hz\n")

    # Save summary to QC folder
    summary_path = subject_qc_dir(subj_id) / "preprocessing_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"  Summary saved -> {summary_path}\n")
    print(">>> Proceed to Cell 7 for spectral analysis.\n")


# ---------------------------------------------------------------------------
# 1.12d  AGGREGATION FUNCTIONS
# ---------------------------------------------------------------------------

def aggregate_subject_excels(excel_dir: Path = EXCEL_DIR) -> Optional[Path]:
    """
    Aggregate all per-subject Excel files into a single global Excel.

    Returns the path to the global Excel file, or None if no files found.
    """
    subject_excels = list(excel_dir.glob("subject_*_eeg_bandpower.xlsx"))

    if not subject_excels:
        print("  No per-subject Excel files found.")
        return None

    print(f"  Found {len(subject_excels)} subject files")

    all_long, all_seg_avg, all_region_avg = [], [], []
    all_qc_mk, all_qc_bad, all_qc_ica = [], [], []

    for excel_path in sorted(subject_excels):
        try:
            xl = pd.ExcelFile(excel_path)
            if "bandpower_long" in xl.sheet_names:
                all_long.append(pd.read_excel(xl, "bandpower_long"))
            if "bandpower_segment_avg" in xl.sheet_names:
                all_seg_avg.append(pd.read_excel(xl, "bandpower_segment_avg"))
            if "bandpower_region_avg" in xl.sheet_names:
                all_region_avg.append(pd.read_excel(xl, "bandpower_region_avg"))
            if "qc_markers" in xl.sheet_names:
                all_qc_mk.append(pd.read_excel(xl, "qc_markers"))
            if "qc_bad_channels" in xl.sheet_names:
                all_qc_bad.append(pd.read_excel(xl, "qc_bad_channels"))
            if "qc_ica" in xl.sheet_names:
                all_qc_ica.append(pd.read_excel(xl, "qc_ica"))
        except Exception as e:
            print(f"    Error reading {excel_path.name}: {e}")

    # Concatenate
    df_long = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    df_seg = pd.concat(all_seg_avg, ignore_index=True) if all_seg_avg else pd.DataFrame()
    df_region = pd.concat(all_region_avg, ignore_index=True) if all_region_avg else pd.DataFrame()
    df_qc_mk = pd.concat(all_qc_mk, ignore_index=True) if all_qc_mk else pd.DataFrame()
    df_qc_bad = pd.concat(all_qc_bad, ignore_index=True) if all_qc_bad else pd.DataFrame()
    df_qc_ica = pd.concat(all_qc_ica, ignore_index=True) if all_qc_ica else pd.DataFrame()

    # Save global Excel
    global_excel = excel_dir / "all_subjects_eeg_bandpower.xlsx"
    with pd.ExcelWriter(global_excel, engine="openpyxl") as writer:
        if not df_long.empty:
            df_long.to_excel(writer, sheet_name="bandpower_long", index=False)
        if not df_seg.empty:
            df_seg.to_excel(writer, sheet_name="bandpower_segment_avg", index=False)
        if not df_region.empty:
            df_region.to_excel(writer, sheet_name="bandpower_region_avg", index=False)
        if not df_qc_mk.empty:
            df_qc_mk.to_excel(writer, sheet_name="qc_markers", index=False)
        if not df_qc_bad.empty:
            df_qc_bad.to_excel(writer, sheet_name="qc_bad_channels", index=False)
        if not df_qc_ica.empty:
            df_qc_ica.to_excel(writer, sheet_name="qc_ica", index=False)

    n_subjects = df_long["subject"].nunique() if not df_long.empty else 0
    print(f"  Global Excel: {global_excel}")
    print(f"  Total subjects: {n_subjects}, Total rows: {len(df_long):,}")

    return global_excel


# ---------------------------------------------------------------------------
# 1.12e  TOPOGRAPHIC MAPS
# ---------------------------------------------------------------------------

def create_topomap_info() -> mne.Info:
    """
    Create an MNE Info object with standard 10-20 montage positions
    for our EEG channels (excluding TP9 which is used as EOG).
    """
    # Create info with our EEG channels
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=1000, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    return info


def plot_subject_topomaps(
    df_seg_avg: pd.DataFrame,
    subj_id: Union[int, str],
    save_dir: Optional[Path] = None,
    power_type: str = "abs_power_mean",
) -> None:
    """
    Create topographic maps for a single subject showing spectral power
    per segment type and frequency band.

    Creates ONE figure per segment type (rows=1, cols=bands).
    Each figure is saved to disk and displayed.

    Parameters
    ----------
    df_seg_avg : DataFrame with columns: subject, seg_type, channel, band, power
    subj_id    : subject identifier
    save_dir   : directory to save figures (if None, just displays)
    power_type : "abs_power_mean" or "rel_power_mean"
    """
    # Robust subject filter (handles str/int mismatch)
    df = df_seg_avg[df_seg_avg["subject"].astype(str) == str(subj_id)].copy()
    if df.empty:
        print(f"  No data for subject {subj_id}")
        return

    info = create_topomap_info()
    seg_types = [s for s in ["LB", "Intero", "BH", "Recov"] if s in df["seg_type"].values]
    bands = list(FREQ_BANDS.keys())
    unit = "µV²/Hz" if power_type == "abs_power_mean" else "ratio"

    if not seg_types:
        print(f"  No segments found for subject {subj_id}")
        return

    for seg_type in seg_types:
        df_seg = df[df["seg_type"] == seg_type]
        if df_seg.empty:
            continue

        n_bands = len(bands)
        fig, axes = plt.subplots(1, n_bands, figsize=(3.8 * n_bands, 4.5))
        if n_bands == 1:
            axes = [axes]

        fig.suptitle(f"Subject {subj_id} — {seg_type}", fontsize=14, fontweight="bold")

        for i, band in enumerate(bands):
            df_band = df_seg[df_seg["band"] == band]
            data = np.zeros(len(EEG_CHANNELS))
            for j, ch in enumerate(EEG_CHANNELS):
                ch_vals = df_band[df_band["channel"] == ch][power_type]
                if not ch_vals.empty:
                    data[j] = float(ch_vals.mean())  # average across seg_idx

            ax = axes[i]
            try:
                im, _ = mne.viz.plot_topomap(
                    data, info, axes=ax, show=False,
                    cmap="RdBu_r", contours=6,
                )
                ax.set_title(f"{band}\n({FREQ_BANDS[band][0]}-{FREQ_BANDS[band][1]} Hz)",
                             fontsize=10)
                cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                                    fraction=0.06, pad=0.08, aspect=18, shrink=0.85)
                cbar.ax.tick_params(labelsize=6)
                cbar.set_label(unit, fontsize=7)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:40]}", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8)
                ax.set_title(band)

        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.15, top=0.85, wspace=0.4)

        # Save to file
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig_path = save_dir / f"topomap_S{subj_id}_{seg_type}.png"
            fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
            print(f"    Saved: {fig_path.name}")

        # Display — use IPython display() so all figures show (not just the last)
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            plt.show()
        plt.close(fig)


def plot_subject_topomaps_comparison(
    df_seg_avg: pd.DataFrame,
    subj_id: Union[int, str],
    save_dir: Optional[Path] = None,
    power_type: str = "abs_power_mean",
    robust_percentile: float = 2.0,
) -> None:
    """
    Create a single comparison grid: rows = segment types, columns = bands.
    One colorbar per band column at the bottom.

    Uses **robust percentile-based clipping** (default 2nd–98th percentile)
    so that outlier values in one segment do not squash the colour range for
    all other segments.

    Parameters
    ----------
    robust_percentile : float
        Lower percentile for vlim (upper = 100 - robust_percentile).
        Set to 0 to use plain min/max (not recommended).
    """
    # Robust subject filter
    df = df_seg_avg[df_seg_avg["subject"].astype(str) == str(subj_id)].copy()
    if df.empty:
        print(f"  No data for subject {subj_id}")
        return

    info = create_topomap_info()
    seg_types = [s for s in ["LB", "Intero", "BH", "Recov"] if s in df["seg_type"].values]
    bands = list(FREQ_BANDS.keys())
    unit = "µV²/Hz" if power_type == "abs_power_mean" else "proportion"

    if not seg_types:
        print(f"  No segments found for subject {subj_id}")
        return

    n_rows = len(seg_types)
    n_cols = len(bands)

    # --- Build a data matrix so we can compute robust colour limits ----------
    # data_matrix[band] = 2-D array (n_segs × n_channels)
    data_matrix: Dict[str, np.ndarray] = {}
    for band in bands:
        mat = np.zeros((n_rows, len(EEG_CHANNELS)))
        for ri, seg_type in enumerate(seg_types):
            df_seg = df[df["seg_type"] == seg_type]
            df_band = df_seg[df_seg["band"] == band]
            for ci, ch in enumerate(EEG_CHANNELS):
                ch_vals = df_band[df_band["channel"] == ch][power_type]
                if not ch_vals.empty:
                    mat[ri, ci] = float(ch_vals.mean())
        data_matrix[band] = mat

    # Robust percentile-based colour limits per band (across all segments)
    band_vmin, band_vmax = {}, {}
    for band in bands:
        vals = data_matrix[band].ravel()
        if robust_percentile > 0 and len(vals) > 0:
            band_vmin[band] = float(np.percentile(vals, robust_percentile))
            band_vmax[band] = float(np.percentile(vals, 100.0 - robust_percentile))
        else:
            band_vmin[band] = float(vals.min()) if len(vals) else 0
            band_vmax[band] = float(vals.max()) if len(vals) else 1
        # Safety: ensure vmin < vmax
        if band_vmax[band] <= band_vmin[band]:
            band_vmax[band] = band_vmin[band] + 1e-6

    # --- Create figure -------------------------------------------------------
    fig = plt.figure(figsize=(3.5 * n_cols, 3.2 * n_rows + 1.5))
    gs = fig.add_gridspec(n_rows + 1, n_cols,
                          height_ratios=[1] * n_rows + [0.06],
                          hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.92, bottom=0.04)

    fig.suptitle(f"Subject {subj_id} — Spectral Power Comparison  ({unit})",
                 fontsize=14, fontweight="bold")

    band_images: Dict[str, Any] = {}

    for row_i, seg_type in enumerate(seg_types):
        for col_i, band in enumerate(bands):
            ax = fig.add_subplot(gs[row_i, col_i])
            data = data_matrix[band][row_i]

            try:
                im, _ = mne.viz.plot_topomap(
                    data, info, axes=ax, show=False,
                    cmap="RdBu_r", contours=4,
                    vlim=(band_vmin[band], band_vmax[band]),
                )
                if band not in band_images:
                    band_images[band] = im
            except Exception:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)

            # Column titles (top row only)
            if row_i == 0:
                ax.set_title(f"{band}\n({FREQ_BANDS[band][0]}-{FREQ_BANDS[band][1]} Hz)",
                             fontsize=10)
            # Row labels (left column only)
            if col_i == 0:
                ax.text(-0.15, 0.5, seg_type, transform=ax.transAxes,
                        fontsize=12, fontweight="bold", va="center", ha="right")

    # Colorbar row at the bottom
    for col_i, band in enumerate(bands):
        cax = fig.add_subplot(gs[n_rows, col_i])
        if band in band_images:
            fig.colorbar(band_images[band], cax=cax, orientation="horizontal")
            cax.tick_params(labelsize=6)
        else:
            cax.set_visible(False)

    # Save to file
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "abs" if "abs" in power_type else "rel"
        fig_path = save_dir / f"topomap_comparison_{suffix}_S{subj_id}.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"    Saved: {fig_path.name}")

    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        plt.show()
    plt.close(fig)


def plot_segment_overview(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    seg_name: str,
    start: float,
    end: float,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Static inline overview for one segment:
      - Top panel: butterfly plot (all EEG channels overlaid)
      - Bottom panel: PSD (Welch) per channel

    The figure is displayed inline and optionally saved to disk.
    """
    sfreq = raw.info["sfreq"]
    i0, i1 = int(start * sfreq), min(int(end * sfreq), raw.n_times)
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)
    data = raw.get_data(picks=eeg_picks, start=i0, stop=i1) * 1e6  # -> uV
    times = np.arange(data.shape[1]) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(f"Subject {subj_id}  |  {seg_name}  "
                 f"({end - start:.1f} s)", fontsize=12)

    # -- Top: butterfly plot --
    ax = axes[0]
    for ch_i in range(data.shape[0]):
        ax.plot(times, data[ch_i], linewidth=0.3, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title("Butterfly plot (all EEG channels)")

    # -- Bottom: PSD --
    ax = axes[1]
    n_fft = min(WELCH_N_FFT, data.shape[1])
    n_overlap = min(WELCH_N_OVERLAP, n_fft - 1)
    try:
        psd, freqs = mne.time_frequency.psd_array_welch(
            data * 1e-6,  # back to V for MNE
            sfreq=sfreq, fmin=PSD_FMIN, fmax=PSD_FMAX,
            n_fft=n_fft, n_overlap=n_overlap, verbose=False,
        )
        psd_uv2 = psd * 1e12  # V^2/Hz -> uV^2/Hz
        for ch_i in range(psd_uv2.shape[0]):
            ax.semilogy(freqs, psd_uv2[ch_i], linewidth=0.4, alpha=0.6)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (uV^2/Hz)")
        ax.set_title("Power Spectral Density (Welch)")
    except Exception:
        ax.text(0.5, 0.5, "PSD computation failed (segment too short?)",
                transform=ax.transAxes, ha="center")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = seg_name.replace(" ", "_")
        fig.savefig(str(save_dir / f"overview_{safe_name}.png"), dpi=120)

    # Show inline (works in IPython); suppress warning for non-interactive backends
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.show()

    plt.close(fig)


def launch_interactive_viewer(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    seg_label: str,
    start: float,
    end: float,
    *,
    tag: str = "raw",
    apply_edits: bool = True,
) -> Dict[str, Any]:
    """
    Launch the interactive EEG segment viewer in a **separate process**.

    This saves a temporary .fif file and calls ``eeg_segment_viewer.py``
    (which uses the TkAgg backend) via :func:`subprocess.run`, so the
    interactive MNE browser window pops up even when the main script runs
    inside an IPython / VSCode cell.

    The call is **blocking** — the cell waits until you close the viewer
    window, then reads back any edits (bad channels, BAD annotations) and
    optionally applies them to the original ``raw`` object.

    Parameters
    ----------
    raw         : loaded Raw object (not cropped — cropping is done in the viewer)
    subj_id     : subject identifier (for the window title)
    seg_label   : human-readable segment label, e.g. "LB", "BH 2"
    start       : segment start in seconds
    end         : segment end in seconds
    tag         : "raw" or "clean" (used in the temp filename)
    apply_edits : if True, bad channels and annotations are applied to ``raw``

    Returns
    -------
    dict with keys:
        bad_channels_added : list of channel names marked bad
        annotations_added  : list of dicts with onset/duration/description
    """
    viewer_script = Path(__file__).with_name("eeg_segment_viewer.py")
    if not viewer_script.exists():
        print(f"  [!] Viewer script not found: {viewer_script}")
        return {"bad_channels_added": [], "annotations_added": []}

    # --- Save a temporary .fif so the viewer can load it -----------------
    tmp_dir = OUTPUT_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe_label = seg_label.replace(" ", "_")
    tmp_fif = tmp_dir / f"S{subj_id}_{tag}_{safe_label}_eeg.fif"
    raw.save(str(tmp_fif), overwrite=True, verbose=False)

    # Output JSON for edits
    edits_json = tmp_dir / f"S{subj_id}_{tag}_{safe_label}_edits.json"

    title = f"S{subj_id} | {seg_label} ({tag})  [{start:.1f} – {end:.1f} s]"
    cmd = [
        sys.executable,
        str(viewer_script),
        str(tmp_fif),
        str(start),
        str(end),
        "--title", title,
        "--output", str(edits_json),
    ]
    print(f"  >> Opening interactive viewer for {seg_label} …")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"  [!] Error launching viewer: {exc}")
        return {"bad_channels_added": [], "annotations_added": []}

    # --- Read back edits -------------------------------------------------
    edits = {"bad_channels_added": [], "annotations_added": []}
    if edits_json.exists():
        try:
            edits = json.loads(edits_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    # --- Apply edits to the original raw object --------------------------
    if apply_edits:
        # Bad channels
        for ch in edits.get("bad_channels_added", []):
            if ch not in raw.info["bads"]:
                raw.info["bads"].append(ch)

        # BAD annotations (absolute times already)
        for ann in edits.get("annotations_added", []):
            raw.annotations.append(
                onset=ann["onset"],
                duration=ann["duration"],
                description=ann["description"],
            )

    return edits


# ---------------------------------------------------------------------------
# 1.13  WINDOWING
# ---------------------------------------------------------------------------

def apply_segment_rules(
    seg_type: str, start: float, end: Optional[float],
) -> Tuple[float, float]:
    """
    Apply segment-specific truncation / trimming rules.

    - LB / Recov : keep at most 300 s from start.
    - BH         : drop the last 10 s.
    - Intero     : use full duration.
    """
    if end is None:
        return start, start
    dur = end - start
    if seg_type == "LB" and dur > LB_MAX_SEC:
        end = start + LB_MAX_SEC
    elif seg_type == "Recov" and dur > RECOV_MAX_SEC:
        end = start + RECOV_MAX_SEC
    elif seg_type == "BH":
        trimmed = end - BH_TRIM_END_SEC
        if trimmed > start:
            end = trimmed
    return start, end


def make_windows(
    start: float, end: float, win_sec: float = WINDOW_SEC,
) -> List[Tuple[float, float]]:
    """
    Divide ``[start, end)`` into non-overlapping windows of ``win_sec``.
    The last window may be shorter.
    """
    if end <= start:
        return []
    wins: List[Tuple[float, float]] = []
    t = start
    while t + win_sec <= end + 0.001:
        wins.append((t, min(t + win_sec, end)))
        t += win_sec
    if t < end - 0.001:
        wins.append((t, end))
    return wins


# ---------------------------------------------------------------------------
# 1.14  SPECTRAL ANALYSIS
# ---------------------------------------------------------------------------

def compute_bandpower_welch(
    data: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]] = FREQ_BANDS,
    n_fft: int = WELCH_N_FFT,
    n_overlap: int = WELCH_N_OVERLAP,
    fmin: float = PSD_FMIN,
    fmax: float = PSD_FMAX,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Welch PSD -> absolute and relative band power.

    Parameters
    ----------
    data : (n_channels, n_times) in **Volts**

    Returns
    -------
    abs_power : {band: (n_channels,)}  in **uV^2**
    rel_power : {band: (n_channels,)}  dimensionless (0-1)
    """
    n_times = data.shape[1]
    eff_nfft = min(n_fft, n_times)
    eff_noverlap = min(n_overlap, eff_nfft - 1)

    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax,
        n_fft=eff_nfft, n_overlap=eff_noverlap, verbose=False,
    )
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    total = np.sum(psd, axis=1) * df
    abs_p: Dict[str, np.ndarray] = {}
    rel_p: Dict[str, np.ndarray] = {}
    for bname, (blow, bhigh) in bands.items():
        mask = (freqs >= blow) & (freqs < bhigh)
        bp = np.sum(psd[:, mask], axis=1) * df
        abs_p[bname] = bp * 1e12
        rel_p[bname] = np.where(total > 0, bp / total, 0.0)
    return abs_p, rel_p


# ---------------------------------------------------------------------------
# 1.15  PROCESS ONE SEGMENT'S WINDOWS -> result rows
# ---------------------------------------------------------------------------

def window_overlaps_bad(
    raw: mne.io.Raw,
    win_start: float,
    win_end: float,
    threshold: float = BAD_OVERLAP_THRESHOLD,
) -> bool:
    """
    Return True if the fraction of BAD time within [win_start, win_end]
    exceeds ``threshold``.

    With threshold=0.25 (default), a 30 s window is rejected only if more
    than 7.5 s of it is covered by BAD annotations. Small, isolated BAD
    marks (e.g., 0.2 s blink artefacts) do NOT cause the whole window to
    be discarded.
    """
    win_dur = win_end - win_start
    if win_dur <= 0:
        return True

    bad_total = 0.0
    for ann in raw.annotations:
        if "BAD" not in ann["description"].upper():
            continue
        a_start = ann["onset"]
        a_end = a_start + ann["duration"]
        # Overlap duration
        ov_start = max(a_start, win_start)
        ov_end = min(a_end, win_end)
        if ov_end > ov_start:
            bad_total += ov_end - ov_start

    return (bad_total / win_dur) > threshold


def process_segment_windows(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    seg_type: str,
    seg_idx: int,
    start: float,
    end: Optional[float],
    eeg_ch_names: List[str],
) -> List[Dict[str, Any]]:
    """Return a list of flat dicts (one per channel x band x window).
    Windows where BAD time exceeds BAD_OVERLAP_THRESHOLD are skipped.
    """
    adj_s, adj_e = apply_segment_rules(seg_type, start, end)
    if adj_e <= adj_s:
        logger.warning(
            f"S{subj_id} | {seg_type} {seg_idx}: 0-duration after rules, skip"
        )
        return []

    windows = make_windows(adj_s, adj_e)
    if not windows:
        return []

    sfreq = raw.info["sfreq"]
    eeg_picks = mne.pick_channels(raw.ch_names, eeg_ch_names)
    rows: List[Dict[str, Any]] = []
    n_skipped = 0

    for wi, (ws, we) in enumerate(windows):
        # Skip windows where BAD fraction > BAD_OVERLAP_THRESHOLD
        if window_overlaps_bad(raw, ws, we):
            n_skipped += 1
            continue
        i0, i1 = int(ws * sfreq), int(we * sfreq)
        if i1 <= i0:
            continue
        i1 = min(i1, raw.n_times)
        data = raw.get_data(picks=eeg_picks, start=i0, stop=i1)

        win_dur = (i1 - i0) / sfreq
        flag = ""
        if win_dur < WINDOW_SEC - 0.5:
            flag = "short_window"

        if APPLY_DETREND:
            from scipy.signal import detrend
            data = detrend(data, axis=1, type="linear")

        ptp_uv = (data.max(axis=1) - data.min(axis=1)) * 1e6
        if ptp_uv.max() > ARTIFACT_PTP_THRESH_UV:
            flag = f"{flag};high_ptp" if flag else "high_ptp"

        abs_p, rel_p = compute_bandpower_welch(data, sfreq)

        for ci, ch in enumerate(eeg_ch_names):
            if ci >= data.shape[0]:
                break
            for bname in FREQ_BANDS:
                rows.append({
                    "subject":        subj_id,
                    "seg_type":       seg_type,
                    "seg_idx":        seg_idx,
                    "win_idx":        wi,
                    "channel":        ch,
                    "band":           bname,
                    "abs_power_uV2":  float(abs_p[bname][ci]),
                    "rel_power":      float(rel_p[bname][ci]),
                    "win_start_s":    ws,
                    "win_end_s":      we,
                    "win_duration_s": win_dur,
                    "flag":           flag,
                })

    n_kept = len(windows) - n_skipped
    logger.info(
        f"S{subj_id} | {seg_type} {seg_idx}: "
        f"{len(windows)} win total, {n_kept} kept, {n_skipped} rejected "
        f"(>{BAD_OVERLAP_THRESHOLD:.0%} BAD), {len(rows)} rows"
    )
    return rows


# ---------------------------------------------------------------------------
# 1.16  SPECTRAL DRIVER  (process all segments for one subject)
# ---------------------------------------------------------------------------

def process_all_segments_spectral(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> List[Dict[str, Any]]:
    """
    Loop over all segment types and intervals, compute band power,
    return all result rows.
    """
    avail_eeg = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    all_rows: List[Dict[str, Any]] = []

    # Count total intervals for progress tracking
    seg_types_to_process = [st for st in ("LB", "Intero", "BH", "Recov") if segments.get(st)]
    total_seg_types = len(seg_types_to_process)

    for seg_idx, seg_type in enumerate(("LB", "Intero", "BH", "Recov"), start=1):
        intervals = segments.get(seg_type, [])
        if not intervals:
            print(f"  {seg_type:<8s}:  no segments found")
            continue

        # Progress indicator for segment type
        print(f"  Processing {seg_type}... ", end="", flush=True)

        seg_rows: List[Dict[str, Any]] = []
        n_intervals = len(intervals)
        for si, (start, end) in enumerate(intervals, start=1):
            if end is None:
                print(f"\n    {seg_type} {si}: unpaired marker, skip")
                continue
            # Show interval progress for segment types with multiple intervals
            if n_intervals > 1:
                print(f"[{si}/{n_intervals}] ", end="", flush=True)
            rows = process_segment_windows(
                raw, subj_id, seg_type, si, start, end, avail_eeg,
            )
            seg_rows.extend(rows)

        n_win = len({(r["seg_idx"], r["win_idx"]) for r in seg_rows})
        print(f"done  ({len(intervals)} interval(s), {n_win} window(s) kept, {len(seg_rows)} rows)")
        if n_win == 0 and len(intervals) > 0:
            print(f"    !! All windows rejected (>{BAD_OVERLAP_THRESHOLD:.0%} BAD). "
                  f"Consider increasing BAD_OVERLAP_THRESHOLD.")
        all_rows.extend(seg_rows)

    return all_rows


def _save_segment_epochs(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    seg_type: str,
    intervals: List[Tuple[float, Optional[float]]],
    eeg_chs: List[str],
) -> None:
    """Save 30-s Epochs per segment for traceability (optional)."""
    sfreq = raw.info["sfreq"]
    events_list = []
    for si, (start, end) in enumerate(intervals, start=1):
        if end is None:
            continue
        adj_s, adj_e = apply_segment_rules(seg_type, start, end)
        for ws, we in make_windows(adj_s, adj_e):
            if we - ws >= WINDOW_SEC - 0.5:
                events_list.append([int(ws * sfreq), 0, si])
    if not events_list:
        return
    events = np.array(events_list, dtype=int)
    try:
        epochs = mne.Epochs(
            raw, events, tmin=0, tmax=WINDOW_SEC - 1 / sfreq,
            baseline=None, preload=True, picks=eeg_chs, verbose=False,
        )
        out = subject_fif_dir(subj_id) / f"epochs_{seg_type}-epo.fif"
        epochs.save(str(out), overwrite=OVERWRITE_FIF, verbose=False)
        logger.debug(f"S{subj_id} | Saved {out.name}")
    except Exception as exc:
        logger.warning(f"S{subj_id} | Epochs save failed: {exc}")


# ---------------------------------------------------------------------------
# 1.17  AGGREGATION
# ---------------------------------------------------------------------------

def compute_segment_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Average band power across windows -> one row per segment x channel x band."""
    grp = ["subject", "seg_type", "seg_idx", "channel", "band"]
    return (
        df.groupby(grp, as_index=False)
        .agg(
            abs_power_mean=("abs_power_uV2", "mean"),
            rel_power_mean=("rel_power", "mean"),
            n_windows=("win_idx", "nunique"),
        )
    )


def compute_region_averages(
    df_seg: pd.DataFrame,
    region_map: Dict[str, List[str]] = REGION_MAP,
) -> pd.DataFrame:
    """Average segment-level band power across channels within each region."""
    rows: List[Dict[str, Any]] = []
    grp_cols = ["subject", "seg_type", "seg_idx", "band"]
    for _, g in df_seg.groupby(grp_cols):
        for region, chs in region_map.items():
            mask = g["channel"].isin(chs)
            if mask.sum() == 0:
                continue
            sub = g.loc[mask]
            rows.append({
                "subject":        g["subject"].iloc[0],
                "seg_type":       g["seg_type"].iloc[0],
                "seg_idx":        g["seg_idx"].iloc[0],
                "band":           g["band"].iloc[0],
                "region":         region,
                "abs_power_mean": sub["abs_power_mean"].mean(),
                "rel_power_mean": sub["rel_power_mean"].mean(),
                "n_channels":     int(mask.sum()),
                "n_windows":      int(sub["n_windows"].iloc[0]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1.18  QC DATAFRAMES
# ---------------------------------------------------------------------------

def build_qc_markers_df(
    all_seg: Dict, 
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid, segs in all_seg.items():
        for lab, ivs in segs.items():
            for i, (s, e) in enumerate(ivs):
                dur = (e - s) if e is not None else None
                flag = ""
                if e is None:
                    flag = "unpaired"
                elif dur is not None and dur <= 0:
                    flag = "negative_duration"
                elif dur is not None and dur < 2.0:
                    flag = "very_short"
                rows.append({
                    "subject": sid, "seg_type": lab, "seg_idx": i + 1,
                    "start_s": s, "end_s": e, "duration_s": dur, "flag": flag,
                })
    return pd.DataFrame(rows)


def build_qc_bad_channels_df(
    all_bads: Dict,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid, bads in all_bads.items():
        if bads:
            for ch in bads:
                rows.append({"subject": sid, "bad_channel": ch, "interpolated": True})
        else:
            rows.append({"subject": sid, "bad_channel": "none", "interpolated": False})
    return pd.DataFrame(rows)


def build_qc_ica_df(
    all_ica: Dict,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid, info in all_ica.items():
        rows.append({
            "subject": sid,
            "n_components": info.get("n_components", None),
            "excluded_indices": str(info.get("excluded", [])),
            "method": info.get("method", ICA_METHOD),
            "figures_dir": str(subject_qc_dir(sid)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1.19  EXCEL EXPORT
# ---------------------------------------------------------------------------

def export_excel(
    df_long: pd.DataFrame,
    df_seg: pd.DataFrame,
    df_reg: pd.DataFrame,
    df_qc_mk: pd.DataFrame,
    df_qc_bad: pd.DataFrame,
    df_qc_ica: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(path), engine="openpyxl") as w:
        df_long.to_excel(w, sheet_name="bandpower_long", index=False)
        df_seg.to_excel(w, sheet_name="bandpower_segment_avg", index=False)
        df_reg.to_excel(w, sheet_name="bandpower_region_avg", index=False)
        df_qc_mk.to_excel(w, sheet_name="qc_markers", index=False)
        df_qc_bad.to_excel(w, sheet_name="qc_bad_channels", index=False)
        df_qc_ica.to_excel(w, sheet_name="qc_ica", index=False)
    logger.info(f"Excel saved -> {path}")


# ========================  END OF CELL 1  ==================================


#%%
# =============================================================================
# CELL 2 — LOAD RAW + ANNOTATIONS + SEGMENT INSPECTION  (BREAKPOINT)
# =============================================================================
# This cell ONLY loads the data and inspects markers.  No preprocessing yet.
#
# 1. Edit SUJETO and EEG_FILE below.
# 2. Run this cell.
# 3. Review the annotation list and segment inspection output.
# 4. If annotations need fixing, add a new cell below this one to patch
#    ALL_SEGMENTS before continuing.  Example:
#
#        ALL_SEGMENTS[SUJETO]["BH"] = ALL_SEGMENTS[SUJETO]["BH"][:4]
#        save_segments_json(SUJETO, ALL_SEGMENTS[SUJETO])
#
# =============================================================================

# ============================================================================
# >>> EDIT HERE: Subject ID and EEG file path <<<
# ============================================================================
SUJETO = "16"
EEG_FILE = r"C:\Users\marag\Desktop\eeg_bh\participantes_eeg_bh\16eeg.vhdr"
# ============================================================================

# --- runtime initialisation ------------------------------------------------
mne.set_log_level("WARNING")
setup_logging()
ensure_output_dirs()

_eeg_path = Path(EEG_FILE)
if not _eeg_path.exists():
    raise FileNotFoundError(f"EEG file not found: {EEG_FILE}")

# --- Validate SUJETO matches EEG_FILE --------------------------------------
# Extract numeric part from filename (e.g., "10eeg.vhdr" -> "10")
import re as _re
_filename_match = _re.search(r"(\d+)", _eeg_path.stem)
if _filename_match:
    _file_subj_id = _filename_match.group(1)
    if _file_subj_id != str(SUJETO):
        warnings.warn(
            f"\n  !! MISMATCH WARNING: SUJETO='{SUJETO}' but EEG file is '{_eeg_path.name}' "
            f"(extracted ID: '{_file_subj_id}')\n"
            f"  !! Please verify that SUJETO and EEG_FILE are correct before proceeding.\n",
            UserWarning,
        )

# --- Step 0: load raw (no preprocessing) ----------------------------------
print(f"Loading subject {SUJETO}  ({EEG_FILE})...")
RAW = step0_load_raw(SUJETO)
print(f"  {len(RAW.ch_names)} channels, {RAW.info['sfreq']:.0f} Hz, "
      f"{RAW.times[-1]:.1f} s\n")
save_step(RAW, SUJETO, "step0_loaded")

# --- List all raw annotations ----------------------------------------------
ann_report = list_raw_annotations(RAW, SUJETO)
(subject_qc_dir(SUJETO) / "raw_annotations.txt").write_text(
    ann_report, encoding="utf-8",
)

# --- Reconstruct & inspect segments ----------------------------------------
_label_onsets = parse_annotations(RAW)
ALL_SEGMENTS: Dict[str, Dict[str, List[Tuple[float, Optional[float]]]]] = {}
ALL_SEGMENTS[SUJETO] = reconstruct_segments(_label_onsets)
save_segments_json(SUJETO, ALL_SEGMENTS[SUJETO])

_seg_report = inspect_segments(SUJETO, ALL_SEGMENTS[SUJETO], RAW.info["sfreq"])
(subject_qc_dir(SUJETO) / "segment_inspection.txt").write_text(
    _seg_report, encoding="utf-8",
)

print("\n>>> Review the annotations and segments above.")
print(">>> If corrections are needed, add a cell below to patch ALL_SEGMENTS")
print(">>> before running Cell 3.\n")


# =============================================================================
# Track which segments have been inspected/cleaned
# =============================================================================
SEGMENTS_INSPECTED: Dict[str, bool] = {
    "LB": False, "Intero": False, "BH": False, "Recov": False,
}


#%%
# =============================================================================
# CELL 3a — INSPECT RAW: LB (Baseline)
# =============================================================================
# Visualize LB segment (first 300s). Mark bad channels in the interactive viewer.
# =============================================================================

INTERACTIVE_VIEW = True   # <<< set to False to skip interactive windows

_qc_dir = subject_qc_dir(SUJETO)
_segments = ALL_SEGMENTS[SUJETO]

# In the interactive viewer: press 'a' then click-drag to mark BAD sections
# BAD annotations are automatically applied to RAW when you close the viewer
inspect_and_mark_segment(
    RAW, SUJETO, "LB", _segments,
    tag="raw", interactive=INTERACTIVE_VIEW, apply_time_rules=True,
)
SEGMENTS_INSPECTED["LB"] = True


#%%
# =============================================================================
# CELL 3b — INSPECT RAW: Intero
# =============================================================================

inspect_and_mark_segment(
    RAW, SUJETO, "Intero", _segments,
    tag="raw", interactive=INTERACTIVE_VIEW, apply_time_rules=False,
)
SEGMENTS_INSPECTED["Intero"] = True


#%%
# =============================================================================
# CELL 3c — INSPECT RAW: BH (Breath Holds)
# =============================================================================

inspect_and_mark_segment(
    RAW, SUJETO, "BH", _segments,
    tag="raw", interactive=INTERACTIVE_VIEW, apply_time_rules=False,
)
SEGMENTS_INSPECTED["BH"] = True


#%%
# =============================================================================
# CELL 3d — INSPECT RAW: Recov (Recovery)
# =============================================================================

inspect_and_mark_segment(
    RAW, SUJETO, "Recov", _segments,
    tag="raw", interactive=INTERACTIVE_VIEW, apply_time_rules=True,
)
SEGMENTS_INSPECTED["Recov"] = True


#%%
# =============================================================================
# CELL 3e — SUMMARY OF MANUAL EDITS
# =============================================================================

print("=" * 70)
print(f"  Subject {SUJETO} -- Summary of manual edits")
print("=" * 70)

print(f"\n  Bad channels marked: {RAW.info['bads'] if RAW.info['bads'] else 'None'}")

# Per-segment BAD summary
_segments = ALL_SEGMENTS[SUJETO]
_summary_rows = []
_total_bad = 0.0
_total_dur = 0.0

for _seg_type in ["LB", "Intero", "BH", "Recov"]:
    _segs = get_segments_by_type(_segments, _seg_type, apply_time_rules=True)
    for _seg_name, _seg_idx, _start, _end in _segs:
        _seg_dur = _end - _start
        _bad_time = 0.0
        _n_bad = 0
        for _a in RAW.annotations:
            if not _a["description"].upper().startswith("BAD"):
                continue
            _a_start = _a["onset"]
            _a_end = _a_start + _a["duration"]
            _ov_start = max(_a_start, _start)
            _ov_end = min(_a_end, _end)
            if _ov_end > _ov_start:
                _bad_time += _ov_end - _ov_start
                _n_bad += 1
        _pct = (_bad_time / _seg_dur * 100) if _seg_dur > 0 else 0
        _total_bad += _bad_time
        _total_dur += _seg_dur
        _summary_rows.append({
            "Segment": _seg_name,
            "Duration (s)": f"{_seg_dur:.1f}",
            "BAD (s)": f"{_bad_time:.1f}",
            "BAD (%)": f"{_pct:.1f}%",
            "# marks": _n_bad,
        })

_df_pre = pd.DataFrame(_summary_rows)
print(f"\n{_df_pre.to_string(index=False)}")

_total_pct = (_total_bad / _total_dur * 100) if _total_dur > 0 else 0
print(f"\n  Total: {_total_bad:.1f}s BAD out of {_total_dur:.1f}s ({_total_pct:.1f}%)")
print("\n>>> Proceed to Cell 4 for preprocessing.\n")


#%%
# =============================================================================
# CELL 4 — PREPROCESSING  (Steps 1-7)
# =============================================================================
# Runs the full signal-processing chain on the loaded raw:
#   Step 1: Channel selection + types
#   Step 2: Montage
#   Step 3: Notch + bandpass filtering
#   Step 4: Average re-reference
#   Step 5: Bad-channel detection + interpolation (includes manual bads from Cell 3)
#   Step 6: Gross-artifact annotation
#   Step 7: ICA (EOG component removal)
#
# Intermediate .fif files are saved after each step.
# =============================================================================

print("=" * 70)
print(f"  Subject {SUJETO} -- Preprocessing pipeline")
print("=" * 70 + "\n")

ALL_BAD_CHANNELS: Dict[str, List[str]] = {}
ALL_ICA_INFO: Dict[str, Dict[str, Any]] = {}

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        RAW_CLEAN, _bads, _ica_info = run_preprocessing_steps(RAW, SUJETO)
    ALL_BAD_CHANNELS[SUJETO] = _bads
    ALL_ICA_INFO[SUJETO] = _ica_info
    print(f"\n{'='*70}")
    print(f"  PREPROCESSING COMPLETE  --  Subject {SUJETO}")
    print(f"{'='*70}\n")
except Exception as _exc:
    print(f"\n!! PREPROCESSING FAILED: {_exc}")
    logger.error(f"S{SUJETO} | PREPROCESSING FAILED:\n{traceback.format_exc()}")
    RAW_CLEAN = None


#%%
# =============================================================================
# CELL 5a — INSPECT CLEANED: LB (Baseline)
# =============================================================================
# Verify preprocessing. Mark any remaining artifacts in the interactive viewer.
# =============================================================================

INTERACTIVE_VIEW_CLEAN = True   # <<< set to False to skip interactive windows

if RAW_CLEAN is None:
    print("!! Cleaned data not available. Fix Cell 4 errors first.")
else:
    inspect_and_mark_segment(
        RAW_CLEAN, SUJETO, "LB", ALL_SEGMENTS[SUJETO],
        tag="clean", interactive=INTERACTIVE_VIEW_CLEAN, apply_time_rules=True,
    )


#%%
# =============================================================================
# CELL 5b — INSPECT CLEANED: Intero
# =============================================================================

if RAW_CLEAN is None:
    print("!! Cleaned data not available.")
else:
    inspect_and_mark_segment(
        RAW_CLEAN, SUJETO, "Intero", ALL_SEGMENTS[SUJETO],
        tag="clean", interactive=INTERACTIVE_VIEW_CLEAN, apply_time_rules=False,
    )


#%%
# =============================================================================
# CELL 5c — INSPECT CLEANED: BH (Breath Holds)
# =============================================================================

if RAW_CLEAN is None:
    print("!! Cleaned data not available.")
else:
    inspect_and_mark_segment(
        RAW_CLEAN, SUJETO, "BH", ALL_SEGMENTS[SUJETO],
        tag="clean", interactive=INTERACTIVE_VIEW_CLEAN, apply_time_rules=False,
    )


#%%
# =============================================================================
# CELL 5d — INSPECT CLEANED: Recov (Recovery)
# =============================================================================

if RAW_CLEAN is None:
    print("!! Cleaned data not available.")
else:
    inspect_and_mark_segment(
        RAW_CLEAN, SUJETO, "Recov", ALL_SEGMENTS[SUJETO],
        tag="clean", interactive=INTERACTIVE_VIEW_CLEAN, apply_time_rules=True,
    )


#%%
# =============================================================================
# CELL 6 — PREPROCESSING SUMMARY & SAVE CLEANED DATA
# =============================================================================
# Shows a summary table of all changes applied during preprocessing,
# then saves the cleaned data to output/clean_data/ for future analyses.
# =============================================================================

save_and_report_clean(
    RAW_CLEAN, SUJETO, ALL_SEGMENTS[SUJETO],
    ALL_BAD_CHANNELS.get(SUJETO, []),
    ALL_ICA_INFO.get(SUJETO, {}),
)


#%%
# =============================================================================
# CELL 7 — SPECTRAL POWER COMPUTATION  (only inspected segments)
# =============================================================================
# Computes Welch PSD band power ONLY for segments that were inspected.
# This ensures the Excel output contains only cleaned data.
# =============================================================================

# Load cleaned data if not already in memory
if "RAW_CLEAN" not in dir() or RAW_CLEAN is None:
    print("  Loading cleaned data from disk...")
    RAW_CLEAN = load_clean_data(SUJETO)

if RAW_CLEAN is None:
    print("!! Cleaned data not available. Run preprocessing first.")
else:
    # Filter segments to only those that were inspected
    _inspected_types = [k for k, v in SEGMENTS_INSPECTED.items() if v]
    if not _inspected_types:
        print("!! No segments were inspected. Run Cell 3a-3d first.")
        ALL_RESULTS = []
    else:
        print("=" * 70)
        print(f"  Subject {SUJETO} -- Spectral band-power computation")
        print(f"  Segments: {_inspected_types}")
        print("=" * 70 + "\n")

        # Build filtered segments dict with only inspected types
        _filtered_segments = {
            k: v for k, v in ALL_SEGMENTS[SUJETO].items()
            if k in _inspected_types
        }

        ALL_RESULTS = process_all_segments_spectral(
            RAW_CLEAN, SUJETO, _filtered_segments,
        )

        _n_win = len({(r["seg_type"], r["seg_idx"], r["win_idx"]) for r in ALL_RESULTS})
        print(f"\n  Total: {_n_win} windows, {len(ALL_RESULTS)} result rows\n")


#%%
# =============================================================================
# CELL 8 — EXCEL EXPORT (per-subject)
# =============================================================================
# Exports the spectral power results for this subject.
# Only includes segments that were inspected and cleaned.
# =============================================================================

print("=" * 70)
print(f"  Subject {SUJETO} -- Exporting per-subject Excel")
print("=" * 70)

df_long = pd.DataFrame(ALL_RESULTS)

if df_long.empty:
    print("\n  !! No spectral results to export.  Check previous cells for errors.\n")
else:
    # --- Segment-level averages --------------------------------------------
    df_seg_avg = compute_segment_averages(df_long)

    # --- Region-level averages ---------------------------------------------
    df_region_avg = compute_region_averages(df_seg_avg)

    # --- QC tables ---------------------------------------------------------
    df_qc_mk  = build_qc_markers_df(ALL_SEGMENTS)
    df_qc_bad = build_qc_bad_channels_df(ALL_BAD_CHANNELS)
    df_qc_ica = build_qc_ica_df(ALL_ICA_INFO)

    # --- Per-subject Excel -------------------------------------------------
    _excel_path = EXCEL_DIR / f"subject_{SUJETO}_eeg_bandpower.xlsx"
    export_excel(
        df_long, df_seg_avg, df_region_avg,
        df_qc_mk, df_qc_bad, df_qc_ica,
        _excel_path,
    )

    print(f"\n  Rows exported : {len(df_long):,}")
    print(f"  Bands         : {df_long['band'].nunique()}")
    print(f"  Channels      : {df_long['channel'].nunique()}")
    print(f"  Segments      : {list(df_long['seg_type'].unique())}")
    print(f"\n  Excel -> {_excel_path}")

print("\n  Subject processing finished.\n")


#%%
# =============================================================================
# CELL 9 — AGGREGATE ALL SUBJECTS (run after processing all subjects)
# =============================================================================
# Run this cell AFTER you have processed all subjects individually.
# =============================================================================

print("=" * 70)
print("  Aggregating all subjects into global Excel")
print("=" * 70 + "\n")

aggregate_subject_excels(EXCEL_DIR)

print("\n  Done.\n")


#%%
# =============================================================================
# CELL 10 — TOPOGRAPHIC MAPS (per-subject visualization)
# =============================================================================
# Creates topographic maps showing the spatial distribution of spectral power
# for each frequency band and segment type.
#
# Two visualization options:
#   1. Separate figures per segment (detailed view)
#   2. Single comparison grid (all segments × all bands)
#
# Run this AFTER Cell 8 (Excel export) since it uses the segment-averaged data.
# =============================================================================

print("=" * 70)
print(f"  Subject {SUJETO} -- Topographic Maps")
print("=" * 70)

# Check if we have the segment-averaged data
if "df_seg_avg" not in dir() or df_seg_avg is None or df_seg_avg.empty:
    # Try to load from the Excel file
    _excel_path = EXCEL_DIR / f"subject_{SUJETO}_eeg_bandpower.xlsx"
    if _excel_path.exists():
        print(f"  Loading data from {_excel_path.name}...")
        df_seg_avg = pd.read_excel(_excel_path, sheet_name="bandpower_segment_avg")
    else:
        print(f"  !! Excel file not found: {_excel_path}")
        print("  !! Run Cell 8 first to generate the data.")
        df_seg_avg = pd.DataFrame()

if not df_seg_avg.empty:
    # Diagnostic: show what segments are in the data
    _found_segs = sorted(df_seg_avg[
        df_seg_avg["subject"].astype(str) == str(SUJETO)
    ]["seg_type"].unique())
    print(f"  Segments in data: {_found_segs}")
    if len(_found_segs) == 0:
        print(f"  !! No segments found for subject {SUJETO} in df_seg_avg.")
        print(f"     Unique subjects in data: {sorted(df_seg_avg['subject'].unique())}")
        print(f"     Make sure Cell 7 & 8 ran with all segments inspected.")
    else:
        _topo_dir = subject_qc_dir(SUJETO)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # 1) Individual segment topomaps (absolute power, per-segment colour scale)
            print("\n  1. Individual segment topomaps (absolute power)...")
            plot_subject_topomaps(df_seg_avg, SUJETO, save_dir=_topo_dir,
                                  power_type="abs_power_mean")

            # 2) Comparison grid — absolute power (robust percentile scaling)
            print("\n  2. Comparison grid — absolute power (robust colour scale)...")
            plot_subject_topomaps_comparison(df_seg_avg, SUJETO, save_dir=_topo_dir,
                                             power_type="abs_power_mean")

            # 3) Comparison grid — relative power (normalised within each window)
            print("\n  3. Comparison grid — relative power...")
            plot_subject_topomaps_comparison(df_seg_avg, SUJETO, save_dir=_topo_dir,
                                             power_type="rel_power_mean")

        print(f"\n  Topomaps saved to: {_topo_dir}")
else:
    print("  !! No data available for topomaps.")

print("\n  Done.\n")

# %%
