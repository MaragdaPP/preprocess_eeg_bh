# -*- coding: utf-8 -*-
"""
EEG Pre-processing Pipeline for BrainVision (32-electrode) Data
================================================================
Experiment: Interoception / Breath-Hold study
Segments  : LB (baseline), BH 1-4, CPT (full duration), Recov (recovery)
Output    : Cleaned EEG .fif files plus QC summaries for downstream analysis

This script is organised in VSCode-style cells (#%%).
Cell 1 contains ALL imports, configuration, and function definitions.
Subsequent cells follow an interactive inspect-first workflow:

  Cell 1 — Imports, config, ALL function definitions
  Cell 2 — Load raw + list annotations + segment inspection  (BREAKPOINT)
  Cell 3 — Visualize RAW signal segment by segment
  Cell 4 — Preprocessing (Steps 1–8)
  Optional — ICA component viewer (LB segment only; read-only QC)
  Cell 5 — Visualize CLEANED signal segment by segment
  Cell 6 — Preprocessing summary + save cleaned data
  Downstream spectral analysis lives in ``eeg_spectral_analysis.py``

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
EOG_CHANNEL: str = "TP9"            # relocated below left eye, used as EOG proxy
ACQUISITION_REFERENCE: str = "FCz"  # nominal (actual varies per subject)

# --- Filtering parameters --------------------------------------------------
NOTCH_FREQ: float         = 50.0    # Hz  (line noise)
NOTCH_HARMONICS: bool     = True    # also notch at 100 Hz
ICA_FIT_HIGH_PASS: float  = 1.0     # Hz  (temporary dataset used only to fit ICA)
FINAL_HIGH_PASS: float    = 0.1     # Hz  (final cleaned dataset keeps low frequencies)
FILTER_LOW_PASS: float    = 40.0    # Hz  (shared low-pass for ICA-fit and final dataset)

# --- ICA parameters --------------------------------------------------------
ICA_METHOD: str                      = "fastica"
ICA_N_COMPONENTS: Optional[float]    = 0.999999  # variance-based (avoids unstable mixing matrix)
ICA_RANDOM_STATE: int                = 42
ICA_MAX_ITER: int                    = 1000

# --- Bad-channel detection --------------------------------------------------
USE_PYPREP: bool           = False
BAD_CHAN_ZSCORE_THRESH: float = 4.0   # robust z-score on windowed variance; higher = less aggressive
FLATLINE_THRESH_UV: float  = 0.5     # uV - channels with std < this
BAD_CHAN_WINDOW_SEC: float = 2.0     # variance is evaluated in short windows
BAD_CHAN_PERSIST_RATIO: float = 0.30  # channel must be outlier in >=30% of windows
BAD_CHAN_MAX_INTERP_RATIO: float = 0.10  # >10% bad channels → warn, likely reject segment

# --- Artifact annotation ---------------------------------------------------
# Peak-to-peak threshold per 1-s window (uV). Increase if too many segments
# are flagged as artifacts (200-250 is moderate; 300-400 is more lenient).
# Set to 0 or very high (e.g., 9999) to disable automatic artifact detection.
ARTIFACT_PTP_THRESH_UV: float = 250.0   # less aggressive; avoids flagging too many blinks
AUTO_ARTIFACT_DETECTION: bool = True    # Set to False to skip automatic detection
# Muscle artifact detection: gamma-band (30-40 Hz) power threshold per 1-s window.
# Windows where ANY channel exceeds this are flagged BAD_muscle.
# Units: uV^2/Hz. ~5.0 is a reasonable starting threshold; lower = more aggressive.
MUSCLE_GAMMA_THRESH_UV2: float = 7.5   # uV^2/Hz; slightly less aggressive than before

# --- Inspection / artifact helper parameters -------------------------------
MUSCLE_GAMMA_BAND: Tuple[float, float] = (30.0, 40.0)
INSPECT_PSD_N_FFT: int = 2048
INSPECT_PSD_N_OVERLAP: int = 1024
INSPECT_PSD_FMIN: float = 1.0
INSPECT_PSD_FMAX: float = 40.0

# --- Segment time rules ----------------------------------------------------
LB_MAX_SEC: float       = 300.0    # LB: keep first 5 min max
RECOV_MAX_SEC: float    = 300.0    # Recov: keep first 5 min max
# CPT: no duration cap — all available data is preprocessed
BH_TRIM_END_SEC: float  = 10.0     # BH: drop last 10 s
PREPROCESS_SEG_TYPES: Tuple[str, ...] = ("LB", "BH", "Recov")
CONCAT_BH_FOR_PREPROC: bool = True  # treat BH 1-4 as one preprocessing unit

# --- Segment-label expectations (soft; used only for warnings) -------------
# BH: 4 segments (8 markers)
EXPECTED_MARKER_COUNTS: Dict[str, int] = {
    "LB": 2, "Recov": 2, "BH": 8, "CPT": 2,
}

# --- File saving flags ------------------------------------------------------
# Set to False to reduce disk usage (only final cleaned data will be saved)
SAVE_INTERMEDIATE_FIF: bool = True    # Save .fif after each step (needed for optional ICA sources viewer)
OVERWRITE_FIF: bool         = True
# Controls QC/diagnostic files: ICA solution .fif, ICA topomap PNGs,
# segment overview PNGs, and preprocessing_summary.csv.
# The clean data .fif is always saved regardless.
# ica_suggest.json is also always saved (needed for manual ICA override).
SAVE_QC_FILES: bool         = True    # Set False to keep only cleaned data

# --- Label normalisation map ------------------------------------------------
_LABEL_NORMALIZE: Dict[str, str] = {
    "lb":       "LB",
    "bh":       "BH",
    "recov":    "Recov",
    "recovery": "Recov",
    "cpt":      "CPT",
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
    console_fmt = logging.Formatter("%(levelname)s | %(message)s")

    # Console: only warnings by default (keep output clean)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(console_fmt)
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
    for d in (OUTPUT_DIR, FIF_DIR, QC_DIR):
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
        # Strip BrainVision Type/ prefix (e.g. "Comment/CPT" -> "CPT")
        if "/" in desc:
            desc = desc.split("/", 1)[-1].strip()
        # Normalize key: collapse whitespace, lower case (handles "CPT ", "  cpt", etc.)
        key = " ".join(desc.lower().split()) if desc else ""
        canonical = _LABEL_NORMALIZE.get(key)
        # Fallback: try without spaces (e.g. "C P T" from some export)
        if canonical is None and key:
            canonical = _LABEL_NORMALIZE.get(key.replace(" ", ""))
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
    for label in ("LB", "BH", "Recov", "CPT"):
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
        ("LB",    "LB",    1),
        ("BH",    "BH",    4),
        ("CPT",   "CPT",   1),
        ("Recov", "Recov", 1),
    ]

    for label, prefix, expected_n in display_order:
        intervals = segments.get(label, [])
        if not intervals:
            lines.append(f"  {prefix:<14s}:  NOT FOUND (no markers)")
            warns.append(f"{label}: no markers found")
            continue

        n_markers = sum(2 if e is not None else 1 for _, e in intervals)
        if label == "BH" and segments.get("_BH_labels"):
            exp_mk = 2 * len(intervals)
        else:
            exp_mk = EXPECTED_MARKER_COUNTS.get(label)
        if exp_mk and n_markers != exp_mk:
            warns.append(f"{label}: expected {exp_mk} markers, found {n_markers}")

        custom_labels = segments.get("_BH_labels") if label == "BH" else None
        for idx, (start, end) in enumerate(intervals):
            if custom_labels and idx < len(custom_labels):
                seg_label = custom_labels[idx]
            else:
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
        if not lab.startswith("_")
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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
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
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    high_pass: float,
    low_pass: float,
    dataset_label: str,
) -> mne.io.Raw:
    logger.info(f"S{subj_id} | Step 3  Filtering ({dataset_label})")
    freqs = [NOTCH_FREQ]
    if NOTCH_HARMONICS:
        freqs.append(NOTCH_FREQ * 2)
    raw.notch_filter(freqs, verbose=False)
    raw.filter(high_pass, low_pass, verbose=False)
    logger.info(
        f"S{subj_id} | {dataset_label}: Notch {freqs} Hz, BP {high_pass}-{low_pass} Hz"
    )
    return raw


# ---- Step 5: Bad channels ------------------------------------------------

def _robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad = max(mad, 1e-12)
    return (x - med) / (mad * 1.4826)


def step5_bad_channels(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    context: Optional[str] = None,
) -> Tuple[mne.io.Raw, List[str]]:
    """Detect bad EEG channels (windowed variance / flatline heuristic).

    Only marks channels in raw.info["bads"] — does NOT interpolate.
    Interpolation happens after ICA in step_interpolate_bad_channels().

    FCz (ACQUISITION_REFERENCE) is excluded from detection: it is flat by
    design (re-added as zero-signal channel) and handled in step_rereference.
    """
    scope = f" [{context}]" if context else ""
    logger.info(f"S{subj_id} | Step 4  Bad-channel detection{scope}")

    eeg_idx = mne.pick_types(raw.info, eeg=True, eog=False)
    eeg_names = [raw.ch_names[i] for i in eeg_idx]

    # Exclude ACQUISITION_REFERENCE from detection — it is flat by design
    ref_ch = ACQUISITION_REFERENCE
    eeg_names_detect = [n for n in eeg_names if n != ref_ch]
    detect_idx = [eeg_idx[eeg_names.index(n)] for n in eeg_names_detect]

    data = raw.get_data(picks=detect_idx)
    bads: List[str] = []

    if USE_PYPREP and HAS_PYPREP:
        try:
            nd = NoisyChannels(raw, do_detrend=True, random_state=ICA_RANDOM_STATE)
            nd.find_all_bads()
            pyprep_bads = [b for b in nd.get_bads() if b != ref_ch]
            bads = list(pyprep_bads)
            logger.info(f"S{subj_id} | pyprep bads: {bads}")
        except Exception as exc:
            logger.warning(f"S{subj_id} | pyprep failed ({exc}), using heuristic")

    if not bads:
        ch_std = np.std(data, axis=1)
        for i, name in enumerate(eeg_names_detect):
            if ch_std[i] * 1e6 < FLATLINE_THRESH_UV:
                bads.append(name)
                logger.info(f"S{subj_id} | Flatline: {name}  std={ch_std[i]*1e6:.4f} uV")

        win_samp = max(1, int(BAD_CHAN_WINDOW_SEC * raw.info["sfreq"]))
        n_windows = data.shape[1] // win_samp
        if n_windows > 0:
            bad_counts = np.zeros(len(eeg_names_detect), dtype=int)
            for win_idx in range(n_windows):
                start = win_idx * win_samp
                stop = start + win_samp
                win_var = np.var(data[:, start:stop], axis=1)
                zsc = _robust_zscore(win_var)
                bad_counts += (zsc > BAD_CHAN_ZSCORE_THRESH).astype(int)

            bad_ratio = bad_counts / n_windows
            for i, name in enumerate(eeg_names_detect):
                if name in bads:
                    continue
                if bad_ratio[i] >= BAD_CHAN_PERSIST_RATIO:
                    bads.append(name)
                    logger.info(
                        f"S{subj_id} | Persistent variance outlier: {name}  "
                        f"{bad_counts[i]}/{n_windows} windows ({bad_ratio[i]*100:.0f}%)"
                    )

    raw.info["bads"] = bads
    logger.info(f"S{subj_id} | Bad channels marked{scope}: {bads if bads else 'none'}")
    logger.info(f"S{subj_id} | Step 4 done")
    return raw, bads


# ---- Step 6: Artifact annotation -----------------------------------------

def step6_annotate_artifacts(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    context: Optional[str] = None,
) -> Tuple[mne.io.Raw, mne.Annotations]:
    """Mark 1-s windows as BAD based on two automatic checks:

    1. **BAD_artifact**: peak-to-peak amplitude exceeds ARTIFACT_PTP_THRESH_UV (gross
       blinks, movement, electrode pops).
    2. **BAD_muscle**: gamma-band (30-40 Hz) power exceeds MUSCLE_GAMMA_THRESH_UV2 on
       any channel (high-frequency bursts from muscle activity).

    Windows already flagged as BAD_artifact are not double-flagged as BAD_muscle.
    If AUTO_ARTIFACT_DETECTION is False, this step is skipped entirely.
    """
    scope = f" [{context}]" if context else ""
    logger.info(f"S{subj_id} | Step 5  Artifact annotation{scope}")

    if not AUTO_ARTIFACT_DETECTION:
        logger.info(f"S{subj_id} | Auto artifact detection disabled, skipping{scope}")
        return raw, mne.Annotations([], [], [], orig_time=raw.annotations.orig_time)

    eeg_idx = mne.pick_types(raw.info, eeg=True, eog=False)
    sfreq = raw.info["sfreq"]
    win_samp = int(1.0 * sfreq)
    data = raw.get_data(picks=eeg_idx)
    n_samp = data.shape[1]
    thresh_v = ARTIFACT_PTP_THRESH_UV * 1e-6

    # Gamma band limits for muscle detection
    gamma_low, gamma_high = MUSCLE_GAMMA_BAND
    muscle_thresh_v2hz = MUSCLE_GAMMA_THRESH_UV2 * 1e-12  # uV^2/Hz -> V^2/Hz

    ptp_onsets, ptp_durs = [], []
    muscle_onsets, muscle_durs = [], []

    for s in range(0, n_samp - win_samp, win_samp):
        seg = data[:, s : s + win_samp]
        t_onset = s / sfreq

        # --- PTP check ---
        ptp = seg.max(axis=1) - seg.min(axis=1)
        if ptp.max() > thresh_v:
            ptp_onsets.append(t_onset)
            ptp_durs.append(1.0)
            continue  # don't also flag as muscle if already flagged

        # --- Gamma-band muscle check ---
        try:
            eff_nfft = min(INSPECT_PSD_N_FFT, win_samp)
            eff_noverlap = min(INSPECT_PSD_N_OVERLAP, eff_nfft - 1)
            psd_win, freqs_win = mne.time_frequency.psd_array_welch(
                seg, sfreq=sfreq,
                fmin=gamma_low, fmax=gamma_high,
                n_fft=eff_nfft, n_overlap=eff_noverlap,
                verbose=False,
            )
            # psd_win: (n_ch, n_freqs) in V^2/Hz; average across gamma band
            gamma_mean = psd_win.mean(axis=1)  # per channel
            if gamma_mean.max() > muscle_thresh_v2hz:
                muscle_onsets.append(t_onset)
                muscle_durs.append(1.0)
        except Exception:
            pass  # too short or degenerate window — skip silently

    auto_annotations = mne.Annotations([], [], [], orig_time=raw.annotations.orig_time)
    new_annotations = raw.annotations
    if ptp_onsets:
        ptp_annotations = mne.Annotations(
            onset=ptp_onsets, duration=ptp_durs,
            description=["BAD_artifact"] * len(ptp_onsets),
            orig_time=raw.annotations.orig_time,
        )
        auto_annotations = auto_annotations + ptp_annotations
        new_annotations = new_annotations + ptp_annotations
        logger.info(f"S{subj_id} | {len(ptp_onsets)} s flagged as BAD_artifact{scope}")
    else:
        logger.info(f"S{subj_id} | No gross artefacts detected (PTP){scope}")

    if muscle_onsets:
        muscle_annotations = mne.Annotations(
            onset=muscle_onsets, duration=muscle_durs,
            description=["BAD_muscle"] * len(muscle_onsets),
            orig_time=raw.annotations.orig_time,
        )
        auto_annotations = auto_annotations + muscle_annotations
        new_annotations = new_annotations + muscle_annotations
        logger.info(f"S{subj_id} | {len(muscle_onsets)} s flagged as BAD_muscle{scope}")
    else:
        logger.info(f"S{subj_id} | No muscle artefacts detected (gamma){scope}")

    safe_set_annotations(raw, new_annotations)
    logger.info(f"S{subj_id} | Step 5 done")
    return raw, auto_annotations


# ---- Step 6: ICA ---------------------------------------------------------

def step7_ica(
    raw_ica: mne.io.Raw,
    raw_apply: mne.io.Raw,
    subj_id: Union[int, str],
    fit_segments: Optional[List[str]] = None,
) -> mne.io.Raw:
    """
    Fit ICA on a temporary high-pass-filtered dataset and apply it to the final dataset.

    Decision logic:
    * Save auto-suggestion to ``ica_suggest.json``.
    * If ``ica_exclude.json`` exists (manual override), use it.
    * Otherwise use the suggestion and write a template for future manual edits.
    """
    logger.info(f"S{subj_id} | Step 6  ICA")
    qdir = subject_qc_dir(subj_id)

    rank_dict = mne.compute_rank(raw_ica, rank="info", verbose=False)
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
    ica.fit(raw_ica, picks="eeg", reject_by_annotation=True, verbose=False)
    logger.info(f"S{subj_id} | ICA fitted ({ica.n_components_} components)")

    if SAVE_QC_FILES:
        ica.save(str(qdir / "ica_solution-ica.fif"), overwrite=True, verbose=False)

    # EOG component detection
    eog_idx: List[int] = []
    eog_scores = np.array([])
    if EOG_CHANNEL in raw_ica.ch_names:
        try:
            eog_idx, eog_scores = ica.find_bads_eog(
                raw_ica, ch_name=EOG_CHANNEL, verbose=False,
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
        "fit_segments": fit_segments or [],
    }
    (qdir / "ica_suggest.json").write_text(
        json.dumps(suggestion, indent=2), encoding="utf-8",
    )

    # Save topography plots (non-interactive)
    if SAVE_QC_FILES:
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

    # Decide which components to exclude (ocular auto-suggestions only).
    auto_exclude = sorted(set([int(x) for x in eog_idx]))
    override_path = qdir / "ica_exclude.json"
    if override_path.exists():
        try:
            override = json.loads(override_path.read_text(encoding="utf-8"))
            exclude = [int(x) for x in override.get("exclude", auto_exclude)]
            logger.info(f"S{subj_id} | Manual ICA override -> {exclude}")
        except Exception:
            exclude = auto_exclude
    else:
        exclude = auto_exclude
        (qdir / "ica_exclude.json").write_text(
            json.dumps({
                "exclude": exclude,
                "_comment": (
                    "Edit 'exclude' list to override ICA component removal. "
                    "auto_eog indices are suggestions only — "
                    "verify with ica_topomap_*.png before re-running Cell 4."
                ),
                "auto_eog_indices": [int(x) for x in eog_idx],
            }, indent=2),
            encoding="utf-8",
        )

    ica.exclude = exclude
    raw_apply = ica.apply(raw_apply, verbose=False)
    logger.info(
        f"S{subj_id} | Step 6 done - removed {len(exclude)} component(s)"
    )
    return raw_apply


# ---- Step 7: Interpolate bad channels (after ICA) -------------------------

def step_interpolate_bad_channels(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    bads: List[str],
    context: Optional[str] = None,
) -> mne.io.Raw:
    """Interpolate bad channels detected in step5_bad_channels.

    Called AFTER ICA so that bad channels do not corrupt the decomposition.
    If more than BAD_CHAN_MAX_INTERP_RATIO (10%) of EEG channels are bad,
    a warning is printed — the segment may need to be rejected downstream.
    """
    scope = f" [{context}]" if context else ""
    logger.info(f"S{subj_id} | Step 7  Interpolating bad channels{scope}")

    # Re-set bads in case ICA modified raw.info["bads"]
    raw.info["bads"] = bads

    if not bads:
        logger.info(f"S{subj_id} | No bad channels to interpolate{scope}")
        logger.info(f"S{subj_id} | Step 7 done")
        return raw

    n_eeg = len(mne.pick_types(raw.info, eeg=True, eog=False))
    ratio = len(bads) / n_eeg if n_eeg > 0 else 0.0
    if ratio > BAD_CHAN_MAX_INTERP_RATIO:
        msg = (
            f"S{subj_id} | {context or 'Segment'}: {len(bads)}/{n_eeg} EEG channels bad "
            f"({ratio*100:.0f}% > {BAD_CHAN_MAX_INTERP_RATIO*100:.0f}%). "
            f"Consider excluding it from analysis."
        )
        logger.warning(msg)

    logger.info(f"S{subj_id} | Interpolating {len(bads)} channel(s){scope}: {bads}")
    raw.interpolate_bads(reset_bads=True, verbose=False)
    logger.info(f"S{subj_id} | Step 7 done")
    return raw


# ---- Step 8: Average re-reference (after ICA + interpolation) -------------

def step_rereference(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    context: Optional[str] = None,
) -> mne.io.Raw:
    """Re-reference all EEG channels to the common average.

    Applied AFTER ICA and bad-channel interpolation to avoid rank deficiency
    issues during ICA fitting (Kim et al. 2023; Bigdely-Shamlo et al. 2015).

    TP9 (typed eog) is automatically excluded from the average computation
    by MNE, but remains in the data for any downstream use.

    FCz handling: the acquisition reference is treated as a non-physiological
    channel in this montage. It is always excluded from the average-reference
    computation and interpolated afterward so the final dataset preserves the
    original channel layout without contaminating the average.
    """
    scope = f" [{context}]" if context else ""
    logger.info(f"S{subj_id} | Step 8  Average reference{scope}")

    ref_ch = ACQUISITION_REFERENCE  # typically "FCz"
    excluded_ref = False
    if ref_ch in raw.ch_names:
        if ref_ch not in raw.info["bads"]:
            raw.info["bads"].append(ref_ch)
        excluded_ref = True
        logger.info(
            f"S{subj_id} | Excluding acquisition reference '{ref_ch}' from "
            "average reference"
        )

    raw, _ = mne.set_eeg_reference(raw, ref_channels="average", verbose=False)

    if excluded_ref and ref_ch in raw.info["bads"]:
        logger.info(f"S{subj_id} | Interpolating '{ref_ch}' after average reference{scope}")
        raw.interpolate_bads(reset_bads=True, verbose=False)

    logger.info(f"S{subj_id} | Step 8 done")
    return raw


# ---------------------------------------------------------------------------
# 1.11  PREPROCESSING DRIVER  (Steps 1-8, takes already-loaded raw)
# ---------------------------------------------------------------------------

def run_preprocessing_steps(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> Tuple[mne.io.Raw, List[str], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Run Steps 1-8 on an already-loaded raw object.

    Pipeline order (Kim 2023; Klug 2024; Makoto):
      1. Channel setup
      2. Montage
      3. Create two filtered copies after montage:
         - ICA-fit dataset: 1-40 Hz + notch
         - Final dataset: 0.1-40 Hz + notch
      4. Detect bad channels by preprocessing unit (LB / BH / Recov)
      5. Annotate gross artifacts by segment on the ICA-fit dataset
      6. Fit ICA on concatenated analysis segments and apply it to the final dataset
      7. Interpolate bad channels per preprocessing unit
      8. Average re-reference per preprocessing unit

    Returns (raw_clean, bad_channels_union, ica_info, segment_qc).
    Raises on failure (caller should handle).
    """
    # Step 1 - channels
    print(f"  [Step 1] Setting up channels...")
    raw = step1_setup_channels(raw, subj_id)
    save_step(raw, subj_id, "step1_channels")

    # Step 2 - montage
    print(f"\n  [Step 2] Applying montage (standard_1020)...")
    raw = step2_set_montage(raw, subj_id)
    save_step(raw, subj_id, "step2_montage")

    # Step 3 - filtering
    print(
        f"\n  [Step 3a] Creating ICA-fit filtered copy "
        f"({ICA_FIT_HIGH_PASS:.1f}-{FILTER_LOW_PASS:.1f} Hz + notch)..."
    )
    raw_ica = step3_filter(
        raw.copy(), subj_id,
        high_pass=ICA_FIT_HIGH_PASS,
        low_pass=FILTER_LOW_PASS,
        dataset_label="ICA fit",
    )
    save_step(raw_ica, subj_id, "step3_ica_filtered")

    print(
        f"\n  [Step 3b] Creating final filtered copy "
        f"({FINAL_HIGH_PASS:.1f}-{FILTER_LOW_PASS:.1f} Hz + notch)..."
    )
    raw_final = step3_filter(
        raw.copy(), subj_id,
        high_pass=FINAL_HIGH_PASS,
        low_pass=FILTER_LOW_PASS,
        dataset_label="final output",
    )
    save_step(raw_final, subj_id, "step3_final_filtered")
    save_step(raw_final, subj_id, "step3_filtered")

    units = build_preprocessing_units(segments)
    if not units:
        raise RuntimeError("No valid LB/BH/Recov segments available for preprocessing.")

    # Step 4 - bad-channel detection by segment/unit
    print(f"\n  [Step 4] Detecting bad channels by segment...")
    segment_qc: Dict[str, Dict[str, Any]] = {}
    all_bads_set = set()
    unit_ica_raws: List[mne.io.Raw] = []
    auto_annotations_abs = mne.Annotations([], [], [], orig_time=raw_ica.annotations.orig_time)

    for unit in units:
        unit_raw_for_bads, _ = crop_and_concat_segments(raw_ica, unit["segments"])
        _, unit_bads = step5_bad_channels(unit_raw_for_bads, subj_id, context=unit["display_name"])

        annotated_parts: List[mne.io.Raw] = []
        artifact_seconds = 0.0
        for seg in unit["segments"]:
            seg_raw = raw_ica.copy().crop(tmin=seg["start"], tmax=seg["end"])
            seg_raw, auto_ann = step6_annotate_artifacts(seg_raw, subj_id, context=seg["name"])
            annotated_parts.append(seg_raw)
            artifact_seconds += float(np.sum(auto_ann.duration)) if len(auto_ann) else 0.0
            auto_annotations_abs = auto_annotations_abs + shift_annotations(
                auto_ann, seg["start"], raw_ica.annotations.orig_time
            )

        unit_raw_ica = annotated_parts[0] if len(annotated_parts) == 1 else mne.concatenate_raws(
            annotated_parts, preload=True, verbose=False
        )
        unit_ica_raws.append(unit_raw_ica)
        segment_qc[unit["label"]] = {
            "display_name": unit["display_name"],
            "segments": unit["segments"],
            "segment_names": [seg["name"] for seg in unit["segments"]],
            "bads": sorted(unit_bads),
            "artifact_seconds": artifact_seconds,
        }
        all_bads_set.update(unit_bads)

        if unit_bads:
            print(f"           {unit['display_name']}: {sorted(unit_bads)}")
        else:
            print(f"           {unit['display_name']}: no bad channels detected")

    bads = sorted(all_bads_set)
    raw_ica.info["bads"] = list(bads)
    raw_final.info["bads"] = list(bads)
    save_bad_channel_report(subj_id, bads, segment_qc)
    save_step(raw_ica, subj_id, "step4_bads_detected")

    # Step 5 - artifact annotation by segment
    print(f"\n  [Step 5] Annotating gross artifacts by segment...")
    safe_set_annotations(raw_ica, raw_ica.annotations + auto_annotations_abs)
    safe_set_annotations(raw_final, raw_final.annotations + auto_annotations_abs)
    for unit in units:
        qc = segment_qc[unit["label"]]
        print(
            f"           {qc['display_name']}: "
            f"{qc['artifact_seconds']:.1f} s flagged as BAD_artifact/BAD_muscle"
        )
    save_step(raw_ica, subj_id, "step5_annotated")

    # Step 6 - ICA on concatenated analysis segments
    print(f"\n  [Step 6] Running ICA...")
    raw_ica_fit = unit_ica_raws[0] if len(unit_ica_raws) == 1 else mne.concatenate_raws(
        unit_ica_raws, preload=True, verbose=False
    )
    fit_segment_names = [unit["display_name"] for unit in units]
    raw_final = step7_ica(raw_ica_fit, raw_final, subj_id, fit_segments=fit_segment_names)
    print(f"           Fit on concatenated segments: {', '.join(fit_segment_names)}")
    save_step(raw_final, subj_id, "step6_ica")

    # Step 7 + Step 8 - clean each processing unit independently
    print(f"\n  [Step 7] Interpolating bad channels by segment...")
    for unit in units:
        qc = segment_qc[unit["label"]]
        unit_raw_final, unit_lengths = crop_and_concat_segments(raw_final, unit["segments"])
        unit_raw_final = step_interpolate_bad_channels(
            unit_raw_final, subj_id, qc["bads"], context=qc["display_name"]
        )
        write_unit_data_back(raw_final, unit_raw_final, unit["segments"], unit_lengths)
        if qc["bads"]:
            print(f"           {qc['display_name']}: interpolated {qc['bads']}")
        else:
            print(f"           {qc['display_name']}: no interpolation needed")
    save_step(raw_final, subj_id, "step7_interpolated")

    print(f"\n  [Step 8] Applying average reference by segment...")
    for unit in units:
        qc = segment_qc[unit["label"]]
        unit_raw_final, unit_lengths = crop_and_concat_segments(raw_final, unit["segments"])
        unit_raw_final = step_rereference(unit_raw_final, subj_id, context=qc["display_name"])
        write_unit_data_back(raw_final, unit_raw_final, unit["segments"], unit_lengths)
        print(f"           {qc['display_name']}: average reference applied")
    save_step(raw_final, subj_id, "step8_refavg")

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
    return raw_final, bads, ica_info, segment_qc


# ---------------------------------------------------------------------------
# 1.12  VISUALIZATION FUNCTIONS
# ---------------------------------------------------------------------------

def _iter_valid_segments(
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> List[Tuple[str, int, float, float]]:
    """Yield (seg_type, seg_idx_1based, start, end) for all valid intervals."""
    order = [
        ("LB", 1), ("BH", 4), ("Recov", 1), ("CPT", 1),
    ]
    result = []
    for label, expected_n in order:
        intervals = segments.get(label, [])
        custom_labels = segments.get("_BH_labels") if label == "BH" else None
        if custom_labels and len(custom_labels) != len(intervals):
            custom_labels = None
        for idx, (start, end) in enumerate(intervals):
            if end is None or end <= start:
                continue
            if custom_labels and idx < len(custom_labels):
                seg_name = custom_labels[idx]
            else:
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

    seg_type : one of "LB", "BH", "Recov"
    apply_time_rules : if True, apply LB/Recov 300s truncation and BH 10s trim
    """
    # Only BH has multiple segments (4)
    expected_n = {"LB": 1, "BH": 4, "Recov": 1}.get(seg_type, 1)
    intervals = segments.get(seg_type, [])
    custom_labels = segments.get("_BH_labels") if seg_type == "BH" else None
    if custom_labels and len(custom_labels) != len(intervals):
        custom_labels = None
    result = []
    for idx, (start, end) in enumerate(intervals):
        if end is None or end <= start:
            continue
        # Apply time rules if requested
        if apply_time_rules:
            start, end = apply_segment_rules(seg_type, start, end)
        if custom_labels and idx < len(custom_labels):
            seg_name = custom_labels[idx]
        else:
            seg_name = f"{seg_type} {idx+1}" if expected_n > 1 else seg_type
        result.append((seg_name, idx + 1, start, end))
    return result


def build_preprocessing_units(
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> List[Dict[str, Any]]:
    """Build processing units used from Step 4 onward.

    LB and Recov are processed independently. BH can be treated as one
    concatenated unit so that bad-channel detection / ICA training sees all
    breath-hold periods together.
    """
    units: List[Dict[str, Any]] = []
    for seg_type in PREPROCESS_SEG_TYPES:
        segs = get_segments_by_type(segments, seg_type, apply_time_rules=True)
        if not segs:
            continue
        if seg_type == "BH" and CONCAT_BH_FOR_PREPROC:
            units.append({
                "label": "BH",
                "display_name": "BH",
                "segments": [
                    {"name": seg_name, "idx": seg_idx, "start": start, "end": end}
                    for seg_name, seg_idx, start, end in segs
                ],
            })
            continue
        for seg_name, seg_idx, start, end in segs:
            units.append({
                "label": seg_name,
                "display_name": seg_name,
                "segments": [{
                    "name": seg_name,
                    "idx": seg_idx,
                    "start": start,
                    "end": end,
                }],
            })
    return units


def crop_and_concat_segments(
    raw: mne.io.Raw,
    segments: List[Dict[str, Any]],
) -> Tuple[mne.io.Raw, List[int]]:
    """Crop one or more segments and concatenate them into a single Raw."""
    parts: List[mne.io.Raw] = []
    lengths: List[int] = []
    for seg in segments:
        part = raw.copy().crop(tmin=seg["start"], tmax=seg["end"])
        parts.append(part)
        lengths.append(part.n_times)
    if not parts:
        raise ValueError("No valid segments to crop.")
    if len(parts) == 1:
        return parts[0], lengths
    merged = mne.concatenate_raws(parts, preload=True, verbose=False)
    return merged, lengths


def shift_annotations(
    annotations: mne.Annotations,
    onset_offset: float,
    orig_time: Optional[datetime],
) -> mne.Annotations:
    """Shift annotations by a fixed number of seconds."""
    if len(annotations) == 0:
        return mne.Annotations([], [], [], orig_time=orig_time)
    return mne.Annotations(
        onset=[float(onset + onset_offset) for onset in annotations.onset],
        duration=[float(duration) for duration in annotations.duration],
        description=[str(desc) for desc in annotations.description],
        orig_time=orig_time,
    )


def safe_set_annotations(raw: mne.io.Raw, annotations: mne.Annotations) -> None:
    """Apply annotations while hiding benign crop-range RuntimeWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw.set_annotations(annotations)


def save_bad_channel_report(
    subj_id: Union[int, str],
    all_bads: List[str],
    segment_qc: Dict[str, Dict[str, Any]],
) -> None:
    """Persist union bad channels plus per-unit details for downstream QC."""
    payload = {
        "bad_channels": all_bads,
        "by_unit": {
            label: {
                "display_name": info["display_name"],
                "segments": list(info["segment_names"]),
                "bad_channels": list(info["bads"]),
                "artifact_seconds": float(info["artifact_seconds"]),
            }
            for label, info in segment_qc.items()
        },
    }
    (subject_qc_dir(subj_id) / "bad_channels.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )


def write_unit_data_back(
    raw_target: mne.io.Raw,
    cleaned_unit: mne.io.Raw,
    segments: List[Dict[str, Any]],
    segment_lengths: List[int],
) -> None:
    """Write concatenated unit data back into the corresponding time spans."""
    sfreq = raw_target.info["sfreq"]
    src_start = 0
    for seg, seg_len in zip(segments, segment_lengths):
        tgt_start = int(round(seg["start"] * sfreq))
        tgt_stop = tgt_start + seg_len
        src_stop = src_start + seg_len
        raw_target._data[:, tgt_start:tgt_stop] = cleaned_unit.get_data(start=src_start, stop=src_stop)
        src_start = src_stop


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
    seg_type : "LB", "BH", or "Recov"
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
    elif seg_type == "BH" and apply_time_rules:
        max_sec = f"  [last {BH_TRIM_END_SEC:.0f}s excluded]"
    
    print("=" * 70)
    print(f"  Subject {subj_id} -- {tag.upper()}: {seg_type}{max_sec}")
    print("=" * 70)
    
    for seg_name, seg_idx, start, end in segs:
        print(f"\n--- {seg_name} ---  ({end - start:.1f} s)  [absolute: {start:.1f}s - {end:.1f}s]")
        plot_segment_overview(raw, subj_id, f"{tag}_{seg_name}", start, end,
                             save_dir=qc_dir if SAVE_QC_FILES else None)
        
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
    segment_qc: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Print and return a summary of preprocessing results.
    
    Returns a DataFrame with segment-level summary.
    """
    print("=" * 70)
    print(f"  Subject {subj_id} -- PREPROCESSING SUMMARY")
    print("=" * 70)

    summary_rows = []
    for seg_type in ["LB", "BH", "Recov"]:
        segs = get_segments_by_type(segments, seg_type)
        for seg_name, seg_idx, start, end in segs:
            orig_dur = end - start
            trunc_start, trunc_end = apply_segment_rules(seg_type, start, end)
            final_dur = trunc_end - trunc_start
            qc_key = "BH" if seg_type == "BH" and segment_qc and "BH" in segment_qc else seg_name
            qc_info = (segment_qc or {}).get(qc_key, {})
            qc_bads = ", ".join(qc_info.get("bads", [])) if qc_info.get("bads") else "-"

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
                "Bad channels": qc_bads,
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
    segment_qc: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Print summary, save cleaned data, and save summary CSV.
    """
    if raw_clean is None:
        print("!! Cleaned data not available. Cannot save.")
        return

    df_summary = print_preprocessing_summary(
        raw_clean, subj_id, segments, bad_channels, ica_info, segment_qc
    )

    # Save cleaned data
    print("\n" + "-" * 70)
    clean_path = save_clean_data(raw_clean, subj_id)
    print(f"  Saved -> {clean_path}")
    print(f"  Channels: {len(raw_clean.ch_names)}")
    print(f"  Duration: {raw_clean.times[-1]:.1f} s")
    print(f"  Sfreq:    {raw_clean.info['sfreq']:.0f} Hz\n")

    # Save summary to QC folder
    if SAVE_QC_FILES:
        summary_path = subject_qc_dir(subj_id) / "preprocessing_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"  Summary saved -> {summary_path}\n")
    print(">>> Preprocessing finished. Run `eeg_spectral_analysis.py` for downstream analyses.\n")


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
    n_fft = min(INSPECT_PSD_N_FFT, data.shape[1])
    n_overlap = min(INSPECT_PSD_N_OVERLAP, n_fft - 1)
    try:
        psd, freqs = mne.time_frequency.psd_array_welch(
            data * 1e-6,  # back to V for MNE
            sfreq=sfreq, fmin=INSPECT_PSD_FMIN, fmax=INSPECT_PSD_FMAX,
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

        # BAD annotations: use annotations_final (full replacement) if available,
        # otherwise fall back to annotations_added (append-only, legacy behaviour).
        seg_tmin = edits.get("segment_tmin", start)
        seg_tmax = edits.get("segment_tmax", end)
        annotations_final = edits.get("annotations_final")

        if annotations_final is not None:
            # Remove all existing BAD annotations that overlap this segment
            keep = []
            for ann in raw.annotations:
                if ann["description"].upper().startswith("BAD"):
                    ann_start = float(ann["onset"])
                    ann_end = ann_start + float(ann["duration"])
                    if ann_end > seg_tmin and ann_start < seg_tmax:
                        continue
                keep.append(ann)

            new_annotations = mne.Annotations(
                onset=[float(a["onset"]) for a in keep],
                duration=[float(a["duration"]) for a in keep],
                description=[a["description"] for a in keep],
                orig_time=raw.annotations.orig_time,
            )

            # Add back BAD annotations as they are after the user edited them
            for ann in annotations_final:
                new_annotations.append(
                    onset=ann["onset"],
                    duration=ann["duration"],
                    description=ann["description"],
                )

            raw.set_annotations(new_annotations)
        else:
            # Legacy: only add new annotations
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
    - CPT        : no cap — all available data is used.
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


# ========================  END OF CELL 1  ==================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================





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
# >>> EDIT HERE: Subject ID (EEG file path is derived: participantes_eeg_bh/{SUJETO}eeg.vhdr)
# ============================================================================
SUJETO = "3"
EEG_FILE = INPUT_DIR / f"{SUJETO}eeg.vhdr"
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
# If CPT (or another new type) is missing from the report below, re-run Cell 1 so updated _LABEL_NORMALIZE and reconstruct_segments are loaded.
ALL_SEGMENTS: Dict[str, Dict[str, List[Tuple[float, Optional[float]]]]] = {}
ALL_SEGMENTS[SUJETO] = reconstruct_segments(_label_onsets)
save_segments_json(SUJETO, ALL_SEGMENTS[SUJETO])

# --- Subject-specific segment exclusions (before Cell 3) --------------------
# S7: Build Recov from first to last Recovery label (3287.26–3676.94 s) — one segment, then truncated to 300s by segment time rules
if str(SUJETO) == "7":
    recov_list = ALL_SEGMENTS[SUJETO].get("Recov", [])
    if len(recov_list) >= 1:
        # Original: [(3287.26, 3408.90), (3676.94, None)]. New: single interval first label → last label
        start_new = recov_list[0][0]
        end_new = recov_list[-1][0] if recov_list[-1][1] is None else recov_list[-1][1]
        ALL_SEGMENTS[SUJETO]["Recov"] = [(start_new, end_new)]
        save_segments_json(SUJETO, ALL_SEGMENTS[SUJETO])
        print("  [S7] Recov = first to last Recovery label (single segment).\n")

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
    "LB": False, "BH": False, "Recov": False,
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
# CELL 3b — INSPECT RAW: BH (Breath Holds)
# =============================================================================
inspect_and_mark_segment(
    RAW, SUJETO, "BH", _segments,
    tag="raw", interactive=INTERACTIVE_VIEW, apply_time_rules=True,
)
SEGMENTS_INSPECTED["BH"] = True
#%%
# Set to True and run this cell if after inspecting BH you decided not to include it.
EXCLUIR_BH = True
SEGMENTS_INSPECTED["BH"] = not EXCLUIR_BH


#%%
# =============================================================================
# CELL 3c — INSPECT RAW: CPT
# =============================================================================
inspect_and_mark_segment(
    RAW, SUJETO, "CPT", _segments,
    tag="raw", interactive=INTERACTIVE_VIEW, apply_time_rules=True,
)
SEGMENTS_INSPECTED["CPT"] = True
#%%
# Set to True and run this cell if after inspecting CPT you decided not to include it.
EXCLUIR_CPT = False
SEGMENTS_INSPECTED["CPT"] = not EXCLUIR_CPT


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
# Set to True and run this cell if after inspecting Recov you decided not to include it.
EXCLUIR_RECOV = True
SEGMENTS_INSPECTED["Recov"] = not EXCLUIR_RECOV



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

for _seg_type in ["LB", "BH", "Recov"]:
    if not SEGMENTS_INSPECTED.get(_seg_type, True):
        continue
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

_excluded = [st for st in ["LB", "BH", "Recov", "CPT"] if not SEGMENTS_INSPECTED.get(st, True)]
if _excluded:
    print(f"\n  >> Excluded from downstream analysis: {', '.join(_excluded)}")
else:
    print(f"\n  >> All segments included for downstream analysis.")
print("\n>>> Proceed to Cell 4 for preprocessing.\n")


#%%
# =============================================================================
# CELL 4 — PREPROCESSING  (Steps 1-8)
# =============================================================================
# Runs the full signal-processing chain on the loaded raw:
#   Step 1: Channel selection + types
#   Step 2: Montage
#   Step 3a: Filter ICA-fit dataset (1-40 Hz + notch)
#   Step 3b: Filter final dataset (0.1-40 Hz + notch)
#   Step 4: Detect bad channels by preprocessing unit (LB / BH / Recov)
#   Step 5: Annotate gross artifacts by segment on the ICA-fit dataset
#   Step 6: Fit ICA on concatenated analysis segments, apply to final dataset
#   Step 7: Interpolate bad channels by preprocessing unit
#   Step 8: Average re-reference by preprocessing unit
#
# Order follows Kim et al. 2023 / Klug et al. 2024 / Makoto's Pipeline.
# This preserves low frequencies in the final cleaned signal while keeping the
# ICA fit stable on a temporary high-pass-filtered copy.
# Intermediate .fif files are saved after each step.
# =============================================================================

print("=" * 70)
print(f"  Subject {SUJETO} -- Preprocessing pipeline")
print("=" * 70 + "\n")

ALL_BAD_CHANNELS: Dict[str, List[str]] = {}
ALL_ICA_INFO: Dict[str, Dict[str, Any]] = {}
ALL_SEGMENT_QC: Dict[str, Dict[str, Dict[str, Any]]] = {}

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        RAW_CLEAN, _bads, _ica_info, _segment_qc = run_preprocessing_steps(
            RAW, SUJETO, ALL_SEGMENTS[SUJETO]
        )
    ALL_BAD_CHANNELS[SUJETO] = _bads
    ALL_ICA_INFO[SUJETO] = _ica_info
    ALL_SEGMENT_QC[SUJETO] = _segment_qc
    print(f"\n{'='*70}")
    print(f"  PREPROCESSING COMPLETE  --  Subject {SUJETO}")
    print(f"{'='*70}\n")
except Exception as _exc:
    print(f"\n!! PREPROCESSING FAILED: {_exc}")
    logger.error(f"S{SUJETO} | PREPROCESSING FAILED:\n{traceback.format_exc()}")
    RAW_CLEAN = None


#%%
# =============================================================================
# OPTIONAL — ICA component time courses (LB only; QC / documentation)
# =============================================================================
# Loads saved ICA + step5_annotated_raw.fif (ICA-fit, same as ICA fit), crops to
# LB (≤300 s), runs ica.plot_sources(). Red overlays = BAD annotations excluded
# from ICA via reject_by_annotation. Requires SAVE_INTERMEDIATE_FIF and
# SAVE_QC_FILES. ICA_SOURCES_PICKS: None = all components.
# =============================================================================

ICA_SOURCES_PICKS = None   # None = all; or e.g. [0, 1, 5] for ICA001, ICA002, ICA006

if RAW_CLEAN is None:
    print("!! Preprocessing did not complete. Run Cell 4 first.")
else:
    _step6_path = subject_fif_dir(SUJETO) / "step5_annotated_raw.fif"
    if not _step6_path.exists():
        print(
            "!! ICA sources viewer: step5 raw not found. "
            "Set SAVE_INTERMEDIATE_FIF = True and re-run Cell 4."
        )
    else:
        _ica_path = subject_qc_dir(SUJETO) / "ica_solution-ica.fif"
        if not _ica_path.exists():
            print(
                "!! ICA sources viewer: ICA solution not found. "
                "Run Cell 4 with SAVE_QC_FILES = True."
            )
        else:
            _ica = mne.preprocessing.read_ica(str(_ica_path), verbose=False)
            _raw_pre_ica = mne.io.read_raw_fif(str(_step6_path), preload=True, verbose=False)
            _lb_segs = get_segments_by_type(ALL_SEGMENTS[SUJETO], "LB", apply_time_rules=True)
            if not _lb_segs:
                print("!! LB segment not found in ALL_SEGMENTS. Check Cell 2.")
            else:
                _seg_name, _idx, _start, _end = _lb_segs[0]
                _raw_lb = _raw_pre_ica.copy().crop(tmin=_start, tmax=_end)
                print(
                    f"  Optional ICA viewer — LB (baseline): "
                    f"{_start:.1f}s – {_end:.1f}s ({_end - _start:.1f}s)"
                )
                # Interactive backend for scrollable MNE browser (PyQt5 recommended)
                %matplotlib qt
                _ica.plot_sources(_raw_lb, picks=ICA_SOURCES_PICKS, show=True, block=True)


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
# CELL 5b — INSPECT CLEANED: BH (Breath Holds)
# =============================================================================

if RAW_CLEAN is None:
    print("!! Cleaned data not available.")
else:
    inspect_and_mark_segment(
        RAW_CLEAN, SUJETO, "BH", ALL_SEGMENTS[SUJETO],
        tag="clean", interactive=INTERACTIVE_VIEW_CLEAN, apply_time_rules=True,
    )


#%%
# =============================================================================
# CELL 5c — INSPECT CLEANED: Recov (Recovery)
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
# CELL 5d — INSPECT CLEANED: CPT
# =============================================================================

if RAW_CLEAN is None:
    print("!! Cleaned data not available.")
else:
    inspect_and_mark_segment(
        RAW_CLEAN, SUJETO, "CPT", ALL_SEGMENTS[SUJETO],
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
    ALL_SEGMENT_QC.get(SUJETO, {}),
)


#%%
# =============================================================================
# DOWNSTREAM ANALYSIS
# =============================================================================
# Spectral power, Excel export, aggregation, and topomaps were moved out of
# this script so that `eeg_preprocessing_pipeline.py` remains focused on data
# cleaning only.
#
# Use `eeg_spectral_analysis.py` after Cell 6 to analyse the cleaned FIF files
# saved in `output/clean_data/` together with the segment JSON files in
# `output/qc/`.
# =============================================================================

