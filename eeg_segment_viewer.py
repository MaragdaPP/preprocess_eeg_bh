"""
Interactive EEG segment viewer  (launched as a subprocess)
==========================================================
Uses TkAgg backend so the MNE raw browser opens as a proper interactive window
even when the main script runs inside an IPython / VSCode cell.

Features:
  - Scroll with arrow keys or mouse wheel
  - Click a channel name to mark it as bad (toggles)
  - Press 'a' to toggle annotation mode, then click-drag to add BAD annotations

On close, the viewer writes a JSON file with any bad channels and annotations
that were added during the session.  The main pipeline reads this file and
applies the changes to the full Raw object.

Usage (called automatically by the pipeline):
    python eeg_segment_viewer.py <fif_path> <tmin> <tmax> [--title TITLE] [--output JSON_PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import mne


def main(
    fif_path: str,
    tmin: float,
    tmax: float,
    title: str = "",
    output_json: str = "",
) -> None:
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # Remember initial state
    initial_bads = set(raw.info["bads"])

    # Crop to segment boundaries (clamp to data range)
    tmax_clamped = min(tmax, raw.times[-1])
    tmin_clamped = max(tmin, raw.times[0])
    segment = raw.copy().crop(tmin=tmin_clamped, tmax=tmax_clamped)

    # Track BAD annotations that already exist in this segment so we only
    # export annotations created during this viewer session.
    initial_bad_ann_keys = set()
    for ann in segment.annotations:
        if ann["description"].upper().startswith("BAD"):
            initial_bad_ann_keys.add(
                (round(float(ann["onset"]), 6), round(float(ann["duration"]), 6), str(ann["description"]))
            )

    # Open the MNE interactive browser (blocking)
    print(f"\n  [Viewer] Segment: {tmin_clamped:.1f} – {tmax_clamped:.1f} s")
    print("           Click channel name to toggle BAD")
    print("           Press 'a' then click-drag to annotate BAD time segments")
    print("           NOTE: to DELETE a BAD mark, zoom in on it first,")
    print("                 then right-click directly on the coloured region.")
    print("           Close window when done.\n")

    fig = segment.plot(
        title=title or f"EEG Segment  [{tmin_clamped:.1f} - {tmax_clamped:.1f} s]",
        n_channels=31,
        scalings="auto",
        block=True,
        show=True,
    )

    # After window closes, collect changes
    new_bads = [ch for ch in segment.info["bads"] if ch not in initial_bads]

    # New BAD annotations in the cropped segment.
    # IMPORTANT: ann["onset"] is already on the same absolute time axis used by
    # the parent Raw object here, so we should NOT add tmin again.
    new_annotations = []
    for ann in segment.annotations:
        if not ann["description"].upper().startswith("BAD"):
            continue
        key = (
            round(float(ann["onset"]), 6),
            round(float(ann["duration"]), 6),
            str(ann["description"]),
        )
        if key in initial_bad_ann_keys:
            continue
        new_annotations.append({
            "onset": float(ann["onset"]),
            "duration": float(ann["duration"]),
            "description": ann["description"],
        })

    # NEW — collect full final BAD state in segment (before vs after handled by pipeline)
    final_annotations = []
    for ann in segment.annotations:
        if ann["description"].upper().startswith("BAD"):
            final_annotations.append({
                "onset":       float(ann["onset"]),
                "duration":    float(ann["duration"]),
                "description": str(ann["description"]),
            })

    # Write output JSON
    result = {
        "segment_tmin":      tmin_clamped,
        "segment_tmax":      tmax_clamped,
        "bad_channels_added": new_bads,
        "annotations_added": new_annotations,   # kept for backward compat
        "annotations_final": final_annotations, # NEW: full state after user edits
    }

    if output_json:
        out_path = Path(output_json)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  [Viewer] Saved edits -> {out_path.name}")
    else:
        # Print to stdout so caller can capture if needed
        print(json.dumps(result))

    if new_bads:
        print(f"  [Viewer] Bad channels marked: {new_bads}")
    if new_annotations:
        print(f"  [Viewer] BAD annotations added: {len(new_annotations)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive EEG segment viewer")
    parser.add_argument("fif_path", help="Path to the .fif file")
    parser.add_argument("tmin", type=float, help="Segment start (seconds)")
    parser.add_argument("tmax", type=float, help="Segment end (seconds)")
    parser.add_argument("--title", type=str, default="", help="Window title")
    parser.add_argument("--output", type=str, default="", help="Output JSON path for edits")
    args = parser.parse_args()
    main(args.fif_path, args.tmin, args.tmax, args.title, args.output)
