"""Stream UCI 251 (wind tunnel) into a compact per-measurement feature table.

The archive is 8.4 GB compressed and 107 GB uncompressed across 18,151 files, so
nothing is ever extracted to disk. Each file is decompressed in memory, reduced
to per-sensor summary statistics, and discarded. Output is a single parquet of
roughly 18k rows, which is what everything downstream actually uses.

## File layout

92 columns per row, sampled at ~52 Hz over 260 s (about 13.6k rows per file):

    col 0        time in ms
    cols 1-8     control/valve channels
    col 9        temperature (degC)
    col 10       relative humidity (%)
    cols 11+     9 modules x (1 constant flag + 8 sensors) = 81 -> 72 sensors

Filenames carry the design: gas, concentration, wind-tunnel location L1-L6,
heater voltage 400-600 V, fan speed 000/060/100.

## Gas onset

Control columns 3, 4 and 5 switch from 0 to 100 at t = 20.2 s and back later.
That is the gas valve. Baseline is therefore t < 20 s and exposure is the valve
window, both read per file rather than hardcoded, because a handful of files may
differ. If the valve channels are unusable in a given file the record is still
emitted with `onset_source="fallback"` and a fixed 20 s split, so a parsing
oddity degrades one row instead of aborting the run.

## Caveat that must reach the writeup

Every gas in this dataset was run at exactly one concentration (CO is the sole
exception, at 1000 and 4000 ppm), and those concentrations span 100 ppm butanol
to 10,000 ppm ammonia, a 100x range. So in UCI 251 molecule identity is
confounded with concentration. The five analytes that come only from this
dataset (methanol, butan-1-ol, benzene, methane, carbon monoxide) have their
signature measured at a single concentration and its stability across
concentration cannot be checked. UCI 270's six analytes have 30-47 levels each.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(
    r"WTD_upload/(?P<gas>[A-Za-z]+)_(?P<conc>\d+)/(?P<loc>L\d+)/"
    r"(?P<ts>\d+)_board_setPoint_(?P<volt>\d+)V_fan_setPoint_(?P<fan>\d+)_mfc_setPoint_"
)

GAS_TO_ANALYTE = {
    "Acetaldehyde": "acetaldehyde",
    "Acetone": "acetone",
    "Ammonia": "ammonia",
    "Benzene": "benzene",
    "Butanol": "butan-1-ol",
    "CO": "carbon_monoxide",
    "Ethylene": "ethylene",
    "Methane": "methane",
    "Methanol": "methanol",
    "Toluene": "toluene",
}

N_SENSORS = 72
VALVE_COLS = (3, 4, 5)
BASELINE_S = 20.0

# Per-sensor summary statistics. Deliberately generous: re-running the 107 GB
# pass to add a feature is expensive, re-deriving a target from stored features
# is free.
FEATURE_NAMES = [
    "max_r",        # peak relative response
    "mean_r",       # mean relative response during exposure
    "p90_r",        # 90th percentile, robust alternative to the peak
    "std_r",        # response variability (turbulence-sensitive)
    "area_r",       # integral of the relative response over exposure
    "rise_slope",   # slope over the first 10 s after onset
    "t_to_max",     # seconds from onset to peak, normalised by exposure length
    "recovery",     # mean relative response over the final 20 s
]


def sensor_matrix(a: np.ndarray) -> np.ndarray:
    """Drop the per-module constant flag columns, leaving 72 sensor channels."""
    block = a[:, 11:]
    return np.delete(block, slice(None, None, 9), axis=1)


def find_windows(a: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (baseline_mask, exposure_mask, source) using the valve channels."""
    for c in VALVE_COLS:
        if c >= a.shape[1]:
            continue
        v = a[:, c]
        on = v > (v.max() / 2.0) if v.max() > 0 else None
        if on is None or not on.any() or on.all():
            continue
        onset = t[np.argmax(on)]
        if not (5.0 < onset < 120.0):
            continue
        return t < onset, on, "valve"
    return t < BASELINE_S, t >= BASELINE_S, "fallback"


def features_for_file(raw: bytes) -> tuple[np.ndarray, dict] | None:
    a = np.loadtxt(io.BytesIO(raw))
    if a.ndim != 2 or a.shape[1] < 11 + 81:
        return None
    t = a[:, 0] / 1000.0
    base_m, exp_m, source = find_windows(a, t)
    if base_m.sum() < 10 or exp_m.sum() < 50:
        return None

    s = sensor_matrix(a)
    if s.shape[1] != N_SENSORS:
        return None

    b = np.median(s[base_m], axis=0)
    b = np.where(np.abs(b) < 1e-9, 1e-9, b)
    r = (s - b) / b

    re_ = r[exp_m]
    te = t[exp_m]
    te0 = te - te[0]
    dur = max(te0[-1], 1e-6)

    first10 = te0 <= 10.0
    if first10.sum() >= 2:
        slope = np.polyfit(te0[first10], re_[first10], 1)[0]
    else:
        slope = np.zeros(N_SENSORS)

    last20 = te0 >= (dur - 20.0)
    imax = np.argmax(np.abs(re_), axis=0)

    feats = np.stack([
        re_[imax, np.arange(N_SENSORS)],
        re_.mean(0),
        np.percentile(re_, 90, axis=0),
        re_.std(0),
        np.trapezoid(re_, te0, axis=0) / dur,
        slope,
        te0[imax] / dur,
        re_[last20].mean(0) if last20.any() else re_[-1],
    ], axis=0)

    meta = {
        "onset_source": source,
        "duration_s": float(t[-1]),
        "n_samples": int(len(t)),
        "temperature": float(np.median(a[:, 9])),
        "humidity": float(np.median(a[:, 10])),
    }
    return feats, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path,
                    default=Path("data/raw/uci251.zip"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/uci251_features.parquet"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N files (smoke test)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(args.zip)
    files = [i.filename for i in z.infolist()
             if not i.is_dir() and NAME_RE.match(i.filename)]
    if args.limit:
        files = files[: args.limit]

    cols = [f"s{i:02d}_{f}" for f in FEATURE_NAMES for i in range(N_SENSORS)]
    rows, meta_rows, failed = [], [], []
    t0 = time.time()

    for k, name in enumerate(files, 1):
        m = NAME_RE.match(name)
        try:
            got = features_for_file(z.open(name).read())
        except Exception as exc:  # noqa: BLE001
            failed.append((name, repr(exc)))
            continue
        if got is None:
            failed.append((name, "unusable_shape_or_window"))
            continue
        feats, meta = got
        rows.append(feats.reshape(-1))
        d = m.groupdict()
        meta_rows.append({
            "analyte": GAS_TO_ANALYTE[d["gas"]],
            "gas_raw": d["gas"],
            "concentration_ppm": int(d["conc"]),
            "location": d["loc"],
            "heater_v": int(d["volt"]),
            "fan": int(d["fan"]),
            "timestamp": d["ts"],
            "source_file": name,
            **meta,
        })
        if k % 500 == 0:
            el = time.time() - t0
            print(f"{k}/{len(files)}  {el/60:.1f} min  "
                  f"eta {(el/k)*(len(files)-k)/60:.1f} min  failed={len(failed)}",
                  flush=True)

    df = pd.concat([pd.DataFrame(meta_rows),
                    pd.DataFrame(np.asarray(rows, dtype=np.float32), columns=cols)],
                   axis=1)
    df.to_parquet(args.out, index=False)

    print(f"\nwrote {args.out}  rows={len(df)}  cols={df.shape[1]}  "
          f"({args.out.stat().st_size/1e6:.1f} MB)")
    print(f"failed: {len(failed)}")
    for n, why in failed[:10]:
        print("   ", why, n)
    print(df.groupby('analyte').size().to_string())
    print("onset source:", df.onset_source.value_counts().to_dict())


if __name__ == "__main__":
    sys.exit(main())
