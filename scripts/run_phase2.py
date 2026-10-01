"""Phase 2/3: run every model under both split types and emit the results table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtn.assemble import (  # noqa: E402
    all_molecules, load_platform_a, load_platform_a2, load_platform_b)
from rtn.evaluate import EvalConfig, run_fold  # noqa: E402

MODELS = ("mean", "ecfp_1nn", "ecfp_mlp", "gcn", "gat", "gat_pretrained")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "phase2_results.csv")
    ap.add_argument("--center-molecules", action="store_true",
                    help="remove the shared common-mode component before scoring")
    args = ap.parse_args()
    args.out.parent.mkdir(exist_ok=True, parents=True)

    A = load_platform_a(ROOT / "data_audit" / "cache" / "uci270.zip")
    B = load_platform_b(ROOT / "data" / "raw" / "zenodo_drift.zip")
    platforms = {"A": A, "B": B}

    a2_path = ROOT / "data" / "processed" / "uci251_features.parquet"
    if a2_path.exists():
        platforms["A2"] = load_platform_a2(a2_path)
    else:
        print("NOTE: UCI 251 features not found, running without platform A2",
              flush=True)
    molecules = all_molecules(*platforms.values())
    for key, p in platforms.items():
        print(f"platform {key:2s}: {len(p):4d} rows, {p.dim:4d}d, "
              f"{len(set(p.molecules)):2d} molecules", flush=True)
    print(f"union: {len(molecules)} molecules -> {len(molecules)} LMO folds\n", flush=True)

    cfg = EvalConfig(epochs=args.epochs, seeds=tuple(args.seeds),
                     pretrained_dir=ROOT / "artifacts",
                     center_molecules=args.center_molecules)
    print(f"common-mode removal: {args.center_molecules}\n", flush=True)

    results = []

    # ---- leave-molecule-out ------------------------------------------------
    for mol in molecules:
        train = {k: p.molecules != mol for k, p in platforms.items()}
        test = {k: p.molecules == mol for k, p in platforms.items()}
        if not any(t.sum() for t in test.values()):
            continue
        # a molecule measured on only one platform still trains the other head
        r = run_fold(platforms, train, test, "leave_molecule_out", mol, cfg, MODELS)
        results.extend(r)
        got = {x.model: x.cosine for x in r if x.seed == cfg.seeds[0]}
        print(f"LMO {mol:18s} " +
              "  ".join(f"{k}={v:+.3f}" for k, v in got.items()), flush=True)

    # ---- batch-wise random split ------------------------------------------
    for seed in cfg.seeds:
        rng = np.random.default_rng(1000 + seed)
        train, test = {}, {}
        for k, p in platforms.items():
            runs = np.array(sorted(set(p.runs)))
            rng.shuffle(runs)
            held = set(runs[: max(1, int(round(0.2 * len(runs))))].tolist())
            m = np.array([r in held for r in p.runs])
            train[k], test[k] = ~m, m
        r = run_fold(platforms, train, test, "random_batchwise",
                     f"seed{seed}", EvalConfig(epochs=args.epochs, seeds=(seed,),
                                               pretrained_dir=ROOT / "artifacts",
                                               center_molecules=args.center_molecules),
                     MODELS)
        results.extend(r)
        print(f"random seed{seed}: {len(r)} scores", flush=True)

    df = pd.DataFrame([vars(x) for x in results])
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")

    print("\n" + "=" * 78)
    print("PHASE 2 RESULTS")
    print("=" * 78)
    for split in df.split.unique():
        sub = df[df.split == split]
        tbl = sub.groupby("model").agg(
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            rank_mean=("rank", "mean"),
            top1=("rank", lambda s: (s == 1).mean()),
            n=("cosine", "size"),
        ).reindex(MODELS)
        print(f"\n--- {split} ---")
        print(tbl.to_string(float_format=lambda x: f"{x:.3f}"))

    # Platform B has only 3 molecules, so a leave-molecule-out fold there trains
    # on 2, whose signatures are close to antipodal (mean pairwise cosine -0.98
    # and -0.74). Those folds cannot support learning and inflate the variance of
    # the pooled mean, so report the platforms separately as well.
    print("\n--- leave-molecule-out, split by platform ---")
    lmo_all = df[df.split == "leave_molecule_out"]
    for plat in sorted(lmo_all.platform.unique()):
        sub = lmo_all[lmo_all.platform == plat]
        print(f"\nplatform {plat}  ({sub.fold.nunique()} folds, "
              f"~{sub.n_candidates.max() - 1} training molecules per fold)")
        print(sub.groupby("model").agg(
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            rank_mean=("rank", "mean"),
            top1=("rank", lambda s: (s == 1).mean()),
            n=("cosine", "size"),
        ).reindex(MODELS).to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n--- leave-molecule-out, per fold (cosine, mean over seeds) ---")
    lmo = df[df.split == "leave_molecule_out"]
    piv = lmo.pivot_table(index="fold", columns="model", values="cosine",
                          aggfunc="mean").reindex(columns=MODELS)
    print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
