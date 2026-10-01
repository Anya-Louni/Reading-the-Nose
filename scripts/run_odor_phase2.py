"""Phase 2 on the odour task: baseline table over 4,975 molecules.

Same baseline ordering and the same two-split structure as the sensor task, so
the two tables can be read side by side.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtn.datasets.openpom import load as load_openpom  # noqa: E402
from rtn.odor_eval import (  # noqa: E402
    MODELS, OdorResult, constant_prediction, macro_scores, random_folds,
    scaffold_folds, train_eval_ecfp, train_eval_graph)

ART = ROOT / "artifacts"
SENSOR_ENCODER = ART / "encoder_sensor_task.pt"


def ensure_sensor_encoder() -> Path | None:
    """Fit the 13-molecule sensor model once and keep its encoder.

    This is the transfer arm that replaces the circular "pretrained on OpenPOM"
    slot. Cheap: 13 graphs, full batch.
    """
    if SENSOR_ENCODER.exists():
        return SENSOR_ENCODER
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from rtn.assemble import (load_platform_a, load_platform_a2,
                                  load_platform_b)
        from rtn.evaluate import EvalConfig
        from run_phase4 import train_interpolation_model

        platforms = {
            "A": load_platform_a(ROOT / "data_audit" / "cache" / "uci270.zip"),
            "A2": load_platform_a2(ROOT / "data" / "processed" / "uci251_features.parquet"),
            "B": load_platform_b(ROOT / "data" / "raw" / "zenodo_drift.zip"),
        }
        # use_pretrained=False is mandatory here. The Phase 4 interpolation model
        # normally warm-starts from the OpenPOM encoder; carrying that into a
        # transfer arm evaluated on OpenPOM would leak the corpus back in, and
        # in a smoke test it did exactly that (AUROC 0.747 vs 0.645 for every
        # other model at the same epoch budget).
        net, _ = train_interpolation_model(
            platforms, EvalConfig(epochs=300, pretrained_dir=ART), seed=0,
            use_pretrained=False)
        torch.save(net.encoder.state_dict(), SENSOR_ENCODER)
        print(f"saved sensor-task encoder -> {SENSOR_ENCODER.name}", flush=True)
        return SENSOR_ENCODER
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not build sensor encoder ({exc!r}); "
              f"transfer arm will be skipped", flush=True)
        return None


def run(graphs, descriptors, folds, split_name, seeds, epochs, sensor_enc):
    n_out = len(descriptors)
    results: list[OdorResult] = []

    for fi, test_idx in enumerate(folds):
        test_set = set(test_idx.tolist())
        rest = np.array([i for i in range(len(graphs)) if i not in test_set])
        # carve a validation slice out of the training portion for early stopping
        rng = np.random.default_rng(1234 + fi)
        perm = rng.permutation(rest)
        n_val = max(1, int(0.12 * len(perm)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        train_g = [graphs[i] for i in train_idx]
        val_g = [graphs[i] for i in val_idx]
        test_g = [graphs[i] for i in test_idx]
        y_test = np.concatenate([g.y.numpy() for g in test_g])

        print(f"\n[{split_name}] fold {fi + 1}/{len(folds)}  "
              f"train {len(train_g)}  val {len(val_g)}  test {len(test_g)}", flush=True)

        for seed in seeds:
            t0 = time.time()
            for name in MODELS:
                if name == "gat_sensor_pretrained" and sensor_enc is None:
                    continue
                if name == "constant":
                    if seed != seeds[0]:
                        continue          # deterministic, one run is enough
                    pred, ran = constant_prediction(train_g, test_g, n_out), 0
                elif name == "ecfp_mlp":
                    pred, ran = train_eval_ecfp(train_g, val_g, test_g, n_out, seed=seed)
                else:
                    pred, ran = train_eval_graph(
                        train_g, val_g, test_g, n_out,
                        attention=(name != "gcn"),
                        init_encoder=sensor_enc if name == "gat_sensor_pretrained" else None,
                        epochs=epochs, seed=seed)

                auc, ap, nd = macro_scores(y_test, pred)
                results.append(OdorResult(name, split_name, fi, seed, auc, ap,
                                          len(test_g), nd, ran))
                print(f"    seed{seed} {name:22s} AUROC {auc:.4f}  AP {ap:.4f}  "
                      f"({ran} ep, {time.time() - t0:.0f}s)", flush=True)
    return results


def summarise(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\n" + "=" * 78 + "\nODOUR TASK: PHASE 2 RESULTS\n" + "=" * 78)
    for split in df.split.unique():
        sub = df[df.split == split]
        tbl = sub.groupby("model").agg(
            auroc=("macro_auroc", "mean"), auroc_sd=("macro_auroc", "std"),
            ap=("macro_ap", "mean"), ap_sd=("macro_ap", "std"),
            n=("macro_auroc", "size")).reindex(MODELS).dropna(how="all")
        print(f"\n--- {split} ---")
        print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- paired vs ECFP+MLP (scaffold split, per fold+seed) ---")
    sc = df[df.split == "scaffold"]
    base = sc[sc.model == "ecfp_mlp"].set_index(["fold", "seed"]).macro_auroc
    for m in ("gcn", "gat", "gat_sensor_pretrained"):
        o = sc[sc.model == m].set_index(["fold", "seed"]).macro_auroc
        d = (o - base).dropna()
        if d.empty:
            continue
        lo, hi = np.percentile(
            [np.mean(np.random.default_rng(s).choice(d.values, len(d)))
             for s in range(4000)], [2.5, 97.5])
        print(f"  {m:22s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"wins {int((d > 0).sum())}/{len(d)}")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--folds", type=int, default=3)
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap_.add_argument("--epochs", type=int, default=60)
    ap_.add_argument("--out", type=Path, default=ART / "odor_phase2_results.csv")
    args = ap_.parse_args()
    ART.mkdir(exist_ok=True)

    graphs, descriptors, dropped = load_openpom(ROOT / "data" / "raw" / "openpom_gslf.csv")
    print(f"corpus {len(graphs)} molecules, {len(descriptors)} descriptors "
          f"({len(dropped)} excluded: {dropped.reason.value_counts().to_dict()})",
          flush=True)

    sensor_enc = ensure_sensor_encoder()

    results = []
    results += run(graphs, descriptors, scaffold_folds(graphs, args.folds),
                   "scaffold", args.seeds, args.epochs, sensor_enc)
    results += run(graphs, descriptors, random_folds(len(graphs), args.folds),
                   "random", args.seeds, args.epochs, sensor_enc)

    df = pd.DataFrame([vars(r) for r in results])
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")
    summarise(df)


if __name__ == "__main__":
    main()
