"""Phase 4: train the interpolation model, then explain it and check the checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtn.analytes import INTERPRETABLE, INTERPRETABLE_STRICT  # noqa: E402
from rtn.assemble import (  # noqa: E402
    all_molecules, load_platform_a, load_platform_a2, load_platform_b)
from rtn.evaluate import TASK_FOR, _build_net, _candidate_signatures, _train_torch  # noqa: E402
from rtn.evaluate import EvalConfig, _graph_batch, _fit_scalers  # noqa: E402
from rtn.explain import (  # noqa: E402
    attention_scores, deletion_auc, integrated_gradients, sanity_check)

OUT = ROOT / "artifacts"


def train_interpolation_model(platforms, cfg, seed: int, use_pretrained: bool = True):
    """Fit on every molecule. This is the interpolation setting by construction.

    `use_pretrained=False` skips the OpenPOM warm start. Required when the
    resulting encoder will be evaluated on OpenPOM itself: warm-starting there
    and then transferring back is training on the test set, one hop removed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    all_mask = {k: np.ones(len(p), dtype=bool) for k, p in platforms.items()}
    scalers = _fit_scalers(platforms, all_mask)

    batches = []
    for key, p in platforms.items():
        cand = _candidate_signatures(p, scalers[key], all_mask[key])
        names = sorted(cand)
        tgt = np.stack([cand[n] for n in names])
        batches.append((_graph_batch(names, cfg.device), TASK_FOR[key],
                        torch.tensor(tgt, dtype=torch.float, device=cfg.device)))

    net = _build_net(platforms, attention=True, cfg=cfg).to(cfg.device)
    w = cfg.pretrained_dir / f"encoder_pretrained_seed{seed}.pt"
    if use_pretrained and w.exists():
        net.encoder.load_state_dict(torch.load(w, map_location=cfg.device))
    _train_torch(net, batches, cfg)
    return net, batches


def main() -> None:
    OUT.mkdir(exist_ok=True)
    platforms = {
        "A": load_platform_a(ROOT / "data_audit" / "cache" / "uci270.zip"),
        "A2": load_platform_a2(ROOT / "data" / "processed" / "uci251_features.parquet"),
        "B": load_platform_b(ROOT / "data" / "raw" / "zenodo_drift.zip"),
    }
    cfg = EvalConfig(epochs=300, pretrained_dir=OUT)
    molecules = all_molecules(*platforms.values())

    # which head to explain each molecule through
    home = {}
    for m in molecules:
        for key in ("A", "A2", "B"):
            if key in platforms and m in set(platforms[key].molecules):
                home[m] = TASK_FOR[key]
                break

    scope = [m for m in INTERPRETABLE if m in home]
    print(f"interpolation model over {len(molecules)} molecules; "
          f"explaining {len(scope)}: {scope}\n", flush=True)

    attributions, sanity, faith, agree = [], [], [], []

    for seed in (0, 1, 2):
        print(f"=== seed {seed} ===", flush=True)
        net, _ = train_interpolation_model(platforms, cfg, seed)

        for mol in scope:
            task = home[mol]
            att = attention_scores(net, mol, task, seed=seed)
            ig = integrated_gradients(net, mol, task, seed=seed)

            for a in (att, ig):
                for i, (sym, sc) in enumerate(zip(a.atom_symbols, a.atom_scores)):
                    attributions.append({"seed": seed, "molecule": mol,
                                         "method": a.method, "atom_idx": i,
                                         "symbol": sym, "score": float(sc)})

            # cross-method agreement
            if len(att.atom_scores) >= 3:
                from scipy.stats import spearmanr
                agree.append({"seed": seed, "molecule": mol,
                              "spearman_attention_vs_ig": float(
                                  spearmanr(att.atom_scores, ig.atom_scores).statistic),
                              "n_atoms": len(att.atom_scores)})

            for method in (attention_scores, integrated_gradients):
                sanity.append({"seed": seed, **sanity_check(net, mol, task, method, seed)})

            faith.append({"seed": seed, "method": "attention",
                          **deletion_auc(net, mol, task, att.atom_scores, seed=seed)})
            faith.append({"seed": seed, "method": "integrated_gradients",
                          **deletion_auc(net, mol, task, ig.atom_scores, seed=seed)})

        print(f"  done {len(scope)} molecules", flush=True)

    for name, rows in (("phase4_attributions", attributions),
                       ("phase4_sanity", sanity),
                       ("phase4_faithfulness", faith),
                       ("phase4_agreement", agree)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 74 + "\nPHASE 4 RESULTS\n" + "=" * 74)

    print("\n--- sanity check (Adebayo): Spearman(trained, weight-randomised) ---")
    print("    high correlation = the map survives destroying the weights = FAILS")
    sdf = pd.DataFrame(sanity)
    print(sdf.groupby(["method", "molecule"]).spearman.mean()
          .unstack(0).to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\n  per-method mean:")
    print(sdf.groupby("method").spearman.agg(["mean", "std"])
          .to_string(float_format=lambda x: f"{x:+.3f}"))

    print("\n--- cross-method agreement: Spearman(attention, IG) ---")
    adf = pd.DataFrame(agree)
    print(adf.groupby("molecule").spearman_attention_vs_ig.agg(["mean", "std"])
          .to_string(float_format=lambda x: f"{x:+.3f}"))
    print(f"\n  overall mean {adf.spearman_attention_vs_ig.mean():+.3f}")

    print("\n--- faithfulness: deletion AUC vs random masking (lower is better) ---")
    fdf = pd.DataFrame(faith)
    print(fdf.groupby(["method", "molecule"])[["auc_attr", "auc_random", "gap"]]
          .mean().to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\n  fraction of (molecule, seed) cases beating the random control:")
    print(fdf.groupby("method").faithful.mean().to_string(
        float_format=lambda x: f"{x:.3f}"))

    print(f"\n  strict subset (>= 5 heavy atoms) {INTERPRETABLE_STRICT}:")
    strict = fdf[fdf.molecule.isin(INTERPRETABLE_STRICT)]
    print(strict.groupby("method").faithful.mean().to_string(
        float_format=lambda x: f"{x:.3f}"))

    (OUT / "phase4_summary.json").write_text(json.dumps({
        "n_molecules_explained": len(scope),
        "scope": scope,
        "sanity_spearman_by_method": sdf.groupby("method").spearman.mean().to_dict(),
        "agreement_attention_vs_ig": float(adf.spearman_attention_vs_ig.mean()),
        "faithful_fraction": fdf.groupby("method").faithful.mean().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
