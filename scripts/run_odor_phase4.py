"""Phase 4 on the odour task: the same checks, on a dataset big enough to mean something.

On the sensor task these checks ran over 8 molecules. Here they run over a
sample of held-out molecules from a scaffold-disjoint test fold, which is the
first time the Adebayo randomisation check in this project has had real
statistical power.

The harness is `rtn.explain`, unchanged apart from letting `_batch` resolve a
raw SMILES as well as a registry name.

Three questions, in order of importance:

1. **Is the attention uniform again?** On the sensor task it was, to float32
   precision (max spread 2e-8), which is why no attention map was shown anywhere.
   With 383x the molecules and a model that demonstrably learned something, this
   is the direct test of whether that collapse was also about data.
2. **Do the explanations survive weight randomisation?** Reported as a
   distribution over molecules, not a handful of cases.
3. **Are they faithful?** Deletion AUC against a random-masking control, again
   as a distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtn.datasets.openpom import load as load_openpom  # noqa: E402
from rtn.explain import (  # noqa: E402
    attention_scores, deletion_auc, integrated_gradients, randomize_weights)
from rtn.odor_eval import scaffold_folds, train_eval_graph  # noqa: E402
from rtn.model import NoseNet  # noqa: E402

ART = ROOT / "artifacts"
TASK = "odor"


def fit_model(train_g, val_g, n_out, seed, epochs):
    """Train a GAT on the odour task and hand back the fitted model itself.

    `train_eval_graph` returns predictions rather than the model, so the loop is
    repeated here. Kept deliberately identical to that function so the explained
    model is the same object the Phase 2 table scored.
    """
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    from rtn.odor_eval import _pos_weight, macro_scores

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NoseNet(n_odor=n_out, attention=True, hidden=128, layers=3,
                    heads=4, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = _pos_weight(torch.cat([g.y for g in train_g]))
    tl = DataLoader(train_g, batch_size=128, shuffle=True)
    vl = DataLoader(val_g, batch_size=256)
    best, best_state, best_ep = -np.inf, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tl:
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(
                model(batch, task=TASK), batch.y, pos_weight=pw).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in vl:
                ps.append(torch.sigmoid(model(batch, task=TASK)).numpy())
                ys.append(batch.y.numpy())
        auc, _, _ = macro_scores(np.concatenate(ys), np.concatenate(ps))
        if auc > best:
            best, best_ep = auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_ep >= 8:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-molecules", type=int, default=250)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-random", type=int, default=20)
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)

    graphs, descriptors, _ = load_openpom(ROOT / "data" / "raw" / "openpom_gslf.csv")
    folds = scaffold_folds(graphs, 3)
    test_idx = folds[1]                       # the balanced middle fold
    rest = np.array([i for i in range(len(graphs)) if i not in set(test_idx.tolist())])
    rng = np.random.default_rng(1235)
    perm = rng.permutation(rest)
    n_val = max(1, int(0.12 * len(perm)))
    val_g = [graphs[i] for i in perm[:n_val]]
    train_g = [graphs[i] for i in perm[n_val:]]

    # Explain a random sample of held-out molecules with enough atoms to rank.
    pool = [i for i in test_idx if graphs[i].x.shape[0] >= 4]
    sample = rng.choice(pool, size=min(args.n_molecules, len(pool)), replace=False)
    smiles = [graphs[i].smiles for i in sample]
    print(f"corpus {len(graphs)} | train {len(train_g)} val {len(val_g)} "
          f"test {len(test_idx)} | explaining {len(smiles)} held-out molecules",
          flush=True)

    rows, spread_rows = [], []
    for seed in args.seeds:
        t0 = time.time()
        model, val_auc = fit_model(train_g, val_g, len(descriptors), seed, args.epochs)
        rnd = randomize_weights(model, seed)
        print(f"\nseed {seed}: val macro AUROC {val_auc:.4f} "
              f"({time.time() - t0:.0f}s) -> explaining", flush=True)

        for k, smi in enumerate(smiles, 1):
            try:
                att = attention_scores(model, smi, TASK, seed=seed)
                ig = integrated_gradients(model, smi, TASK, seed=seed)
                att_r = attention_scores(rnd, smi, TASK, seed=seed)
                ig_r = integrated_gradients(rnd, smi, TASK, seed=seed)
            except Exception as exc:  # noqa: BLE001
                print(f"   skip {smi}: {exc!r}", flush=True)
                continue

            n = len(att.atom_scores)
            spread_rows.append({
                "seed": seed, "smiles": smi, "n_atoms": n,
                "attention_spread": float(np.ptp(att.atom_scores)),
                "ig_spread": float(np.ptp(ig.atom_scores)),
                "attention_uniform_ref": 1.0 / n,
            })

            for name, tr, rd in (("attention", att, att_r),
                                 ("integrated_gradients", ig, ig_r)):
                sane = (float("nan") if np.ptp(tr.atom_scores) == 0
                        or np.ptp(rd.atom_scores) == 0
                        else float(spearmanr(tr.atom_scores, rd.atom_scores).statistic))
                f = deletion_auc(model, smi, TASK, tr.atom_scores,
                                 n_random=args.n_random, seed=seed)
                rows.append({"seed": seed, "smiles": smi, "method": name,
                             "n_atoms": n, "sanity_spearman": sane,
                             "auc_attr": f["auc_attr"], "auc_random": f["auc_random"],
                             "gap": f["gap"], "faithful": f["faithful"]})

            if np.ptp(att.atom_scores) > 0 and np.ptp(ig.atom_scores) > 0:
                rows[-1]["agreement"] = float(
                    spearmanr(att.atom_scores, ig.atom_scores).statistic)

            if k % 50 == 0:
                print(f"   {k}/{len(smiles)}  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    sp = pd.DataFrame(spread_rows)
    df.to_csv(ART / "odor_phase4_checks.csv", index=False)
    sp.to_csv(ART / "odor_phase4_spread.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78 + "\nODOUR TASK: PHASE 4 RESULTS\n" + "=" * 78)

    print("\n--- 1. Is the attention uniform, as it was on the sensor task? ---")
    print("    sensor task reference: max spread 1.99e-08 over every molecule and seed")
    print(sp[["attention_spread", "ig_spread"]].describe(
        percentiles=[.5, .9, .99]).to_string(float_format=lambda x: f"{x:.3e}"))
    frac = (sp.attention_spread < 1e-6).mean()
    print(f"\n    molecules with attention spread < 1e-6 (i.e. uniform): {frac:.3f}")
    print(f"    median attention spread relative to a uniform weight: "
          f"{(sp.attention_spread / sp.attention_uniform_ref).median():.3f}")

    print("\n--- 2. Adebayo sanity check: Spearman(trained, weight-randomised) ---")
    print("    high = the map survives destroying the weights = FAILS")
    s = df.groupby("method").sanity_spearman.agg(
        ["count", "mean", "std",
         lambda x: (x > 0.8).mean(), lambda x: (x.abs() < 0.3).mean()])
    s.columns = ["n", "mean", "sd", "frac_fail_>0.8", "frac_near_zero_<0.3"]
    print(s.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- 3. Faithfulness: deletion AUC vs random masking ---")
    f = df.groupby("method").agg(
        auc_attr=("auc_attr", "mean"), auc_random=("auc_random", "mean"),
        gap=("gap", "mean"), frac_faithful=("faithful", "mean"),
        n=("gap", "size"))
    print(f.to_string(float_format=lambda x: f"{x:.4f}"))

    if "agreement" in df:
        a = df.agreement.dropna()
        print(f"\n--- 4. Cross-method agreement Spearman(attention, IG): "
              f"mean {a.mean():+.4f}, median {a.median():+.4f}, n {len(a)} ---")

    (ART / "odor_phase4_summary.json").write_text(json.dumps({
        "n_molecules": len(smiles), "seeds": args.seeds,
        "attention_spread_median": float(sp.attention_spread.median()),
        "attention_spread_max": float(sp.attention_spread.max()),
        "ig_spread_median": float(sp.ig_spread.median()),
        "frac_attention_uniform": float(frac),
        "sanity_mean": df.groupby("method").sanity_spearman.mean().to_dict(),
        "faithful_frac": df.groupby("method").faithful.mean().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
