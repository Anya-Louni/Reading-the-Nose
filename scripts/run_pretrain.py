"""Stage 1: pretrain on OpenPOM and save the encoder weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtn.datasets.openpom import load  # noqa: E402
from rtn.pretrain import pretrain  # noqa: E402

OUT = ROOT / "artifacts"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    graphs, descriptors, dropped = load(ROOT / "data" / "raw" / "openpom_gslf.csv")
    print(f"corpus: {len(graphs)} molecules, {len(descriptors)} descriptors, "
          f"{len(dropped)} dropped ({dropped.reason.value_counts().to_dict()})",
          flush=True)

    for seed in (0, 1, 2):
        print(f"\n=== seed {seed} ===", flush=True)
        model, res = pretrain(graphs, n_descriptors=len(descriptors), epochs=60,
                              seed=seed, hidden=128, layers=3, heads=4)
        torch.save(model.encoder.state_dict(), OUT / f"encoder_pretrained_seed{seed}.pt")
        (OUT / f"pretrain_seed{seed}.json").write_text(json.dumps({
            "macro_auc": res.macro_auc,
            "best_epoch": res.best_epoch,
            "epochs_run": res.epochs_run,
            "n_molecules": len(graphs),
            "n_descriptors": len(descriptors),
            "history": res.history,
        }, indent=2))
        print(f"seed {seed}: val macro AUC {res.macro_auc:.4f} "
              f"(best epoch {res.best_epoch}/{res.epochs_run})", flush=True)


if __name__ == "__main__":
    main()
