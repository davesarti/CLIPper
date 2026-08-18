"""Recover a CPAS-MLP training history from a run log.

scripts/train_cpas.py prints one line per epoch but writes only the checkpoint,
so the learning curves the report asks for exist solely in the stdout of the run
that produced the released model. This turns that log back into the CSV the
notebook plots, with the columns its training cell would have written:

    epoch, phase, loss, proxy_r1, val_r10

It also checks the recovered history against the checkpoint: if the epoch with
the best val R@10 is not the epoch the checkpoint records, the log belongs to a
different run and the curves would be describing a model nobody released.

Run:  conda run -n clipper python scripts/parse_training_log.py ~/pilot.log
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+\[(?P<phase>\w+)\]\s+"
    r"loss\s+(?P<loss>[\d.]+)\s+"
    r"proxy r@1\s+(?P<proxy_r1>[\d.]+)\s+\|\s+"
    r"val R@10\s+(?P<val_r10>[\d.]+)"
)

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("log", type=Path, help="stdout of a scripts/train_cpas.py run")
parser.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "results" / "mlp_final_s0.pt",
                    help="checked against the recovered history; pass a missing "
                         "path to skip the check")
parser.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "cpas_training_history.csv")
args = parser.parse_args()

if not args.log.is_file():
    sys.exit(f"no such log: {args.log}")

rows = [m.groupdict() for m in LINE.finditer(args.log.read_text())]
if not rows:
    sys.exit(f"{args.log} holds no 'epoch ... val R@10 ...' lines; wrong file?")

history = pd.DataFrame(rows).astype(
    {"epoch": int, "loss": float, "proxy_r1": float, "val_r10": float})

# A log may contain several runs back to back (the campaign scripts loop over
# seeds into one file). Epoch numbers restart, so keep only the last run.
starts = history.index[history["epoch"] == 0].tolist()
if len(starts) > 1:
    print(f"{len(starts)} runs in this log; keeping the last one")
    history = history.loc[starts[-1]:].reset_index(drop=True)

best = history.loc[history["val_r10"].idxmax()]
print(f"{len(history)} epochs, best epoch {int(best.epoch)} "
      f"(val R@10 {best.val_r10:.4f}), phases: "
      f"{history.phase.value_counts().to_dict()}")

if args.checkpoint.is_file():
    saved = torch.load(args.checkpoint, weights_only=True)
    ok = int(best.epoch) == saved.get("epoch")
    print(f"checkpoint {args.checkpoint.name}: epoch {saved.get('epoch')}, "
          f"val R@10 {saved.get('val_r10', float('nan')):.4f} -> "
          f"{'matches' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit("this log did not produce that checkpoint; plotting it would "
                 "describe a model that was never released")
else:
    print(f"no checkpoint at {args.checkpoint}: consistency check skipped")

args.out.parent.mkdir(exist_ok=True)
history.to_csv(args.out, index=False)
print(f"Wrote {args.out}")
