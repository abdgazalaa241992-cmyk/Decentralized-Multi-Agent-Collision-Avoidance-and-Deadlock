"""
Ablation studies of the proposed method (circular swap, 30 seeds each).

  1. maximum rotation angle  theta_max in {60, 90, 120, 150} deg  (N = 10, 12)
  2. unwinding: none (angle never decreases), default (2x), instant reset
     (N = 4, 10)

Usage (from this folder):  python ablation.py
Writes results/ablation.csv and prints a summary.
"""
import csv, math, os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from run_trial import run_trial

FIELDS = ["study", "setting", "N", "seed", "success", "T_goal", "path_ratio", "d_min", "theta_max"]


def job(a):
    study, setting, N, seed, params = a
    r = run_trial("proposed", N, seed=seed, params=params)
    return dict(study=study, setting=setting, N=N, seed=seed,
                **{k: r[k] for k in FIELDS[4:]})


if __name__ == "__main__":
    jobs = []
    for deg in [60, 90, 120, 150]:
        for N in [10, 12]:
            jobs += [("theta_max", deg, N, s, {"MAX_ROT_ANGLE": math.radians(deg)}) for s in range(1, 31)]
    for name, f in [("none", 0.0), ("default", 2.0), ("instant", 1000.0)]:
        for N in [4, 10]:
            jobs += [("unwinding", name, N, s, {"DECAY_FACTOR": f}) for s in range(1, 31)]
    os.makedirs("results", exist_ok=True)
    rows = []
    with Pool(os.cpu_count()) as p:
        for i, r in enumerate(p.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    with open("results/ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    d = pd.DataFrame(rows)
    d["success"] = d.success.astype(bool)
    g = d.groupby(["study", "setting", "N"])
    out = g.agg(success=("success", lambda x: 100 * x.mean()),
                T_goal=("T_goal", "mean"), path_ratio=("path_ratio", "mean"),
                d_min=("d_min", "min"))
    print(out.round(3).to_string())
    out.round(3).to_csv("results/ablation_summary.csv")
