"""
Perfectly symmetric circular swaps (no perturbation), one run per method and N.
Usage (from this folder):  python symmetric.py
"""
import pandas as pd
from run_trial import run_trial
from montecarlo import PARAMS

rows = []
for N in [2, 4, 6, 8, 10, 12]:
    for m in ["proposed", "rsvc", "noise", "sbc"]:
        r = run_trial(m, N, seed=0, pos_noise=0.0, heading_noise=0.0, params=PARAMS.get(m))
        rows.append({k: r[k] for k in ["method", "N", "success", "T_goal", "path_ratio",
                                       "d_min", "theta_max", "collision_steps"]})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/symmetric.csv", index=False)
