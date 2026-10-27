"""
Monte Carlo evaluation of the circular-swap scenario.

Each trial perturbs the initial positions (std 2 cm) and headings
(std 0.1 rad) with a different random seed, runs one method in the
headless Robotarium simulator, and records the metrics of run_trial().

Usage (from this folder):
    python montecarlo.py --trials 30 --Ns 2 4 6 8 10 12 --out results/mc.csv
"""
import argparse, csv, os, time
from multiprocessing import Pool
import numpy as np
from run_trial import run_trial

FIELDS = ["method", "N", "seed", "success", "reached_all", "collision_steps",
          "boundary_steps", "d_min", "T_goal", "path_ratio", "theta_max", "steps"]
PARAMS = {"noise": {"NOISE_STD": 0.1}}


def job(args):
    method, N, seed, save_dir = args
    res = run_trial(method, N, seed=seed, params=PARAMS.get(method))
    if save_dir and seed == 1:                      # keep one example run per (method, N)
        np.savez_compressed(os.path.join(save_dir, f"traj_{method}_N{N}.npz"),
                            traj=res["traj"], theta=res["theta"])
    return {k: res[k] for k in FIELDS}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--Ns", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--methods", nargs="+", default=["proposed", "rsvc", "noise", "sbc"])
    ap.add_argument("--out", default="results/mc.csv")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    jobs = [(m, N, s, os.path.dirname(a.out)) for m in a.methods for N in a.Ns
            for s in range(1, a.trials + 1)]
    t0 = time.time()
    with open(a.out, "w", newline="") as f, Pool(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(job, jobs), 1):
            w.writerow(row); f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} trials done ({time.time() - t0:.0f} s)", flush=True)
    print("finished in", round(time.time() - t0), "s")
