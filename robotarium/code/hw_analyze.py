#!/usr/bin/env python3
"""
Figures and metrics for the physical Robotarium experiment (thesis Section 5.2.3).

    python3 hw_analyze.py log.txt                  (log returned by the Robotarium)
    python3 hw_analyze.py rsvc_robotarium_data.npz (older logging script)

Writes ../figures/robotarium_hw_traj.pdf, ../figures/robotarium_hw_theta.pdf
and ../experiments/results/robotarium_hw_table.tex; recompile the thesis
afterwards (the section includes them automatically).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "experiments", "results")
GOAL_TOL = 0.05
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "legend.frameon": False})

def load_log(path):
    """Parse the H/G/D lines printed by rsvc_robotarium_submit.py."""
    hdr, goals, rows = {}, None, []
    for line in open(path, errors="ignore"):
        i = line.find("H,N=")
        if i >= 0:
            hdr = dict(kv.split("=") for kv in line[i + 2:].strip().split(","))
            continue
        for tag in ("G,", "D,"):
            j = line.find(tag)
            if j >= 0 and (tag == "G," and goals is None or tag == "D,"):
                vals = line[j + 2:].strip().split(",")
                try:
                    vals = [float(v) for v in vals]
                except ValueError:
                    break
                if tag == "G,":
                    goals = np.array(vals).reshape(-1, 2)
                else:
                    rows.append(vals)
                break
    N = int(hdr["N"]); dt = float(hdr["DT"])
    rows = [r for r in rows if len(r) == 1 + 6 * N]
    A = np.array(rows)
    k = A[:, 0]
    P = A[:, 1:1 + 3 * N].reshape(-1, N, 3).transpose(0, 2, 1)       # (T, 3, N)
    th = A[:, 1 + 5 * N:].T                                          # (N, T)
    return P, th, goals, dt, k * dt


src = sys.argv[1] if len(sys.argv) > 1 else "log.txt"
if src.endswith(".npz"):
    d = np.load(src)
    P, th, goals, dt = d["poses"], d["theta_deg"], d["goals"], float(d["dt"])
    t = np.arange(P.shape[0]) * dt
else:
    P, th, goals, dt, t = load_log(src)
T, _, N = P.shape
xy = P[:, :2, :]                                   # (T, 2, N)

dist_goal = np.linalg.norm(xy - goals.T[None], axis=1)          # (T, N)
reached = dist_goal <= GOAL_TOL
first = [np.argmax(reached[:, i]) if reached[:, i].any() else -1 for i in range(N)]
success = all(f >= 0 for f in first)
T_goal = t[max(first)] if success else float("nan")
D = np.linalg.norm(xy[:, :, :, None] - xy[:, :, None, :], axis=1)  # (T, N, N)
D[:, np.arange(N), np.arange(N)] = np.inf
d_min = float(D.min())
path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum(axis=0)
straight = np.linalg.norm(goals.T - xy[0], axis=0)
ratio = float(np.mean(path / straight))
theta_max = float(th.max())
print(f"success={success} T_goal={T_goal:.1f}s path_ratio={ratio:.2f} d_min={d_min:.3f}m "
      f"theta_max={theta_max:.1f}deg steps={T}")

os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
open(os.path.join(RES, "robotarium_hw_table.tex"), "w").write(
    "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
    "\\textbf{Run} & \\textbf{Success} & $T_{\\mathrm{goal}}$ (s) & \\textbf{Path ratio} & "
    "$d_{\\min}$ (m) & $\\theta_{\\max}$ ($^\\circ$) \\\\\n\\midrule\n"
    f"Physical Robotarium & {'yes' if success else 'no'} & "
    f"{'--' if not success else f'{T_goal:.1f}'} & {ratio:.2f} & {d_min:.3f} & {theta_max:.1f} \\\\\n"
    "\\bottomrule\n\\end{tabular}\n")

cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(5, 5))
for i in range(N):
    ax.plot(xy[:, 0, i], xy[:, 1, i], color=cmap(i), lw=1.4, label=f"Robot {i + 1}")
    ax.plot(*xy[0, :, i], "o", color=cmap(i), mec="black", mew=0.5, ms=6)
    ax.plot(*xy[-1, :, i], "^", color=cmap(i), mec="black", mew=0.5, ms=6)
    ax.plot(*goals[i], "x", color=cmap(i), ms=8, mew=2)
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(fontsize=7, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "robotarium_hw_traj.pdf")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.2))
for i in range(N):
    ax.plot(t[:th.shape[1]], th[i][:len(t)], color=cmap(i), lw=1.4, label=f"Robot {i + 1}")
ax.axhline(90, color="#52514e", ls="--", lw=1)
ax.set_xlabel("Time (s)"); ax.set_ylabel(r"Rotation angle $\theta_i$ (deg)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "robotarium_hw_theta.pdf")); plt.close(fig)
print("written to", os.path.abspath(FIG), "and", os.path.abspath(RES))
