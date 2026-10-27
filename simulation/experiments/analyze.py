"""
Summarise the Monte Carlo results: LaTeX table + figures for Chapter 5.

Usage (from this folder):  python analyze.py results/mc.csv
Outputs: results/mc_table.tex, results/mc_summary.csv and
         ../figures/mc_success.pdf, mc_time.pdf, mc_path.pdf, mc_dmin.pdf,
         ../figures/mc_example_traj.pdf
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/mc.csv"
OUT = os.path.dirname(CSV)
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

ORDER = ["proposed", "rsvc", "noise", "sbc"]
LABEL = {"proposed": "RSVC + rotation (proposed)", "rsvc": "RSVC (baseline)",
         "noise": "RSVC + random perturbation", "sbc": "Barrier certificates (SBC)"}
SHORT = {"proposed": "Proposed", "rsvc": "RSVC", "noise": "RSVC + noise", "sbc": "SBC"}
# categorical slots 1-4 of the reference palette, fixed order
COLOR = {"proposed": "#2a78d6", "rsvc": "#eb6834", "noise": "#1baf7a", "sbc": "#eda100"}
MARK = {"proposed": "o", "rsvc": "s", "noise": "^", "sbc": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})

df = pd.read_csv(CSV)
df["success"] = df["success"].astype(str).str.lower() == "true"
df["warn"] = df["collision_steps"] > 0
Ns = sorted(df.N.unique())

rows = []
for m in ORDER:
    for N in Ns:
        d = df[(df.method == m) & (df.N == N)]
        if d.empty:
            continue
        s = d[d.success]
        rows.append(dict(method=m, N=N, trials=len(d), success=100 * d.success.mean(),
                         T_mean=s.T_goal.mean(), T_std=s.T_goal.std(),
                         path_mean=s.path_ratio.mean(), path_std=s.path_ratio.std(),
                         dmin_min=d.d_min.min(), dmin_mean=d.d_min.mean(),
                         warn=100 * d.warn.mean(), theta_max=d.theta_max.mean()))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, "mc_summary.csv"), index=False)
print(S.round(3).to_string(index=False))

# ---------------- LaTeX table ------------------------------------------
def pm(a, b, f="{:.1f}"):
    return "--" if np.isnan(a) else (f.format(a) + (r" $\pm$ " + f.format(b) if not np.isnan(b) else ""))

lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"$N$ & \textbf{Method} & \textbf{Success (\%)} & $T_{\mathrm{goal}}$ (s) & \textbf{Path ratio} & $d_{\min}$ (m) & \textbf{Warn.\ (\%)} \\",
         r"\midrule"]
for N in Ns:
    for k, m in enumerate(ORDER):
        r = S[(S.method == m) & (S.N == N)]
        if r.empty:
            continue
        r = r.iloc[0]
        name = r"\textbf{" + SHORT[m] + "}" if m == "proposed" else SHORT[m]
        lines.append(f"{N if k == 0 else ''} & {name} & {r.success:.0f} & {pm(r.T_mean, r.T_std)} & "
                     f"{pm(r.path_mean, r.path_std, '{:.2f}')} & {r.dmin_min:.3f} & {r.warn:.0f} \\\\")
    lines.append(r"\midrule" if N != Ns[-1] else r"\bottomrule")
lines.append(r"\end{tabular}")
open(os.path.join(OUT, "mc_table.tex"), "w").write("\n".join(lines) + "\n")

# ---------------- Figures ------------------------------------------------
def line_chart(col, ylabel, fname, ylim=None, only=None, ref=None):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for m in (only or ORDER):
        d = S[S.method == m].sort_values("N")
        y = d[col].values
        if np.all(np.isnan(y)):
            continue
        ax.plot(d.N, y, marker=MARK[m], color=COLOR[m], label=LABEL[m],
                markeredgecolor="white", markeredgewidth=1.2)
    if ref:
        for val, txt in ref:
            ax.axhline(val, color=INK2, lw=1, ls="--")
            ax.text(Ns[-1], val, " " + txt, va="bottom", ha="right", color=INK2, fontsize=8)
    ax.set_xlabel("Number of agents $N$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(Ns)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)

line_chart("success", "Success rate (%)", "mc_success.pdf", ylim=(-5, 105))
line_chart("T_mean", "Completion time (s)", "mc_time.pdf")
line_chart("path_mean", "Path-length ratio", "mc_path.pdf")
line_chart("dmin_min", "Minimum centre distance (m)", "mc_dmin.pdf",
           ref=[(0.17, "$d_s=0.17$ m"), (0.11, "robot diameter 0.11 m")])

# example trajectories, N = 10, seed 1
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
for ax, m in zip(axes, ["rsvc", "proposed"]):
    f = os.path.join(OUT, f"traj_{m}_N10.npz")
    if not os.path.exists(f):
        continue
    tr = np.load(f)["traj"]
    cmap = plt.get_cmap("tab10")
    for i in range(tr.shape[1]):
        ax.plot(tr[:, i, 0], tr[:, i, 1], lw=1.3, color=cmap(i))
        ax.plot(*tr[0, i], "o", color=cmap(i), ms=5, mec="black", mew=0.5)
        ax.plot(*tr[-1, i], "^", color=cmap(i), ms=5, mec="black", mew=0.5)
    ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, ls="--", color=INK2, lw=0.8))
    ax.set_aspect("equal"); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_title(LABEL[m], fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "mc_example_traj.pdf"))
print("figures written to", os.path.abspath(FIG))
