#!/usr/bin/env python3
"""
Turn ROS 2 / Gazebo run logs into the figures and table of thesis Section 5.3.

    python3 ros2_analyze.py ~/rsvc_logs/proposed_2026... ~/rsvc_logs/rsvc_2026... \
            --labels "Proposed" "RSVC (baseline)"

By default the outputs go straight into the thesis project:
    ../figures/ros2_traj.pdf, ros2_theta.pdf, ros2_cmd.pdf, ros2_dist.pdf
    ../experiments/results/ros2_table.tex, ros2_metrics.csv
Recompile the thesis afterwards; Section 5.3 includes them automatically.
The FIRST run given is the proposed method (used for the profile plots).
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.8,
})


def load_run(d):
    info = yaml.safe_load(open(os.path.join(d, "run_info.yaml")))
    cfg = info["config"]
    robots = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        if len(df):
            robots[name] = df
    return info, cfg, robots


def metrics(cfg, robots):
    c = cfg["controller"]
    goal_tol = c.get("goal_tol", 0.2)
    t0 = min(df.t_abs.iloc[0] for df in robots.values())
    t1 = max(df.t_abs.iloc[-1] for df in robots.values())
    grid = np.arange(t0, t1, 0.1)
    P = {n: np.c_[np.interp(grid, df.t_abs, df.x), np.interp(grid, df.t_abs, df.y)]
         for n, df in robots.items()}
    names = list(P)
    # sanity check: odometry of all robots must be in ONE world frame
    starts = np.array([[df.x.iloc[0], df.y.iloc[0]] for df in robots.values()])
    spread = np.linalg.norm(starts[:, None] - starts[None], axis=2)
    common_frame = bool(spread[np.triu_indices(len(starts), 1)].min() > 0.3)
    if not common_frame:
        print("WARNING: two or more robots start at the same position -> the /odom of each "
              "robot is relative to its own spawn pose. Set odom_offset in rsvc_params.yaml "
              "and run again; d_centre and the trajectory plot are not meaningful.")
    dmin_t = np.full(len(grid), np.inf)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dmin_t = np.minimum(dmin_t, np.linalg.norm(P[names[i]] - P[names[j]], axis=1))
    # distance to the closest object; NaN when nothing is within the LiDAR window
    # (no interpolation across "nothing seen", which produced spurious spikes)
    dl = []
    for df in robots.values():
        d = df.d_obs.replace([np.inf, -np.inf], np.nan).to_numpy(float)
        idx = np.clip(np.searchsorted(df.t_abs.to_numpy(), grid), 0, len(df) - 1)
        dl.append(d[idx])
    with np.errstate(all="ignore"):
        dlidar_t = np.nanmin(np.vstack(dl), axis=0)
    reach, ratios = [], []
    for n, df in robots.items():
        g = np.array(cfg["robots"][n]["goal"], float)
        ok = np.where(df.dist_goal <= goal_tol)[0]
        reach.append(df.t_abs.iloc[ok[0]] - t0 if len(ok) else np.nan)
        L = np.sum(np.hypot(np.diff(df.x), np.diff(df.y)))
        straight = np.hypot(g[0] - df.x.iloc[0], g[1] - df.y.iloc[0]) - goal_tol
        ratios.append(L / straight if straight > 0 else np.nan)
    success = bool(np.all(np.isfinite(reach)))
    dw = np.concatenate([np.diff(df.omega) / np.diff(df.t_abs).clip(1e-3) for df in robots.values()])
    return dict(
        success=success,
        T_goal=float(np.nanmax(reach)) if success else np.nan,
        path_ratio=float(np.nanmean(ratios)) if success else np.nan,
        d_centre=float(dmin_t.min()) if common_frame else np.nan,
        common_frame=common_frame,
        d_lidar=float(min(df.d_obs.replace(np.inf, np.nan).min() for df in robots.values())),
        theta_max=float(max(df.theta_deg.max() for df in robots.values())),
        omega_rms_acc=float(np.sqrt(np.mean(dw ** 2))),
        estop_steps=int(sum(df.estop.sum() for df in robots.values() if "estop" in df)),
        duration=float(t1 - t0),
    ), (grid - t0, dlidar_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "..", "figures"))
    ap.add_argument("--res-dir", default=os.path.join(HERE, "..", "experiments", "results"))
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(os.path.normpath(r)) for r in a.runs]
    os.makedirs(a.fig_dir, exist_ok=True); os.makedirs(a.res_dir, exist_ok=True)

    runs = [load_run(r) for r in a.runs]
    rows, dist_curves = [], []
    for lab, (info, cfg, robots) in zip(labels, runs):
        m, curve = metrics(cfg, robots)
        rows.append(dict(run=lab, method=info["method"], **m))
        dist_curves.append(curve)
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(a.res_dir, "ros2_metrics.csv"), index=False)
    print(M.round(3).to_string(index=False))

    # ---------------- table --------------------------------------------
    f = lambda v, fmt: "--" if not np.isfinite(v) else fmt.format(v)
    lines = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
             r"\textbf{Run} & \textbf{Success} & $T_{\mathrm{goal}}$ (s) & \textbf{Path ratio} & "
             r"$d_{\mathrm{centre}}$ (m) & $d_{\mathrm{LiDAR}}$ (m) & $\theta_{\max}$ ($^\circ$) & "
             r"RMS $\dot\omega$ (rad/s$^2$) & \textbf{E-stop} & \textbf{Duration} (s) \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['run']} & {'yes' if r['success'] else 'no'} & {f(r['T_goal'], '{:.1f}')} & "
                     f"{f(r['path_ratio'], '{:.2f}')} & {f(r['d_centre'], '{:.2f}')} & {f(r['d_lidar'], '{:.2f}')} & "
                     f"{r['theta_max']:.1f} & {r['omega_rms_acc']:.2f} & {r['estop_steps']} & {r['duration']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.res_dir, "ros2_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---------------- trajectories --------------------------------------
    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 3.8), squeeze=False)
    for ax, lab, (info, cfg, robots) in zip(axes[0], labels, runs):
        for k, (n, df) in enumerate(robots.items()):
            col = SERIES[k % len(SERIES)]
            ax.plot(df.x, df.y, color=col, label=n)
            ax.plot(df.x.iloc[0], df.y.iloc[0], "o", color=col, mec="black", mew=0.5, ms=6)
            ax.plot(df.x.iloc[-1], df.y.iloc[-1], "^", color=col, mec="black", mew=0.5, ms=6)
            g = cfg["robots"][n]["goal"]
            ax.plot(g[0], g[1], "x", color=col, ms=8, mew=2)
        ax.set_aspect("equal"); ax.set_title(lab, fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(a.fig_dir, "ros2_traj.pdf")); plt.close(fig)

    # ---------------- profiles of the proposed run ---------------------
    info, cfg, robots = runs[0]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for k, (n, df) in enumerate(robots.items()):
        ax.plot(df.t, df.theta_deg, color=SERIES[k % 8], label=n)
    ax.axhline(cfg["controller"].get("theta_max_deg", 90), color=INK2, ls="--", lw=1)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(r"Rotation angle $\theta_i$ (deg)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig_dir, "ros2_theta.pdf")); plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True)
    for k, (n, df) in enumerate(robots.items()):
        a1.plot(df.t, df.v, color=SERIES[k % 8], label=n)
        a2.plot(df.t, df.omega, color=SERIES[k % 8], label=n)
    a1.set_ylabel("Linear speed $v_i$ (m/s)"); a2.set_ylabel(r"Angular speed $\omega_i$ (rad/s)")
    a2.set_xlabel("Time (s)"); a1.legend(fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig_dir, "ros2_cmd.pdf")); plt.close(fig)

    # ---------------- minimum LiDAR distance --------------------------
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for k, (lab, (t, d)) in enumerate(zip(labels, dist_curves)):
        ax.plot(t, d, color=SERIES[k % 8], label=lab)
    c = runs[0][1]["controller"]
    d_s = c["robot_radius"] + c["margin"]
    ax.set_ylim(0, c["avoid_radius"] + 0.35)
    ax.set_xlim(left=0)
    ax.axhline(d_s, color=INK2, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], d_s, f" $d_s$ = {d_s:.2f} m ", ha="right", va="bottom",
            fontsize=8, color=INK2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Min. distance to any object (m)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig_dir, "ros2_dist.pdf")); plt.close(fig)
    print("written:", os.path.abspath(a.fig_dir), os.path.abspath(a.res_dir))


if __name__ == "__main__":
    main()
