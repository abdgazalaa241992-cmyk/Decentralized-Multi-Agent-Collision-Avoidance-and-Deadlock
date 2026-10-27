"""
Validation of Proposition 4.8: escape angle versus angular margin.

For every deadlock episode of every robot (a maximal interval in which the
rotation angle theta_i > 0) we record
  * the angular margin  alpha_2 - phi  at the onset of the episode: the angle,
    measured counter-clockwise, from the goal direction phi to the most
    counter-clockwise bearing of an active neighbour (Proposition 4.8);
  * the peak rotation angle of the episode (the escape angle).
Proposition 4.8 predicts, for a static neighbourhood with w_i = 1, an escape
angle of  margin + arcsin(eps_v / v_max) = margin + 30 deg,  capped at 90 deg.

Usage (from this folder):  python margin_study.py
Writes results/margin_episodes.csv and ../figures/margin_vs_escape.pdf
"""
import math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rsvc_agent import RSVCAgent, DEFAULT_PARAMS
from run_trial import circle_swap, GOAL_TOL

P = DEFAULT_PARAMS
OFFSET = math.degrees(math.asin(P["DEADLOCK_SPEED"] / P["MAX_SPEED"]))   # 30 deg


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(N, seed, max_steps=3000):
    rng = np.random.default_rng(seed)
    x0, goals = circle_swap(N, pos_noise=0.02, heading_noise=0.1, rng=rng)
    r = robotarium.Robotarium(number_of_robots=N, show_figure=False, sim_in_real_time=False,
                              initial_conditions=x0, skip_initialization=True)
    si_to_uni = create_si_to_uni_dynamics()
    agents = [RSVCAgent(i, x0[:2, i], goals[:, i]) for i in range(N)]
    margins = np.full((max_steps, N), np.nan)
    for k in range(max_steps):
        poses = r.get_poses()
        for i in range(N):
            agents[i].pos = poses[:2, i]
        P2 = poses[:2].T
        if np.all(np.linalg.norm(P2 - goals.T, axis=1) <= GOAL_TOL):
            break
        for i in range(N):                       # angular margin seen by robot i now
            g = goals[:, i] - P2[i]
            phi = math.atan2(g[1], g[0])
            rel = []
            for j in range(N):
                if j == i:
                    continue
                d = P2[j] - P2[i]
                if np.linalg.norm(d) <= P["SENSING_RADIUS"]:
                    rel.append(wrap(math.atan2(d[1], d[0]) - phi))
            rel = np.array(rel)
            pos = rel[(rel >= 0) & (rel < np.pi)]
            if len(pos):
                margins[k, i] = math.degrees(pos.max())
        dxi = np.zeros((2, N))
        for i in range(N):
            u = agents[i].solve_rsvc(agents[i].get_nominal_velocity(), agents)
            dxi[:, i] = u
            agents[i].u_prev = u
        r.set_velocities(np.arange(N), si_to_uni(dxi, poses))
        r.step()
    theta = np.array([a.angle_history for a in agents]).T            # (steps, N)
    rows = []
    for i in range(N):
        th = theta[:, i]
        k = 0
        while k < len(th):
            if th[k] > 0:
                k0 = k
                while k < len(th) and th[k] > 0:
                    k += 1
                peak = th[k0:k].max()
                m = margins[max(k0 - 1, 0), i]
                rows.append(dict(N=N, seed=seed, robot=i, onset=k0, margin=m,
                                 escape=peak, saturated=peak >= 89.9))
            k += 1
    return rows


if __name__ == "__main__":
    rows = []
    for N in [2, 4, 6, 8, 10]:
        for s in range(1, 11):
            rows += run(N, s)
    df = pd.DataFrame(rows).dropna(subset=["margin"])
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/margin_episodes.csv", index=False)
    ok = df[~df.saturated]
    pred = np.minimum(ok.margin + OFFSET, 90)
    r = np.corrcoef(ok.margin, ok.escape)[0, 1]
    mae = np.mean(np.abs(ok.escape - pred))
    print(f"episodes={len(df)} unsaturated={len(ok)} corr={r:.2f} "
          f"MAE(pred)={mae:.1f} deg  mean(escape-margin)={np.mean(ok.escape - ok.margin):.1f} deg")

    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(5.6, 4))
    cols = {2: "#2a78d6", 4: "#eb6834", 6: "#1baf7a", 8: "#eda100", 10: "#e87ba4"}
    for N, g in df.groupby("N"):
        ax.scatter(g.margin, g.escape, s=18, color=cols[N], alpha=0.75, label=f"$N={N}$",
                   edgecolors="white", linewidths=0.4)
    xs = np.linspace(0, 100, 200)
    ax.plot(xs, np.minimum(xs + OFFSET, 90), color="#0b0b0b", lw=1.5,
            label=r"static prediction $\min(\alpha_2-\varphi+30^\circ,\,90^\circ)$")
    ax.plot(xs, xs, color="#52514e", lw=1, ls="--", label=r"escape angle = margin")
    ax.set_xlim(0, 100); ax.set_ylim(0, 95)
    ax.set_xlabel(r"Angular margin $\alpha_2-\varphi$ at deadlock onset (deg)")
    ax.set_ylabel(r"Escape angle $\max\theta_i$ (deg)")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join("..", "figures", "margin_vs_escape.pdf"))
