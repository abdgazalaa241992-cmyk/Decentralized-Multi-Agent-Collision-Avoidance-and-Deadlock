"""
One trial of the circular-swap scenario in the (headless) Robotarium simulator.

Methods
-------
proposed : RSVC + dynamic rotation of u0 (the thesis method, original code)
rsvc     : RSVC without deadlock handling (baseline of Berkane & Dimarogonas)
noise    : RSVC + random perturbation of u0 while deadlocked
sbc      : Robotarium safety barrier certificates (Wang, Ames & Egerstedt 2017)
           applied to the nominal goal-seeking controller
"""
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary

from rsvc_agent import RSVCAgent, DEFAULT_PARAMS

GOAL_TOL = 0.05


def circle_swap(N, radius=0.8, pos_noise=0.0, heading_noise=0.0, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    ang = 2 * np.pi * np.arange(N) / N
    start = radius * np.vstack([np.cos(ang), np.sin(ang)])
    goal = -start.copy()
    start = start + rng.normal(0.0, pos_noise, start.shape)
    heading = ang + np.pi + rng.normal(0.0, heading_noise, N)
    return np.vstack([start, heading]), goal


def run_trial(method, N, seed=0, pos_noise=0.02, heading_noise=0.1,
              max_steps=3000, freeze_steps=300, fix_projection=False, params=None):
    rng = np.random.default_rng(seed)
    x0, goals = circle_swap(N, pos_noise=pos_noise, heading_noise=heading_noise, rng=rng)
    r = robotarium.Robotarium(number_of_robots=N, show_figure=False, sim_in_real_time=False,
                              initial_conditions=x0, skip_initialization=True)
    si_to_uni = create_si_to_uni_dynamics()
    cert = create_uni_barrier_certificate_with_boundary() if method == "sbc" else None
    p = dict(DEFAULT_PARAMS, **(params or {}))

    agents = [RSVCAgent(i, x0[:2, i], goals[:, i], params=p,
                        method=(method if method != "sbc" else "rsvc"),
                        rng=np.random.default_rng(seed * 1000 + i),
                        fix_projection=fix_projection) for i in range(N)]

    traj = [x0[:2].T.copy()]
    reach_step = np.full(N, -1)
    d_min = np.inf
    for k in range(max_steps):
        poses = r.get_poses()
        for i in range(N):
            agents[i].pos = poses[0:2, i]
        # minimum centre distance
        P = poses[:2].T
        D = np.linalg.norm(P[:, None] - P[None], axis=2) + np.eye(N) * 1e9
        d_min = min(d_min, D.min())

        dist_goal = np.linalg.norm(P - goals.T, axis=1)
        newly = (dist_goal <= GOAL_TOL) & (reach_step < 0)
        reach_step[newly] = k
        if np.all(dist_goal <= GOAL_TOL):
            break

        dxi = np.zeros((2, N))
        if method == "sbc":
            for i in range(N):
                u = -p["GAIN_K"] * (agents[i].pos - agents[i].goal)
                s = np.linalg.norm(u)
                dxi[:, i] = u if s <= p["MAX_SPEED"] else u / s * p["MAX_SPEED"]
            dxu = cert(si_to_uni(dxi, poses), poses)
        else:
            for i in range(N):
                u_nom = agents[i].get_nominal_velocity()
                u_safe = agents[i].solve_rsvc(u_nom, agents)
                dxi[:, i] = u_safe
                agents[i].u_prev = u_safe
            dxu = si_to_uni(dxi, poses)      # as in the original script (no barrier certificate)

        r.set_velocities(np.arange(N), dxu)
        r.step()
        traj.append(r._poses[:2].T.copy())

        # stop early if no unfinished robot has moved more than 5 mm in the
        # last `freeze_steps` steps (permanent deadlock)
        if k >= freeze_steps and k % 50 == 0:
            disp = np.linalg.norm(traj[-1] - traj[-1 - freeze_steps], axis=1)
            if np.all(disp[dist_goal > GOAL_TOL] < 0.005):
                break

    traj = np.array(traj)                                     # (T, N, 2)
    poses = r._poses
    dist_goal = np.linalg.norm(poses[:2].T - goals.T, axis=1)
    success_goal = bool(np.all(dist_goal <= GOAL_TOL))
    collisions = int(r._errors.get("robots_too_close", 0))
    path_len = np.linalg.norm(np.diff(traj, axis=0), axis=2).sum(axis=0)
    straight = np.linalg.norm(goals.T - x0[:2].T, axis=1)
    theta = (np.array([a.angle_history for a in agents]) if method in ("proposed",)
             else np.zeros((N, 1)))
    return dict(
        method=method, N=N, seed=seed,
        success=success_goal and d_min > 0.11,   # no physical contact (robot diameter 0.11 m)
        reached_all=success_goal,
        collision_steps=collisions,
        boundary_steps=int(r._errors.get("robots_outside_boundaries", 0)),
        d_min=float(d_min),
        T_goal=(float(reach_step.max() * p["DT"]) if success_goal else np.nan),
        path_ratio=(float(np.mean(path_len / straight)) if success_goal else np.nan),
        theta_max=float(theta.max()) if theta.size else 0.0,
        steps=k + 1,
        traj=traj, theta=theta,
    )


if __name__ == "__main__":
    import time
    for m in ["proposed", "rsvc", "noise", "sbc"]:
        t = time.time()
        res = run_trial(m, 10, seed=0, pos_noise=0.0, heading_noise=0.0)
        print(m, {k: v for k, v in res.items() if k not in ("traj", "theta")},
              f"{time.time() - t:.1f}s")
