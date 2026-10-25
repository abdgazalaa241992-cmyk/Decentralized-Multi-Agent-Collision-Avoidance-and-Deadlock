"""
RSVC with dynamic vector steering -- ROS-independent controller core.

This is the same control law as the Robotarium code (Chapter 4 of the
thesis), written per robot and in physical time units so that it can run in
a ROS 2 node at any control rate:

    u0      = sat(k_g (g - x))                       nominal controller
    z       = R(theta) u0                            rotated preferred velocity
    w       = max_j w(d_j)                           smooth activation
    u       = (1 - w) z + w Proj_C(A)(z)             blended RSVC law
    v       = acceleration / speed filter of u
    theta   += rate * dt      if ||v_prev|| < eps_v   (deadlock flag)
    theta   -= 2 rate * dt    otherwise               (unwinding)

Neighbours are obtained from the LiDAR scan: consecutive scan points are
grouped into clusters (one cluster per object), and each cluster is
represented by its closest point. Distances are measured from the robot
centre to the closest point of the object's surface, so the safety distance
is  d_s = robot_radius + margin.
"""
import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RSVCParams:
    v_max: float = 0.26            # m/s   (TurtleBot3 Waffle)
    a_max: float = 0.5             # m/s^2
    k_g: float = 0.5               # 1/s   nominal gain
    robot_radius: float = 0.22     # m     circumscribed radius of a Waffle
    margin: float = 0.10           # m     -> d_s = 0.32 m
    avoid_radius: float = 0.80     # m     R_a (interaction radius)
    rot_rate_min: float = 0.24     # rad/s rotation rate far from the neighbour
    rot_rate_max: float = 0.27     # rad/s rotation rate close to the neighbour
    decay_factor: float = 2.0      # unwinding is this many times faster
    theta_max_deg: float = 90.0
    deadlock_speed: float = 0.13   # m/s  (= 0.5 v_max)
    goal_tol: float = 0.20         # m
    cluster_gap: float = 0.15      # m    max jump between points of one cluster
    method: str = "proposed"       # "proposed" or "rsvc" (no deadlock handling)

    @property
    def d_s(self):
        return self.robot_radius + self.margin


def project_onto_cone(z, A):
    """Euclidean projection of z onto C(A) = {u | A u <= 0} in R^2 (closed form,
    Algorithm 1 of the thesis): try the empty active set, then single
    constraints with a non-negative multiplier; otherwise the projection is 0."""
    if A.shape[0] == 0 or np.all(A @ z <= 1e-9):
        return z.copy()
    best, best_d = np.zeros(2), np.inf
    for a in A:
        lam = a @ z
        if lam <= 0:
            continue
        p = z - lam * a
        if np.all(A @ p <= 1e-9):
            d = np.linalg.norm(z - p)
            if d < best_d:
                best, best_d = p, d
    return best


def cluster_scan(rel_points, gap):
    """Group consecutive scan points (N x 2, in scan order, robot frame or any
    frame centred on the robot) into objects; return the closest point of each.
    Handles the wrap-around between the last and the first ray."""
    n = len(rel_points)
    if n == 0:
        return np.zeros((0, 2))
    jumps = np.linalg.norm(np.diff(rel_points, axis=0), axis=1) > gap
    labels = np.concatenate([[0], np.cumsum(jumps)])
    if n > 2 and labels[-1] != 0 and np.linalg.norm(rel_points[-1] - rel_points[0]) <= gap:
        labels[labels == labels[-1]] = 0          # merge the wrap-around cluster
    reps = []
    dists = np.linalg.norm(rel_points, axis=1)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        reps.append(rel_points[idx[np.argmin(dists[idx])]])
    return np.array(reps)


@dataclass
class RSVCController:
    p: RSVCParams = field(default_factory=RSVCParams)
    theta: float = 0.0
    v_prev: np.ndarray = field(default_factory=lambda: np.zeros(2))
    deadlock: bool = False
    d_min_prev: float = np.inf

    def step(self, pos, goal, obstacles_rel, dt):
        """pos, goal: world positions (2,). obstacles_rel: (M, 2) closest points
        of the detected objects relative to the robot centre, in the world frame.
        Returns the planar velocity command and diagnostic values."""
        p = self.p
        # ---- 1. rotation angle (flag from the previous step) -------------
        rs = p.d_s
        prox = 1.0 - (self.d_min_prev - rs) / (p.avoid_radius - rs + 1e-9)
        prox = float(np.clip(prox, 0.0, 1.0)) if np.isfinite(self.d_min_prev) else 0.0
        rate = p.rot_rate_min + (p.rot_rate_max - p.rot_rate_min) * prox
        th_max = math.radians(p.theta_max_deg)
        if self.deadlock and p.method == "proposed":
            self.theta = min(self.theta + rate * dt, th_max)
        else:
            self.theta = max(self.theta - p.decay_factor * rate * dt, 0.0)

        # ---- 2. nominal controller ----------------------------------------
        e = goal - pos
        dist_goal = float(np.linalg.norm(e))
        u0 = p.k_g * e
        s = np.linalg.norm(u0)
        if s > p.v_max:
            u0 = u0 / s * p.v_max

        # ---- 3. active neighbours and smooth activation -------------------
        d = np.linalg.norm(obstacles_rel, axis=1) if len(obstacles_rel) else np.zeros(0)
        act = d <= p.avoid_radius
        self.d_min_prev = float(d.min()) if len(d) else np.inf
        if np.any(act):
            A = obstacles_rel[act] / d[act, None]
            L = p.avoid_radius - p.d_s
            ww = np.where(d[act] <= p.d_s, 1.0,
                          0.5 + 0.5 * np.cos(np.pi * (d[act] - p.d_s) / L))
            w = float(ww.max())
            c, sn = math.cos(self.theta), math.sin(self.theta)
            z = np.array([c * u0[0] - sn * u0[1], sn * u0[0] + c * u0[1]])
            u = (1.0 - w) * z + w * project_onto_cone(z, A)
            n_act = int(act.sum())
        else:
            w, u, n_act = 0.0, u0, 0

        # ---- 4. acceleration and speed limits -----------------------------
        du = u - self.v_prev
        if np.linalg.norm(du) > p.a_max * dt:
            du = du / np.linalg.norm(du) * p.a_max * dt
        v = self.v_prev + du
        if np.linalg.norm(v) > p.v_max:
            v = v / np.linalg.norm(v) * p.v_max
        if dist_goal <= p.goal_tol:
            v = np.zeros(2)

        # ---- 5. deadlock flag for the next step ---------------------------
        self.deadlock = bool(np.linalg.norm(v) < p.deadlock_speed and dist_goal > p.goal_tol
                             and n_act > 0)
        self.v_prev = v
        return v, dict(theta_deg=math.degrees(self.theta), w=w, n_active=n_act,
                       deadlock=int(self.deadlock), d_obs=self.d_min_prev,
                       dist_goal=dist_goal, ux=u[0], uy=u[1])


def si_to_unicycle(v, yaw, v_lim, w_lim, omega_gain=math.pi):
    """Same mapping as the Robotarium create_si_to_uni_dynamics (Eq. si2uni):
    linear speed = projection on the heading; angular speed proportional to
    the heading error (omega_gain at 90 degrees)."""
    ef = np.array([math.cos(yaw), math.sin(yaw)])
    ep = np.array([-math.sin(yaw), math.cos(yaw)])
    lin = float(ef @ v)
    if np.linalg.norm(v) < 1e-6:
        return 0.0, 0.0
    ang = omega_gain * math.atan2(float(ep @ v), lin) / (math.pi / 2)
    return float(np.clip(lin, -v_lim, v_lim)), float(np.clip(ang, -w_lim, w_lim))
