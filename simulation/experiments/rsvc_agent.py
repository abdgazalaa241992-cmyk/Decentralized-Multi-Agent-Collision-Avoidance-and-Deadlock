"""
RSVC agent with dynamic vector steering (u0 rotation) -- Monte Carlo version.

This is the controller of `simulation/code/rsvc_steering.py` (the Robotarium script
used for the thesis figures), refactored so that it can be run many times
without a GUI.  The control logic is unchanged; the only differences are:
  * parameters are passed in a dict instead of module-level globals,
  * the debug `print` was removed,
  * a `method` switch selects the baseline / noise / proposed variants,
  * `fix_projection` optionally fixes a small bug in `_project_onto_cone`
    (see the comment there).  The default (False) reproduces the original.
"""
import math
import numpy as np

DEFAULT_PARAMS = dict(
    DT=0.033,
    GAIN_K=1.0,
    AGENT_RADIUS=0.08,
    SENSING_RADIUS=0.35,     # = avoidance radius R_a in the thesis
    MAX_SPEED=0.2,
    A_MAX=0.5,
    EPSILON=0.01,
    MIN_ROT_SPEED=0.008,     # rad per step (~0.46 deg)
    MAX_ROT_SPEED=0.009,     # rad per step (~0.52 deg)
    DECAY_FACTOR=2.0,        # unwinding is DECAY_FACTOR x faster than winding
    MAX_ROT_ANGLE=math.pi / 2,
    DEADLOCK_SPEED=0.1,      # deadlock flag when ||u_safe|| < 0.1 m/s (= 0.5 v_max)
    NOISE_STD=0.05,          # only for method="noise": std of the random perturbation (m/s)
)


def apply_actuator_constraints(u, u_prev, u_max, a_max, dt):
    """1. acceleration limit, 2. speed limit (identical to the original)."""
    delta_u = u - u_prev
    max_delta = a_max * dt
    norm_delta = np.linalg.norm(delta_u)
    if norm_delta > max_delta:
        delta_u = (delta_u / norm_delta) * max_delta
    u_limited = u_prev + delta_u
    speed = np.linalg.norm(u_limited)
    if speed > u_max:
        u_limited = (u_limited / speed) * u_max
    return u_limited


def rotate_point(point, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]]) @ point


class RSVCAgent:
    """method: 'proposed' (rotation of u0), 'rsvc' (no deadlock handling),
    'noise' (random perturbation of u0 while deadlocked)."""

    def __init__(self, id, start_pos, goal_pos, params=None, method="proposed",
                 rng=None, fix_projection=False):
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        self.id = id
        self.pos = np.array(start_pos, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.radius = self.p["AGENT_RADIUS"]
        self.method = method
        self.deadlock_en = (method == "proposed")
        self.deadlock = False
        self.deadlock_angle = 0.0
        self.smooth_en = True
        self.u_prev = np.zeros(2)
        self.angle_history = []          # degrees, one value per step
        self.min_dist_to_hazard = 20
        self.r_safe = 0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.fix_projection = fix_projection

    # ------------------------------------------------------------------ #
    def get_nominal_velocity(self):
        p = self.p
        SR = p["SENSING_RADIUS"]
        proximity = 1.0 - ((self.min_dist_to_hazard - self.r_safe) / (SR - self.r_safe + 1e-9))
        proximity = np.clip(proximity, 0.0, 1.0)
        rotation_speed = p["MIN_ROT_SPEED"] + (p["MAX_ROT_SPEED"] - p["MIN_ROT_SPEED"]) * proximity
        if self.deadlock and self.deadlock_en:
            self.deadlock_angle = min(self.deadlock_angle + rotation_speed, p["MAX_ROT_ANGLE"])
        else:
            self.deadlock_angle = max(self.deadlock_angle - rotation_speed * p["DECAY_FACTOR"], 0.0)
        self.angle_history.append(np.rad2deg(self.deadlock_angle))

        u_nom = -p["GAIN_K"] * (self.pos - self.goal)
        speed = np.linalg.norm(u_nom)
        if speed > p["MAX_SPEED"]:
            u_nom = (u_nom / speed) * p["MAX_SPEED"]
        if self.method == "noise" and self.deadlock:
            u_nom = u_nom + self.rng.normal(0.0, p["NOISE_STD"], 2)
        return u_nom

    def solve_rsvc(self, u_nom, all_agents):
        p = self.p
        SR, EPS = p["SENSING_RADIUS"], p["EPSILON"]
        A_rows = []
        w_max = 0.0 if self.smooth_en else 1.0
        self.min_dist_to_hazard = SR
        r_safe_closest = self.radius

        for other in all_agents:
            if other.id == self.id:
                continue
            relative_pos = other.pos - self.pos
            dist = np.linalg.norm(relative_pos)
            if dist <= SR:
                A_rows.append(relative_pos / (dist + 1e-9))
                if dist < self.min_dist_to_hazard:
                    self.min_dist_to_hazard = dist
                    r_safe_closest = self.radius + other.radius
                self.r_safe = self.radius + other.radius
                if dist <= self.r_safe + EPS:
                    w = 1.0
                else:
                    w = 0.5 + 0.5 * math.cos(math.pi * (dist - (self.r_safe + EPS)) /
                                             (SR - (self.r_safe + EPS) + 1e-9))
                w_max = max(w_max, w)

        if not A_rows:
            self.min_dist_to_hazard = 20
            self.deadlock = False
            return apply_actuator_constraints(u_nom, self.u_prev, p["MAX_SPEED"], p["A_MAX"], p["DT"])

        A = np.vstack(A_rows)
        u_rot = rotate_point(u_nom, self.deadlock_angle)
        u_projected = self._project_onto_cone(u_rot, A)
        u_safe = (1 - w_max) * u_rot + w_max * u_projected      # smooth blending
        u_safe = apply_actuator_constraints(u_safe, self.u_prev, p["MAX_SPEED"], p["A_MAX"], p["DT"])

        self.deadlock = np.linalg.norm(u_safe) < p["DEADLOCK_SPEED"]
        return u_safe

    def _project_onto_cone(self, u0, A):
        if np.all(A @ u0 <= 1e-9):
            return u0
        best = np.zeros_like(u0)
        best_dist = np.inf
        for a in A:
            aa = a @ a
            if aa < 1e-12:
                continue
            if self.fix_projection and a @ u0 <= 0:
                continue                      # multiplier must be >= 0 (KKT)
            u = u0 - (a @ u0) / aa * a
            if np.all(A @ u <= 1e-9):
                d = np.linalg.norm(u0 - u)
                if self.fix_projection:
                    if d < best_dist:
                        best, best_dist = u, d
                else:
                    # Original code: `best_dist` is never updated, so the LAST
                    # feasible candidate is returned (still inside the cone,
                    # but not always the closest point).
                    best = u
        return best
