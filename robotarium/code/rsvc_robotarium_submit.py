"""
RSVC + dynamic preferred-velocity rotation -- submission script for the
physical Robotarium testbed (10 GRITSbots, antipodal circle swap).

Controller: identical to Appendix A of the thesis (same parameters).
Changes with respect to the first submission:
  1. The positions, the commands and the rotation angle of every robot are
     PRINTED to the experiment log (log.txt), because the testbed does not
     always return data files. Lines start with "D," and are read by
     hw_analyze.py.
  2. The same data are also saved with np.save (returned only if the
     testbed returns files).
  3. The per-step debug print and the matplotlib figures were removed
     (figures are produced offline by hw_analyze.py).
  4. The unused barrier-certificate call was removed; the command sent to
     the robots is the same as before (no barrier certificate is applied).
"""
import rps.robotarium as robotarium
from rps.utilities.transformations import *
import numpy as np
import math

# --- Hyper-parameters adapted to the dimensions of the real Robotarium arena ---
NUM_AGENTS = 10            # number of robots (suitable for the size of the physical arena)
DT = 0.033                # default time step of the Robotarium Python simulator
GAIN_K = 1.0
AGENT_RADIUS = 0.08       # robot radius (8 cm, suitable for the real GRITSbots)
SENSING_RADIUS = 0.35      # interaction (avoidance) radius
MAX_SPEED = 0.2           # maximum safe speed in the simulator
A_MAX = 0.5               # maximum acceleration, to protect the real motors
EPSILON = 0.01            # small safety margin for smooth motion

def apply_actuator_constraints(u, u_prev, u_max, a_max, dt):
    """ Actuator constraints: 1. acceleration limit, 2. speed limit """
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

def rotate_point(point, angle_degrees):
    """Rotate a 2D point around the origin by the given angle (in degrees)."""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])
    return rotation_matrix @ point
MIN_ROT_SPEED = 0.008
MAX_ROT_SPEED = 0.009
class RSVCAgent:
    def __init__(self, id, start_pos, goal_pos, radius=AGENT_RADIUS):
        self.id = id
        self.pos = np.array(start_pos, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.radius = radius
        self.deadlock_en = True
        self.deadlock = False
        self.deadlock_angle = 0.0  # accumulated rotation angle (added dynamically)
        self.smooth_en = True
        self.u_prev = np.zeros(2)
        self.Rotation_matrix_history = []
        self.min_dist_to_hazard=20
        self.r_safe=0


    def get_nominal_velocity(self):

        proximity = 1.0  -((self.min_dist_to_hazard - self.r_safe) / (SENSING_RADIUS - self.r_safe + 1e-9))
        proximity = np.clip(proximity, 0.0, 1.0)
        
        rotation_speed = MIN_ROT_SPEED + (MAX_ROT_SPEED - MIN_ROT_SPEED) * proximity
        max_rotation_angle=math.pi/2
        if self.deadlock and self.deadlock_en:
            self.deadlock_angle = min(self.deadlock_angle + rotation_speed,max_rotation_angle )
            # self.deadlock_angle = max(self.deadlock_angle ,np.deg2rad(60) )
        else:
            self.deadlock_angle = max(self.deadlock_angle - rotation_speed*2, 0.0)
        self.Rotation_matrix_history.append(np.rad2deg(self.deadlock_angle) )
        """ Nominal velocity toward the goal """
        u_nom = -GAIN_K * (self.pos - self.goal)
        speed = np.linalg.norm(u_nom)
        if speed > MAX_SPEED:
            u_nom = (u_nom / speed) * MAX_SPEED
        return u_nom

    def solve_rsvc(self, u_nom, all_agents):
        """ Safe velocity using the (smoothly activated) RSVC projection """
        A_rows = []
        w_max = 0.0 if self.smooth_en else 1.0
        
        # track the distance to the closest neighbor
        self.min_dist_to_hazard = SENSING_RADIUS
        r_safe_closest = self.radius
        
        for other in all_agents:
            if other.id == self.id:
                continue

            relative_pos = other.pos - self.pos
            dist = np.linalg.norm(relative_pos)
            if dist <= SENSING_RADIUS:
                bearing = relative_pos / (dist + 1e-9)
                A_rows.append(bearing)
                
                # record the closest neighbor
                if dist < self.min_dist_to_hazard:
                    self.min_dist_to_hazard = dist
                    r_safe_closest = self.radius + other.radius
                
                self.r_safe = self.radius + other.radius
                if dist <= self.r_safe + EPSILON:
                    w = 1.0
                else:
                    w = 0.5 + 0.5 * math.cos(math.pi * (dist - (self.r_safe + EPSILON)) / (SENSING_RADIUS - (self.r_safe + EPSILON) + 1e-9))
                w_max = max(w_max, w)

        if not A_rows:
            self.min_dist_to_hazard=20
            best_normal=apply_actuator_constraints(u_nom, self.u_prev, MAX_SPEED, A_MAX, DT)
            self.deadlock = False
            return best_normal

        A = np.vstack(A_rows)
        u_norm_R=rotate_point(u_nom,np.rad2deg(self.deadlock_angle))
        # project the rotated preferred velocity onto the safety cone
        u_projected = self._project_onto_cone(u_norm_R, A, r_safe_closest)
        u_safe = (1 - w_max) * u_norm_R + w_max * u_projected

        u_safe = apply_actuator_constraints(u_safe, self.u_prev, MAX_SPEED, A_MAX, DT)

        
        normal_speed = np.linalg.norm(u_safe)
        
        # 2. detect entering / leaving a deadlock
        if normal_speed < 1e-1:
            self.deadlock = True
        else:
        # elif normal_speed > 0.25 * MAX_SPEED:
            self.deadlock = False

        return u_safe

    def _project_onto_cone(self, u0, A, r_safe):
        """ Projection of u0 onto the safety velocity cone """
        best = np.zeros_like(u0)
        best_dist = np.linalg.norm(u0 - best)
        
        # 1. unconstrained case: u0 already inside the cone
        if np.all(A @ u0 <= 1e-9):
            best_normal = u0
        else:
            best_normal = np.zeros_like(u0)
            best_dist_n = np.linalg.norm(u0 - best_normal)
            for a in A:
                aa = np.dot(a, a)
                if aa < 1e-12: continue
                u = u0 - (np.dot(a, u0) / aa) * a
                if np.all(A @ u <= 1e-9):
                    d = np.linalg.norm(u0 - u)
                    if d < best_dist_n:
                        best_normal = u
                        best_dist = d

        return best_normal

# --- Scenario: robots on a circle, goals at the antipodal points ---
radius_circle = 0.8
initial_positions = np.zeros((3, NUM_AGENTS))
agents_objects = []

for i in range(NUM_AGENTS):
    angle = 2 * np.pi * i / NUM_AGENTS
    x_start = radius_circle * np.cos(angle)
    y_start = radius_circle * np.sin(angle)
    x_goal = -x_start
    y_goal = -y_start
    
    initial_positions[0, i] = x_start
    initial_positions[1, i] = y_start
    initial_positions[2, i] = angle + np.pi
    
    agents_objects.append(RSVCAgent(i, [x_start, y_start], [x_goal, y_goal]))

# --- Robotarium set-up ---
r = robotarium.Robotarium(number_of_robots=NUM_AGENTS, show_figure=True,
                          sim_in_real_time=True, initial_conditions=initial_positions)
si_to_uni_dyn = create_si_to_uni_dynamics()

LOG_EVERY = 2          # log every 2nd step (about 15 Hz)
print("H,N=%d,DT=%.3f,R=%.3f,RS=%.3f,VMAX=%.3f,AMAX=%.3f,LOG_EVERY=%d"
      % (NUM_AGENTS, DT, AGENT_RADIUS, SENSING_RADIUS, MAX_SPEED, A_MAX, LOG_EVERY))
print("G," + ",".join("%.4f,%.4f" % (a.goal[0], a.goal[1]) for a in agents_objects))

pose_log, cmd_log = [], []
iterations = 1500
success_step = -1
for k in range(iterations):
    poses = r.get_poses()
    for i in range(NUM_AGENTS):
        agents_objects[i].pos = poses[0:2, i]

    dxi = np.zeros((2, NUM_AGENTS))
    for i in range(NUM_AGENTS):
        u_nom = agents_objects[i].get_nominal_velocity()
        u_safe = agents_objects[i].solve_rsvc(u_nom, agents_objects)
        dxi[:, i] = u_safe
        agents_objects[i].u_prev = u_safe

    pose_log.append(poses.copy())
    cmd_log.append(dxi.copy())
    if k % LOG_EVERY == 0:
        th = [math.degrees(a.deadlock_angle) for a in agents_objects]
        print("D,%d," % k
              + ",".join("%.4f,%.4f,%.3f" % (poses[0, i], poses[1, i], poses[2, i])
                         for i in range(NUM_AGENTS))
              + "," + ",".join("%.4f,%.4f" % (dxi[0, i], dxi[1, i]) for i in range(NUM_AGENTS))
              + "," + ",".join("%.2f" % v for v in th))

    dxu = si_to_uni_dyn(dxi, poses)
    r.set_velocities(np.arange(NUM_AGENTS), dxu)
    r.step()

    all_done = all(np.linalg.norm(a.pos - a.goal) <= 0.05 for a in agents_objects)
    if all_done:
        success_step = k
        print("[SUCCESS] All robots reached their targets at step %d (t = %.2f s)" % (k, k * DT))
        break

if success_step < 0:
    print("[END] Not all robots reached their targets after %d steps" % iterations)

# --- also try to save the data as files (returned only if the testbed allows it) ---
try:
    np.save("rsvc_poses.npy", np.array(pose_log))
    np.save("rsvc_commands.npy", np.array(cmd_log))
    np.save("rsvc_theta_deg.npy", np.array([a.Rotation_matrix_history for a in agents_objects]))
    print("[INFO] data saved to rsvc_*.npy")
except Exception as e:
    print("[INFO] could not save data files:", e)

# required call at the end of every Robotarium script
r.debug()
