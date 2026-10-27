import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
import numpy as np
import math
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        print("deadlock_angle)",np.rad2deg(self.deadlock_angle),"id",self.id,"RS",rotation_speed,"prox",proximity)
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

# --- Robotarium simulator set-up ---
r = robotarium.Robotarium(number_of_robots=NUM_AGENTS, show_figure=True, sim_in_real_time=True, initial_conditions=initial_positions)
si_to_uni_dyn = create_si_to_uni_dynamics()

# Robotarium barrier certificate (created but NOT applied: see set_velocities below)
uni_barrier_cert = create_uni_barrier_certificate_with_boundary()
trajectory_x=[[] for _ in range(NUM_AGENTS)]
trajectory_y=[[] for _ in range(NUM_AGENTS)]
# run the simulation for at most 1500 steps
iterations = 1500
for k in range(iterations):
    poses = r.get_poses()
    for i in range(NUM_AGENTS):
        agents_objects[i].pos = poses[0:2, i]
    
        # -- Store trajectory ------------------------------------------------ #
    for i in range(NUM_AGENTS):
        trajectory_x[i].append(poses[0, i])
        trajectory_y[i].append(poses[1, i])

    dxi = np.zeros((2, NUM_AGENTS))
    for i in range(NUM_AGENTS):
        u_nom = agents_objects[i].get_nominal_velocity()
        u_safe = agents_objects[i].solve_rsvc(u_nom, agents_objects)
        dxi[:, i] = u_safe
        agents_objects[i].u_prev = u_safe
        
    dxu = si_to_uni_dyn(dxi, poses)
    
    # barrier certificate output (not used)
    dxu_safe = uni_barrier_cert(dxu, poses)
    
    r.set_velocities(np.arange(NUM_AGENTS), dxu)
    r.step()

    # -- Check if all goals reached -------------------------------------- #
    all_done = True
    for i in range(NUM_AGENTS):
        dist_to_goal = np.linalg.norm(agents_objects[i].pos - agents_objects[i].goal)
        if dist_to_goal > 0.05:
            all_done = False

    if all_done:
        print(f"[SUCCESS] All robots reached their targets at step {k}!")
        break

colors = plt.cm.tab10(np.linspace(0, 1, NUM_AGENTS))
 # ---- Plot 1: Robot Trajectories ---------------------------------------- #
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.set_aspect('equal')
ax1.set_title('RSVC Collision Avoidance - Robot Trajectories\n'
                f'N={NUM_AGENTS} robots, Circle-Crossing Scenario', fontsize=13, fontweight='bold')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.add_patch(plt.Circle((0, 0), radius_circle, color='lightgray', fill=False,
                            linestyle='--', linewidth=1, label='Initial circle'))
ax1.grid(True, alpha=0.3)

legend_handles = []
for i in range(NUM_AGENTS):
    # Trajectory
    ax1.plot(trajectory_x[i], trajectory_y[i], color=colors[i],
                linewidth=1.5, alpha=0.8)
    # Start
    ax1.plot(trajectory_x[i][0], trajectory_y[i][0], 'o',
                color=colors[i], markersize=10, markeredgecolor='black')
    # End
    ax1.plot(trajectory_x[i][-1], trajectory_y[i][-1], '^',
                color=colors[i], markersize=10, markeredgecolor='black')
    # Target
    ax1.plot(agents_objects[i].goal[0], agents_objects[i].goal[1], 'x',
                color=colors[i], markersize=12, markeredgewidth=2.5)
    legend_handles.append(
        mpatches.Patch(color=colors[i], label=f'Robot {i+1}')
    )

# Legend icons
ax1.plot([], [], 'ok', markersize=8, label='Start (o)')
ax1.plot([], [], '^k', markersize=8, label='Final position (^)')
ax1.plot([], [], 'xk', markersize=8, markeredgewidth=2, label='Target (x)')

ax1.legend(handles=legend_handles + [
    mpatches.Patch(color='k', label='Start (o)'),
    mpatches.Patch(color='k', label='Final (^)'),
    mpatches.Patch(color='k', label='Target (x)'),
], loc='upper right', fontsize=8)

plt.tight_layout()
fig1.savefig('trajectories.png', dpi=150, bbox_inches='tight')
print("\n[INFO] Trajectory plot saved to 'trajectories.png'")
# ---- Plot 4: Rotation Matrix History Over Time ------------------------- #

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.set_title('Rotation Matrix Values Over Simulation Time', fontsize=13, fontweight='bold')
ax4.set_xlabel('Simulation Step')
ax4.set_ylabel('Rotation Angle (degrees)')
steps = range(len(agents_objects[0].Rotation_matrix_history))

# Plot each robot's rotation history

# print("rot",Rotation_matrix_history) # shape (steps, N)
for i in range(NUM_AGENTS):
    rotation_matrix_history_array = np.array(agents_objects[i].Rotation_matrix_history) 
    ax4.plot(steps, agents_objects[i].Rotation_matrix_history, color=colors[i], 
             linewidth=2, label=f'Robot {i+1}max rotation angle {np.max(rotation_matrix_history_array)}', alpha=0.8)
# ax4.plot(steps, agents_objects[0].Rotation_matrix_history, color=colors[0], 
#                 linewidth=2, label=f'Robot {1}max rotation angle {np.max(rotation_matrix_history_array)}', alpha=0.8)
ax4.axhline(90, color='red', linestyle='--', linewidth=1.5, 
            label='Max rotation (90°)', alpha=0.5)
ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
ax4.set_ylim(-10, 190)
ax4.legend(loc='right', fontsize=9)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
fig4.savefig('rotation_matrix_history.png', dpi=150, bbox_inches='tight')
print("[INFO] Rotation matrix history plot saved to 'rotation_matrix_history.png'")
plt.show(block=True)

# required call at the end of every Robotarium script
r.debug()