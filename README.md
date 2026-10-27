# Decentralized Multi-Agent Collision Avoidance and Deadlock Resolution

Code, data, figures and videos for the experimental part of a master's thesis on
**decentralized collision avoidance for multi-robot systems with deadlock
resolution**. The method extends **RSVC** (a velocity-cone collision-avoidance
law) with **dynamic vector steering**: when a robot detects that it is stuck
(a deadlock), it gradually rotates its preferred velocity until it escapes, and
then "unwinds" the rotation back to zero once it moves freely again.

The same control law is validated on three platforms:

| Platform | Robots | Scenario | Folder |
|---|---|---|---|
| **Robotarium Python simulator** (Monte Carlo study) | 2–12 GRITSbots | Antipodal circle swap, 30 randomized trials per team size; comparison with RSVC, RSVC + noise and the Robotarium barrier certificates (SBC) | [`simulation/`](simulation/) |
| **Robotarium** (Georgia Tech, physical testbed) | 10 GRITSbots | Antipodal circle swap (each robot goes to the opposite point of a circle of radius 0.8 m) | [`robotarium/`](robotarium/) |
| **ROS 2 Jazzy + Gazebo Harmonic** (simulation) | 4 TurtleBot3 Waffle | Diagonal corner swap on a 4 × 4 m square, neighbours detected with LiDAR | [`ROS2/`](ROS2/) |

> **ملخص بالعربية:** هذا المستودع يحتوي على الأكواد والبيانات والنتائج والفيديوهات
> الخاصة بالجزء العملي من رسالة ماجستير حول **تجنب التصادم اللامركزي لأنظمة
> الروبوتات المتعددة مع حل مشكلة الانسداد (Deadlock)**. الطريقة المقترحة تعتمد على
> خوارزمية RSVC مع **تدوير ديناميكي لمتجه السرعة المفضّلة** عند اكتشاف الانسداد.
> تم اختبار الطريقة على منصة **Robotarium** الحقيقية (10 روبوتات) وفي محاكاة
> **ROS 2 / Gazebo** (4 روبوتات TurtleBot3). لإعادة التجارب اتبع الخطوات في
> الأقسام أدناه.

---

## The control law (summary)

For each robot, at every control step:

```
u0    = sat(k_g (g - x))                 nominal go-to-goal controller
z     = R(theta) u0                      preferred velocity rotated by theta
w     = max_j w(d_j)                     smooth activation from neighbour distances
u     = (1 - w) z + w Proj_C(A)(z)       blended RSVC law (projection onto the safe cone)
v     = acceleration / speed limits of u
theta += rate * dt      if ||v|| < eps_v (deadlock detected)
theta -= 2 rate * dt    otherwise        (unwinding back to 0)
```

`Proj_C(A)` is the closed-form Euclidean projection onto the cone
`C(A) = {u | A u <= 0}` built from the directions to the neighbours.
The baseline (`rsvc`) is the same law with `theta` fixed at 0.

---

## Repository structure

```
.
├── README.md                     ← this file
├── simulation/                   Robotarium-simulator study (Monte Carlo, ablations)
│   ├── code/
│   │   └── rsvc_steering.py            simulator script of the ten-agent results
│   ├── experiments/
│   │   ├── rsvc_agent.py               the controller, with switches for the baselines
│   │   ├── run_trial.py                one trial: scenario, simulation, metrics
│   │   ├── montecarlo.py               all trials → CSV
│   │   ├── ablation.py                 ablation of theta_max and of the unwinding
│   │   ├── symmetric.py                perfectly symmetric cases (no noise)
│   │   ├── margin_study.py             angular margin vs. escape analysis
│   │   ├── analyze.py                  CSV → table + figures
│   │   └── results/                    CSV files and trajectories reported in the thesis
│   └── figures/                        mc_*.pdf, margin_vs_escape.pdf
│
├── robotarium/                   Physical Robotarium experiment (10 GRITSbots)
│   ├── code/
│   │   ├── rsvc_robotarium_submit.py   script submitted to the Robotarium testbed
│   │   └── hw_analyze.py               log → figures + LaTeX table
│   ├── data/
│   │   ├── log.txt                     experiment log returned by the testbed (used for the results)
│   │   ├── rsvc_poses.npy              poses        (steps × 3 × N)
│   │   ├── rsvc_commands.npy           commands     (steps × 2 × N)
│   │   ├── rsvc_theta_deg.npy          rotation θ   (N × steps)
│   │   └── video_run/                  .npy files of the run shown in the video
│   ├── figures/                        robotarium_hw_traj.pdf, robotarium_hw_theta.pdf
│   └── experiments/results/            robotarium_hw_table.tex
│
├── ROS2/                         ROS 2 / Gazebo experiment (4 TurtleBot3)
│   ├── ros2/                     ← the controller (see ROS2/ros2/README.md)
│   │   ├── rsvc_core.py                controller, no ROS dependency
│   │   ├── rsvc_node.py                ROS 2 node (one per robot, or --all)
│   │   ├── rsvc_params.yaml            robots, goals and parameters
│   │   └── ros2_analyze.py             logs → figures + LaTeX table
│   ├── tb3_multi_robot/          multi-TurtleBot3 Gazebo world (ROS 2 package)
│   ├── figures/                        ros2_traj/theta/cmd/dist.pdf
│   └── experiments/results/            ros2_table.tex, ros2_metrics.csv
│
└── videos/
    ├── robotarium/robotarium_hardware_run.mp4
    └── ros2/
        ├── ros2_deadlock_static_obstacle.mp4
        └── ros2_deadlock_solved.mp4
```

---

## 0. Reproducing the simulation study (Monte Carlo)

### Requirements
```bash
pip install numpy scipy matplotlib pandas cvxopt
pip install git+https://github.com/robotarium/robotarium_python_simulator.git
```

### Run
```bash
cd simulation/code
python rsvc_steering.py                      # ten-agent circle swap with graphics

cd ../experiments
python run_trial.py                          # quick test: one run of each method, N = 10
python montecarlo.py --trials 30 --out results/mc.csv   # main study
python ablation.py                           # theta_max and unwinding ablations
python symmetric.py                          # perfectly symmetric cases
python analyze.py results/mc.csv             # table + figures
```
Outputs go to `simulation/experiments/results/` and `simulation/figures/`.
More details: [`simulation/experiments/README.md`](simulation/experiments/README.md).

---

## 1. Reproducing the Robotarium experiment

### Requirements
- Python 3.8+ with `numpy` and `matplotlib`
- To simulate locally before submitting: the
  [Robotarium Python simulator](https://github.com/robotarium/robotarium_python_simulator)
  (`rps` package)

### Run the controller
1. **Local simulation:** install the simulator, then
   ```bash
   cd robotarium/code
   python rsvc_robotarium_submit.py
   ```
2. **Physical testbed:** create an account on
   [robotarium.gatech.edu](https://www.robotarium.gatech.edu), submit
   `rsvc_robotarium_submit.py` as a new experiment, and download the returned
   `log.txt` (and the `.npy` files if returned). The script prints every robot's
   pose, command and rotation angle to the log (lines starting with `D,`), so the
   results can be recovered even when the testbed does not return data files.

### Regenerate the figures and the table
```bash
cd robotarium/code
python hw_analyze.py ../data/log.txt
```
Outputs go to `robotarium/figures/` and `robotarium/experiments/results/`.

### Main parameters (`rsvc_robotarium_submit.py`)
| Parameter | Value | Meaning |
|---|---|---|
| `NUM_AGENTS` | 10 | number of robots |
| `AGENT_RADIUS` | 0.08 m | robot radius |
| `SENSING_RADIUS` | 0.35 m | interaction (avoidance) radius |
| `MAX_SPEED` | 0.2 m/s | speed limit |
| `A_MAX` | 0.5 m/s² | acceleration limit |
| `DT` | 0.033 s | time step |

---

## 2. Reproducing the ROS 2 / Gazebo experiment

### Requirements
- Ubuntu 24.04, [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html),
  Gazebo Harmonic (or use the Docker image in `ROS2/tb3_multi_robot/docker/`)
- Python packages: `numpy`, `pyyaml`, `pandas`, `matplotlib`

### Step 1 — build and launch the Gazebo world
```bash
mkdir -p ~/robot_ws/src
cp -r ROS2/tb3_multi_robot ~/robot_ws/src/
cd ~/robot_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch tb3_multi_robot tb3_world.launch.py
```
The four robots `tb1 … tb4` are spawned at the corners (±2, ±2) as configured in
`ROS2/tb3_multi_robot/config/robots.yaml`. See
[`ROS2/tb3_multi_robot/README.md`](ROS2/tb3_multi_robot/README.md) for
CycloneDDS settings and Docker instructions.

### Step 2 — run the controller (new terminal, ROS 2 sourced)
```bash
cd ROS2/ros2
python3 rsvc_node.py --all --method proposed --run-name proposed_1
```
Reset the simulation to the same start poses, then run the baseline:
```bash
python3 rsvc_node.py --all --method rsvc --run-name rsvc_1
```
The baseline is expected to stay stuck in the centre (deadlock); stop it with
Ctrl+C after about 60 s. Logs are written to `~/rsvc_logs/<run-name>/`.

### Step 3 — regenerate the figures and the table
```bash
python3 ros2_analyze.py ~/rsvc_logs/proposed_1 ~/rsvc_logs/rsvc_1 \
        --labels "Proposed" "RSVC (baseline)"
```
Outputs go to `ROS2/figures/` and `ROS2/experiments/results/`.

All controller parameters (speeds, safety margin, avoidance radius, rotation
rates, goals) are in [`ROS2/ros2/rsvc_params.yaml`](ROS2/ros2/rsvc_params.yaml).
More details: [`ROS2/ros2/README.md`](ROS2/ros2/README.md).

---

## 3. Results

### ROS 2 / Gazebo (4 TurtleBot3, diagonal swap)
| Run | Success | T_goal (s) | Path ratio | d_centre min (m) | θ_max (°) |
|---|---|---|---|---|---|
| **Proposed** | yes | 38.5 | 1.12 | 0.58 | 76.6 |
| RSVC (baseline) | no — deadlock | – | – | 0.46 | 0 |

### Physical Robotarium (10 GRITSbots, circle swap)
| Run | Success | Path ratio | d_min (m) | θ_max (°) |
|---|---|---|---|---|
| Proposed | no (not all robots within 5 cm of their goals) | 1.32 | 0.149 | 90.0 |

`d_min` is the smallest centre-to-centre distance between any two robots over
the run. Full tables: `*/experiments/results/*.tex`; figures: `*/figures/*.pdf`;
videos: `videos/`.

---

## Credits

- `ROS2/tb3_multi_robot/` is based on
  [arshadlab/tb3_multi_robot](https://github.com/arshadlab/tb3_multi_robot)
  (ROS 2 Jazzy branch), with the robot spawn poses changed for this experiment.
  Original credit for that package and the TurtleBot3/Nav2 files it contains
  remains with their authors.
- The physical experiments were run on the
  [Robotarium](https://www.robotarium.gatech.edu) at Georgia Tech.

## Citation

If you use this code, please cite the thesis:

```
<Author name>, "<Thesis title>", Master's thesis, <University>, <Year>.
```
