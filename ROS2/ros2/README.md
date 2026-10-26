# ROS 2 / Gazebo experiments (TurtleBot3, tb3_multi_robot)

Decentralized RSVC + dynamic vector steering, the same control law as the
Robotarium code (thesis Chapter 4), with neighbours detected by the LiDAR.

| File | Content |
|---|---|
| `rsvc_core.py` | Controller (no ROS): LiDAR clustering, blended RSVC law, rotation/unwinding, SI -> unicycle mapping |
| `rsvc_node.py` | ROS 2 node, one per robot (`--robot tb1`) or all robots in one process (`--all`); writes CSV logs |
| `rsvc_params.yaml` | Robots, goals, parameters |
| `ros2_analyze.py` | Logs -> figures + LaTeX table, written directly into the thesis project |

## 1. Run in Gazebo

Start your `tb3_multi_robot` world with the four robots (`tb1` ... `tb4`)
as usual, then in another terminal (ROS 2 sourced):

```bash
cd ros2
pip install pyyaml          # if needed
python3 rsvc_node.py --all --method proposed --run-name proposed_1
```

When the run ends (all robots at their goals, or `max_time`), reset the
simulation to the same start poses and run the baseline:

```bash
python3 rsvc_node.py --all --method rsvc --run-name rsvc_1
```

The baseline is expected to stop in the centre; stop it with Ctrl+C after
~60 s (the log is still saved). Logs go to `~/rsvc_logs/<run-name>/`.

**Check at start-up:** each node prints its start pose. If every robot
prints `x=0 y=0`, the odometry is relative to the spawn pose: put the spawn
poses in `odom_offset` in `rsvc_params.yaml` and run again.

## 2. Produce the figures and table

```bash
python3 ros2_analyze.py ~/rsvc_logs/proposed_1 ~/rsvc_logs/rsvc_1 \
        --labels "Proposed" "RSVC (baseline)"
```

This writes `../figures/ros2_traj.pdf`, `ros2_theta.pdf`, `ros2_cmd.pdf`,
`ros2_dist.pdf` and `../experiments/results/ros2_table.tex`. Recompile the
thesis: Section 5.3 includes them automatically.
