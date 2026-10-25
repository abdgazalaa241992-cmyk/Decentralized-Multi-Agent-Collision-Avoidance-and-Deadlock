#!/usr/bin/env python3
"""
Decentralized RSVC + dynamic vector steering for TurtleBot3 robots (ROS 2).

One RSVCRobot node controls ONE robot and uses only that robot's own
odometry and LiDAR scan, so the controller is fully decentralized.  For
convenience all robots can be started in one process (each robot is still
a separate node with its own data):

    # all robots in one process (default config: rsvc_params.yaml)
    python3 rsvc_node.py --all --config rsvc_params.yaml --method proposed

    # baseline without deadlock handling, same scenario
    python3 rsvc_node.py --all --config rsvc_params.yaml --method rsvc

    # one robot per process (e.g. from a launch file)
    python3 rsvc_node.py --robot tb1 --config rsvc_params.yaml

Every run writes  <log_dir>/<run_name>/<robot>.csv  (one row per control
step) and  <run_name>/run_info.yaml.  Use ros2_analyze.py on these folders
to produce the thesis figures and table.

Tested with ROS 2 Jazzy + Gazebo (tb3_multi_robot); TwistStamped commands.
"""
import argparse, csv, math, os, sys, time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from geometry_msgs.msg import TwistStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsvc_core import RSVCParams, RSVCController, cluster_scan, si_to_unicycle

LOG_FIELDS = ["t", "t_abs", "x", "y", "yaw", "vx_cmd", "vy_cmd", "v", "omega", "theta_deg", "w",
              "deadlock", "n_active", "d_obs", "dist_goal", "estop"]


def yaw_from_quat(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class RSVCRobot(Node):
    def __init__(self, name, cfg, method, run_dir):
        super().__init__(f"rsvc_{name}", parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, bool(cfg.get("use_sim_time", True)))])
        self.name = name
        rc = cfg["robots"][name]
        self.goal = np.array(rc["goal"], dtype=float)
        # odometry -> world frame (odom of each robot usually starts at its spawn pose)
        off = rc.get("odom_offset", [0.0, 0.0, 0.0])
        self.off_xy, self.off_yaw = np.array(off[:2], float), float(off[2])
        self.lidar_offset = np.array(cfg.get("lidar_offset", [-0.064, 0.0]), float)
        pp = dict(cfg.get("controller", {}))
        pp["method"] = method
        self.ctrl = RSVCController(RSVCParams(**pp))
        self.w_max = float(cfg.get("omega_max", 1.82))
        self.omega_gain = float(cfg.get("omega_gain", math.pi))
        self.dt = 1.0 / float(cfg.get("rate_hz", 10.0))
        self.use_stamped = bool(cfg.get("use_twist_stamped", True))
        # emergency stop (implementation safety layer, logged): no translation
        # toward an object closer than estop_distance in the driving direction
        self.estop_dist = float(cfg.get("estop_distance", 0.0))
        self.estop_half = math.radians(float(cfg.get("estop_half_angle_deg", 60.0)))
        self.pts_robot = np.zeros((0, 2))

        self.pose = None          # (x, y, yaw) in world frame
        self.scan = None
        self.done = False
        self.t0 = None

        ns = cfg.get("namespace_format", "/{name}")
        topic = lambda t: f"{ns.format(name=name)}/{t}"
        self.create_subscription(Odometry, topic("odom"), self.on_odom, 10)
        self.create_subscription(LaserScan, topic("scan"), self.on_scan, qos_profile_sensor_data)
        self.pub = self.create_publisher(TwistStamped if self.use_stamped else Twist,
                                         topic("cmd_vel"), 10)
        self.create_timer(self.dt, self.control_step)

        os.makedirs(run_dir, exist_ok=True)
        self.log_file = open(os.path.join(run_dir, f"{name}.csv"), "w", newline="")
        self.log = csv.DictWriter(self.log_file, fieldnames=LOG_FIELDS)
        self.log.writeheader()
        self.get_logger().info(f"{name}: goal={self.goal.tolist()} method={method}")

    # ------------------------------------------------------------------
    def on_odom(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        c, s = math.cos(self.off_yaw), math.sin(self.off_yaw)
        xw = self.off_xy + np.array([c * x - s * y, s * x + c * y])
        self.pose = (xw[0], xw[1], yaw_from_quat(msg.pose.pose.orientation) + self.off_yaw)

    def on_scan(self, msg):
        self.scan = msg

    def obstacles_world(self):
        """Closest point of each object seen by the LiDAR, relative to the robot
        centre, expressed in the world frame."""
        msg = self.scan
        r = np.asarray(msg.ranges, dtype=float)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        pts = np.c_[r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])] + self.lidar_offset
        self.pts_robot = pts                     # robot frame, origin at the base centre
        # only points that can matter (saves time, keeps clusters local)
        pts = pts[np.linalg.norm(pts, axis=1) <= self.ctrl.p.avoid_radius + 0.3]
        reps = cluster_scan(pts, self.ctrl.p.cluster_gap)
        yaw = self.pose[2]
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        return reps @ R.T

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def publish(self, v, w):
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        else:
            msg = Twist()
            msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)

    def control_step(self):
        if self.pose is None or self.scan is None:
            return
        t = self.now()
        if t <= 0.0:            # simulation clock not received yet
            return
        if self.t0 is None:
            self.get_logger().info(f"{self.name}: start pose x={self.pose[0]:.2f} "
                                   f"y={self.pose[1]:.2f} yaw={math.degrees(self.pose[2]):.0f} deg")
            self.t0 = t
        x, y, yaw = self.pose
        obs = self.obstacles_world()
        vel, info = self.ctrl.step(np.array([x, y]), self.goal, obs, self.dt)
        lin, ang = si_to_unicycle(vel, yaw, self.ctrl.p.v_max, self.w_max, self.omega_gain)
        estop = 0
        if self.estop_dist > 0 and abs(lin) > 1e-3 and len(self.pts_robot):
            P = self.pts_robot if lin > 0 else -self.pts_robot      # driving direction = +x
            d = np.linalg.norm(P, axis=1)
            bear = np.abs(np.arctan2(P[:, 1], P[:, 0]))
            if np.any((d < self.estop_dist) & (bear < self.estop_half)):
                lin, estop = 0.0, 1                                  # keep turning, stop translating
                self.ctrl.v_prev = np.zeros(2)
        self.done = info["dist_goal"] <= self.ctrl.p.goal_tol
        if self.done:
            lin, ang = 0.0, 0.0
        self.publish(lin, ang)
        self.log.writerow(dict(t=round(t - self.t0, 3), t_abs=round(t, 3), x=x, y=y, yaw=yaw,
                               vx_cmd=vel[0], vy_cmd=vel[1], v=lin, omega=ang,
                               theta_deg=info["theta_deg"], w=info["w"],
                               deadlock=info["deadlock"], n_active=info["n_active"],
                               d_obs=info["d_obs"], dist_goal=info["dist_goal"], estop=estop))

    def stop(self):
        try:
            self.publish(0.0, 0.0)
        except Exception:
            pass
        self.log_file.flush()
        self.log_file.close()


# ----------------------------------------------------------------------
def write_run_info(run_dir, method, names, cfg):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_info.yaml"), "w") as f:
        yaml.safe_dump(dict(method=method, robots=names, config=cfg), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "rsvc_params.yaml"))
    ap.add_argument("--method", default=None, choices=["proposed", "rsvc"])
    ap.add_argument("--robot", default=None, help="control a single robot")
    ap.add_argument("--all", action="store_true", help="control all robots of the config")
    ap.add_argument("--run-name", default=None)
    args, ros_args = ap.parse_known_args()

    cfg = yaml.safe_load(open(args.config))
    method = args.method or cfg.get("method", "proposed")
    names = list(cfg["robots"]) if args.all or not args.robot else [args.robot]
    run_name = args.run_name or f"{method}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(os.path.expanduser(cfg.get("log_dir", "~/rsvc_logs")), run_name)
    os.makedirs(run_dir, exist_ok=True)
    write_run_info(run_dir, method, names, cfg)

    rclpy.init(args=ros_args)
    nodes = [RSVCRobot(n, cfg, method, run_dir) for n in names]
    ex = MultiThreadedExecutor()
    for n in nodes:
        ex.add_node(n)
    max_time = float(cfg.get("max_time", 300.0))
    t_start = time.time()
    try:
        while rclpy.ok():
            ex.spin_once(timeout_sec=0.05)
            if all(n.done for n in nodes):
                print(f"[SUCCESS] all robots reached their goals -> logs in {run_dir}")
                break
            if time.time() - t_start > max_time:
                print(f"[TIMEOUT] {max_time:.0f} s wall time -> logs in {run_dir}")
                break
    except KeyboardInterrupt:
        print(f"[STOPPED] logs in {run_dir}")
    finally:
        for n in nodes:
            n.stop()
            n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
