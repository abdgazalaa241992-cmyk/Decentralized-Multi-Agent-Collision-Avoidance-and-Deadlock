pkill -9 -f "ros2|gazebo|gz|nav2|amcl|bt_navigator|nav_to_pose|rviz2|assisted_teleop|cmd_vel_relay|robot_state_publisher|joint_state_publisher|move_to_free|mqtt|autodock|cliff_detection|moveit|move_group|basic_navigator"
cd ~/robot_ws
build
export TURTLEBOT3_MODEL=waffle

source install/setup.bash
ros2 launch tb3_multi_robot tb3_world.launch.py
