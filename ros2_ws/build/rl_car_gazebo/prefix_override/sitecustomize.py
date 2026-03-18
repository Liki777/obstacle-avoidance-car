import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/liki777/graduation_project/ros2_ws/install/rl_car_gazebo'
