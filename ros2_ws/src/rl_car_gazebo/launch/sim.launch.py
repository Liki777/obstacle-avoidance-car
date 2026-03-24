from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, 
    ExecuteProcess, 
    TimerAction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    world = LaunchConfiguration("world")
    use_sim_time = LaunchConfiguration("use_sim_time")
    gui = LaunchConfiguration("gui")
    server = LaunchConfiguration("server")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")

    pkg_gazebo = FindPackageShare("rl_car_gazebo")
    pkg_desc = FindPackageShare("rl_car_description")

    world_path = PathJoinSubstitution([pkg_gazebo, "worlds", world])
    robot_urdf = PathJoinSubstitution([pkg_desc, "robot.urdf"])

    # 启动 gzserver，显式加载 factory 插件
    gzserver = ExecuteProcess(
        cmd=[
            "gzserver",
            "--verbose",
            world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ],
        output="screen",
        additional_env={
            # WSL/无声卡环境下抑制 OpenAL 设备报错
            "ALSOFT_DRIVERS": "null",
        },
        condition=IfCondition(server),
    )

    gzclient = ExecuteProcess(
        cmd=["gzclient"],
        output="screen",
        additional_env={
            "ALSOFT_DRIVERS": "null",
        },
        condition=IfCondition(gui),
    )

    robot_description = ParameterValue(Command(["cat ", robot_urdf]), value_type=str)
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "robot_description": robot_description}],
    )

    # spawn 节点 - 带重试机制
    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-entity", "rl_car",
            "-x", x, "-y", y, "-z", z,
            "-timeout", "30",  # 增加超时时间
        ],
    )

    # 延迟 spawn，确保 gzserver 完全就绪
    delayed_spawn = TimerAction(
        period=5.0,  # 等待 5 秒
        actions=[spawn],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value="maze.world"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("server", default_value="true"),
            DeclareLaunchArgument("x", default_value="0.0"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument("z", default_value="0.1"),
            gzserver,
            gzclient,
            rsp,
            delayed_spawn,  # 使用延迟版本
        ]
    )