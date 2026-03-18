from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration("world")
    use_sim_time = LaunchConfiguration("use_sim_time")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")

    pkg_gazebo = FindPackageShare("rl_car_gazebo")
    pkg_desc = FindPackageShare("rl_car_description")

    world_path = PathJoinSubstitution([pkg_gazebo, "worlds", world])
    robot_urdf = PathJoinSubstitution([pkg_desc, "robot.urdf"])

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])
        ),
        launch_arguments={"world": world_path}.items(),
    )

    robot_description = ParameterValue(Command(["cat ", robot_urdf]), value_type=str)
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "robot_description": robot_description}],
    )

    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=["-topic", "robot_description", "-entity", "rl_car", "-x", x, "-y", y, "-z", z],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value="maze.world"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("x", default_value="0.0"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument("z", default_value="0.1"),
            gazebo_launch,
            rsp,
            spawn,
        ]
    )

