from setuptools import setup

package_name = "rl_car_description"

setup(
    name=package_name,
    version="0.0.1",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}",
            [
                "robot.urdf",
                "robot.urdf.xacro",
                "robot_core.xacro",
                "gazebo_control.xacro",
                "lidar.xacro",
                "depth_camera.xacro",
                "inertial_macros.xacro",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)

