from setuptools import setup

package_name = "rl_car_gazebo"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/sim.launch.py"]),
        (
            f"share/{package_name}/worlds",
            [
                "empty.world",
                "maze.world",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)

