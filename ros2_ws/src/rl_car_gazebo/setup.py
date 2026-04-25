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
                "worlds/avoid_training_course.world",
                "worlds/curriculum_s2_one_obstacle.world",
                "worlds/curriculum_s3_narrow.world",
                "worlds/level0_map.world",
                "worlds/level0_arena_8x8.world",
                "worlds/level0_road_straight_8x8.world",
                "worlds/level0_road_straight_2way_2lane_8x8.world",
                "worlds/level0_road_straight_long_2way_2lane_each_8x20.world",
                "worlds/level1_map.world",
                "worlds/arena_8x8.world",
                "worlds/level1_arena_8x8.world",
                "worlds/level1_arena_8x8_fast.world",
                "worlds/level1_arena_8x8_irregular_tight.world",
                "worlds/level2_arena_8x8.world",
                "worlds/level3_arena_8x8.world",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)

