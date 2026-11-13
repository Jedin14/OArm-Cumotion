from glob import glob

from setuptools import setup

package_name = "l2_perception"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="l2",
    maintainer_email="user@example.com",
    description="Stationary Unitree L2 perception: drift-free cloud and obstacle detection",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "l2_stabilizer = l2_perception.stabilizer_node:main",
            "l2_obstacles = l2_perception.obstacle_node:main",
            "l2_sim = l2_perception.sim_node:main",
            "l2_sync_check = l2_perception.sync_check:main",
            "l2_dense = l2_perception.dense_node:main",
        ],
    },
)
