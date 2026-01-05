#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import xacro


def generate_launch_description():
    # Get the URDF file from pepper_description
    pepper_description_path = get_package_share_directory('pepper_description')
    
    # Assuming your URDF is in pepper_description/urdf/pepper.urdf or similar
    # Adjust the path according to your actual file structure
    urdf_file = os.path.join(pepper_description_path, 'urdf', 'pepper.urdf')
    
    # If you're using xacro instead:
    # xacro_file = os.path.join(pepper_description_path, 'urdf', 'pepper.urdf.xacro')
    # robot_description = xacro.process_file(xacro_file).toxml()
    
    # Read the URDF file
    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'pepper',
            '-topic', 'robot_description',
            '-x', '8.5',  # Position beside blackboard
            '-y', '1.5',
            '-z', '0.0',
            '-Y', '3.14159'  # Face forward
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    return LaunchDescription([
        robot_state_publisher,
        spawn_entity
    ])