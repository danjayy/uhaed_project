import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package directories
    pkg_dir = get_package_share_directory('sim_environment')
    pepper_description_path = get_package_share_directory('pepper_description')
    pepper_meshes_path = get_package_share_directory('pepper_meshes')

    # Set the path to the world file
    world_file = os.path.join(pkg_dir, 'worlds', 'classroom.sdf')
    
    # Path to Pepper URDF (adjust according to your actual structure)
    urdf_file = os.path.join(pepper_description_path, 'urdf', 'pepper.urdf')
    
    # If using xacro:
    # urdf_file = os.path.join(pepper_description_path, 'urdf', 'pepper.urdf.xacro')
    
    # Read the URDF
    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()
    
    # Set GZ_SIM_RESOURCE_PATH to include pepper_meshes and sim_environment models
    models_dir = os.path.join(pkg_dir, 'models')
    pepper_meshes_parent = os.path.dirname(pepper_meshes_path)  # This gives us /share
    
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.pathsep.join([
            models_dir,
            pepper_meshes_parent,  # Changed: parent directory
            os.environ.get('GZ_SIM_RESOURCE_PATH', '')
        ])
    )


    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={
            'gz_args': f'-r {world_file}'
        }.items()
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }]
    )

    # Spawn Pepper robot in Gazebo (with delay to ensure Gazebo is ready)
    spawn_pepper = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'pepper',
                    '-topic', 'robot_description',
                    '-x', '8.5',      # Position beside blackboard
                    '-y', '1.5',      # To the left of blackboard
                    '-z', '0.85',
                    '-Y', '3.14159'   # Facing forward (toward students)
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        gazebo,
        robot_state_publisher,
        spawn_pepper
    ])