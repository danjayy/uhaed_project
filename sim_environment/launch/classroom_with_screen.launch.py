import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
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
    
    # Path to Pepper URDF
    urdf_file = os.path.join(pepper_description_path, 'urdf', 'pepper.urdf')
    
    # Read the URDF
    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    # Set GZ_SIM_RESOURCE_PATH
    models_dir = os.path.join(pkg_dir, 'models')
    pepper_meshes_parent = os.path.dirname(pepper_meshes_path)
    
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.pathsep.join([
            models_dir,
            pepper_meshes_parent,
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

    # Spawn Pepper robot in Gazebo
    spawn_pepper = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'pepper',
                    '-topic', 'robot_description',
                    '-x', '8.5',
                    '-y', '1.5',
                    '-z', '0.85',
                    '-Y', '3.14159'
                ],
                output='screen'
            )
        ]
    )

    # Spawn tablet screen (after Pepper)
    spawn_screen = TimerAction(
        period=7.0,  # After Pepper is spawned
        actions=[
            Node(
                package='sim_environment',
                executable='spawn_tablet_screen.py',
                name='tablet_screen_spawner',
                output='screen'
            )
        ]
    )

    # Tablet screen publisher node
    tablet_screen_publisher = Node(
        package='sim_environment',
        executable='tablet_screen_publisher.py',
        name='tablet_screen_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # ROS-Gazebo bridge for image topic (if needed for visualization)
    image_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/pepper/tablet_screen/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/world/classroom/create@ros_gz_interfaces/srv/SpawnEntity'
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        gz_resource_path,
        gazebo,
        robot_state_publisher,
        spawn_pepper,
        spawn_screen,
        tablet_screen_publisher,
        image_bridge  # Uncomment if you want to bridge the image to Gazebo
    ])