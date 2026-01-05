#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SpawnEntity
from geometry_msgs.msg import Pose
import time
import os
from ament_index_python.packages import get_package_share_directory


class TabletScreenSpawner(Node):
    def __init__(self):
        super().__init__('tablet_screen_spawner')
        
        # Wait for Gazebo and Pepper to be ready
        self.get_logger().info('Waiting for Gazebo and Pepper to spawn...')
        time.sleep(6)
        
        # self.spawn_screen()
        
    def spawn_screen(self):
        """Spawn the screen as a separate model in Gazebo"""
        
        # Create spawn service client
        spawn_client = self.create_client(SpawnEntity, '/world/classroom/create')
        while not spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /world/classroom/create service...')
        
        # Create a simple screen SDF inline
        screen_sdf = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="tablet_screen">
    <static>true</static>
    <link name="screen_link">
      <visual name="screen_visual">
        <geometry>
          <box>
            <size>0.15 0.001 0.22</size>
          </box>
        </geometry>
        <material>
          <ambient>0.2 0.3 0.5 1</ambient>
          <diffuse>0.3 0.4 0.6 1</diffuse>
          <emissive>0.5 0.6 0.8 1</emissive>
          <specular>0.8 0.8 0.8 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        
        # Create spawn request
        request = SpawnEntity.Request()
        request.entity_factory.name = 'tablet_screen'
        request.entity_factory.sdf = screen_sdf
        
        # Position the screen
        # These coordinates need to be adjusted based on Pepper's tablet position
        # For now, using estimated values
        pose = Pose()
        pose.position.x = 8.5      # Same as Pepper's x
        pose.position.y = 1.5      # Slightly forward from Pepper
        pose.position.z = 0.8     # Approximate tablet height
        
        # Orientation - facing forward like Pepper
        pose.orientation.x = -1.579
        pose.orientation.y = 0.0
        pose.orientation.z = 1.579   # Facing same direction as Pepper
        # pose.orientation.w = 0.0
        
        request.entity_factory.pose = pose
        
        # Call spawn service
        self.get_logger().info('Spawning tablet screen...')
        future = spawn_client.call_async(request)
        
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('✓ Tablet screen spawned successfully!')
        else:
            self.get_logger().info(future.result())
            error_msg = future.result().status_message if future.result() else "Timeout"
            self.get_logger().error(f'✗ Failed to spawn screen: {error_msg}')


# def main(args=None):
#     rclpy.init(args=args)
#     node = TabletScreenSpawner()
    
#     # Keep node alive briefly then shutdown
#     time.sleep(1)
    
#     node.destroy_node()
#     rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = TabletScreenSpawner()
    
    # Do NOT call spawn_screen in __init__
    # Call it here, AFTER the node is initialized
    node.spawn_screen()
    
    # Then clean up
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()