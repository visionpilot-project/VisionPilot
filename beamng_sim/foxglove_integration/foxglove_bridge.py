import foxglove
from foxglove import Channel
from foxglove.channels import SceneUpdateChannel
from foxglove.schemas import (
    Color,
    CubePrimitive,
    SceneEntity,
    SceneUpdate,
    Vector3,
    ModelPrimitive,
    Pose,
)
import numpy as np
import os


class FoxgloveBridge:
    def __init__(self):
        self._initialized = False
        self._server_started = False
        # JSON channels
        self.sign_channel = None
        self.traffic_light_channel = None
        self.lane_channel = None
        self.vehicle_channel = None
        self.vehicle_state_channel = None
        # PointCloud channel
        self.lidar_channel = None
    
    def start_server(self):
        """Start the Foxglove WebSocket server"""
        if self._server_started:
            return
        
        print("[FoxgloveBridge] Starting Foxglove WebSocket server...")
        foxglove.set_log_level("INFO")
        foxglove.start_server()
        self._server_started = True
        print("[FoxgloveBridge] Foxglove server started on ws://localhost:8765")
    
    def initialize_channels(self):
        """Initialize channels after server has started"""
        if self._initialized:
            return
        
        print("[FoxgloveBridge] Initializing channels...")
        
        # JSON channels
        self.sign_channel = Channel("/detections/sign", message_encoding="json")
        self.traffic_light_channel = Channel("/detections/traffic_light", message_encoding="json")
        self.lane_channel = Channel("/detections/lane", message_encoding="json")
        self.vehicle_channel = Channel("/detections/vehicle", message_encoding="json")
        self.vehicle_state_channel = Channel("/vehicle/state", message_encoding="json")
        self.lidar_channel = Channel("/lidar/points", message_encoding="json")
        self.scene_channel = SceneUpdateChannel("/scene")
        
        self._initialized = True
        print("[FoxgloveBridge] Channels initialized successfully")
    
    def send_sign_detection(self, sign_type, x, y, confidence):
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping sign detection")
            return
        
        message = {
            "type": sign_type,
            "x": x,
            "y": y,
            "confidence": confidence
        }
        try:
            print(f"[FoxgloveBridge] Sending sign detection: {message}")
            self.sign_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending sign detection: {e}")
    
    # def send_traffic_light_detection(self, state, x, y, confidence):
    #     self.traffic_light_channel.log({
    #         "state": state,
    #         "x": x,
    #         "y": y,
    #         "confidence": confidence
    #     })
    
    def send_lane_detection(self, lane_center, vehicle_center, deviation, confidence, left_lane_points=None, right_lane_points=None):
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping lane detection")
            return
        
        message = {
            "lane_center": lane_center,
            "vehicle_center": vehicle_center,
            "deviation": deviation,
            "confidence": confidence
        }
        if left_lane_points is not None:
            message["left_lane_points"] = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}
                for p in left_lane_points
            ]
        if right_lane_points is not None:
            message["right_lane_points"] = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}
                for p in right_lane_points
            ]
        try:
            print(f"[FoxgloveBridge] Sending lane detection: {message}")
            self.lane_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending lane detection: {e}")

    def send_vehicle_detection(self, detection_type, x, y, width, height, confidence):
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping vehicle detection")
            return
        
        message = {
            "type": detection_type,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "confidence": confidence
        }
        try:
            print(f"[FoxgloveBridge] Sending vehicle detection: {message}")
            self.vehicle_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending vehicle detection: {e}")
    
    def send_vehicle_state(self, speed_kph, steering, throttle, x, y, z):
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping vehicle state")
            return
        
        # Ensure all values are standard Python floats for JSON compatibility
        message = {
            "speed_kph": float(speed_kph),
            "steering": float(steering),
            "throttle": float(throttle),
            "x": float(x),
            "y": float(y),
            "z": float(z)
        }
        try:
            print(f"[FoxgloveBridge] Sending vehicle state: {message}")
            self.vehicle_state_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending vehicle state: {e}")
    
    def send_lidar(self, points):
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping LiDAR")
            return
        
        if points is None or len(points) == 0:
            print("[FoxgloveBridge] No LiDAR points to send.")
            return
        try:
            points_array = np.asarray(points, dtype=np.float32)
            
            # Convert to list of dicts for JSON serialization
            points_data = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for p in points_array
            ]
            
            message = {
                "points": points_data,
                "count": len(points_data)
            }
            
            print(f"[FoxgloveBridge] Sending LiDAR: {len(points_data)} points to /lidar/points")
            print(f"[FoxgloveBridge] Sample point: {points_data[0] if points_data else 'N/A'}")
            self.lidar_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending LiDAR: {e}")
    
    def send_vehicle_3d(self, x, y, z, yaw=0.0):
        """
        Send vehicle position and orientation with BMW X5 GLB models.
        
        Args:
            x, y, z: Vehicle position in world coordinates
            yaw: Vehicle yaw angle in radians (rotation around Z axis)
        """
        try:
            # Convert yaw to quaternion (rotation around Z axis)
            half_yaw = yaw / 2.0
            qx = 0.0
            qy = 0.0
            qz = np.sin(half_yaw)
            qw = np.cos(half_yaw)
            
            entities = []
            
            # Base directory for GLB files
            model_dir = os.path.join(
                os.path.dirname(__file__),
                "model", "bmw_x5", "meshes"
            )
            
            # Car body
            body_glb_path = os.path.join(model_dir, "car_body.glb")
            if os.path.exists(body_glb_path):
                body_entity = SceneEntity(
                    id="bmw_body",
                    frame_id="world",
                    pose=Pose(
                        position=Vector3(x=float(x), y=float(y), z=float(z)),
                        orientation={"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
                    ),
                    model=ModelPrimitive(url=f"file://{body_glb_path}"),
                )
                entities.append(body_entity)
            
            # Wheel positions from URDF (relative to body)
            wheels = [
                ("front_left_wheel", 1.3, 0.8, -0.3, "wheel_front_left.glb"),
                ("front_right_wheel", 1.3, -0.8, -0.3, "wheel_front_right.glb"),
                ("rear_left_wheel", -1.3, 0.8, -0.3, "wheel_rear_left.glb"),
                ("rear_right_wheel", -1.3, -0.8, -0.3, "wheel_rear_right.glb"),
            ]
            
            for wheel_id, rel_x, rel_y, rel_z, mesh_file in wheels:
                glb_path = os.path.join(model_dir, mesh_file)
                if os.path.exists(glb_path):
                    # Transform wheel position to world coordinates
                    cos_yaw = np.cos(yaw)
                    sin_yaw = np.sin(yaw)
                    world_x = x + (rel_x * cos_yaw - rel_y * sin_yaw)
                    world_y = y + (rel_x * sin_yaw + rel_y * cos_yaw)
                    world_z = z + rel_z
                    
                    wheel_entity = SceneEntity(
                        id=wheel_id,
                        frame_id="world",
                        pose=Pose(
                            position=Vector3(x=float(world_x), y=float(world_y), z=float(world_z)),
                            orientation={"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
                        ),
                        model=ModelPrimitive(url=f"file://{glb_path}"),
                    )
                    entities.append(wheel_entity)
            
            if entities:
                scene_update = SceneUpdate(entities=entities)
                print(f"[FoxgloveBridge] Sending BMW X5 3D model at ({x:.1f}, {y:.1f}, {z:.1f}), yaw={np.degrees(yaw):.1f}°")
                self.scene_channel.log(scene_update)
            else:
                print(f"[FoxgloveBridge] Warning: No GLB files found in {model_dir}")
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending vehicle 3D: {e}")
    