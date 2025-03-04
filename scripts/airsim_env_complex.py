# Work in Progress
from . import airsim
import os
import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class AirSimDroneEnvCheck(gym.Env):
    def __init__(self, ip_address, image_shape, env_config, input_mode):
        self.image_shape = image_shape
        self.sections = env_config["sections"]
        self.input_mode = input_mode

        self.drone = airsim.MultirotorClient(ip=ip_address)

        if self.input_mode == "depth":
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(image_shape[0], image_shape[1], 1),
                dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(image_shape[0], image_shape[1], 3),
                dtype=np.uint8)

        # Expanded action space: [forward/back, left/right, up/down, yaw]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.info = {"collision": False}
        self.collision_time = 0
        self.random_start = True
        self.last_position = None
        self.last_orientation = None
        self.setup_flight()
        
        # For tracking progress
        self.initial_distance = None
        self.previous_distance = None
        self.step_count = 0
        self.max_steps = 500  # Prevent episodes from running too long

    def save_observation(self, obs):
        if self.input_mode == "multi_rgb":
            stacked_image = np.hstack((
                self.obs_stack[:, :, 0],
                self.obs_stack[:, :, 1],
                self.obs_stack[:, :, 2]
            ))
            filename = 'multi_rgb_stacked.png'
            cv2.imwrite(filename, stacked_image)
            print(f'Saved multi_rgb stacked image as {filename}')

        elif self.input_mode == "single_rgb":
            rgb_image = self.get_rgb_image()
            filename = 'single_rgb.png'
            cv2.imwrite(filename, rgb_image)
            print(f'Saved single_rgb image as {filename}')

        elif self.input_mode == "depth":
            depth_image = self.get_depth_image(thresh=10.0)  # Increased threshold for better distance sensing
            depth_image = ((depth_image / 10.0) * 255).astype(np.uint8)
            filename = 'depth_image.png'
            cv2.imwrite(filename, depth_image)
            print(f'Saved depth image as {filename}')

    def step(self, action):
        # Execute the action
        self.do_action(action)
        
        # Get observation
        obs, self.info = self.get_obs()
        
        # Compute reward
        reward = self.compute_reward(action)
        
        # Check if episode is done
        done = False
        
        # Done if collision detected
        if self.info["collision"]:
            done = True
            self.info["success"] = False
            self.info["flight_duration"] = self.step_count * 0.1  # Approx time in seconds
            
            # Calculate flight distance
            current_pos = self.drone.simGetVehiclePose().position
            pos = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            self.info["flight_distance"] = self.initial_distance - self.previous_distance
        
        # Done if target reached (within 5 meter radius)
        current_pos = self.drone.simGetVehiclePose().position
        pos = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])
        distance_to_target = np.linalg.norm(pos - self.target_pos)
        
        if distance_to_target < 5.0:  # 5 meter radius for success
            done = True
            reward += 200.0  # Big bonus for reaching target
            self.info["success"] = True
            self.info["flight_duration"] = self.step_count * 0.1  # Approx time in seconds
            self.info["flight_distance"] = self.initial_distance  # Full distance traveled
        
        # Done if max steps reached
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            self.info["success"] = False
            self.info["flight_duration"] = self.step_count * 0.1  # Approx time in seconds
            self.info["flight_distance"] = self.initial_distance - self.previous_distance
        
        return obs, reward, done, self.info

    def reset(self):
        self.drone.reset()
        self.setup_flight()
        
        # Reset step counter
        self.step_count = 0
        
        # Reset collision info
        self.info["collision"] = False
        self.collision_time = 0
        
        # Get first target from sections as spawn position
        spawn_pos = self.sections[0]["target"]
        
        # Set target to the second section's target
        if len(self.sections) > 1:
            target_section = self.sections[1]
        else:
            target_section = self.sections[0]  # Fallback if only one section
        
        self.target_pos = np.array([
            target_section["target"]["x"],
            target_section["target"]["y"],
            target_section["target"]["z"]
        ])
        
        # Teleport drone to spawn position
        pose = self.drone.simGetVehiclePose()
        pose.position.x_val = spawn_pos["x"]
        pose.position.y_val = spawn_pos["y"]
        pose.position.z_val = spawn_pos["z"]
        
        # Point drone toward target
        direction = self.target_pos - np.array([spawn_pos["x"], spawn_pos["y"], spawn_pos["z"]])
        yaw = np.arctan2(direction[1], direction[0])
        
        # Set orientation using quaternion
        pose.orientation.w_val = np.cos(yaw/2)
        pose.orientation.z_val = np.sin(yaw/2)
        pose.orientation.x_val = 0
        pose.orientation.y_val = 0
        
        self.drone.simSetVehiclePose(pose, True)
        
        # Reset distance tracking
        self.initial_distance = np.linalg.norm(
            np.array([spawn_pos["x"], spawn_pos["y"], spawn_pos["z"]]) - self.target_pos
        )
        self.previous_distance = self.initial_distance
        
        # Get initial observation
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Hover at starting position
        self.drone.moveToZAsync(-2, 1).join()

        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

        if self.random_start:
            self.target_pos_idx = np.random.randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        section = self.sections[self.target_pos_idx]
        
        # Extract target position from section
        if isinstance(section["target"], dict):
            # Handle the case where target is a dictionary with x, y, z keys
            self.target_pos = np.array([
                section["target"]["x"],
                section["target"]["y"],
                section["target"]["z"]
            ])
        else:
            # Handle the case where target is already a list/array
            self.target_pos = np.array(section["target"])
        
        # Random starting position with some distance from target
        start_x = self.target_pos[0] - np.random.uniform(10, 20)
        start_y = self.target_pos[1] - np.random.uniform(-10, 10)
        start_z = self.target_pos[2] - np.random.uniform(-5, 5)
        
        # Ensure we're not starting too close to the ground
        if start_z > -1.5:
            start_z = -2.0
            
        pose = airsim.Pose(airsim.Vector3r(start_x, start_y, start_z))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
        # Wait for the drone to stabilize
        time.sleep(0.5)
        
        # Get initial position and distance to target
        current_pos = self.drone.simGetVehiclePose().position
        self.last_position = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])
        self.initial_distance = np.linalg.norm(self.last_position - self.target_pos)
        self.previous_distance = self.initial_distance
        
        # Initialize orientation tracking
        orientation = self.drone.simGetVehiclePose().orientation
        self.last_orientation = np.array([orientation.w_val, orientation.x_val, 
                                          orientation.y_val, orientation.z_val])

        if self.input_mode == "multi_rgb":
            self.obs_stack = np.zeros(self.image_shape)

    def do_action(self, action):
        # Unpack action: [forward/back, left/right, up/down, yaw]
        # Ensure forward movement only (no backward)
        forward_action = max(0, action[0])  # Force non-negative (forward only)
        lateral_action = action[1]
        vertical_action = action[2]
        yaw_action = action[3]
        
        # Scale actions
        forward_speed = forward_action * 5.0  # Scale forward speed (0 to 5 m/s)
        lateral_speed = lateral_action * 3.0  # Scale lateral speed (-3 to 3 m/s)
        vertical_speed = vertical_action * 2.0  # Scale vertical speed (-2 to 2 m/s)
        yaw_rate = yaw_action * 45.0  # Scale yaw rate (-45 to 45 degrees/s)
        
        # Send movement commands to drone
        self.drone.moveByVelocityBodyFrameAsync(
            forward_speed,
            lateral_speed,
            vertical_speed,
            0.1,  # Duration
            airsim.DrivetrainType.ForwardOnly
        ).join()
        
        # Apply yaw
        self.drone.rotateByYawRateAsync(yaw_rate, 0.1).join()
        
        # Small delay to allow physics to stabilize
        time.sleep(0.05)

    def get_obs(self):
        self.info["collision"] = self.is_collision()

        if self.input_mode == "multi_rgb":
            obs_t = self.get_rgb_image()
            try:
                obs_t_gray = cv2.cvtColor(obs_t, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                print(f"Error converting image to grayscale: {e}")
                print("Attempting to convert image to uint8 format and retrying...")
                if obs_t.dtype != np.uint8:
                    obs_t = (obs_t * 255).astype(np.uint8)
                obs_t_gray = cv2.cvtColor(obs_t, cv2.COLOR_BGR2GRAY)
            self.obs_stack[:, :, 0] = self.obs_stack[:, :, 1]
            self.obs_stack[:, :, 1] = self.obs_stack[:, :, 2]
            self.obs_stack[:, :, 2] = obs_t_gray
            obs = np.hstack((
                self.obs_stack[:, :, 0],
                self.obs_stack[:, :, 1],
                self.obs_stack[:, :, 2]))
            obs = np.expand_dims(obs, axis=2)

            # Convert to channel-first format (C, H, W)
            obs = np.transpose(obs, (2, 0, 1))

        elif self.input_mode == "single_rgb":
            obs = self.get_rgb_image()
            # Convert to channel-first format (C, H, W)
            obs = np.transpose(obs, (2, 0, 1))

        elif self.input_mode == "depth":
            obs = self.get_depth_image(thresh=10.0).reshape(self.image_shape)
            obs = ((obs / 10.0) * 255).astype(np.uint8)
            # Add channel dimension for depth (1, H, W)
            obs = np.expand_dims(obs, axis=0)

        # Add distance to target as part of info
        current_pos = self.drone.simGetVehiclePose().position
        pos = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])
        distance_to_target = np.linalg.norm(pos - self.target_pos)
        self.info["distance_to_target"] = distance_to_target

        return obs, self.info

    def compute_reward(self, action):
        # Get current position
        current_pos = self.drone.simGetVehiclePose().position
        pos = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(pos - self.target_pos)
        
        # Calculate progress toward target
        if self.previous_distance is None:
            progress = 0.0
        else:
            progress = self.previous_distance - distance_to_target
        
        # Update previous distance
        self.previous_distance = distance_to_target
        
        # Get drone's orientation
        orientation = self.drone.simGetVehiclePose().orientation
        # Convert quaternion to direction vector
        q = [orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val]
        # Forward vector calculation from quaternion
        forward_x = 2 * (q[1] * q[3] + q[0] * q[2])
        forward_y = 2 * (q[2] * q[3] - q[0] * q[1])
        forward_z = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        forward_vector = np.array([forward_x, forward_y, forward_z])
        forward_vector = forward_vector / np.linalg.norm(forward_vector)
        
        # Calculate direction to target
        direction_to_target = self.target_pos - pos
        direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
        
        # Base reward
        reward = 0.0
        
        # MAJOR reward for progress toward target
        reward += progress * 30.0
        
        # MAJOR reward for alignment with target
        alignment = np.dot(forward_vector, direction_to_target)
        reward += alignment * 20.0
        
        # Get velocity
        velocity = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        velocity_vector = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        speed = np.linalg.norm(velocity_vector)
        
        # Reward for speed (encourage movement)
        reward += speed * 2.0
        
        # Penalty for height deviation
        ideal_height = self.target_pos[2]  # Target's height
        height_error = abs(pos[2] - ideal_height)
        reward -= height_error * 3.0
        
        # Penalty for collision (high penalty)
        if self.info["collision"]:
            reward -= 200.0
            return reward
        
        # Bonus for being close to target
        if distance_to_target < 10.0:
            reward += (10.0 - distance_to_target) * 5.0  # Graduated bonus as we get closer
        
        return reward

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False

    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(
            0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(
            img1d, (responses[0].height, responses[0].width, 3))

        # Sometimes no image returns from API
        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros(self.image_shape)

    def get_depth_image(self, thresh=10.0):
        depth_image_request = airsim.ImageRequest(
            1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(
            responses[0].image_data_float, dtype=np.float32)
        depth_image = depth_image.reshape(
            responses[0].height, responses[0].width)
        depth_image[depth_image > thresh] = thresh
        if len(depth_image) == 0:
            depth_image = np.zeros(self.image_shape)
        return depth_image
    
    def get_lidar_data(self):
        """Get distance measurements in multiple directions to detect obstacles"""
        lidar_data = []
        
        # Use depth image as a simple lidar
        depth_image = self.get_depth_image(thresh=20.0)
        
        # Check if depth_image has the expected shape
        if len(depth_image.shape) != 2:
            # If depth_image is not 2D, reshape it or create an empty array
            try:
                depth_image = depth_image.reshape(self.image_shape[0], self.image_shape[1])
            except:
                print("Warning: Could not reshape depth image. Using empty array.")
                depth_image = np.zeros((self.image_shape[0], self.image_shape[1]))
        
        # Sample points from the depth image
        h, w = depth_image.shape
        center_h, center_w = h // 2, w // 2
        
        # Sample in a grid pattern
        sample_points = [
            (center_h, center_w),  # center
            (center_h - h//4, center_w),  # top
            (center_h + h//4, center_w),  # bottom
            (center_h, center_w - w//4),  # left
            (center_h, center_w + w//4),  # right
            (center_h - h//4, center_w - w//4),  # top-left
            (center_h - h//4, center_w + w//4),  # top-right
            (center_h + h//4, center_w - w//4),  # bottom-left
            (center_h + h//4, center_w + w//4),  # bottom-right
        ]
        
        for y, x in sample_points:
            if 0 <= y < h and 0 <= x < w:
                lidar_data.append(depth_image[y, x])
        
        return np.array(lidar_data)


class TestEnvCheck(AirSimDroneEnvCheck):
    def __init__(
        self,
        ip_address,
        image_shape,
        env_config,
        input_mode,
        test_mode="random"
    ):
        self.test_mode = test_mode
        self.total_traveled = 0
        self.eps_n = 0
        self.eps_success = 0
        self.positions = []
        self.start_pos = -1
        self.trajectory = []

        super(TestEnvCheck, self).__init__(
            ip_address,
            image_shape,
            env_config,
            input_mode
        )

    def setup_flight(self):
        super(TestEnvCheck, self).setup_flight()
        self.trajectory = []  # Reset trajectory for new episode

    def step(self, action):
        obs, reward, done, info = super(TestEnvCheck, self).step(action)
        
        pose = self.drone.simGetVehiclePose().position
        self.trajectory.append((pose.x_val, pose.y_val, pose.z_val))
        
        if done:
            info['trajectory'] = self.trajectory.copy()
            info['target_position'] = self.target_pos.tolist()
            
            final_distance = np.linalg.norm(
                np.array([pose.x_val, pose.y_val, pose.z_val]) - self.target_pos)
            success = final_distance < 3.0 and not self.is_collision()
            
            if success:
                self.eps_success += 1
                
            self.eps_n += 1
            print("-----------------------------------")
            print(f"> Episode {self.eps_n} completed")
            print(f"> Success: {success}")
            print(f"> Final distance to target: {final_distance:.2f} meters")
            print(f"> Success rate: {self.eps_success}/{self.eps_n} ({self.eps_success*100/self.eps_n:.2f}%)")
            print("-----------------------------------\n")
            
            info['success'] = success
            
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs