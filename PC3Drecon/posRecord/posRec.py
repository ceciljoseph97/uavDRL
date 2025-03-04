import airsim
import time
import math
import yaml

def positions_are_close(pos1, pos2, threshold=0.01):
    dx = pos1.x_val - pos2.x_val
    dy = pos1.y_val - pos2.y_val
    dz = pos1.z_val - pos2.z_val
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    return distance < threshold

def orientations_are_close(ori1, ori2, threshold=1.0):
    pitch1, roll1, yaw1 = airsim.to_eularian_angles(ori1)
    pitch2, roll2, yaw2 = airsim.to_eularian_angles(ori2)
    pitch1_deg = pitch1 * (180.0 / math.pi)
    roll1_deg = roll1 * (180.0 / math.pi)
    yaw1_deg = yaw1 * (180.0 / math.pi)
    pitch2_deg = pitch2 * (180.0 / math.pi)
    roll2_deg = roll2 * (180.0 / math.pi)
    yaw2_deg = yaw2 * (180.0 / math.pi)
    dpitch = abs(pitch1_deg - pitch2_deg)
    droll = abs(roll1_deg - roll2_deg)
    dyaw = abs(yaw1_deg - yaw2_deg)
    return dpitch < threshold and droll < threshold and dyaw < threshold
client = airsim.VehicleClient()
client.confirmConnection()

camera_name = '0'
vehicle_name = ''

session_name = input("Enter a name for this session (will be used in output filenames): ").strip()
if not session_name:
    session_name = "session"

positions_filename = f'camera_positions_{session_name}.txt'
with open(positions_filename, 'w') as f:
    f.write('Time(s),X(m),Y(m),Z(m),Pitch(deg),Roll(deg),Yaw(deg)\n')

    try:
        start_time = time.time()
        prev_position = None
        prev_orientation = None
        stable_count = 0
        checkpoints = []
        while True:
            camera_info = client.simGetCameraInfo(camera_name)
            if camera_info is None:
                print(f"Camera '{camera_name}' not found.")
                break

            pose = camera_info.pose
            position = pose.position
            orientation = pose.orientation

            pitch, roll, yaw = airsim.to_eularian_angles(orientation)

            pitch_deg = pitch * (180.0 / math.pi)
            roll_deg = roll * (180.0 / math.pi)
            yaw_deg = yaw * (180.0 / math.pi)

            current_time = time.time() - start_time

            f.write(f"{current_time:.2f},{position.x_val:.2f},{position.y_val:.2f},{position.z_val:.2f},"
                    f"{pitch_deg:.2f},{roll_deg:.2f},{yaw_deg:.2f}\n")

            print(f"Time: {current_time:.2f}s, Position: x={position.x_val:.2f}, y={position.y_val:.2f}, z={position.z_val:.2f}, "
                  f"Pitch: {pitch_deg:.2f}°, Roll: {roll_deg:.2f}°, Yaw: {yaw_deg:.2f}°")

            if prev_position is not None and prev_orientation is not None:
                if positions_are_close(position, prev_position) and orientations_are_close(orientation, prev_orientation):
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count == 2:
                    checkpoint_exists = False
                    for cp in checkpoints:
                        if positions_are_close(position, cp['position']) and orientations_are_close(orientation, cp['orientation']):
                            checkpoint_exists = True
                            break
                    if not checkpoint_exists:
                        checkpoints.append({'time': current_time, 'position': position, 'orientation': orientation})
                        print(f"Checkpoint detected at time {current_time:.2f}s")
            else:
                stable_count = 0

            prev_position = position
            prev_orientation = orientation

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping position recording.")
        if checkpoints:
            print("Checkpoints detected:")
            for idx, cp in enumerate(checkpoints):
                position = cp['position']
                orientation = cp['orientation']
                pitch_rad, roll_rad, yaw_rad = airsim.to_eularian_angles(orientation)
                pitch_deg = pitch_rad * (180.0 / math.pi)
                roll_deg = roll_rad * (180.0 / math.pi)
                yaw_deg = yaw_rad * (180.0 / math.pi)
                print(f"Checkpoint {idx+1}: Time {cp['time']:.2f}s, Position x={position.x_val:.2f}, y={position.y_val:.2f}, z={position.z_val:.2f}, "
                      f"Pitch {pitch_deg:.2f}°, Roll {roll_deg:.2f}°, Yaw {yaw_deg:.2f}°")

            yaml_filename = f'checkpoints_{session_name}.yaml'
            with open(yaml_filename, 'w') as yaml_file:
                yaml_data = {'TrainEnv': {'sections': []}}
                offset = 0
                for idx, cp in enumerate(checkpoints):
                    position = cp['position']
                    section = {
                        'target': {
                            'x': position.x_val,
                            'y': position.y_val,
                            'z': position.z_val
                        },
                        'offset': [offset]
                    }
                    yaml_data['TrainEnv']['sections'].append(section)
                    offset += 4
                yaml.dump(yaml_data, yaml_file)
                print(f"Checkpoints written to {yaml_filename}")
        else:
            print("No checkpoints detected.")
