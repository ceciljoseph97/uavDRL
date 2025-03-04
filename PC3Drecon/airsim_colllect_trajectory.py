import time
import airsim
import numpy as np
import pandas as pd
import csv
import os
import argparse

def read_trajectory(file_path):
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() in ['.csv', '.txt']:
            df = pd.read_csv(file_path)
            possible_x_columns = ['X', 'X(m)']
            possible_y_columns = ['Y', 'Y(m)']
            x_col = next((col for col in possible_x_columns if col in df.columns), None)
            y_col = next((col for col in possible_y_columns if col in df.columns), None)
            if x_col is None or y_col is None:
                raise ValueError("File must contain 'X' or 'X(m)' and 'Y' or 'Y(m)' columns.")
            return df[[x_col, y_col]].rename(columns={x_col: 'X', y_col: 'Y'})
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or TXT file.")
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame(columns=['X', 'Y'])

def connect_airsim():
    client = airsim.VehicleClient()
    client.reset()
    client.confirmConnection()
    return client

def move_drone(client, x, y, z, pitch, roll, yaw):
    pose = airsim.Pose(
        airsim.Vector3r(x, -y, -z),
        airsim.to_quaternion(pitch, roll, yaw)
    )
    client.simSetVehiclePose(pose, True)

def main(file_path, z_min, z_max, delay=0.2):
    df = read_trajectory(file_path)
    if df.empty:
        print("No valid trajectory data found. Exiting.")
        return

    yaw = 0 * np.pi / 180
    pitch = -90 * np.pi / 180
    roll = 0 * np.pi / 180

    client = connect_airsim()
    print("Connected to AirSim.")

    client.startRecording()
    print("Started recording in AirSim.")

    for index, row in df.iterrows():
        x = row['X']
        y = row['Y']
        z = np.random.uniform(z_min, z_max)
        move_drone(client, x, y, z, pitch, roll, yaw)
        print(f"Moved to Index: {index}, X: {x}, Y: {y}, Z: {z:.2f}")
        time.sleep(delay)

    client.stopRecording()
    print("Stopped recording in AirSim.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move drone along trajectory with random altitude and record in AirSim.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the trajectory file.")
    parser.add_argument('--z_min', type=float, required=True, help="Minimum Z-axis (altitude) value.")
    parser.add_argument('--z_max', type=float, required=True, help="Maximum Z-axis (altitude) value.")
    parser.add_argument('--delay', type=float, default=0.1, help="Delay in seconds between movements.")
    args = parser.parse_args()

    if args.z_min > args.z_max:
        raise ValueError("z_min should be less than or equal to z_max.")

    main(
        file_path=args.file_path,
        z_min=args.z_min,
        z_max=args.z_max,
        delay=args.delay
    )
