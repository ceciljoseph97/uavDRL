import time

import airsim
import numpy as np


wp_bound_x = [-105.0089340209961, 120.97982788085938]
wp_bound_y = [-118.89655303955078, 1.1254631280899048]
wp_step = 50
wp_z = 50

x = wp_bound_x[0]
y = wp_bound_y[0]
z = wp_z
yaw = 0 * np.pi/180
pitch = -90 * np.pi/180
roll = 0 * np.pi/180


client = airsim.VehicleClient()
client.reset()
client.confirmConnection()

client.startRecording()

d = 0
while y <= wp_bound_y[1]:
    while (d == 0 and x <= wp_bound_x[1]) or (d == 1 and x >= wp_bound_x[0]):

        yaw = d * 180 * np.pi/180
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, -y, -z), airsim.to_quaternion(pitch, -roll, yaw)), True)

        time.sleep(0.2)

        if d == 0:
            x += wp_step
        else:
            x -= wp_step

    if d == 0:
        x -= wp_step
    else:
        x += wp_step

    d = 1 - d
    y += wp_step

client.stopRecording()
