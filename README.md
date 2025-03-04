# Evaluating Reinforcement Learning Algorithms for UAVNavigation in Diverse Simulated Environments

This Study presents a comparative evaluation of different deep reinforcement learning (DRL) algorithms for UAV autonomous navigation under diverse weather conditions, with a focus on assessing
their generalization capabilities. Specifically, the study integrate a deep convolutional neural network (NatureCNN) with two policy-gradient methods Proximal Policy Optimization (PPO) and
Advantage Actor-Critic (A2C) to process multi-modal visual inputs (depth, RGB, and Multi-Grey)
within simulation platforms such as Unreal Engine and Microsoft AirSim. Experiments were conducted across various dynamic weather scenarios (normal, leaves falling, rain and fog, and snow
only) to rigorously analyze performance metrics, including flight distance, target achievement, and
collision rates. The results indicate that weather-conditioned PPO variants consistently outperform
their A2C counterparts, achieving better generalization, higher accuracies, longer flight distances,
and lower collision rates. These findings highlight the potential of lightweight, CNN-based DRL
architectures for robust, real-time UAV navigation in challenging and ever changing environments.


### Inputs
There are three models trained using depth, single RGB, and stacked gray images, respectively. Their sizes are as follows


- **Single RGB image:** 50 x 50 x 3
- **Depth map:** 50 x 50 x 1
- **Depth image:** 50 x 150 x 1


<summary>Details for installation</summary>

<!-- - [Install CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) -->
- [Install Pytorch with the compatible CUDA version](https://pytorch.org/get-started/locally/)

For this project, I used CUDA 11.0 and the following conda installation command to install Pytorch: or simply use the drlnavCheck.yaml to see the requirements and install.

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

</details>

#️⃣ **1. Edit `settings.json`**

Content of the settings.json should be as below:

> The `setting.json` file is located at `Documents\AirSim` folder.
### SimpleEnv Controlled settings
```json
{
    "SettingsVersion": 1.2,
    "LocalHostIp": "127.0.0.1",
    "SimMode": "Multirotor",
    "ClockSpeed": 40,
    "ViewMode": "SpringArmChase",
    "Vehicles": {
        "drone0": {
            "VehicleType": "SimpleFlight",
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
            "Yaw": 0.0
        }
    },
    "CameraDefaults": {
        "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 50,
                "Height": 50,
                "FOV_Degrees": 120
            },
            {
                "ImageType": 2,
                "Width": 50,
                "Height": 50,
                "FOV_Degrees": 120
            }
        ]
    }
}
```
### PCD analysis settings
```json
{
  "SettingsVersion": 1.2,
  "SimMode": "ComputerVision",
  "Recording": {
    "RecordOnMove": true,
    "RecordInterval": 0.05,
    "Cameras": [
        { "CameraName": "0", "ImageType": 0, "PixelsAsFloat": false, "Compress": true },
        { "CameraName": "0", "ImageType": 5, "PixelsAsFloat": false, "Compress": true },
        { "CameraName": "0", "ImageType": 1, "PixelsAsFloat": true, "Compress": false }
    ]
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 256,
        "Height": 192,
        "FOV_Degrees": 90,
        "AutoExposureSpeed": 100,
        "MotionBlurAmount": 0
      },
	  {
        "ImageType": 1,
        "Width": 256,
        "Height": 192,
        "FOV_Degrees": 90,
        "AutoExposureSpeed": 100,
        "MotionBlurAmount": 0
      },
      {
        "ImageType": 5,
        "Width": 256,
        "Height": 192,
        "FOV_Degrees": 90,
        "AutoExposureSpeed": 100,
        "MotionBlurAmount": 0
      }
    ]
  },
  "SubWindows": [
    {"WindowID": 0, "CameraName": "0", "ImageType": 3, "VehicleName": "", "Visible": true},
    {"WindowID": 1, "CameraName": "0", "ImageType": 5, "VehicleName": "", "Visible": true},
    {"WindowID": 2, "CameraName": "0", "ImageType": 0, "VehicleName": "", "Visible": true}
  ],
  "SegmentationSettings": {
    "InitMethod": "",
    "MeshNamingMethod": "",
    "OverrideExisting": false
  }
}

```

## How to run the training?

#️⃣ **1. Open up any desired SimpleTarget environment's executable file and start the training**

So, inside the repository
```
# for weather randomized training
python trainGen.py \
  --algorithm PPO \
  --cnn NatureCNN \
  --total_timesteps 50000 \
  --train_mode multi_rgb \
  --env_select city \
  --weather \
  --weather_freq 10000 \
  --weather_scenarios Clear HeavyRain FogOnly


# for normal training
python traiGen.py \
  --algorithm PPO \
  --cnn NatureCNN \
  --total_timesteps 35000 \
  --train_mode single_rgb \
  --env_select obstacle


```

## How to run the pretrained model?

#️⃣ **1. Open up environment's executable file and run the trained model**

So, inside the repository
```
python inferenceChecks.py \
  --algorithm A2C \
  --cnn NatureCNN \
  --test_mode depth \
  --test_type custom_scenario \
  --model_path saved_models/a2c_depth_model.zip \
  --suffix "depth_inference_run"

```
