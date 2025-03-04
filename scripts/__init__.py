from .airsim_env import AirSimDroneEnv, TestEnv
from .airsim_env_complex import AirSimDroneEnvCheck, TestEnvCheck
from gym.envs.registration import register

register(
    id="airsim-env-v0", entry_point="scripts:AirSimDroneEnv",
)

register(
    id="test-env-v0", entry_point="scripts:TestEnv",
)

register(
    id="airsim-env-v2", entry_point="scripts:AirSimDroneEnvCheck",
)

register(
    id="test-env-v2", entry_point="scripts:TestEnvCheck",
)
