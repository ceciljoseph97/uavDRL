import subprocess
import time
import yaml
import gym

def launch_environment(exe_path):
    """Launch the environment executable."""
    process = subprocess.Popen(exe_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Started executable: {exe_path}")
    return process

def wait_for_environment_ready(ip_address, image_shape, env_config, input_mode, env_id, timeout=60):
    """Check if the environment is ready by attempting to connect."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Attempt to create the environment with the required arguments
            env = gym.make(
                env_id,
                ip_address=ip_address,
                image_shape=image_shape,
                env_config=env_config["TrainEnv"],
                input_mode=input_mode
            )
            env.reset()
            print("Environment is ready.")
            return True
        except Exception as e:
            print(f"Waiting for environment to be ready: {e}")
            time.sleep(5)
    print("Environment failed to initialize within the timeout period.")
    return False

def terminate_environment(process):
    """Terminate the environment process."""
    print("Terminating environment process.")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("Process did not terminate in time, forcing kill.")
        process.kill()
    print("Environment process terminated.")
