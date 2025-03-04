from stable_baselines3.common.callbacks import BaseCallback
import airsim
import random
from airsim import WeatherParameter

class WeatherChangeCallback(BaseCallback):

    def __init__(self, freq: int, client: airsim.MultirotorClient, selected_scenarios: list,
                 verbose: int = 0, single_weather_condition: bool = False):
        super(WeatherChangeCallback, self).__init__(verbose)
        self.freq = freq
        self.client = client
        self.selected_scenarios = selected_scenarios
        self.single_weather_condition = single_weather_condition

        self.WEATHER_PARAMETER_NAMES = {
            WeatherParameter.Dust: 'Dust',
            WeatherParameter.Enabled: 'Enabled',
            WeatherParameter.Fog: 'Fog',
            WeatherParameter.MapleLeaf: 'MapleLeaf',
            WeatherParameter.Rain: 'Rain',
            WeatherParameter.RoadLeaf: 'RoadLeaf',
            WeatherParameter.RoadSnow: 'RoadSnow',
            WeatherParameter.Roadwetness: 'Roadwetness',
            WeatherParameter.Snow: 'Snow'
        }

        self.all_weather_scenarios = {
            "Clear": {
                WeatherParameter.Rain: 0.0,
                WeatherParameter.Fog: 0.0,
                WeatherParameter.Dust: 0.0,
                WeatherParameter.Snow: 0.0,
                WeatherParameter.MapleLeaf: 0.0,
                WeatherParameter.RoadLeaf: 0.0,
                WeatherParameter.RoadSnow: 0.0,
                WeatherParameter.Roadwetness: 0.0,
                WeatherParameter.Enabled: 1.0
            },
            "LightRain": {
                WeatherParameter.Rain: 0.3,
                WeatherParameter.Fog: 0.1,
                WeatherParameter.Dust: 0.0,
                WeatherParameter.Snow: 0.0,
                WeatherParameter.MapleLeaf: 0.0,
                WeatherParameter.RoadLeaf: 0.0,
                WeatherParameter.RoadSnow: 0.0,
                WeatherParameter.Roadwetness: 0.2,
                WeatherParameter.Enabled: 1.0
            },
            "HeavyRain": {
                WeatherParameter.Rain: 1.0,
                WeatherParameter.Fog: 0.2,
                WeatherParameter.Dust: 0.0,
                WeatherParameter.Snow: 0.0,
                WeatherParameter.MapleLeaf: 0.0,
                WeatherParameter.RoadLeaf: 0.0,
                WeatherParameter.RoadSnow: 0.0,
                WeatherParameter.Roadwetness: 0.8,
                WeatherParameter.Enabled: 1.0
            },
            "FogOnly": {
                WeatherParameter.Rain: 0.0,
                WeatherParameter.Fog: 0.5,
                WeatherParameter.Dust: 0.0,
                WeatherParameter.Snow: 0.0,
                WeatherParameter.MapleLeaf: 0.0,
                WeatherParameter.RoadLeaf: 0.0,
                WeatherParameter.RoadSnow: 0.0,
                WeatherParameter.Roadwetness: 0.0,
                WeatherParameter.Enabled: 1.0
            },
            "SnowOnly": {
                WeatherParameter.Rain: 0.0,
                WeatherParameter.Fog: 0.0,
                WeatherParameter.Dust: 0.0,
                WeatherParameter.Snow: 1.0,
                WeatherParameter.MapleLeaf: 0.0,
                WeatherParameter.RoadLeaf: 0.0,
                WeatherParameter.RoadSnow: 1.0,
                WeatherParameter.Roadwetness: 0.0,
                WeatherParameter.Enabled: 1.0
            },
            "RainAndFog": {
                WeatherParameter.Rain: 1,
                WeatherParameter.Fog: 1,
                WeatherParameter.Dust: 1,
                WeatherParameter.Snow: 0.0,
                WeatherParameter.MapleLeaf: 0.0,
                WeatherParameter.RoadLeaf: 0.0,
                WeatherParameter.RoadSnow: 0.0,
                WeatherParameter.Roadwetness: 1,
                WeatherParameter.Enabled: 1.0
            },
            "LeavesFalling": {
                WeatherParameter.Rain: 0.0,
                WeatherParameter.Fog: 0.0,
                WeatherParameter.Dust: 0.0,
                WeatherParameter.Snow: 0.0,
                WeatherParameter.MapleLeaf: 1,
                WeatherParameter.RoadLeaf: 1,
                WeatherParameter.RoadSnow: 0.0,
                WeatherParameter.Roadwetness: 0.0,
                WeatherParameter.Enabled: 1.0
            },
            "RainAndLeaves": {
                WeatherParameter.Rain: 0.5,
                WeatherParameter.Fog: 0.1,
                WeatherParameter.Dust: 0.0,
                WeatherParameter.Snow: 0.0,
                WeatherParameter.MapleLeaf: 0.3,
                WeatherParameter.RoadLeaf: 0.3,
                WeatherParameter.RoadSnow: 0.0,
                WeatherParameter.Roadwetness: 0.4,
                WeatherParameter.Enabled: 1.0
            },
        }

    def _on_step(self) -> bool:
        if not self.single_weather_condition:
            if self.num_timesteps % self.freq == 0:
                self._change_weather()
        return True

    def _on_episode_start(self):
        if self.single_weather_condition:
            self._set_weather_once()
        else:
            self._change_weather()

    def _set_weather_once(self):
        try:
            scenario_name = self.selected_scenarios[0]
            new_weather = self.all_weather_scenarios.get(scenario_name, self.all_weather_scenarios["Clear"])

            self.client.simEnableWeather(True)

            for param, value in new_weather.items():
                self.client.simSetWeatherParameter(param, value)

            if self.verbose > 0:
                readable_weather = {self.WEATHER_PARAMETER_NAMES.get(param, 'Unknown'): value for param, value in new_weather.items()}
                print(f"[WeatherChangeCallback] Episode Start - Weather set to: {readable_weather}")

        except airsim.rpc.RPCError as rpc_e:
            print(f"[WeatherChangeCallback] RPC Error: {rpc_e}")
        except Exception as e:
            print(f"[WeatherChangeCallback] Unexpected Error: {e}")

    def _change_weather(self):
        try:
            scenario_name = random.choice(self.selected_scenarios)
            new_weather = self.all_weather_scenarios.get(scenario_name, self.all_weather_scenarios["Clear"])

            self.client.simEnableWeather(True)

            for param, value in new_weather.items():
                self.client.simSetWeatherParameter(param, value)

            if self.verbose > 0:
                readable_weather = {self.WEATHER_PARAMETER_NAMES.get(param, 'Unknown'): value for param, value in new_weather.items()}
                print(f"[WeatherChangeCallback] Timesteps: {self.num_timesteps} - Weather changed to: {readable_weather}")
        except airsim.rpc.RPCError as rpc_e:
            print(f"[WeatherChangeCallback] RPC Error: {rpc_e}")
        except Exception as e:
            print(f"[WeatherChangeCallback] Unexpected Error: {e}")
