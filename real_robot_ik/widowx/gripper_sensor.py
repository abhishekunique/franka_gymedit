# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sensor for a sim Robotiq gripper."""
import enum
from typing import Dict

import numpy as np
import typing_extensions
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.moma import sensor as moma_sensor


@typing_extensions.runtime
class Gripper(typing_extensions.Protocol):

    def grasp_sensor_callable(self, physics) -> int:
        pass


@enum.unique
class Observations(enum.Enum):
    """Observations exposed by a Robotiq gripper sensor."""
    # Whether an object is grasped by the gripper.
    GRASP = 'info/{}_grasp'

    def get_obs_key(self, name: str) -> str:
        """Returns the key to the observation in the observables dict."""
        return self.value.format(name)


class GripperSensor(moma_sensor.Sensor):
    """Robotiq gripper sensor for pos, vel, and grasp related observations."""

    def __init__(self, gripper: Gripper, name: str):
        self._gripper = gripper
        self._name = name
        self._observables = None

    def initialize_for_task(self, control_timestep_seconds: float,
                            physics_timestep_seconds: float,
                            physics_steps_per_control_step: int):

        self._observables = {
            self.get_obs_key(Observations.GRASP):
            observable.Generic(self._grasp)
        }
        for obs in self._observables.values():
            obs.enabled = True

    def initialize_episode(self, physics: mjcf.Physics,
                           random_state: np.random.RandomState) -> None:
        pass

    @property
    def observables(self) -> Dict[str, observable.Observable]:
        return self._observables

    @property
    def name(self) -> str:
        return self._name

    def get_obs_key(self, obs: Observations) -> str:
        return obs.get_obs_key(self._name)

    def _grasp(self, physics: mjcf.Physics) -> np.ndarray:
        return np.array([self._gripper.grasp_sensor_callable(physics)],
                        dtype=np.uint8)
