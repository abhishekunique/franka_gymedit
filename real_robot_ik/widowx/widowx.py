# Copyright 2019 The dm_control Authors.
# Modifications 2021 Ilya Kostrikov.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
from typing import List, Optional

import numpy as np
from bridge_mujoco.robots.control_mode import ControlMode
from bridge_mujoco.robots.widowx.gripper_sensor import GripperSensor
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils.transformations import mat_to_quat
from dm_robotics.moma import robot as moma_robot
from dm_robotics.moma.effectors import (arm_effector,
                                        cartesian_4d_velocity_effector,
                                        cartesian_6d_velocity_effector,
                                        default_gripper_effector)
from dm_robotics.moma.models import types
from dm_robotics.moma.models.end_effectors.robot_hands import robot_hand
from dm_robotics.moma.models.robots.robot_arms import robot_arm
from dm_robotics.moma.sensors import mujoco_utils, robot_tcp_sensor

_ASSETS_DIR = os.path.dirname(__file__)
_WX250S_XML_ARM_PATH = os.path.join(_ASSETS_DIR, 'wx250s_arm.xml')
_WX250S_XML_HAND_PATH = os.path.join(_ASSETS_DIR, 'wx250s_hand.xml')

_PAD_GEOM_NAMES = [
    'left_finger',
    'right_finger',
]
_VELOCITY_CTRL_TOL = -0.09  # Since the desired value is -0.1


class WindowX250sArm(robot_arm.RobotArm):
    RESET_JOINT_VALUES = [1.57, -0.6, -0.6, 0, -1.57, 0]
    """A composer entity representing a Jaco arm."""

    def _build(self, name=None):
        """Initializes the JacoArm.

    Args:
      name: String, the name of this robot. Used as a prefix in the MJCF name
        name attributes.
    """
        self._mjcf_root = mjcf.from_path(_WX250S_XML_ARM_PATH)
        self._name = name

        # Find MJCF elements that will be exposed as attributes.
        self._joints = self._mjcf_root.find_all('joint')
        self._bodies = self.mjcf_model.find_all('body')
        self._actuators = self.mjcf_model.find_all('actuator')
        self._wrist_site = self.mjcf_model.find('site', 'wrist_site')
        self._base_site = self.mjcf_model.find('site', 'base_site')

    def name(self) -> str:
        return self._name

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root

    def set_joint_angles(self, physics: mjcf.Physics,
                         joint_angles: np.ndarray) -> None:
        physics_joints = physics.bind(self._joints)
        physics_joints.qpos[:] = joint_angles

    @property
    def base_site(self) -> types.MjcfElement:
        return self._base_site

    @property
    def wrist_site(self) -> types.MjcfElement:
        return self._wrist_site

    def initialize_episode(self, physics: mjcf.Physics,
                           random_state: np.random.RandomState):
        """Function called at the beginning of every episode."""
        del random_state  # Unused.

        # Apply gravity compensation
        body_elements = self.mjcf_model.find_all('body')
        gravity = np.hstack([physics.model.opt.gravity, [0, 0, 0]])
        physics_bodies = physics.bind(body_elements)
        if physics_bodies is None:
            raise ValueError('Calling physics.bind with bodies returns None.')
        physics_bodies.xfrc_applied[:] = -gravity * physics_bodies.mass[...,
                                                                        None]

        self.set_joint_angles(physics, self.RESET_JOINT_VALUES)

    def get_relative_wrist_pose(self, physics):
        pose = mujoco_utils.get_site_relative_pose(physics, self.wrist_site,
                                                   self.base_site)
        pos = pose[:3, 3]
        quat = mat_to_quat(pose)
        return np.concatenate([pos, quat])


def _is_geom_in_collision(physics: mjcf.Physics,
                          geom_name: str,
                          geom_exceptions: Optional[List[str]] = None) -> bool:
    """Returns true if a geom is in collision in the physics object."""
    for contact in physics.data.contact:
        geom1_name = physics.model.id2name(contact.geom1, 'geom')
        geom2_name = physics.model.id2name(contact.geom2, 'geom')
        if contact.dist > 1e-8:
            continue
        if (geom1_name == geom_name and geom2_name not in geom_exceptions) or (
                geom2_name == geom_name and geom1_name not in geom_exceptions):
            return True
    return False


def _are_all_collision_geoms_colliding(physics: mjcf.Physics,
                                       mjcf_root: mjcf.RootElement) -> bool:
    """Returns true if the collision geoms in the model are colliding."""
    collision_geoms = [
        mjcf_root.find('geom', name).full_identifier
        for name in _PAD_GEOM_NAMES
    ]
    return all([
        _is_geom_in_collision(physics, geom, collision_geoms)
        for geom in collision_geoms
    ])


class WindowX250sHand(robot_hand.RobotHand):
    RESET_JOINT_VALUES = [0.037, -0.037]

    def _build(self, name=None):
        self._mjcf_root = mjcf.from_path(_WX250S_XML_HAND_PATH)
        self._name = name

        # Find MJCF elements that will be exposed as attributes.
        self._joints = self._mjcf_root.find_all('joint')
        self._bodies = self.mjcf_model.find_all('body')
        self._actuators = self.mjcf_model.find_all('actuator')
        self._tool_center_point = self.mjcf_model.find('site', 'gripper_ee')

    def name(self) -> str:
        return self._name

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root

    @property
    def tool_center_point(self) -> types.MjcfElement:
        """Tool center point site of the hand."""
        return self._tool_center_point

    def initialize_episode(self, physics: mjcf.Physics,
                           random_state: np.random.RandomState):
        """Function called at the beginning of every episode."""
        del random_state  # Unused.

        # Apply gravity compensation
        body_elements = self.mjcf_model.find_all('body')
        gravity = np.hstack([physics.model.opt.gravity, [0, 0, 0]])
        physics_bodies = physics.bind(body_elements)
        if physics_bodies is None:
            raise ValueError('Calling physics.bind with bodies returns None.')
        physics_bodies.xfrc_applied[:] = -gravity * physics_bodies.mass[...,
                                                                        None]

        physics.bind(self.joints).qpos[:] = self.RESET_JOINT_VALUES

    def get_qpos(self, physics):
        return physics.bind(self.joints).qpos

    def get_qvel(self, physics):
        return physics.bind(self.joints).qvel

    def grasp_sensor_callable(self, physics) -> int:
        """Simulate the robot's gOBJ object detection flag."""

        # No grasp when no collision.
        collision_geoms_colliding = _are_all_collision_geoms_colliding(
            physics, self.mjcf_model)

        if not collision_geoms_colliding:
            return False

        # No grasp when no velocity ctrl command.
        vel = physics.bind(self.actuators[0]).ctrl
        if vel < _VELOCITY_CTRL_TOL:
            return True
        else:
            return False

    def object_between_fingers_callable(self, physics):
        left_finger_site = self.mjcf_model.find('site', 'left_finger_ee')
        left_finger_geom = self.mjcf_model.find('geom', 'left_finger')
        left_finger_geom_id = physics.bind(left_finger_geom).element_id
        left_finger_pos = physics.bind(left_finger_site).xpos.copy()

        right_finger_site = self.mjcf_model.find('site', 'right_finger_ee')
        right_finger_geom = self.mjcf_model.find('geom', 'right_finger')
        right_finger_geom_id = physics.bind(right_finger_geom).element_id
        right_finger_pos = physics.bind(right_finger_site).xpos.copy()

        ray_pos = left_finger_pos
        ray_vec = right_finger_pos - left_finger_pos
        ray_vec /= np.linalg.norm(ray_vec)

        geomid_out = np.array([-1], dtype=np.intc)
        mjbindings.mjlib.mj_ray(physics.model.ptr, physics.data.ptr, ray_pos,
                                ray_vec, None, 1, -1, geomid_out)
        return (geomid_out[0] != right_finger_geom_id
                and geomid_out[0] != left_finger_geom_id)

    def gripper_open_percentage_callable(self, physics):
        for joint in self.joints:
            qpos = physics.bind(joint).qpos[0]
            curr_state = abs(qpos - joint.range[0])
            joint_range = joint.range[1] - joint.range[0]
            if joint.name == 'left_finger':
                left_percentage = min(max(curr_state / joint_range, 0.), 1.)
            elif joint.name == 'right_finger':
                right_percentage = 1 - min(max(curr_state / joint_range, 0.),
                                           1.)

        return min(left_percentage, right_percentage)


def create_wx250s(control_timestep: float,
                  robot_name: str = 'wx250s',
                  control_mode: ControlMode = ControlMode.CARTESIAN_4D,
                  add_sensors: bool = False) -> moma_robot.Robot:
    arm = WindowX250sArm()
    gripper = WindowX250sHand()

    moma_robot.standard_compose(arm=arm, gripper=gripper)

    effector = arm_effector.ArmEffector(arm=arm,
                                        action_range_override=None,
                                        robot_name=robot_name)

    effector_model = cartesian_6d_velocity_effector.ModelParams(
        gripper.tool_center_point, arm.joints)

    effector_control = cartesian_6d_velocity_effector.ControlParams(
        control_timestep_seconds=control_timestep,
        max_lin_vel=0.5,
        max_rot_vel=1.0,
        joint_velocity_limits=np.array([np.pi] * 6),
        nullspace_gain=0.025,
        nullspace_joint_position_reference=arm.RESET_JOINT_VALUES,
        regularization_weight=1e-2,
        enable_joint_position_limits=True,
        minimum_distance_from_joint_position_limit=0.01,
        joint_position_limit_velocity_scale=0.95,
        max_cartesian_velocity_control_iterations=300,
        max_nullspace_control_iterations=300)
    # TODO: Check if nothing is missing from
    # https://github.com/deepmind/rgb_stacking/blob/main/rgb_stacking/task.py#L259

    cart_effector_6d = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
        robot_name, effector, effector_model, effector_control)

    if control_mode == ControlMode.CARTESIAN_4D:
        cart_effector = cartesian_4d_velocity_effector.Cartesian4dVelocityEffector(
            cart_effector_6d,
            element=gripper.tool_center_point,
            rotation_gain=0.01,
            effector_prefix=f'{robot_name}_cart_4d_vel')
    else:
        cart_effector = cart_effector_6d

    gripper_effector = default_gripper_effector.DefaultGripperEffector(
        gripper, robot_name=robot_name)

    if add_sensors:
        tcp_sensor = robot_tcp_sensor.RobotTCPSensor(gripper=gripper,
                                                     name=f'info/{robot_name}')

        pos_key = tcp_sensor.get_obs_key(robot_tcp_sensor.Observations.POS)
        quat_key = tcp_sensor.get_obs_key(robot_tcp_sensor.Observations.QUAT)
        for k, v in tcp_sensor.observables.items():
            if k not in [pos_key, quat_key]:
                v.enabled = False
        """
        wrist_site_sensor = site_sensor.SiteSensor(arm.wrist_site,
                                                   name='wrist')
        """
        gripper_sensor = GripperSensor(gripper=gripper, name='gripper')

        robot_sensors = [tcp_sensor, gripper_sensor]  #, wrist_site_sensor]
    else:
        robot_sensors = []

    return moma_robot.StandardRobot(arm=arm,
                                    arm_base_site_name='base_site',
                                    gripper=gripper,
                                    robot_sensors=robot_sensors,
                                    arm_effector=cart_effector,
                                    gripper_effector=gripper_effector,
                                    name=robot_name)
