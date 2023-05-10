import os

import numpy as np
from dm_control import mjcf
from dm_robotics.moma import robot as moma_robot
from dm_robotics.moma.effectors import (arm_effector,
                                        cartesian_6d_velocity_effector,
                                        default_gripper_effector)
from dm_robotics.moma.sensors import robot_tcp_sensor

from bridge_mujoco.robots.widowx.gripper_sensor import GripperSensor
from bridge_mujoco.robots.widowx.widowx import WindowX250sArm, WindowX250sHand

_ASSETS_DIR = os.path.dirname(__file__)
_FRANKA_ARM_XML_PATH = os.path.join(_ASSETS_DIR, 'franka_arm.xml')
_FRANKA_HAND_XML_PATH = os.path.join(_ASSETS_DIR, 'franka_hand.xml')


class FrankaArm(WindowX250sArm):
    RESET_JOINT_VALUES = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    """A composer entity representing a Jaco arm."""

    def _build(self, name=None):
        """Initializes the JacoArm.
    Args:
      name: String, the name of this robot. Used as a prefix in the MJCF name
        name attributes.
    """
        self._mjcf_root = mjcf.from_path(_FRANKA_ARM_XML_PATH)
        self._name = name

        # Find MJCF elements that will be exposed as attributes.
        self._joints = self._mjcf_root.find_all('joint')
        self._bodies = self.mjcf_model.find_all('body')
        self._actuators = self.mjcf_model.find_all('actuator')
        self._wrist_site = self.mjcf_model.find('site', 'wrist_site')
        self._base_site = self.mjcf_model.find('site', 'base_site')


class FrankaHand(WindowX250sHand):
    RESET_JOINT_VALUES = [0.04, 0.04]

    def _build(self, name=None):
        self._mjcf_root = mjcf.from_path(_FRANKA_HAND_XML_PATH)
        self._name = name

        # Find MJCF elements that will be exposed as attributes.
        self._joints = self._mjcf_root.find_all('joint')
        self._bodies = self.mjcf_model.find_all('body')
        self._actuators = self.mjcf_model.find_all('actuator')
        self._tool_center_point = self.mjcf_model.find('site', 'gripper_ee')


def create_franka(control_timestep: float,
                  robot_name: str = 'wx250s',
                  add_sensors: bool = False) -> moma_robot.Robot:
    arm = FrankaArm()
    gripper = FrankaHand()

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
        joint_velocity_limits=np.array([np.pi] * 7),
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
