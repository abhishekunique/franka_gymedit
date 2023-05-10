import numpy as np
from dm_control import mjcf
from dm_robotics.moma.effectors import (arm_effector,
										cartesian_6d_velocity_effector)
from scipy.spatial.transform import Rotation as R
from real_robot_ik.arm import FrankaArm
import torch

def quat_diff(target, source, return_euler=False):
    result = R.from_quat(target) * R.from_quat(source).inv()
    if return_euler: return result.as_euler('xyz')
    return result.as_quat()

class RobotIKSolver:

	def __init__(self, robot, control_hz=20):
		self._robot = robot
		self._arm = FrankaArm()

		self._physics = mjcf.Physics.from_mjcf_model(self._arm.mjcf_model)
		self._effector = arm_effector.ArmEffector(arm=self._arm,
									action_range_override=None,
									robot_name=self._arm.name)
		
		self._effector_model = cartesian_6d_velocity_effector.ModelParams(
			self._arm.wrist_site, self._arm.joints)

		scaler = 0.1

		self._effector_control = cartesian_6d_velocity_effector.ControlParams(
			control_timestep_seconds=1 / control_hz,
			max_lin_vel=1.0,
			max_rot_vel=1.0,
			joint_velocity_limits=np.array([2.075 * scaler] * 4 + [2.51 * scaler] * 3),
			nullspace_gain=0.025, #1e-2 #Encourages small joint changes
			regularization_weight=1e-2, #1e-2 #Encourages staying near joint centers
			enable_joint_position_limits=True,
			minimum_distance_from_joint_position_limit=0.3, #0.01
			joint_position_limit_velocity_scale=0.95,
			max_cartesian_velocity_control_iterations=300,
			max_nullspace_control_iterations=300)


		self._cart_effector_6d = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
			self._arm.name, self._effector, self._effector_model, self._effector_control)

		self._cart_effector_6d.after_compile(self._arm.mjcf_model, self._physics)

	def compute(self, desired_ee_pos, desired_ee_quat):

		qpos = self._robot.get_joint_positions()
		qvel = self._robot.get_joint_velocities()
		curr_pos, curr_quat = self._robot.get_ee_pose()
		curr_pos, curr_quat = curr_pos, curr_quat

		lin_vel = desired_ee_pos - curr_pos
		rot_vel = quat_diff(desired_ee_quat, curr_quat, return_euler=True)

		action = np.concatenate([lin_vel, rot_vel])
		self._arm.update_state(self._physics, qpos, qvel)
		self._cart_effector_6d.set_control(self._physics, action)
		joint_vel_ctrl = self._physics.bind(self._arm.actuators).ctrl.copy()
		
		desired_qpos = qpos + joint_vel_ctrl
		success = np.any(joint_vel_ctrl) #I think it returns zeros when it fails

		return desired_qpos, success
