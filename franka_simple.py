import numpy as np
from gym import utils
from gym import spaces
import mujoco
from collections import OrderedDict
import gym
from real_robot_ik.robot_ik_solver import RobotIKSolver
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import glfw
import time
import mujoco_viewer

from transformations import euler_to_quat, quat_to_euler

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class FrankaMujocoEnv(gym.Env, utils.EzPickle):
    def __init__(self,
                 max_episode_length=25,
                 has_renderer=True, has_offscreen_renderer=False, video=False,
                 control_hz=20,
                 inject_noise=0.0,
                 control_steps = 5,
                 **kwargs):
        
        assert (has_renderer and has_offscreen_renderer) is False
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer

        self.control_steps = control_steps

        self.render_width = 720
        self.render_height = 720

        self.img_width = 256
        self.img_height = 256

        self.max_episode_length = max_episode_length

        self.video = video
        if self.video:
            self.recording = None
        
        self.frame_skip = 5

        xml_path = 'assets/kitchen_franka/kitchen_assets/free_franka.xml'
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.ik = RobotIKSolver(robot=self, control_hz=control_hz)

        self.viewer_setup()

        self._set_action_space()
        observation = self._get_obs()
        self._set_observation_space(observation)

        self.action_space.flat_dim = len(self.action_space.low)
        self.observation_space.flat_dim = len(self.observation_space.low)
        self.spec = self
        self.seed()

        # mujoco_env.MujocoEnv.__init__(self, , 5)
        utils.EzPickle.__init__(self)
        
    @property
    def parameter_dim(self):
         return len(self.get_parameters())
    
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
    
    def _set_action_space(self):
        self.action_space = spaces.Box(low=-1., high=1., shape=[7], dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space
    
    def get_parameters(self):
        return self._value_to_range(self.zeta, self.zeta_max, self.zeta_min, 1., -1.)
    
    def set_parameters(self, params):
        self.zeta = self._value_to_range(params, 1., -1., self.zeta_max, self.zeta_min)
        self.parameters_set = True
    
    def update_gripper(self, gripper):
        pass

    def update_pose(self, pos, angle):
         
        desired_qpos, success = self.ik.compute(pos, euler_to_quat(angle))
        
        # TODO: Fix syntax
        for _ in range(self.control_steps):
            self.data.ctrl[:len(desired_qpos)] = desired_qpos
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # self.update_joints(desired_qpos)
        self.data.ctrl[:len(desired_qpos)] = desired_qpos

        # advance simulation, use control callback to obtain external force and control.
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        if self.has_renderer:
            self.render()
        return self.get_ee_pos(), self.get_ee_angle()
    
    def update_joints(self, qpos):
        self.data.qpos[:len(qpos)] = qpos
        # forward dynamics: same as mj_step but do not integrate in time.
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        
        angle_euler = R.from_quat(action[3:][[3, 0, 1, 2]]).as_euler('xyz', degrees=False)
        self.update_pose(action[:3], angle_euler)

        next_obs = self._get_obs()
        after_ee_pos, after_ee_quat = self.get_ee_pose()
        reward = - np.linalg.norm(np.array([0.3, 0.3]) - after_ee_pos[:2])

        done = bool(reward < 0.1)

        return next_obs, reward, done, {}
    
    def _get_obs(self):
        return np.concatenate(
            [
                # joint pos
                self.get_joint_positions(),
                # wrist_site ee pos
                *self.get_ee_pose(),
                # self.data.qvel.flat,
            ]
        ).astype(np.float32)

    def _value_to_range(self, value, r_max, r_min, t_max, t_min):
        """scales value in range [r_max, r_min] to range [t_max, t_min]"""
        return (value - r_min) / (r_max - r_min) * (t_max - t_min) + t_min

    def reset(self, *args, **kwargs):
        
        mujoco.mj_resetData(self.model, self.data)

        obs = self._get_obs()
                
        return obs # , {}
    
    def viewer_setup(self):
        if self.has_renderer:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, height=self.render_height, width=self.render_width)
        if self.has_offscreen_renderer:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, "offscreen", height=self.img_height, width=self.img_width)
        
    def render(self):
        if self.has_renderer:
            self.viewer.render()
            return np.zeros(3)
        elif self.has_offscreen_renderer:
            return self.viewer.read_pixels(camid=0)
        else:
            return np.zeros(3)

    def seed(self, seed=0):
        np.random.seed(seed)

    def get_joint_positions(self):
        return self.data.qpos.copy()
    
    def get_joint_velocities(self):
        return self.data.qvel.copy()
    
    def get_ee_pose(self):
        # wrist_site
        return self.get_ee_pos(), self.get_ee_angle(quat=True)

    def get_ee_pos(self):
        # wrist_site
        return self.data.site_xpos[1].copy()
    
    def get_ee_angle(self, quat=False):
        # wrist_site
        ee_angle = R.from_matrix(self.data.site_xmat[1].copy().reshape(3,3))
        ee_angle = ee_angle.as_euler('xyz')
        if quat:
            return euler_to_quat(ee_angle)
        else:
            return ee_angle
        
    def get_gripper_state(self):
        # always closed
        return np.zeros(3)
    
    def read_cameras(self):
        # cam 0
        img = self.render()
        img_dict = {"array": img}
        # cam 1 TBD
        return [img_dict, img_dict]
