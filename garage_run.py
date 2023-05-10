#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.experiment import deterministic
from garage.torch import set_gpu_mode, global_device
from garage.sampler import LocalSampler, RaySampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.envs import GymEnv, normalize

@wrap_experiment(snapshot_mode='none', archive_launch_repo=False)
def ppo_push(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    deterministic.set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
   
    horizon=100
    from robot_env import RobotEnv
    env_ = RobotEnv(
                    hand_centric_view=False, third_person_view=False,
                    qpos=True, ee_pos=True,
                    flat_obs=True,
                    normalize_obs=True,
                    max_path_length=50,
                    goal_state='right_closed',
                    randomize_ee_on_reset=False,
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    sim=True
                )

    env = normalize(GymEnv(env_, max_episode_length=horizon))

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[256, 256],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    # sampler = RaySampler(agents=policy,
    #                      envs=env,
    #                      max_episode_length=env.spec.max_episode_length)
    sampler = LocalSampler(agents=policy,
                         envs=env,
                         n_workers=1,
                         max_episode_length=env.spec.max_episode_length)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               center_adv=False)

    set_gpu_mode(False)

    trainer.setup(algo=algo, env=env)
    trainer.train(n_epochs=500000, batch_size=256)

ppo_push(seed=42)