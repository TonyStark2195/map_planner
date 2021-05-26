import numpy as np
import gym
import os
import sys
from arguments_dqn_baseline import get_args
# from goal_env import *
from dmlab_env.gym_dmlab_maze import *
import random
import torch

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': 1,
            'action_max': 1,
            'action_dim': env.action_space.n
            }
    params['max_timesteps'] = 720 # env._max_episode_steps
    print(params)
    return params

def launch(args):
    env = gym.make(args.env_name,
               map_sizes=[5], map_fix_id=[5, 1],
               disable_goal=True,
               regenerate=True,
               view_topdown=False)
    test_env = gym.make(args.test,
               map_sizes=[5], map_fix_id=[5, 11],
               disable_goal=True,
               regenerate=True,
               view_topdown=False)
    # set random seeds for reproduce
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device is not 'cpu':
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)
    env_params['max_test_timesteps'] = 720 # test_env._max_episode_steps

    from algos.dqn_agent import dqn_agent
    dqn_agent = dqn_agent(args, env, env_params, test_env)
    dqn_agent.learn()

if __name__ == '__main__':
    args = get_args()
    launch(args)