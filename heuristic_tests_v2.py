import os
import pickle
import numpy as np

from IPython.display import clear_output

import matplotlib
import gym
from optical_rl_gym.utils import evaluate_heuristic
from optical_rl_gym.envs.rwa_env_focs_v2 import shortest_available_path_first_fit


current_directory = os.getcwd()
with open(current_directory+'/topologies/nsfnet_chen_5-paths_directional.h5', 'rb') as f:
    topology = pickle.load(f)
# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
       0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
       0.07607608, 0.12012012, 0.01901902, 0.16916917])

load = int(1e10)

# mean_service_holding_time=7.5,
env_args = dict(topology=topology, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=int(1e6),
                episode_length=5000, node_request_probabilities=node_request_probabilities)

env = gym.make('RWAFOCS-v2', **env_args)

mean_reward, std_reward = evaluate_heuristic(env, shortest_available_path_first_fit, n_eval_episodes=1,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
print(mean_reward)
print(std_reward)

num_lps_reused = env.num_lightpaths_reused
print('Load (Erlangs):', load)
print('Service bit rate (Gb/s):', env.service.bit_rate/1e9)
print('Total number of services:', env.services_processed)
print('Total number of accepted services:', env.services_accepted)
print('Blocking probability:', 1 - env.services_accepted/env.services_processed)
print('Number of services on existing lightpaths:', num_lps_reused)
print('Number of services released:', env.num_lightpaths_released)
print('Number of transmitters on each node:', env.num_transmitters)
print('Number of receivers on each node:', env.num_receivers)
