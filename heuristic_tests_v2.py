import os
import pickle
import numpy as np

from IPython.display import clear_output

import matplotlib
import matplotlib.pyplot as plt
import gym
from optical_rl_gym.utils import evaluate_heuristic
from optical_rl_gym.heuristics import kSP_FF, FF_kSP, kSP_MU, CA_MU


current_directory = os.getcwd()
with open(current_directory+'/topologies/nsfnet_chen_5-paths_rounded.h5', 'rb') as f:
    topology_5 = pickle.load(f)
with open(current_directory+'/topologies/nsfnet_chen_4-paths_rounded.h5', 'rb') as f:
    topology_4 = pickle.load(f)
with open(current_directory+'/topologies/nsfnet_chen_3-paths_rounded.h5', 'rb') as f:
    topology_3 = pickle.load(f)
with open(current_directory+'/topologies/nsfnet_chen_2-paths_rounded.h5', 'rb') as f:
    topology_2 = pickle.load(f)
with open(current_directory+'/topologies/nsfnet_chen_1-paths_rounded.h5', 'rb') as f:
    topology_1 = pickle.load(f)
# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
# node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
#        0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
#        0.07607608, 0.12012012, 0.01901902, 0.16916917])
node_request_probabilities = pickle.load(open("node_req_probs/14_node_uniform.pkl",'rb'))
load = int(1e10)
ep_len = 20000
req_br = 100e9
tfb = False
# mean_service_holding_time=7.5,
env_args_1 = dict(topology=topology_1, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=1e8,
                episode_length=ep_len, node_request_probabilities=node_request_probabilities, exp_request_res = req_br,
                term_on_first_block=tfb)
env_args_2 = dict(topology=topology_2, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=1e8,
                episode_length=ep_len, node_request_probabilities=node_request_probabilities, exp_request_res = req_br,
                term_on_first_block=tfb)
env_args_3 = dict(topology=topology_3, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=1e8,
                episode_length=ep_len, node_request_probabilities=node_request_probabilities, exp_request_res = req_br,
                term_on_first_block=tfb)
env_args_4 = dict(topology=topology_4, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=1e8,
                episode_length=ep_len, node_request_probabilities=node_request_probabilities, exp_request_res = req_br,
                term_on_first_block=tfb)
env_args_5 = dict(topology=topology_5, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=1e8,
                episode_length=ep_len, node_request_probabilities=node_request_probabilities, exp_request_res = req_br,
                term_on_first_block=tfb)

# env_1 = gym.make('RWAFOCS-v221', **env_args_1)
# env_2 = gym.make('RWAFOCS-v221', **env_args_2)
# env_3 = gym.make('RWAFOCS-v221', **env_args_3)
# env_4 = gym.make('RWAFOCS-v221', **env_args_4)
# env_5 = gym.make('RWAFOCS-v221', **env_args_5)
# print('RWAFOCS-v221')

env_1 = gym.make('RWAFOCS-v22', **env_args_1)
env_2 = gym.make('RWAFOCS-v22', **env_args_2)
env_3 = gym.make('RWAFOCS-v22', **env_args_3)
env_4 = gym.make('RWAFOCS-v22', **env_args_4)
env_5 = gym.make('RWAFOCS-v22', **env_args_5)
print('RWAFOCS-v22')

# heuristic = kSP_FF
# heuristic = FF_kSP
heuristic = kSP_MU
# heuristic = CA_MU

# mean_reward_1, std_reward_1 = evaluate_heuristic(env_1, heuristic, n_eval_episodes=1,
#                        render=False, callback=None, reward_threshold=None,
#                        return_episode_rewards=False)
# print("mean reward T1:", mean_reward_1)
# print("Std reward T1:", std_reward_1)
# print('Total number of accepted services T1:', env_1.services_accepted)
#
# mean_reward_2, std_reward_2 = evaluate_heuristic(env_2, heuristic, n_eval_episodes=1,
#                        render=False, callback=None, reward_threshold=None,
#                        return_episode_rewards=False)
# print("mean reward T2:", mean_reward_2)
# print("Std reward T2:", std_reward_2)
# print('Total number of accepted services T2:', env_2.services_accepted)
#
mean_reward_3, std_reward_3 = evaluate_heuristic(env_3, heuristic, n_eval_episodes=1,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
print("mean reward T3:", mean_reward_3)
print("Std reward T3:", std_reward_3)
print('Total number of accepted services T3:', env_3.services_accepted)
#
# mean_reward_4, std_reward_4 = evaluate_heuristic(env_4, heuristic, n_eval_episodes=1,
#                        render=False, callback=None, reward_threshold=None,
#                        return_episode_rewards=False)
# print("mean reward T4:", mean_reward_4)
# print("Std reward T4:", std_reward_4)
# print('Total number of accepted services T4:', env_4.services_accepted)

mean_reward_5, std_reward_5 = evaluate_heuristic(env_5, heuristic, n_eval_episodes=1,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
print("mean reward T5:", mean_reward_5)
print("Std reward T5:", std_reward_5)
print('Total number of accepted services T5:', env_5.services_accepted)
