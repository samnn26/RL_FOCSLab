import gym
import optical_rl_gym
from optical_rl_gym.utils import Path
from optical_rl_gym.envs.rwa_env import shortest_available_path_last_fit, shortest_path_first_fit, shortest_available_path_first_fit, least_loaded_path_first_fit, PathOnlyFirstFitAction
from optical_rl_gym.utils import evaluate_heuristic, random_policy

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

# imports for training the RL agent etc...
# import tensorflow as tf
# import stable_baselines
# from stable_baselines.common.callbacks import BaseCallback
# from stable_baselines.results_plotter import load_results, ts2xy
# from stable_baselines import PPO2
# from stable_baselines.bench import Monitor
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines import results_plotter

load = 1e10
logging.getLogger('rwaenv').setLevel(logging.INFO)

seed = 10
episodes = 10
episode_length = 300

monitor_files = []
policies = []

# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'
with open(f'/Users/joshnevin/RL_FOCSLab/topologies/nsfnet_chen_5-paths_directional.h5', 'rb') as f:
    topology = pickle.load(f)

env_args = dict(topology=topology, seed=10, allow_rejection=False, load=load, mean_service_holding_time=1e10, episode_length=episode_length)

# creating an environment
env_rnd = gym.make('RWAFOCS-v0', **env_args)
# env_focs = gym.make('RWAFOCS-v0', **env_args)

# evaluating the environment that acts completely random both for path and wavelength
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, random_policy, n_eval_episodes=episodes)
# env_rnd.render()  # uncomment to visualize a representation of the environment
print('Rnd:', mean_reward_rnd, std_reward_rnd)

# # here goes the arguments of the policy network to be used
# policy_args = dict(net_arch=5*[128], # the neural network has five layers with 128 neurons each
#                    act_fun=tf.nn.elu) # we use the elu activation function
#
# agent = PPO2(MlpPolicy, env_focs, verbose=0, tensorboard_log="./tb/PPO-RWAEnvFOCS-v0/",
#  policy_kwargs=policy_args, gamma=.95, learning_rate=10e-5)
#
# a = agent.learn(total_timesteps=1000, callback=callback)
# results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "RWAEnvFOCS PPO")


rnd_path_action_probability = np.sum(env_rnd.actions_output, axis=1) / np.sum(env_rnd.actions_output)
rnd_wavelength_action_probability = np.sum(env_rnd.actions_output, axis=0) / np.sum(env_rnd.actions_output)
print('Path action probability:', np.sum(env_rnd.actions_output, axis=1) / np.sum(env_rnd.actions_output))
print('Wavelength action probability:', np.sum(env_rnd.actions_output, axis=0) / np.sum(env_rnd.actions_output))

num_lps_reused = env_rnd.num_lightpaths_reused
print('Load (Erlangs):', load)
print('Service bit rate (Gb/s):', env_rnd.service.bit_rate/1e9)
print('Total number of services:', env_rnd.services_processed)
print('Total number of accepted services:', env_rnd.services_accepted)
print('Blocking probability:', 1 - env_rnd.services_accepted/env_rnd.services_processed)
print('Number of services on existing lightpaths:', num_lps_reused)
print('Number of services released:', env_rnd.num_lightpaths_released)
print('Number of transmitters on each node:', env_rnd.num_transmitters)
print('Number of receivers on each node:', env_rnd.num_receivers)





# breakpoint()
# # creating an envionrment that only needs the path selection, then selects the first-fit wavelength automatically
# env_rnd_ff = PathOnlyFirstFitAction(gym.make('RWAFOCS-v0', **env_args))
# mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd_ff, random_policy, n_eval_episodes=episodes)
# # env_rnd.render()  # uncomment to visualize a representation of the environment
# print('Rnd-FF:', mean_reward_rnd, std_reward_rnd)
# rnd_ff_path_action_probability = np.sum(env_rnd_ff.actions_output, axis=1) / np.sum(env_rnd_ff.actions_output)
# rnd_ff_wavelength_action_probability = np.sum(env_rnd_ff.actions_output, axis=0) / np.sum(env_rnd_ff.actions_output)
# breakpoint()
# print('Path action probability:', np.sum(env_rnd.actions_output, axis=1) / np.sum(env_rnd.actions_output))
# print('Wavelength action probability:', np.sum(env_rnd.actions_output, axis=0) / np.sum(env_rnd.actions_output))
#
# env_sp = gym.make('RWA-v0', **env_args)
# mean_reward_sp, std_reward_sp = evaluate_heuristic(env_sp, shortest_path_first_fit, n_eval_episodes=episodes)
# sp_path_action_probability = np.sum(env_sp.actions_output, axis=1) / np.sum(env_sp.actions_output)
# sp_wavelength_action_probability = np.sum(env_sp.actions_output, axis=0) / np.sum(env_sp.actions_output)
# print('SP:', mean_reward_sp, std_reward_sp)
# # print('Path action probability:', np.sum(env_sp.actions_output, axis=1) / np.sum(env_sp.actions_output))
# # print('Wavelength action probability:', np.sum(env_sp.actions_output, axis=0) / np.sum(env_sp.actions_output))
#
# env_sap = gym.make('RWA-v0', **env_args)
# mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap, shortest_available_path_first_fit, n_eval_episodes=episodes)
# sap_path_action_probability = np.sum(env_sap.actions_output, axis=1) / np.sum(env_sap.actions_output)
# sap_wavelength_action_probability = np.sum(env_sap.actions_output, axis=0) / np.sum(env_sap.actions_output)
# print('SAP:', mean_reward_sap, std_reward_sap)
# # print('Path action probability:', np.sum(env_sap.actions_output, axis=1) / np.sum(env_sap.actions_output))
# # print('Wavelength action probability:', np.sum(env_sap.actions_output, axis=0) / np.sum(env_sap.actions_output))
#
# env_sap_lf = gym.make('RWA-v0', **env_args)
# mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap_lf, shortest_available_path_last_fit, n_eval_episodes=episodes)
# sap_lf_path_action_probability = np.sum(env_sap_lf.actions_output, axis=1) / np.sum(env_sap_lf.actions_output)
# sap_lf_wavelength_action_probability = np.sum(env_sap_lf.actions_output, axis=0) / np.sum(env_sap_lf.actions_output)
# print('SAP-LF:', mean_reward_sap, std_reward_sap)
# # print('Path action probability:', np.sum(env_sap.actions_output, axis=1) / np.sum(env_sap.actions_output))
# # print('Wavelength action probability:', np.sum(env_sap.actions_output, axis=0) / np.sum(env_sap.actions_output))
#
# env_llp = gym.make('RWA-v0', **env_args)
# mean_reward_llp, std_reward_llp = evaluate_heuristic(env_llp, least_loaded_path_first_fit, n_eval_episodes=episodes)
# llp_path_action_probability = np.sum(env_llp.actions_output, axis=1) / np.sum(env_llp.actions_output)
# llp_wavelength_action_probability = np.sum(env_llp.actions_output, axis=0) / np.sum(env_llp.actions_output)
# print('LLP:', mean_reward_llp, std_reward_llp)
# # print('Path action probability:', np.sum(env_llp.actions_output, axis=1) / np.sum(env_llp.actions_output))
# # print('Wavelength action probability:', np.sum(env_llp.actions_output, axis=0) / np.sum(env_llp.actions_output))
#
# plt.figure()
# plt.semilogy(rnd_path_action_probability, label='Rnd')
# plt.semilogy(rnd_ff_path_action_probability, label='Rnd-FF')
# plt.semilogy(sp_path_action_probability, label='SP-FF')
# plt.semilogy(sap_path_action_probability, label='SAP-FF')
# plt.semilogy(sap_lf_path_action_probability, label='SAP-LF')
# plt.semilogy(llp_path_action_probability, label='LLP-FF')
# plt.xlabel('Path index')
# plt.ylabel('Probability')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'figures/path_action_probability_{load}.svg')
# plt.close()
#
# plt.figure()
# plt.semilogy(rnd_wavelength_action_probability, label='Rnd')
# plt.semilogy(rnd_ff_wavelength_action_probability, label='Rnd-FF')
# plt.semilogy(sp_wavelength_action_probability, label='SP-FF')
# plt.semilogy(sap_wavelength_action_probability, label='SAP-FF')
# plt.semilogy(sap_lf_wavelength_action_probability, label='SAP-LF')
# plt.semilogy(llp_wavelength_action_probability, label='LLP-FF')
# plt.xlabel('Wavelength index')
# plt.ylabel('Probability')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'figures/wavelength_action_probability_{load}.svg')
# plt.close()
