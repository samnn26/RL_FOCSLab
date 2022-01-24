import os
import pickle
import numpy as np

from IPython.display import clear_output

import matplotlib
import matplotlib.pyplot as plt
#import config InlineBackend.figure_format = 'svg'
import tensorflow as tf

# silencing tensorflow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from datetime import datetime

tf.__version__ # printint out tensorflow version used
import stable_baselines3
import sb3_contrib
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.results_plotter import load_results, ts2xy
from stable_baselines3.common.results_plotter import load_results, ts2xy
# from stable_baselines3.bench import Monitor
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common import results_plotter
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
#stable_baselines.__version__ # printing out stable_baselines version used
import gym
import pickle
import cProfile
import pandas as pd
from optical_rl_gym.heuristics import kSP_FF, FF_kSP, kSP_MU, CA_MU
from optical_rl_gym.utils import evaluate_heuristic, random_policy

import argparse
font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 14}
matplotlib.rc('font', **font)

parser = argparse.ArgumentParser(description='Set up simulation.')
parser.add_argument('--n_eval_eps', default=1, type=int)
parser.add_argument('--exp_id', default="", type=str)
parser.add_argument('--env_id', default="", type=str)
args = parser.parse_args()
n_eval_eps = args.n_eval_eps
exp_id = args.exp_id
env_id = args.env_id

# def make_env(env_args, log_dir):
#     def maker():
#         env = gym.make('RWAFOCS-v4', **env_args)
#         env = Monitor(env, log_dir + 'evaluation', info_keywords=('episode_services_accepted',
#         'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
#         'episode_cum_services_processed', 'throughput'))
#         return env
#     return maker
current_directory = os.getcwd()
# with open('/Users/joshnevin/RL_FOCSLab/topologies/nsfnet_chen_5-paths_directional.h5', 'rb') as f:
with open(current_directory+'/topologies/nsfnet_chen_5-paths_directional.h5', 'rb') as f:
    topology = pickle.load(f)
# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
       0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
       0.07607608, 0.12012012, 0.01901902, 0.16916917])
load = int(1e10)
env_args = dict(topology=topology, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=1e8, # value is not set as in the paper to achieve comparable reward values
                episode_length=1600, node_request_probabilities=node_request_probabilities, exp_request_res=25e9,
                exp_request_lambda=1, term_on_first_block=True)
model_dir = "./tmp/RWAFOCS-ppo/"+exp_id+"/_core_0/"
eval_dir = model_dir + "eval_results/"
os.makedirs(eval_dir, exist_ok=True)

env_0 = gym.make('RWAFOCS-v41', **env_args)
env_0 = Monitor(env_0, eval_dir + 'evaluation_rand', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput'))
env_1 = gym.make('RWAFOCS-v22', **env_args)
env_1 = Monitor(env_1, eval_dir + 'evaluation_kspff', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput'))
env_2 = gym.make('RWAFOCS-v22', **env_args)
env_2 = Monitor(env_2, eval_dir + 'evaluation_ffksp', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput'))
env_3 = gym.make('RWAFOCS-v22', **env_args)
env_3 = Monitor(env_3, eval_dir + 'evaluation_camu', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput'))
env_4 = gym.make('RWAFOCS-v22', **env_args)
env_4 = Monitor(env_4, eval_dir + 'evaluation_kspmu', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput'))

# 2022-01-18tbf10Mdict
# model_dir = "/Users/joshnevin/RL_FOCSLab/tmp/RWAFOCS-ppo/"+exp_id+"/_core_0/"

#env = DummyVecEnv([make_env(env_args, log_dir)])
env = gym.make(env_id, **env_args)
env = Monitor(env, eval_dir + 'evaluation_rl', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput'))

model = MaskablePPO.load(model_dir+'best_model')

mean_reward_rl, std_reward_rl = evaluate_policy(model, env, n_eval_episodes=n_eval_eps,deterministic =True)

mean_reward_rand, std_reward_rand = evaluate_heuristic(env_0, random_policy, n_eval_episodes=n_eval_eps,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
mean_reward_kspff, std_reward_kspff = evaluate_heuristic(env_1, kSP_FF, n_eval_episodes=n_eval_eps,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
mean_reward_ffksp, std_reward_ffksp = evaluate_heuristic(env_2, FF_kSP, n_eval_episodes=n_eval_eps,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
mean_reward_camu, std_reward_camu = evaluate_heuristic(env_3, CA_MU, n_eval_episodes=n_eval_eps,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
mean_reward_kspmu, std_reward_kspmu = evaluate_heuristic(env_4, kSP_MU, n_eval_episodes=n_eval_eps,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)

today = datetime.today().strftime('%Y-%m-%d')
