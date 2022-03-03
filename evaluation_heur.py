import os
import pickle
import numpy as np
import time
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
parser.add_argument('--tfb', dest='tfb', action='store_true')
parser.add_argument('--no-tfb', dest='tfb', action='store_false')
parser.set_defaults(tfb=True)
parser.add_argument('--requestsize', default=25e9, type=float)
parser.add_argument('--numrequests', default=2200, type=int)
parser.add_argument('--holdingtime', default=1e8, type=float)
parser.add_argument('--load', default=1e10, type=float)
parser.add_argument('--evaldir', default="", type=str)
parser.add_argument('--topology', default="nsfnet_chen_5-paths", type=str)
parser.add_argument('--setup_id', default="", type=str)
parser.add_argument('--node_req_prbs', default='nsfnet_chen_gravity', type=str)
args = parser.parse_args()
n_eval_eps = args.n_eval_eps
exp_id = args.exp_id
env_id = args.env_id
termonfirstblock = args.tfb
requestsize = args.requestsize
numrequests = args.numrequests
holdingtime = args.holdingtime
load = args.load
evaldir = args.evaldir
topology_name = args.topology
setup_id = args.setup_id
node_req_prbs = args.node_req_prbs
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
#with open(current_directory+'/topologies/nsfnet_chen_5-paths.h5', 'rb') as f:
with open(current_directory+'/topologies/'+topology_name+'.h5', 'rb') as f:
    topology = pickle.load(f)
# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = pickle.load(open("node_req_probs/"+node_req_prbs+".pkl",'rb'))
env_args = dict(topology=topology, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=holdingtime, # value is not set as in the paper to achieve comparable reward values
                episode_length=numrequests, node_request_probabilities=node_request_probabilities, exp_request_res=requestsize,
                term_on_first_block=termonfirstblock)
eval_dir = "./tmp/RWAFOCS-ppo/"+setup_id+"heuristics_eval/eval_results"+evaldir+"/"
os.makedirs(eval_dir, exist_ok=True)

env_1 = gym.make('RWAFOCS-v22', **env_args)
env_1 = Monitor(env_1, eval_dir + 'evaluation_kspff', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput','service_distribution'))
env_2 = gym.make('RWAFOCS-v22', **env_args)
env_2 = Monitor(env_2, eval_dir + 'evaluation_ffksp', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput','service_distribution'))
env_3 = gym.make('RWAFOCS-v22', **env_args)
env_3 = Monitor(env_3, eval_dir + 'evaluation_camu', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput','service_distribution'))
env_4 = gym.make('RWAFOCS-v22', **env_args)
env_4 = Monitor(env_4, eval_dir + 'evaluation_kspmu', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput','service_distribution'))

# 2022-01-18tbf10Mdict
# model_dir = "/Users/joshnevin/RL_FOCSLab/tmp/RWAFOCS-ppo/"+exp_id+"/_core_0/"

#env = DummyVecEnv([make_env(env_args, log_dir)])

start_kspff = time.time()
mean_reward_kspff, std_reward_kspff = evaluate_heuristic(env_1, kSP_FF, n_eval_episodes=n_eval_eps,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
end_kspff = time.time()
start_ffksp = time.time()
mean_reward_ffksp, std_reward_ffksp = evaluate_heuristic(env_2, FF_kSP, n_eval_episodes=n_eval_eps,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
end_ffksp = time.time()
start_camu = time.time()
mean_reward_camu, std_reward_camu = evaluate_heuristic(env_3, CA_MU, n_eval_episodes=n_eval_eps,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
end_camu = time.time()
start_kspmu = time.time()
mean_reward_kspmu, std_reward_kspmu = evaluate_heuristic(env_4, kSP_MU, n_eval_episodes=n_eval_eps,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False)
end_kspmu = time.time()


pickle.dump(end_kspff-start_kspff, open("eval_timing/"+exp_id+"_kspff.pkl", 'wb'))
pickle.dump(end_ffksp-start_ffksp, open("eval_timing/"+exp_id+"_ffksp.pkl", 'wb'))
pickle.dump(end_camu-start_camu, open("eval_timing/"+exp_id+"_camu.pkl", 'wb'))
pickle.dump(end_kspmu-start_kspmu, open("eval_timing/"+exp_id+"_kspmu.pkl", 'wb'))
