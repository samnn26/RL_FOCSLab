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
#import sb3_contrib
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.results_plotter import load_results, ts2xy
from stable_baselines3.common.results_plotter import load_results, ts2xy
# from stable_baselines3.bench import Monitor
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
#from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3 import PPO
#from sb3_contrib import MaskablePPO
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
parser.add_argument('--requestsize', default=100e9, type=float)
parser.add_argument('--numrequests', default=2500, type=int)
parser.add_argument('--holdingtime', default=1e8, type=float)
parser.add_argument('--load', default=1e10, type=float)
parser.add_argument('--evaldir', default="", type=str)
parser.add_argument('--topology', default="nsfnet_chen_5-paths", type=str)
parser.add_argument('--node_req_prbs', default="nsfnet_chen_gravity", type=str)
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
with open(current_directory+'/topologies/'+topology_name+'.h5', 'rb') as f:
    topology = pickle.load(f)
# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = pickle.load(open("node_req_probs/"+node_req_prbs+".pkl",'rb'))
env_args = dict(topology=topology, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=holdingtime, # value is not set as in the paper to achieve comparable reward values
                episode_length=numrequests, node_request_probabilities=node_request_probabilities, exp_request_res=requestsize,
                term_on_first_block=termonfirstblock)
model_dir = "./tmp/RWAFOCS-ppo/"+exp_id+"/_core_0/"
#eval_dir = model_dir + "eval_results/"
eval_dir = model_dir + "eval_results"+evaldir+"/"
os.makedirs(eval_dir, exist_ok=True)

env = gym.make(env_id, **env_args)
env = Monitor(env, eval_dir + 'evaluation_rl', info_keywords=('episode_services_accepted',
'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
'episode_cum_services_processed', 'throughput', 'service_distribution'))

model = PPO.load(model_dir+'best_model')

mean_reward_rl, std_reward_rl = evaluate_policy(model, env, n_eval_episodes=n_eval_eps, deterministic = False)

