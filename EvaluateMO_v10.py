import os
import pickle
import time

import numpy as np

from IPython.display import clear_output

import matplotlib
import matplotlib.pyplot as plt
import csv
import pandas as pd
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
#from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# import stable_baselines
# from stable_baselines.common.callbacks import BaseCallback
# from stable_baselines.results_plotter import load_results, ts2xy
# from stable_baselines import PPO2
# from stable_baselines.bench import Monitor
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines import results_plotter
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines.common.vec_env import DummyVecEnv
#stable_baselines.__version__ # printing out stable_baselines version used
import gym
import pickle
import cProfile
import pandas as pd
#from optical_rl_gym.heuristics import kSP_FF, FF_kSP, kSP_MU, CA_MU
#from optical_rl_gym.utils import evaluate_heuristic, random_policy


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

def main():
    current_directory = os.getcwd()
    with open('/home/sam/extern/Workspace/opticalGym/examples/topologies/nsfnet_chen_5-paths_rounded.h5', 'rb') as f:
        topology = pickle.load(f)
    # node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
    #node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
    #  0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
    #   0.07607608, 0.12012012, 0.01901902, 0.16916917])
    node_request_probabilities = pickle.load(open("../node_req_probs/" + "14_node_uniform" + ".pkl", 'rb'))

    #
    load = int(1e10)
    env_args_9 = dict(topology=topology, seed=10, load=load,
                      allow_rejection=False,  # the agent cannot proactively reject a request
                      mean_service_holding_time=1e8,  # value is not set as in the paper to achieve comparable reward values
                      episode_length=10000, node_request_probabilities=node_request_probabilities,
                      exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=0.1,weight_failure_score=0.8,weight_transmitter=0.1)

    env_args_10 = dict(topology=topology, seed=10, load=load,
                       allow_rejection=False,  # the agent cannot proactively reject a request
                       mean_service_holding_time=1e8,
                       # value is not set as in the paper to achieve comparable reward values
                       episode_length=10000, node_request_probabilities=node_request_probabilities,
                       exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=0.8,weight_failure_score=0.1,weight_transmitter=0.1)

    env_args_8 = dict(topology=topology, seed=10, load=load,
                      allow_rejection=False,  # the agent cannot proactively reject a request
                      mean_service_holding_time=1e8,
                      # value is not set as in the paper to achieve comparable reward values
                      episode_length=10000, node_request_probabilities=node_request_probabilities,
                      exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=0.1,weight_failure_score=0.1,weight_transmitter=0.8)

    env_args_7 = dict(topology=topology, seed=10, load=load,
                      allow_rejection=False,  # the agent cannot proactively reject a request
                      mean_service_holding_time=1e8,
                      # value is not set as in the paper to achieve comparable reward values
                      episode_length=10000, node_request_probabilities=node_request_probabilities,
                      exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=0.6,weight_failure_score=0.2,weight_transmitter=0.2)

    env_args_6 = dict(topology=topology, seed=10, load=load,
                      allow_rejection=False,  # the agent cannot proactively reject a request
                      mean_service_holding_time=1e8,
                      # value is not set as in the paper to achieve comparable reward values
                      episode_length=10000, node_request_probabilities=node_request_probabilities,
                      exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=0.2,weight_failure_score=0.6,weight_transmitter=0.2)

    env_args_5 = dict(topology=topology, seed=10, load=load,
                      allow_rejection=False,  # the agent cannot proactively reject a request
                      mean_service_holding_time=1e8,
                      # value is not set as in the paper to achieve comparable reward values
                      episode_length=10000, node_request_probabilities=node_request_probabilities,
                      exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=0.2,weight_failure_score=0.2,weight_transmitter=0.6)

    env_args_4 = dict(topology=topology, seed=10, load=load,
                      allow_rejection=False,  # the agent cannot proactively reject a request
                      mean_service_holding_time=1e8,
                      # value is not set as in the paper to achieve comparable reward values
                      episode_length=10000, node_request_probabilities=node_request_probabilities,
                      exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=0.45,weight_failure_score=0.45,weight_transmitter=0.1)

    env_args_3 = dict(topology=topology, seed=10, load=load,
                      allow_rejection=False,  # the agent cannot proactively reject a request
                      mean_service_holding_time=1e8,
                      # value is not set as in the paper to achieve comparable reward values
                      episode_length=10000, node_request_probabilities=node_request_probabilities,
                      exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=0.1,weight_failure_score=0.45,weight_transmitter=0.45)

    env_args_2 = dict(topology=topology, seed=10, load=load,
                      allow_rejection=False,  # the agent cannot proactively reject a request
                      mean_service_holding_time=1e8,
                      # value is not set as in the paper to achieve comparable reward values
                      episode_length=10000, node_request_probabilities=node_request_probabilities,
                      exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=0.33,weight_failure_score=0.33,weight_transmitter=0.33)

    env_args_1 = dict(topology=topology, seed=10, load=load,
                      allow_rejection=False,  # the agent cannot proactively reject a request
                      mean_service_holding_time=1e8,
                      # value is not set as in the paper to achieve comparable reward values
                      episode_length=10000, node_request_probabilities=node_request_probabilities,
                      exp_request_res=100e9, term_on_first_block=False, num_spectrum_resources=100,weight_service=1.0,weight_failure_score=0.0,weight_transmitter=0.0)
    #model_dir_opt = "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-06-04v7nsfUp5100U"
    #model_dir_opt = ""
    # model_dir_10= "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-07-08v8_0.9"
    # model_dir_9 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-07-08v8_0.9"
    # model_dir_8 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-07-09v7nsfUp5100U_0.8"
    # model_dir_7 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-07-08v8_0.7"
    # model_dir_6 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-07-08v8_0.6"
    # model_dir_5 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-07-08v8_0.5"
    # model_dir_4 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-07-08v8_0.4"
    # model_dir_3 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-07-08v8_0.3"
    # model_dir_2 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/RWAFOCS-ppo/2022-07-08v8_0.2"


    model_dir_10 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_0.8_0.1_0.1"
    model_dir_9 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_0.1_0.8_0.1"
    model_dir_8 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_0.1_0.1_0.8"
    model_dir_7 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_0.6_0.2_0.2"
    model_dir_6 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_0.2_0.6_0.2"
    model_dir_5 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_0.2_0.2_0.6"
    model_dir_4 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_0.45_0.45_0.1"
    model_dir_3 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_0.1_0.45_0.45"
    model_dir_2 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_0.33_0.33_0.33"
    model_dir_1 = "/home/sam/extern/Workspace/opticalGym/examples/tmp/nsfnet_gpu_test/v10_1.0_0.0_0.0"

    eval_dir_9 = model_dir_9 + "/eval_results/"
    os.makedirs(eval_dir_9, exist_ok=True)
    eval_dir_10 = model_dir_10 + "/eval_results/"
    os.makedirs(eval_dir_10, exist_ok=True)
    eval_dir_8 = model_dir_8 + "/eval_results/"
    os.makedirs(eval_dir_8, exist_ok=True)
    eval_dir_7 = model_dir_7 + "/eval_results/"
    os.makedirs(eval_dir_7, exist_ok=True)
    eval_dir_6 = model_dir_6 + "/eval_results/"
    os.makedirs(eval_dir_6, exist_ok=True)
    eval_dir_5 = model_dir_5 + "/eval_results/"
    os.makedirs(eval_dir_5, exist_ok=True)
    eval_dir_4 = model_dir_4 + "/eval_results/"
    os.makedirs(eval_dir_4, exist_ok=True)
    eval_dir_3 = model_dir_3 + "/eval_results/"
    os.makedirs(eval_dir_3, exist_ok=True)
    eval_dir_2 = model_dir_2 + "/eval_results/"
    os.makedirs(eval_dir_2, exist_ok=True)
    eval_dir_1 = model_dir_1 + "/eval_results/"
    os.makedirs(eval_dir_1, exist_ok=True)

    #     env_id_9 = 'RWAFOCS-v481'
    # # env_id_opt = 'RWAFOCS-v7'
    #
    #     env_9 = gym.make(env_id_9, **env_args_9)
    #     env_9 = Monitor(env_9, eval_dir_9 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                              'throughput','failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_9 = MaskablePPO.load(model_dir_9 + '/best_model')
    #
    #     env_id_10 = 'RWAFOCS-v481'
    #     # env_id_opt = 'RWAFOCS-v7'
    #
    #     env_10 = gym.make(env_id_10, **env_args_10)
    #     env_10 = Monitor(env_10, eval_dir_10 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                                'throughput', 'failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_10 = MaskablePPO.load(model_dir_10 + '/best_model')
    #
    #     env_id_8 = 'RWAFOCS-v481'
    #     env_8 = gym.make(env_id_8, **env_args_8)
    #     env_8 = Monitor(env_8, eval_dir_8 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                          'throughput','failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_8 = MaskablePPO.load(model_dir_8 + '/best_model')
    #
    #
    #
    #     env_id_7 = 'RWAFOCS-v481'
    #     env_7 = gym.make(env_id_7, **env_args_7)
    #     env_7 = Monitor(env_7, eval_dir_7 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                          'throughput','failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_7 = MaskablePPO.load(model_dir_7 + '/best_model')
    #
    #     env_id_6 = 'RWAFOCS-v481'
    #     env_6 = gym.make(env_id_6, **env_args_6)
    #     env_6 = Monitor(env_6, eval_dir_6 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                          'throughput','failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_6 = MaskablePPO.load(model_dir_6 + '/best_model')
    #
    #     env_id_5 = 'RWAFOCS-v481'
    #     env_5 = gym.make(env_id_5, **env_args_5)
    #     env_5 = Monitor(env_5, eval_dir_5 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                       'throughput','failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_5 = MaskablePPO.load(model_dir_5 + '/best_model')
    #
    #     env_id_4 = 'RWAFOCS-v481'
    #     env_4 = gym.make(env_id_4, **env_args_4)
    #     env_4 = Monitor(env_4, eval_dir_4 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                       'throughput','failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_4 = MaskablePPO.load(model_dir_4 + '/best_model')
    #
    #     env_id_3 = 'RWAFOCS-v481'
    #     env_3 = gym.make(env_id_3, **env_args_3)
    #     env_3 = Monitor(env_3, eval_dir_3 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                       'throughput','failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_3 = MaskablePPO.load(model_dir_3 + '/best_model')
    #
    #     env_id_2 = 'RWAFOCS-v481'
    #     env_2 = gym.make(env_id_2, **env_args_2)
    #     env_2 = Monitor(env_2, eval_dir_2 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                       'throughput','failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_2 = MaskablePPO.load(model_dir_2 + '/best_model')
    #
    #     env_id_1 = 'RWAFOCS-v481'
    #     env_1 = gym.make(env_id_1, **env_args_1)
    #     env_1 = Monitor(env_1, eval_dir_1 + 'evaluation_rand', info_keywords=('episode_services_accepted',
    #                                                                       'throughput','failure_score','per_link_service_distributioin','transmitters','service_distribution'))
    #     model_1 = MaskablePPO.load(model_dir_1 + '/best_model')
    #
    env_id_9 = 'RWAFOCS-v481'
    # env_id_opt = 'RWAFOCS-v7'

    env_9 = gym.make(env_id_9, **env_args_9)
    env_9 = Monitor(env_9, eval_dir_9 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                          'throughput', 'failure_score',
                                                                          'service_distribution'))
    model_9 = MaskablePPO.load(model_dir_9 + '/best_model')

    env_id_10 = 'RWAFOCS-v481'
    # env_id_opt = 'RWAFOCS-v7'

    env_10 = gym.make(env_id_10, **env_args_10)
    env_10 = Monitor(env_10, eval_dir_10 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                             'throughput', 'failure_score',
                                                                             'service_distribution'))
    model_10 = MaskablePPO.load(model_dir_10 + '/best_model')

    env_id_8 = 'RWAFOCS-v481'
    env_8 = gym.make(env_id_8, **env_args_8)
    env_8 = Monitor(env_8, eval_dir_8 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                          'throughput', 'failure_score',
                                                                          'service_distribution'))
    model_8 = MaskablePPO.load(model_dir_8 + '/best_model')

    env_id_7 = 'RWAFOCS-v481'
    env_7 = gym.make(env_id_7, **env_args_7)
    env_7 = Monitor(env_7, eval_dir_7 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                          'throughput', 'failure_score',
                                                                          'service_distribution'))
    model_7 = MaskablePPO.load(model_dir_7 + '/best_model')

    env_id_6 = 'RWAFOCS-v481'
    env_6 = gym.make(env_id_6, **env_args_6)
    env_6 = Monitor(env_6, eval_dir_6 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                          'throughput', 'failure_score',
                                                                          'service_distribution'))
    model_6 = MaskablePPO.load(model_dir_6 + '/best_model')

    env_id_5 = 'RWAFOCS-v481'
    env_5 = gym.make(env_id_5, **env_args_5)
    env_5 = Monitor(env_5, eval_dir_5 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                          'throughput', 'failure_score',
                                                                          'service_distribution'))
    model_5 = MaskablePPO.load(model_dir_5 + '/best_model')

    env_id_4 = 'RWAFOCS-v481'
    env_4 = gym.make(env_id_4, **env_args_4)
    env_4 = Monitor(env_4, eval_dir_4 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                          'throughput', 'failure_score',
                                                                          'service_distribution'))
    model_4 = MaskablePPO.load(model_dir_4 + '/best_model')

    env_id_3 = 'RWAFOCS-v481'
    env_3 = gym.make(env_id_3, **env_args_3)
    env_3 = Monitor(env_3, eval_dir_3 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                          'throughput', 'failure_score',
                                                                          'service_distribution'))
    model_3 = MaskablePPO.load(model_dir_3 + '/best_model')

    env_id_2 = 'RWAFOCS-v481'
    env_2 = gym.make(env_id_2, **env_args_2)
    env_2 = Monitor(env_2, eval_dir_2 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                          'throughput', 'failure_score',
                                                                          'service_distribution'))
    model_2 = MaskablePPO.load(model_dir_2 + '/best_model')

    env_id_1 = 'RWAFOCS-v481'
    env_1 = gym.make(env_id_1, **env_args_1)
    env_1 = Monitor(env_1, eval_dir_1 + 'evaluation_rand', info_keywords=('episode_services_accepted',
                                                                          'throughput', 'failure_score',
                                                                          'service_distribution'))
    model_1 = MaskablePPO.load(model_dir_1 + '/best_model')


    n_eval_eps = 1

    episode_rewards_10, episode_lengths_10 = evaluate_policy(model_10, env_10, n_eval_episodes=n_eval_eps,
                                                         deterministic=True, return_episode_rewards=True)

    episode_rewards_9, episode_lengths_9 = evaluate_policy(model_9, env_9, n_eval_episodes=n_eval_eps,deterministic =True,return_episode_rewards=True)
    episode_rewards_8, episode_lengths_8 = evaluate_policy(model_8, env_8, n_eval_episodes=n_eval_eps,deterministic =True,return_episode_rewards=True)
    episode_rewards_7, episode_lengths_7 = evaluate_policy(model_7, env_7, n_eval_episodes=n_eval_eps,deterministic =True,return_episode_rewards=True)
    episode_rewards_6, episode_lengths_6 = evaluate_policy(model_6, env_6, n_eval_episodes=n_eval_eps,deterministic =True,return_episode_rewards=True)
    episode_rewards_5, episode_lengths_5 = evaluate_policy(model_5, env_5, n_eval_episodes=n_eval_eps,deterministic =True,return_episode_rewards=True)
    episode_rewards_4, episode_lengths_4 = evaluate_policy(model_4, env_4, n_eval_episodes=n_eval_eps,deterministic =True,return_episode_rewards=True)
    episode_rewards_3, episode_lengths_3 = evaluate_policy(model_3, env_3, n_eval_episodes=n_eval_eps,deterministic =True,return_episode_rewards=True)
    episode_rewards_2, episode_lengths_2 = evaluate_policy(model_2, env_2, n_eval_episodes=n_eval_eps,deterministic =True,return_episode_rewards=True)
    episode_rewards_1, episode_lengths_1 = evaluate_policy(model_1, env_1, n_eval_episodes=n_eval_eps,deterministic =True,return_episode_rewards=True)






if __name__ == '__main__':
    main()
