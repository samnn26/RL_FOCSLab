import os
import pickle
import numpy as np

from IPython.display import clear_output

import matplotlib
#import config InlineBackend.figure_format = 'svg'
import tensorflow as tf

# silencing tensorflow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from datetime import datetime

# tf.__version__ # printint out tensorflow version used
import sb3_contrib
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.results_plotter import load_results, ts2xy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from sb3_contrib import MaskablePPO
# from stable_baselines3.bench import Monitor
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common import results_plotter
#stable_baselines.__version__ # printing out stable_baselines version used
import gym
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


# callback from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True

def make_env(env_id, env_args, log_dir):
    def maker():
        env = gym.make(env_id, **env_args)
        env = Monitor(env, log_dir + 'training', info_keywords=('episode_services_accepted',
        'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
        'episode_cum_services_processed', 'throughput'))
        return env
    return maker

def make_env_multiproc(env_id, rank, env_args, log_dirs, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, **env_args)
        env = Monitor(env, log_dirs[rank] + 'training', info_keywords=('episode_services_accepted',
        'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
        'episode_cum_services_processed', 'throughput'))
        env.seed(seed + rank)
        return env
    #set_global_seeds(seed)
    return _init

def main():
    # loading the topology binary file containing the graph and the k-shortest paths
    current_directory = os.getcwd()
    # with open(current_directory+'/topologies/nsfnet_chen_5-paths_directional.h5', 'rb') as f:
    #     topology = pickle.load(f)
    # with open(current_directory+'/topologies/3_node_network_sym.h5', 'rb') as f:
    #     topology = pickle.load(f)
    # node_request_probabilities = np.array([0.333333,0.333333,0.333333])
    with open(f'/Users/joshnevin/RL_FOCSLab/topologies/nsfnet_chen_5-paths.h5', 'rb') as f:
        topology = pickle.load(f)
    node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
           0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
           0.07607608, 0.12012012, 0.01901902, 0.16916917])

    load = int(1e10)
    env_args = dict(topology=topology, seed=10, load = load,
                    allow_rejection=False, # the agent cannot proactively reject a request
                    mean_service_holding_time=1e8, # value is not set as in the paper to achieve comparable reward values
                    episode_length=1600, node_request_probabilities=node_request_probabilities,
                    exp_request_res=100e9, exp_request_lambda=1, term_on_first_block=False, num_spectrum_resources=100)

    # Create log dir
    today = datetime.today().strftime('%Y-%m-%d')
    exp_num = "_v41learnconttest"
    continue_training = False
    number_of_cores = 1
    envID = 'RWAFOCS-v42'
    if continue_training:
        log_dirs = []
        for i in range(number_of_cores):
            log_dirs.append("./tmp/RWAFOCS-ppo/"+today+exp_num+"/_core_"+str(i)+"/")
            os.makedirs("./tmp/RWAFOCS-ppo/"+today+exp_num+"/_core_"+str(i)+"/", exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dirs[0])
        pickle.dump(env_args, open(log_dirs[0] + "env_args.pkl", 'wb'))
        env = SubprocVecEnv([make_env_multiproc(envID, i, env_args, log_dirs) for i in range(number_of_cores)])
        # env = DummyVecEnv([make_env()])
        model_dir = "./tmp/RWAFOCS-ppo/2022-01-31v41f25Gexpbid/_core_0"
        agent = MaskablePPO.load(model_dir+'/best_model')
        agent.set_env(env)
        a = agent.learn(total_timesteps=int(1000), callback=callback)
        pickle.dump(env_args, open(log_dir + "env_args.pkl", 'wb'))
        # env_print = env.envs[0] # get environment from DummyVecEnv
        # print("Whole training process statistics:")
        # rnd_path_action_probability = np.sum(env_print.actions_output, axis=1) / np.sum(env_print.actions_output)
        # rnd_wavelength_action_probability = np.sum(env_print.actions_output, axis=0) / np.sum(env_print.actions_output)
        # print('Path action probability:', np.sum(env_print.actions_output, axis=1) / np.sum(env_print.actions_output))
        # print('Wavelength action probability:', np.sum(env_print.actions_output, axis=0) / np.sum(env_print.actions_output))
        # print('Load (Erlangs):', load)
        # print('Last service bit rate (Gb/s):', env_print.service.bit_rate/1e9)
        # print('Total number of services:', env_print.services_processed)
        # print('Total number of accepted services:', env_print.services_accepted)
        # print('Blocking probability:', 1 - env_print.services_accepted/env_print.services_processed)
        # print('Number of services on existing lightpaths:', env_print.num_lightpaths_reused)
        # print('Number of services released:', env_print.num_lightpaths_released)
        # print('Number of transmitters on each node:', env_print.num_transmitters)
        # print('Number of receivers on each node:', env_print.num_receivers)
        # print('Final throughput (TB/s):', env_print.get_throughput()/1e12)

    else:
        if number_of_cores == 1:
            log_dir = "./tmp/RWAFOCS-ppo/"+today+exp_num+"/"
            os.makedirs(log_dir, exist_ok=True)
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
            env = DummyVecEnv([make_env(envID, env_args, log_dir)])
            net_arch = 2*[64]  # default for MlpPolicy
            policy_args = dict(net_arch=net_arch) # we use the elu activation function
            print(envID)
            # agent = MaskablePPO('MlpPolicy', env, verbose=0, tensorboard_log="./tb/PPO-RWA-v0/", policy_kwargs=policy_args, gamma=.95, learning_rate=10e-5)
            # agent = MaskablePPO('MlpPolicy', env, verbose=0, policy_kwargs=policy_args, gamma=.95, learning_rate=10e-5)
            agent = MaskablePPO('MultiInputPolicy', env, verbose=0, policy_kwargs=policy_args, gamma=.95, learning_rate=10e-5)
            a = agent.learn(total_timesteps=1000, callback=callback)
            #results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "RWA")
            pickle.dump(env_args, open(log_dir + "env_args.pkl", 'wb'))
            env_print = env.envs[0]
            print("Whole training process statistics:")
            #rnd_lightpath_action_probability = env_print.actions_output / env_print.services_processed
            print('Lightpath action probability:', env_print.actions_output / env_print.services_processed)

            num_lps_reused = env_print.num_lightpaths_reused
            print('Load (Erlangs):', load)
            print('Last service bit rate (Gb/s):', env_print.service.bit_rate/1e9)
            print('Total number of services:', env_print.services_processed)
            print('Total number of accepted services:', env_print.services_accepted)
            print('Blocking probability:', 1 - env_print.services_accepted/env_print.services_processed)
            print('Number of services on existing lightpaths:', num_lps_reused)
            print('Number of services released:', env_print.num_lightpaths_released)
            print('Number of transmitters on each node:', env_print.num_transmitters)
            print('Number of receivers on each node:', env_print.num_receivers)
            print('Final throughput (TB/s):', env_print.get_throughput()/1e12)
            print('Episode number of no valid state events:', env_print.episode_no_valid_actions)
            print('Number of no valid state events:', env_print.no_valid_actions)
        else:
            log_dirs = []
            for i in range(number_of_cores):
                log_dirs.append("./tmp/RWAFOCS-ppo/"+today+exp_num+"/_core_"+str(i)+"/")
                os.makedirs("./tmp/RWAFOCS-ppo/"+today+exp_num+"/_core_"+str(i)+"/", exist_ok=True)
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dirs[0])
            pickle.dump(env_args, open(log_dirs[0] + "env_args.pkl", 'wb'))
            env = SubprocVecEnv([make_env_multiproc(envID, i, env_args, log_dirs) for i in range(number_of_cores)])
            print(envID)
            net_arch = 2*[64]  # default for MlpPolicy
            policy_args = dict(net_arch=net_arch)
            agent = MaskablePPO('MultiInputPolicy', env, verbose=0, policy_kwargs=policy_args, gamma=.95, learning_rate=10e-5)
            a = agent.learn(total_timesteps=2000, callback=callback)
if __name__ == '__main__':
    main()
