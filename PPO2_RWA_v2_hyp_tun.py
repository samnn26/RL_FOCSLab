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

tf.__version__ # printint out tensorflow version used
import stable_baselines
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv

#stable_baselines.__version__ # printing out stable_baselines version used
import gym
import pickle
import optuna
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

def make_env():
    def maker():
        env = gym.make('RWAFOCS-v2', **env_args)
        return env
    return maker


def ppo2_params(trial):
    n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1.)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    cliprange = trial.suggest_categorical('cliprange', [0.1, 0.2, 0.3, 0.4])
    noptepochs = trial.suggest_categorical('noptepochs', [1, 5, 10, 20, 30, 50])
    lam = trial.suggest_categorical('lambda', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    number_of_layers = trial.suggest_categorical('num_layers', [2,3,4,5,6,7,8,9,10])
    number_of_neurons =  trial.suggest_categorical('num_neurons', [32,64,128,256])
    policy_kwargs = dict(net_arch=number_of_layers*[number_of_neurons],
                       act_fun=tf.nn.elu)
    return{
        'n_steps': n_steps,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'cliprange': cliprange,
        'noptepochs': noptepochs,
        'lam': lam,
        'policy_kwargs': policy_kwargs
    }

def optimize_agent(trial):
    n_training_envs = 1
    model_params = ppo2_params(trial)
    # envs = DummyVecEnv([make_env() for _ in range(n_training_envs)])
    env = gym.make('RWAFOCS-v2', **env_args)
    model = PPO2('MlpPolicy', env, nminibatches=n_training_envs, **model_params)
    model.learn(int(1e2))
    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=1, deterministic = False)
    return -1 * mean_reward

# loading the topology binary file containing the graph and the k-shortest paths
# if you want to generate your own binary topology file, check examples/create_topology_rmsa.py
current_directory = os.getcwd()
with open(current_directory+'/topologies/nsfnet_chen_5-paths_directional.h5', 'rb') as f:
    topology = pickle.load(f)
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
           0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
           0.07607608, 0.12012012, 0.01901902, 0.16916917])
# with open(current_directory+'/topologies/3_node_network.h5', 'rb') as f:
#     topology = pickle.load(f)
# node_request_probabilities = np.array([0.333333,0.333333,0.333333])

load = 1000
env_args = dict(topology=topology, seed=10, load = load,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=10, # value is not set as in the paper to achieve comparable reward values
                episode_length=50, node_request_probabilities=node_request_probabilities, exp_request_res=25e9, exp_request_lambda=1)

n_training_envs = 1
study = optuna.create_study()
study.optimize(optimize_agent, n_trials=1, n_jobs = n_training_envs)
