import os
import pickle
import numpy as np

from IPython.display import clear_output

import matplotlib
#import config InlineBackend.figure_format = 'svg'
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
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
from sb3_contrib.common.maskable.evaluation import evaluate_policy
# from stable_baselines3.bench import Monitor
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common import results_plotter
#stable_baselines.__version__ # printing out stable_baselines version used
import gym
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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

def make_env(env_id, env_args):
    def maker():
        env = gym.make(env_id, **env_args)
        # env = Monitor(env, log_dir + 'training', info_keywords=('episode_services_accepted',
        # 'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
        # 'episode_cum_services_processed', 'throughput'))
        return env
    return maker

def make_env_multiproc(env_id, rank, env_args, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, **env_args)
        # env = Monitor(env, log_dirs[rank] + 'training', info_keywords=('episode_services_accepted',
        # 'episode_services_processed', 'services_accepted', 'services_processed', 'episode_cum_services_accepted',
        # 'episode_cum_services_processed', 'throughput'))
        env.seed(seed + rank)
        return env
    #set_global_seeds(seed)
    return _init

def mppo_params(trial):
    #n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1.)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    #ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    #clip_range = trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3, 0.4])
    #noptepochs = trial.suggest_categorical('noptepochs', [1, 5, 10, 20, 30, 50])
    #lam = trial.suggest_categorical('lambda', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    number_of_layers = trial.suggest_categorical('num_layers', [1,2,3,4])
    number_of_neurons =  trial.suggest_categorical('num_neurons', [16,32,64,128])
    policy_kwargs = dict(net_arch=number_of_layers*[number_of_neurons])
    return{
        #'n_steps': n_steps,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        #'ent_coef': ent_coef,
        #'clip_range': cliprange,
        #'noptepochs': noptepochs,
        #'lam': lam,
        'policy_kwargs': policy_kwargs
    }

def optimize_agent(trial):

    model_params = mppo_params(trial)
    envs = DummyVecEnv([make_env('RWAFOCS-v4',env_args) for _ in range(number_of_cores)])
    #env = SubprocVecEnv([make_env_multiproc('RWAFOCS-v4', i, env_args) for i in range(number_of_cores)])
    # env = gym.make('RWAFOCS-v4', **env_args)
    model = MaskablePPO('MlpPolicy', envs, **model_params)
    model.learn(int(1e2))
    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=1, deterministic = False)
    return -1 * mean_reward

current_directory = os.getcwd()
with open(current_directory+'/topologies/nsfnet_chen_5-paths_directional.h5', 'rb') as f:
    topology = pickle.load(f)
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
               0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
               0.07607608, 0.12012012, 0.01901902, 0.16916917])
load = int(1e10)
env_args = dict(topology=topology, seed=10, load = load,
            allow_rejection=False, # the agent cannot proactively reject a request
            mean_service_holding_time=1e8, # value is not set as in the paper to achieve comparable reward values
            episode_length=1600, node_request_probabilities=node_request_probabilities,
            exp_request_res=25e9, exp_request_lambda=1, term_on_first_block=False, num_spectrum_resources=100)

number_of_cores = 2
study_name = '1'
def main():
    study = optuna.create_study()
    #study.optimize(optimize_agent, n_trials=1, n_jobs = number_of_cores)  # n_jobs = -1 will make it equal to CPU count
    with ThreadPoolExecutor(max_workers=number_of_cores) as executor:
        for _ in range(number_of_cores):
            executor.submit(study.optimize, optimize_agent, 1)
    pickle.dump(study, open("optuna_results/study"+study_name+".pkl", 'wb'))
if __name__ == '__main__':
    main()
