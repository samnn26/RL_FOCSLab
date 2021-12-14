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

tf.__version__ # printint out tensorflow version used
import stable_baselines
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import results_plotter
stable_baselines.__version__ # printing out stable_baselines version used
import gym
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



# loading the topology binary file containing the graph and the k-shortest paths
# if you want to generate your own binary topology file, check examples/create_topology_rmsa.py
# topology_name = 'nsfnet_chen'
# k_paths = 5
# with open(f'topologies/{topology_name}_{k_paths}-paths.h5', 'rb') as f:
#     topology = pickle.load(f)
with open(f'/Users/joshnevin/RL_FOCSLab/topologies/nsfnet_chen_5-paths_directional.h5', 'rb') as f:
    topology = pickle.load(f)

# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
       0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
       0.07607608, 0.12012012, 0.01901902, 0.16916917])

# mean_service_holding_time=7.5,
env_args = dict(topology=topology, seed=10, load = 1e10,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=7.5, # value is not set as in the paper to achieve comparable reward values
                episode_length=50, node_request_probabilities=node_request_probabilities)
# breakpoint()
# Create log dir
log_dir = "./tmp/RWAFOCS-ppo/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

#env = gym.make('DeepRMSA-v0', **env_args)
env = gym.make('RWAFOCS-v0', **env_args)
# logs will be saved in log_dir/training.monitor.csv
# in this case, on top of the usual monitored things, we also monitor service and bit rate blocking rates
env = Monitor(env, log_dir + 'training', info_keywords=('episode_service_blocking_rate','service_blocking_rate'))
# for more information about the monitor, check https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/bench/monitor.html#Monitor

# here goes the arguments of the policy network to be used
policy_args = dict(net_arch=5*[128], # the neural network has five layers with 128 neurons each
                   act_fun=tf.nn.elu) # we use the elu activation function

agent = PPO2(MlpPolicy, env, verbose=0, tensorboard_log="./tb/PPO-RWA-v0/", policy_kwargs=policy_args, gamma=.95, learning_rate=10e-5)

a = agent.learn(total_timesteps=4000, callback=callback)
results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "RWA")
