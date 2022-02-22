import time
from itertools import islice
import networkx as nx
import numpy as np
import GN_model


class Path:

    def __init__(self, path_id, node_list, weight, length, best_modulation=None, lightpaths=None):
        self.path_id = path_id
        self.node_list = node_list
        self.length = length
        self.best_modulation = best_modulation
        self.hops = len(node_list) - 1
        if not lightpaths:
            lightpaths = {}
        self.lightpaths = lightpaths
        self.weight = weight



class Service:

    def __init__(self, service_id, source, source_id, destination=None, destination_id=None, arrival_time=None,
                 holding_time=None, bit_rate=25e9, best_modulation=None, service_class=None, number_slots=None):
        self.service_id = service_id
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.source = source
        self.source_id = source_id
        self.destination = destination
        self.destination_id = destination_id
        self.bit_rate = bit_rate
        self.service_class = service_class
        self.best_modulation = best_modulation
        self.number_slots = number_slots
        self.route = None
        self.initial_slot = None
        self.accepted = False
        self.wavelength = None
        self.new_lp = None

    def foobar(self):
        return "foobar"

    def __str__(self):
        msg = '{'
        msg += '' if self.bit_rate is None else f'br: {self.bit_rate}, '
        msg += '' if self.service_class is None else f'cl: {self.service_class}, '
        return f'Serv. {self.service_id} ({self.source} -> {self.destination})' + msg


def start_environment(env, steps):
    done = True
    for i in range(steps):
        if done:
            env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
    return env


def get_k_shortest_paths(G, source, target, k, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight='length'):
    return np.sum([graph[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)])


def random_policy(env):
    return env.action_space.sample()


def evaluate_heuristic(env, heuristic, n_eval_episodes=10,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False):
    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action = heuristic(env)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: ' \
                                               '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

class LightPath:
    def __init__(self, channel_id, available_capacity):
        self.channel_id = channel_id
        self.available_capacity = available_capacity

# class network_nsr:
#     def __init__(self,link_nsrs):
#         self.link_nsrs = link_nsrs
#
# class link_nsr:
#     def __init__(self,wavelegth_nsrs):
#         self.wavelength_nsrs = wavelegth_nsrs
#
# def initialise_worst_case_nsr(env,graph):
#     """method to initalise nsr considering worst case """
#     number_of_wavelengths=100
#     link_nsrs = list()
#     for edge in enumerate(graph.edges()):
#         link_length = edge.length
#         wavelength_nsrs = list()
#         for wavelength in range(number_of_wavelengths):
#             wavelen_nsr = gn_model.calculate_per_channel_nsr_for_link(link_length, wavelength)
#             wavelength_nsrs[wavelength] = wavelen_nsr
#             lnsr = link_nsr(wavelength_nsrs)
#         link_nsrs[edge] = lnsr
#     env.nsrs = network_nsr(link_nsrs)
#
#
# def add_wavelen_nsr(env,link_id,link_length,wavelen_id):
#     """method to be called when a new wavelength is been allocated for a new service"""
#     wavelen_nsr = gn_model.calculate_per_channel_nsr_for_link(link_length,wavelen_id)
#     link_nsr = env.nsrs.link_nsrs
#     link_nsr = link_nsr + wavelen_nsr
#     env.topology.graph.nsrs[link_id] = link_nsr
#     #calculate the wavelengths given the light
#
# def remove_wavelen_nsr(env,link_id,link_length,wavelen_id):
#     """method to be called when an existing wavelength is been released - lightpath count for the wavelength becomes 0 using the wavelength"""
#     wavelen_nsr = gn_model.calculate_per_channel_nsr_for_link(link_length, wavelen_id)
#     link_nsr = env.link_nsrs[link_id]
#     link_nsr = link_nsr - wavelen_nsr
#     env.topology.graph.nsrs[link_id] = link_nsr










