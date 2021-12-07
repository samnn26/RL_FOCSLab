from typing import Sequence
import gym

import heapq
import logging
import numpy as np
import matplotlib.pyplot as plt

from optical_rl_gym.utils import Service, Path
from .optical_network_env import OpticalNetworkEnv


class RWAEnv(OpticalNetworkEnv):

    metadata = {
        'metrics': ['service_blocking_rate', 'episode_service_blocking_rate']
    }

    def __init__(self, topology=None,
                 episode_length=1000,
                 load=10,
                 mean_service_holding_time=10800.0,
                 num_spectrum_resources=80,
                 node_request_probabilities=None,
                 allow_rejection=True,
                 k_paths=5,
                 seed=None, reset=True):
        super().__init__(topology=topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed,
                         k_paths=k_paths)

        # vector that stores the service IDs for which the wavelengths are allocated to
        """
        need to change this to allow for multiple service IDs per wavelength - could do as a 2d vector, although this may mean
        we need to define a maximum number of serices per wavelength, which would have to be large enough that it wasn't exceeded...
        """
        self.spectrum_wavelengths_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                 fill_value=-1, dtype=np.int)

        self.reject_action = 1 if allow_rejection else 0

        self.actions_output = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        self.episode_actions_output = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        """
        don't understand why it is + 1 here, rather than self.reject_action. What is the difference?
        """
        self.actions_taken = np.zeros((self.k_paths + 1,
                                        self.num_spectrum_resources + 1), dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + 1,
                                        self.num_spectrum_resources + 1), dtype=int)
        """
        can probably leave this as-is and overwrite it with an action space wrapper, as Carlos does below...
        """
        self.action_space = gym.spaces.MultiDiscrete((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action))
        """
        Again I don't understand the 10 here - what are the 10 things in the space here?
        """
        self.observation_space = gym.spaces.Dict(
            {'topology': gym.spaces.Discrete(10),
             'current_service': gym.spaces.Discrete(10)}
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rwaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    """
    Method that represents a step into the environment, i.e., the provisioning (or rejection) of a service request.
    The action parameter is a is a sequence with two elements, the first representing the path index, and the second representing the wavelength.
    """
    """I don't know what actions_output does - why is it incremented here?"""
    """same for episode_actions"""
    def step(self, action: Sequence[int]):
        """
        steps 1-5 in Algorithm 1   (note: split into two parts to ensure tracking is still present)
        for kpath in self.k_shortest_paths[self.service.source, self.service.destination]:  # for all kSPs between source and destination
            for wavelen in range(self.num_spectrum_resources): # need to search each wavelength on each path
                if self.is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path], wavelength):
                    # if the path is free, then check the capacity - for now we can assume this is first-fit (another agent could even choose this?)
                    if get_path_capacity(self.k_shortest_paths[self.service.source, self.service.destination][path], wavelength) > self.service.bit_rate:
                        self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path], wavelength)

        """
        path, wavelength = action[0], action[1]
        self.actions_output[path, wavelength] += 1
        self.episode_actions_output[path, wavelength] += 1
        if path < self.k_paths and wavelength < self.num_spectrum_resources:  # if the indices are within the bounds
            if self.is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path], wavelength):
                self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path], wavelength)
                self.service.accepted = True
                self.services_accepted += 1
                self.episode_services_accepted += 1

                self.actions_taken[path, wavelength] += 1
                self.episode_actions_taken[path, wavelength] += 1
                self._add_release(self.service)
            else:
                self.service.accepted = False
        else:
            self.service.accepted = False

        if not self.service.accepted:
            self.actions_taken[self.k_paths, self.num_spectrum_resources] += 1

        self.services_processed += 1
        self.episode_services_processed += 1

        self.topology.graph['services'].append(self.service)

        reward = self.reward()
        info = {
            'service_blocking_rate': (self.services_processed - self.services_accepted) / self.services_processed,
            'episode_service_blocking_rate': (self.episode_services_processed - self.episode_services_accepted) / self.episode_services_processed,
            'path_action_probability': np.sum(self.actions_output, axis=1) / np.sum(self.actions_output),
            'wavelength_action_probability': np.sum(self.actions_output, axis=0) / np.sum(self.actions_output)
        }

        self._new_service = False
        self._next_service()

        return self.observation(), reward, self.episode_services_processed == self.episode_length, info

    def reset(self, only_counters=True):
        # resetting counters for the episode
        self.episode_actions_output = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + 1,
                                        self.num_spectrum_resources + 1), dtype=int)
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        if only_counters:
            return self.observation()

        # if not only counters, the whole environment needs to be reset
        super().reset()
        """
        This needs to be modified - the state of each wavelength will no longer be binary, but is represented by a capacity
        """
        # vector that stores the state of each wavelength, 1=available, 0=used
        self.topology.graph["available_wavelengths"] = np.ones((self.topology.number_of_edges(), self.num_spectrum_resources), dtype=int)

        self.spectrum_wavelengths_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                 fill_value=-1, dtype=np.int)

        # saving statistics
        self.actions_output = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        self.actions_taken = np.zeros((self.k_paths + 1,
                                        self.num_spectrum_resources + 1), dtype=int)
        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human', close=False):
        fig = plt.figure(figsize=(18, 12))

        plt.subplot(1, 3, 1)
        plt.pcolor(self.spectrum_wavelengths_allocation.transpose(), cmap=plt.cm.Greys, edgecolors='gray', linewidth=.01)
        plt.ylabel('Wavelength index')
        plt.xlabel('Edge index')

        plt.subplot(1, 3, 2)
        source_destination_representation = np.zeros((self.topology.number_of_nodes(), 2))
        source_destination_representation[self.service.source_id, 0] = 1
        source_destination_representation[self.service.destination_id, 1] = 1
        plt.pcolor(source_destination_representation, cmap=plt.cm.Greys, edgecolors='none', linewidth=.01)
        # plt.xlabel('edge')
        plt.ylabel('node')
        plt.xticks([0.5, 1.5], ['src', 'dst'], rotation=90)

        plt.subplot(1, 3, 3)
        paths = np.zeros((self.k_paths, self.topology.number_of_edges()))
        for idp, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
            for i in range(len(path.node_list) - 1):
                paths[idp, self.topology[path.node_list[i]][path.node_list[i + 1]]['index']] = 1
        plt.pcolor(paths, cmap=plt.cm.Greys, edgecolors='none', linewidth=.01)
        plt.xlabel('path')
        plt.ylabel('Edge index')

        plt.tight_layout()
        plt.show()
        # plt.savefig('./repr.svg')
        plt.close()

    def _next_service(self):
        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the look

        self.service = Service(self.episode_services_processed, src, src_id, destination=dst, destination_id=dst_id,
                               arrival_time=at, holding_time=ht, number_slots=1)
        self._new_service = True

    def observation(self):
        return {'topology': self.topology,
                'service': self.service}
    """
    self.observation_space = spaces.Tuple(( spaces.Discrete(self.k_paths ),
     spaces.Discrete(self.k_paths ), spaces.Discrete(self.num_spectrum_resources)))

    """

    """
    need to modify provision path method to allow for multiple service IDs on one channel
    """

    def _provision_path(self, path: Path, wavelength: int):
        # usage
        if not self.is_path_free(path, wavelength):
            raise ValueError("Wavelength {} of Path {} is not free".format(wavelength, path.node_list))

        for i in range(len(path.node_list) - 1):
            self.topology.graph['available_wavelengths'][self.topology[path.node_list[i]][path.node_list[i + 1]]['index'], wavelength] = 0
            self.spectrum_wavelengths_allocation[self.topology[path.node_list[i]][path.node_list[i + 1]]['index'], wavelength] = self.service.service_id
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(self.service.service_id)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(self.service.service_id)
            self._update_link_stats(path.node_list[i], path.node_list[i + 1])
        self.topology.graph['running_services'].append(self.service.service_id)
        self.service.wavelength = wavelength
        self._update_network_stats()
        self.service.route = path

    def _release_path(self, service: Service):
        for i in range(len(service.route.node_list) - 1):
            self.topology.graph['available_wavelengths'][self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'], service.wavelength] = 1
            self.spectrum_wavelengths_allocation[self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'], service.wavelength] = -1
            try:
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service.service_id)
            except:
                self.logger.warning('error')
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])
        try:
            self.topology.graph['running_services'].remove(service.service_id)
        except:
            self.logger.warning('error')

    def _update_network_stats(self):
        """
        Implement here any network-wide statistics
        :return:
        """
        pass
        # last_update = self.topology.graph['last_update']
        # time_diff = self.current_time - last_update
        # if self.current_time > 0:
        #     for service in self.topology.graph["running_services"]:
        #         cur_throughtput += service.bit_rate
        #     utilization = ((last_throughput * last_update) + (cur_throughtput * time_diff)) / self.current_time
        #     self.topology.graph['throughput'] = utilization

    def _update_link_stats(self, node1, node2):
        last_update = self.topology[node1][node2]['last_update']
        time_diff = self.current_time - self.topology[node1][node2]['last_update']
        if self.current_time > 0:
            last_util = self.topology[node1][node2]['utilization']
            cur_util = (self.num_spectrum_resources - np.sum(
                self.topology.graph['available_wavelengths'][self.topology[node1][node2]['index'], :])) / \
                       self.num_spectrum_resources
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization

        self.topology[node1][node2]['last_update'] = self.current_time

    """
    we want to call is_path_free for each of the k-shortest paths on each wavelength in turn - only if none are free
    do we then move on to the next wavelength
    """
    def is_path_free(self, path: Path, wavelength: int) -> bool:
        # if wavelength is out of range, return false
        if wavelength > self.num_spectrum_resources:
            return False

        # checks over all links if the wavelength is available
        for i in range(len(path.node_list) - 1):
            if self.topology.graph['available_wavelengths'][
                      self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                      wavelength] == 0:
                return False  # if not available, return False
        return True


"""
not sure what this function is doing... it seems to be defined to be used in least_loaded_path_first_fit
"""
def get_path_capacity(env: RWAEnv, path: Path) -> int:
    capacity = 0
    # checks all wavelengths to see which ones are available
    for wavelength in range(env.num_spectrum_resources):
        available = True  # starts assuming wavelength is available
        # tries to find whether at least one is not available
        for i in range(len(path.node_list) - 1):
            # if not available in this link
            if env.topology.graph['available_wavelengths'][env.topology[path.node_list[i]][path.node_list[i + 1]]['index'], wavelength] == 0:
                available = False  # sets available to false
                break  # stops iteration
        if available:  # if available over all links
            capacity += 1  # increments
    return capacity

"""
this is what we need to do - loop first over the wavelengths and then get the agent to choose the path, after we check its availability
we need to modify the tracking of the paths, i.e. is_path_free will become is_path_capacity_sufficient() ...
this is equivalent to Robert's FF-SP algo. - needs another loop over the k paths to make it FF-kSP
"""
def shortest_path_first_fit(env: RWAEnv) -> Sequence[int]:
    for wavelength in range(env.num_spectrum_resources):
        if env.is_path_free(env.k_shortest_paths[env.service.source, env.service.destination][0], wavelength):
            return (0, wavelength)
    # if no path is found, return out-of-bounds indices
    return (env.k_paths, env.num_spectrum_resources)

"""
this is equivalent to kSP-FF algo in Robert's paper...
"""
def shortest_available_path_first_fit(env: RWAEnv) -> Sequence[int]:
    best_hops = np.finfo(0.0).max  # in this case, shortest means least hops
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        if path.hops < best_hops:  # if path is shorter
            # checks all wavelengths
            for wavelength in range(env.num_spectrum_resources):
                if env.is_path_free(path, wavelength):  # if wavelength is found
                    # stores decision and breaks the wavelength loop (first fit)
                    best_hops = path.hops
                    decision = (idp, wavelength)
                    break
    return decision

"""
this performs kSP-LF, i.e. it chooses the wavelength slot for a given path starting with the last slot and scanning backwards
"""
def shortest_available_path_last_fit(env: RWAEnv) -> Sequence[int]:
    best_hops = np.finfo(0.0).max  # in this case, shortest means least hops
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        if path.hops < best_hops:  # if path is shorter
            # checks all wavelengths from the highest to the lowest index
            for wavelength in range(env.num_spectrum_resources-1, 0, -1):
                if env.is_path_free(path, wavelength):  # if wavelength is found
                    # stores decision and breaks the wavelength loop (first fit)
                    best_hops = path.hops
                    decision = (idp, wavelength)
                    break
    return decision


def least_loaded_path_first_fit(env: RWAEnv) -> Sequence[int]:
    best_load = np.finfo(0.0).min
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        cap = get_path_capacity(env, path)
        if cap > best_load:
            # checks all wavelengths
            for wavelength in range(env.num_spectrum_resources):
                if env.is_path_free(path, wavelength):  # if wavelength is found
                    # stores decision and breaks the wavelength loop (first fit)
                    best_load = cap
                    decision = (idp, wavelength)
                    break
    return decision


class PathOnlyFirstFitAction(gym.ActionWrapper):

    def __init__(self, env: RWAEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.env.k_paths + self.env.reject_action)
        self.observation_space = env.observation_space

    """
    This method transforms an action that only selected the path, into an action that selects the path and the first-fit wavelength.
    """
    def action(self, action: int) -> Sequence[int]:
        if action < self.env.k_paths:
            for wavelength in range(self.env.num_spectrum_resources):
                if self.env.is_path_free(self.env.topology.graph['ksp'][self.env.service.source, self.env.service.destination][action], wavelength):  # if wavelength is found
                    return (action, wavelength)
        return (self.env.k_paths, self.env.num_spectrum_resources)  # do this if the action is not one of the k paths

    def step(self, action):
        return self.env.step(self.action(action))

# class FirstFitPathOnlyObservation(gym.ObservationWrapper):
#
#     def __init__(self, env: RWAEnv):
#         super().__init__(env)
#         self.observation_space = env.observation_space
#
#     def observation(self, obs):
#         for wavelength in range(self.env.num_spectrum_resources):  # scan over wavelengths to find the first available
#             viable_paths = []
#             for path in range(self.k_paths):
#                 if self.env.is_path_free(self.env.topology.graph['ksp'][self.env.service.source, self.env.service.destination][path], wavelength):
#                     viable_paths.append(path) # save the paths that are viable (will modify this later to be capacity-based)
#             if len(viable_paths) > 0: # if at least one k shortest path for this wavelength
#                 return {'viablekSPs': self.topology.graph['ksp'][self.env.service.source, self.env.service.destination][viable_paths],
#                         'service': self.service}
#         if len(viable_paths) == 0: # if none of the wavelengths actually fit
#             raise ValueError('Ran out of wavelengths!')  # need to work out how to handle this - it should block...


# def observation(self):
#     return {'topology': self.topology,
#             'service': self.service}
