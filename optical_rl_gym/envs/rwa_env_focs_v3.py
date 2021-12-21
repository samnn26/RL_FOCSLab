from typing import Sequence
import gym
from collections import defaultdict
import heapq
import logging
import numpy as np
import matplotlib.pyplot as plt

import GN_model
from optical_rl_gym.utils import Service, Path, LightPath
from .optical_network_env import OpticalNetworkEnv
import pdb


class RWAEnvFOCSV3(OpticalNetworkEnv):

    metadata = {
        'metrics': ['service_blocking_rate', 'episode_service_blocking_rate']
    }

    def __init__(self, topology=None,
                 episode_length=1000,
                 load=10,
                 mean_service_holding_time=1e100,
                 num_spectrum_resources=100,
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
        for now spectrum_wavelengths_allocation is removed - if we want to track this we need to
        allow for multiple service IDs per wavelength per edge...
        """
        # self.max_services_allocation = 1000 # define this to be large enough to never be exceeded
        # self.spectrum_wavelengths_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources, self.max_services_allocation),
        #  fill_value=-1, dtype=np.int)
        # array that tracks how many services are allocated to each lightpath, indexed by path ID and wavelength
        self.lightpath_service_allocation = np.zeros([self.topology.number_of_nodes()*
        (self.topology.number_of_nodes()-1)*self.k_paths, self.num_spectrum_resources], dtype=int)
        self.num_transmitters = np.zeros(self.topology.number_of_nodes(),)
        self.num_receivers = np.zeros(self.topology.number_of_nodes(),)
        self.episode_num_transmitters = np.zeros(self.topology.number_of_nodes(),)
        self.episode_num_receivers = np.zeros(self.topology.number_of_nodes(),)
        self.reject_action = 1 if allow_rejection else 0

        self.actions_output = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        self.episode_actions_output = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        self.num_lightpaths_reused = 0
        self.episode_num_lightpaths_reused = 0

        self.num_lightpaths_released = 0  # tracks every time we call _release_path
        self.episode_num_lightpaths_released = 0
        self.actions_taken = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        """
        can probably leave this as-is and overwrite it with an action space wrapper, as Carlos does below...
        """
        self.action_space = gym.spaces.MultiDiscrete((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action))

        nodes = self.topology.number_of_nodes()
        number_of_bitrates = 100 # test
        self.observation_space= gym.spaces.MultiDiscrete((number_of_bitrates,nodes,nodes))
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rwaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)
        self.initialise_lightpath_capacities()
    """
    Method that represents a step into the environment, i.e., the provisioning (or rejection) of a service request.
    The action parameter is a is a sequence with two elements, the first representing the path index, and the second representing the wavelength.
    """
    """I don't know what actions_output does - why is it incremented here?"""
    """same for episode_actions"""


    def update_available_lightpath_capacity(self, path, channel_ind, capacity_allocated, provision):

        ligthpath = path.lightpaths[channel_ind]
        if provision:
            new_capacity = ligthpath.available_capacity - capacity_allocated/1e12 # convert bps to Tbps
        else:  # if we are releasing
            new_capacity = ligthpath.available_capacity + capacity_allocated/1e12 # convert bps to Tbps
        ligthpath.available_capacity = new_capacity
        #print("available capacity updated for lightpath ", channel_id, " new capacity ", ligthpath.available_capacity, " Tbps")

    def get_available_lightpath_capacity(self, source, dest, path_ind, channel_ind):

        p = self.k_shortest_paths[source, dest][path_ind]
        c = p.lightpaths[channel_ind]
        c_bps = c.available_capacity*1e12
        #print("available capacity for ligthpath ", channel_id, " is ", c_bps, "bps")
        return c_bps #converted to bps(from Tbps)

    def initialise_lightpath_capacities(self):
        # access through the channels of k shortest paths and initialise to max capacity
        nch = 101  # call a method in gn model to retriev this
        channel_capacities = None
        for idn1, n1 in enumerate(self.topology.nodes()):
            for idn2, n2 in enumerate(self.topology.nodes()):
                if idn1 != idn2:
                    for path in range(self.k_paths):
                        p = self.k_shortest_paths[n1, n2][path]
                        for ch in range(nch):
                            capacity = GN_model.calculate_lightpath_capacity(p.length,ch)
                            ligthpath = LightPath(ch, capacity)
                            p.lightpaths[ch] = ligthpath

    def step(self, action: Sequence[int]):

        path, wavelength = action[0], action[1]
        self.actions_output[path, wavelength] += 1
        self.episode_actions_output[path, wavelength] += 1
        if path < self.k_paths and wavelength < self.num_spectrum_resources:  # if the indices are within the bounds
            if self.is_lightpath_free(self.k_shortest_paths[self.service.source, self.service.destination][path],
            wavelength) and self.get_available_lightpath_capacity(self.service.source, self.service.destination,
            path, wavelength) > self.service.bit_rate:  # if path is free and has sufficient capacity
                self.num_transmitters[int(self.service.source)-1] += 1  # only for new lightpaths do we need to count these
                self.num_receivers[int(self.service.destination)-1] += 1
                self.episode_num_transmitters[int(self.service.source)-1] += 1
                self.episode_num_receivers[int(self.service.destination)-1] += 1
                self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path], wavelength)

                self.service.accepted = True
                self.services_accepted += 1
                self.episode_services_accepted += 1

                self.actions_taken[path, wavelength] += 1
                self.episode_actions_taken[path, wavelength] += 1
                self._add_release(self.service)

            elif self.does_lightpath_exist(self.k_shortest_paths[self.service.source, self.service.destination][path],
            wavelength) and self.get_available_lightpath_capacity(self.service.source, self.service.destination,
            path, wavelength) > self.service.bit_rate:
                self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path], wavelength)
                self.num_lightpaths_reused += 1
                self.episode_num_lightpaths_reused += 1
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
            self.actions_taken[self.k_paths, self.num_spectrum_resources] += 1  # dont understand this...

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
        if only_counters:
            print("true")
        else:
            print("false")

        self.episode_actions_output = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + 1,
                                        self.num_spectrum_resources + 1), dtype=int)
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_num_transmitters = np.zeros(self.topology.number_of_nodes())
        self.episode_num_receivers = np.zeros(self.topology.number_of_nodes())
        self.episode_num_lightpaths_reused = 0
        self.episode_num_lightpaths_released = 0
        if only_counters:
            return self.observation()

        # if not only counters, the whole environment needs to be reset
        super().reset()
        self.initialise_lightpath_capacities()
        """
        Old version of available wavelengths: array that stores the state of each wavelength on each edge, 1=available, 0=used
        New version of available wavelengths: array that stores the number of services on each wavelength on each edge.
        """
        #
        self.topology.graph['available_wavelengths'] = np.zeros((self.topology.number_of_edges(), self.num_spectrum_resources), dtype=int)

        self.lightpath_service_allocation = np.zeros([self.topology.number_of_nodes()*
        (self.topology.number_of_nodes()-1)*self.k_paths, self.num_spectrum_resources], dtype=int)
        # self.spectrum_wavelengths_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources, self.max_services_allocation),
        #                                          fill_value=-1, dtype=np.int)
        # saving statistics
        self.actions_output = np.zeros((self.k_paths + self.reject_action,
                                        self.num_spectrum_resources + self.reject_action), dtype=int)
        self.actions_taken = np.zeros((self.k_paths + 1,
                                        self.num_spectrum_resources + 1), dtype=int)
        self.num_transmitters = np.zeros(self.topology.number_of_nodes())
        self.num_receivers = np.zeros(self.topology.number_of_nodes())
        self.num_lightpaths_reused = 0
        self.num_lightpaths_released = 0
        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human', close=False):
        fig = plt.figure(figsize=(18, 12))

        # plt.subplot(1, 3, 1)
        # plt.pcolor(self.spectrum_wavelengths_allocation.transpose(), cmap=plt.cm.Greys, edgecolors='gray', linewidth=.01)
        # plt.ylabel('Wavelength index')
        # plt.xlabel('Edge index')

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
        #at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        at = self.current_time + self.mean_service_inter_arrival_time
        self.current_time = at

        #ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        ht = self.mean_service_holding_time
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

        return [self.service.bit_rate,self.service.source_id,self.service.destination_id]
    """
    self.observation_space = spaces.Tuple(( spaces.Discrete(self.k_paths ),
     spaces.Discrete(self.k_paths ), spaces.Discrete(self.num_spectrum_resources)))

    """

    """
    need to modify provision path method to allow for multiple service IDs on one channel
    """

    def _provision_path(self, path: Path, wavelength: int):

        self.lightpath_service_allocation[path.path_id, wavelength] += 1
        for i in range(len(path.node_list) - 1):
            self.topology.graph['available_wavelengths'][self.topology[path.node_list[i]][path.node_list[i + 1]]['index'], wavelength] += 1
            # ind = next(x for x, val in enumerate(self.spectrum_wavelengths_allocation[self.topology[path.node_list[i]][path.node_list[i + 1]]['index'], wavelength]) if x != -1)
            # try:
            #     self.spectrum_wavelengths_allocation[self.topology[path.node_list[i]][path.node_list[i + 1]]['index'], wavelength][ind] = self.service.service_id  # add service id to first slot
            # except:
            #     raise ValueError("Ran out of service id slots, increase this value")

            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(self.service.service_id)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['service_wavelengths'].append(wavelength)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(self.service.service_id)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_service_wavelengths'].append(wavelength)
            self._update_link_stats(path.node_list[i], path.node_list[i + 1])

        self.update_available_lightpath_capacity(path, wavelength, self.service.bit_rate, True)
        self.topology.graph['running_services'].append(self.service.service_id)
        self.topology.graph['running_service_wavelengths'].append(wavelength)
        self.service.wavelength = wavelength
        self._update_network_stats()
        self.service.route = path


    def _release_path(self, service: Service):

        self.num_lightpaths_released += 1
        self.lightpath_service_allocation[service.route.path_id, service.wavelength] -= 1
        for i in range(len(service.route.node_list) - 1):
            self.topology.graph['available_wavelengths'][self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'], service.wavelength] -= 1
            try:
                ind_edge = self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].index(service.service_id)
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service.service_id)
                del self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_service_wavelengths'][ind_edge]
            except:
                self.logger.warning('error')
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])
        try:
            ind_top = self.topology.graph['running_services'].index(service.service_id)
            self.topology.graph['running_services'].remove(service.service_id)
            del self.topology.graph['running_service_wavelengths'][ind_top]
        except:
            self.logger.warning('error')
        self.update_available_lightpath_capacity(service.route, service.wavelength, self.service.bit_rate, False)
        self.num_transmitters[int(service.source)-1] -= 1
        self.num_receivers[int(service.destination)-1] -= 1
        self.episode_num_transmitters[int(service.source)-1] -= 1
        self.episode_num_receivers[int(service.destination)-1] -= 1

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

            # cur_util = (self.num_spectrum_resources - np.sum(
            #     self.topology.graph['available_wavelengths'][self.topology[node1][node2]['index'], :])) / \
            #            self.num_spectrum_resources
            # utilisation = fraction of unused wavelengths on edge - definition used by Robert
            cur_util = (self.num_spectrum_resources -  np.count_nonzero(self.topology.graph['available_wavelengths']
                    [self.topology[node1][node2]['index'], :])) / self.num_spectrum_resources
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization

        self.topology[node1][node2]['last_update'] = self.current_time

    """
    we want to call is_lightpath_free for each of the k-shortest paths on each wavelength in turn - only if none are free
    do we then move on to the next wavelength
    """

    def is_lightpath_free(self, path: Path, wavelength: int) -> bool:
         # if wavelength is out of range, return false
        if wavelength > self.num_spectrum_resources:
            return False
        # checks over all links if the wavelength is available
        for i in range(len(path.node_list) - 1):
            if self.topology.graph['available_wavelengths'][
                      self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                      wavelength] != 0: # 0 means completely unoccupied
                return False
        return True


    def does_lightpath_exist(self, path: Path, wavelength: int) -> bool:

        if self.lightpath_service_allocation[path.path_id, wavelength] != 0:
            return True
        else:
            return False

"""
not sure what this function is doing... it seems to be defined to be used in least_loaded_path_first_fit
"""
def get_path_capacity(env: RWAEnvFOCSV3, path: Path) -> int:
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
we need to modify the tracking of the paths, i.e. is_lightpath_free will become is_path_capacity_sufficient() ...
this is equivalent to Robert's FF-SP algo. - needs another loop over the k paths to make it FF-kSP
"""
def shortest_path_first_fit(env: RWAEnvFOCSV3) -> Sequence[int]:
    for wavelength in range(env.num_spectrum_resources):
        if env.is_lightpath_free(env.k_shortest_paths[env.service.source, env.service.destination][0], wavelength):
            return (0, wavelength)
    # if no path is found, return out-of-bounds indices
    return (env.k_paths, env.num_spectrum_resources)

"""
this is equivalent to kSP-FF algo in Robert's paper...
"""
def shortest_available_path_first_fit(env: RWAEnvFOCSV3) -> Sequence[int]:
    best_hops = np.finfo(0.0).max  # in this case, shortest means least hops
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        if path.hops < best_hops:  # if path is shorter
            # checks all wavelengths
            for wavelength in range(env.num_spectrum_resources):
                if env.is_lightpath_free(path, wavelength):  # if wavelength is found
                    # stores decision and breaks the wavelength loop (first fit)
                    best_hops = path.hops
                    decision = (idp, wavelength)
                    break
    return decision

"""
this performs kSP-LF, i.e. it chooses the wavelength slot for a given path starting with the last slot and scanning backwards
"""
def shortest_available_path_last_fit(env: RWAEnvFOCSV3) -> Sequence[int]:
    best_hops = np.finfo(0.0).max  # in this case, shortest means least hops
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        if path.hops < best_hops:  # if path is shorter
            # checks all wavelengths from the highest to the lowest index
            for wavelength in range(env.num_spectrum_resources-1, 0, -1):
                if env.is_lightpath_free(path, wavelength):  # if wavelength is found
                    # stores decision and breaks the wavelength loop (first fit)
                    best_hops = path.hops
                    decision = (idp, wavelength)
                    break
    return decision


def least_loaded_path_first_fit(env: RWAEnvFOCSV3) -> Sequence[int]:
    best_load = np.finfo(0.0).min
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        cap = get_path_capacity(env, path)
        if cap > best_load:
            # checks all wavelengths
            for wavelength in range(env.num_spectrum_resources):
                if env.is_lightpath_free(path, wavelength):  # if wavelength is found
                    # stores decision and breaks the wavelength loop (first fit)
                    best_load = cap
                    decision = (idp, wavelength)
                    break
    return decision


class PathOnlyFirstFitAction(gym.ActionWrapper):

    def __init__(self, env: RWAEnvFOCSV3):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.env.k_paths + self.env.reject_action)
        self.observation_space = env.observation_space

    """
    This method transforms an action that only selected the path, into an action that selects the path and the first-fit wavelength.
    """
    def action(self, action: int) -> Sequence[int]:
        if action < self.env.k_paths:
            for wavelength in range(self.env.num_spectrum_resources):
                if self.env.is_lightpath_free(self.env.topology.graph['ksp'][self.env.service.source, self.env.service.destination][action], wavelength):  # if wavelength is found
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
#                 if self.env.is_lightpath_free(self.env.topology.graph['ksp'][self.env.service.source, self.env.service.destination][path], wavelength):
#                     viable_paths.append(path) # save the paths that are viable (will modify this later to be capacity-based)
#             if len(viable_paths) > 0: # if at least one k shortest path for this wavelength
#                 return {'viablekSPs': self.topology.graph['ksp'][self.env.service.source, self.env.service.destination][viable_paths],
#                         'service': self.service}
#         if len(viable_paths) == 0: # if none of the wavelengths actually fit
#             raise ValueError('Ran out of wavelengths!')  # need to work out how to handle this - it should block...


# def observation(self):
#     return {'topology': self.topology,
#             'service': self.service}
