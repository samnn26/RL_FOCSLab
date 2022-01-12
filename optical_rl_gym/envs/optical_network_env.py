import gym
import copy
import heapq
import random
import numpy as np
import networkx as nx

import GN_model
from optical_rl_gym.utils import Service, Path, get_k_shortest_paths, get_path_weight
from GN_model import calculate_capacity

class OpticalNetworkEnv(gym.Env):



    def __init__(self, topology=None, episode_length=1000, load=10, mean_service_holding_time=100.0,
                 num_spectrum_resources=100, allow_rejection=False,
                 node_request_probabilities=None, seed=None, k_paths=5):
        """
        Initalises the optical network env
        :param topology: the input network topology
        :param episode_length: the length of the episode RL agent is running
        :param load: the input traffic load in Erlang
        :param mean_service_holding_time: the mean of the exponential distribution holding time is generated from
        :param num_spectrum_resources: number of spectrum slots/channels per each link
        :param allow_rejection: whether we allow proative rejection
        :param node_request_probabilities:  the probabilities of input traffic at each node-this can be used to generate different traffic distributions
        :param seed: random seed for the run
        :param k_paths: number of shortest paths per each source destination pair
        """
        assert topology is None or 'ksp' in topology.graph
        assert topology is None or 'k_paths' in topology.graph
        self._events = []
        self.current_time = 0
        self.episode_length = episode_length
        self.services_processed = 0
        self.services_accepted = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        self.current_service = None
        self._new_service = False
        self.allow_rejection = allow_rejection

        self.load = 0
        self.mean_service_holding_time = 0
        self.mean_service_inter_arrival_time = 0
        self.set_load(load=load, mean_service_holding_time=mean_service_holding_time)
        self.service = None

        self.rand_seed = None
        self.rng = None
        self.seed(seed=seed)


        if topology is None:
            # defines a dummy topology
            self.topology_name = 'simple'
            self.topology = nx.Graph()
            self.topology.add_node("A", pos=(0, 1))
            self.topology.add_node("B", pos=(1, 2))
            self.topology.add_node("C", pos=(1, 0))
            self.topology.add_node("D", pos=(2, 2))
            self.topology.add_node("E", pos=(2, 0))
            self.topology.add_node("F", pos=(3, 1))
            self.topology.add_edge("A", "B", index=0, weight=1, length=200)
            self.topology.add_edge("A", "C", index=1, weight=1, length=500)
            self.topology.add_edge("B", "C", index=2, weight=1, length=600)
            self.topology.add_edge("B", "D", index=3, weight=1, length=700)
            self.topology.add_edge("C", "E", index=4, weight=1, length=300)
            self.topology.add_edge("D", "E", index=5, weight=1, length=200)
            self.topology.add_edge("D", "F", index=6, weight=1, length=400)
            self.topology.add_edge("E", "F", index=6, weight=1, length=500)
            self.topology.graph["node_indices"] = []

            for idx, node in enumerate(self.topology.nodes()):
                self.topology.graph["node_indices"].append(node)
            k_shortest_paths = {}
            idp = 0
            for idn1, n1 in enumerate(self.topology.nodes()):
                for idn2, n2 in enumerate(self.topology.nodes()):
                    if idn1 < idn2:
                        paths = get_k_shortest_paths(self.topology, n1, n2, k_paths)
                       # print(n1, n2, len(paths))
                        lengths = [get_path_weight(self.topology, path) for path in paths]
                        objs = []
                        for path, length in zip(paths, lengths):
                            objs.append(Path(idp, path, length))
                            idp += 1
                        k_shortest_paths[n1, n2] = objs
                        k_shortest_paths[n2, n1] = objs

            self.topology.graph["ksp"] = k_shortest_paths
            self.k_shortest_paths = k_shortest_paths
            self.topology.graph['k_paths'] = k_paths
            self.k_paths = k_paths
        else:
            self.topology = copy.deepcopy(topology)
            self.topology_name = topology.graph['name']
            self.k_paths = self.topology.graph['k_paths']
            self.k_shortest_paths = self.topology.graph['ksp']  # just as a more convenient way to access it
            #print capacities######################################################################################################
            #print("source id, destination id, path length, physical layer capacity")
            for idn1, n1 in enumerate(self.topology.nodes()):
                for idn2, n2 in enumerate(self.topology.nodes()):
                    if (n1 != n2):
                        paths = self.k_shortest_paths[n1, n2]
                        for i in range(len(paths)):
                            path = paths[i]
                            path_capacity = GN_model.calculate_capacity(path.length)
                            path_capacity_rounded = "{:.2f}".format(path_capacity)
                            #print(n1, ",", n2, ",", path.length, ",", path_capacity_rounded)
#################################################################################################################################
        assert node_request_probabilities is None or len(node_request_probabilities) == self.topology.number_of_nodes()
        self.num_spectrum_resources = num_spectrum_resources
        self.topology.graph['num_spectrum_resources'] = num_spectrum_resources
        self.topology.graph['available_spectrum'] = np.full((self.topology.number_of_edges()),
                                                            fill_value=self.num_spectrum_resources,
                                                            dtype=int)
        if node_request_probabilities is not None:
            self.node_request_probabilities = node_request_probabilities
        else:
            self.node_request_probabilities = np.full((self.topology.number_of_nodes()),
                                                      fill_value=1. / self.topology.number_of_nodes())


    def set_load(self, load=None, mean_service_holding_time=None):
        """
        Sets the load to be used to generate requests.
        :param load: The load to be generated, in Erlangs
        :param mean_service_holding_time: The mean service holding time to be used to generate the requests
        :return: None
        """
        if load is not None:
            self.load = load
        if mean_service_holding_time is not None:
            self.mean_service_holding_time = mean_service_holding_time  # current_service holding time in seconds
        self.mean_service_inter_arrival_time = 1 / float(self.load / float(self.mean_service_holding_time))

    def _plot_topology_graph(self, ax):
        """
        this method plots the topology graph,
        :param ax: refers to Matplotlib Axes object, optional Draw the graph in the specified Matplotlib axes.
        """
        pos = nx.get_node_attributes(self.topology, 'pos')
        nx.draw_networkx_edges(self.topology, pos, ax=ax)
        nx.draw_networkx_nodes(self.topology, pos,
                               nodelist=[x for x in self.topology.nodes() if x in [self.current_service.source, self.current_service.destination]],
                               label=[x for x in self.topology.nodes()],
                               node_shape='s',
                               node_color='white', edgecolors='black', ax=ax)
        nx.draw_networkx_nodes(self.topology, pos,
                               nodelist=[x for x in self.topology.nodes() if
                                         x not in [self.current_service.source, self.current_service.destination]],
                               label=[x for x in self.topology.nodes()],
                               node_shape='o',
                               node_color='white', edgecolors='black', ax=ax)
        nx.draw_networkx_labels(self.topology, pos)
        nx.draw_networkx_edge_labels(self.topology, pos,
                                     edge_labels={(i, j): '{}'.format(self.available_spectrum[self.topology[i][j]['index']]) for i, j in self.topology.edges()})
        # TODO: implement a trigger (a flag) that tells whether to plot the edge labels
        # set also an universal label dictionary inside the edge dictionary, e.g., (self.topology[a][b]['plot_label']

    def _add_release(self, service: Service):
        """
        Adds an event to the event list of the simulator.
        This implementation is based on the functionalities of heapq: https://docs.python.org/2/library/heapq.html

        :param event:
        :return: None
        """
        heapq.heappush(self._events, (service.arrival_time + service.holding_time, service))

    def _get_node_pair(self):
        """
        Uses the `node_request_probabilities` variable to generate a source and a destination.

        :return: source node, source node id, destination node, destination node id
        """
        src = self.rng.choices([x for x in self.topology.nodes()], weights=self.node_request_probabilities)[0]
        src_id = self.topology.graph['node_indices'].index(src)
        new_node_probabilities = np.copy(self.node_request_probabilities)
        new_node_probabilities[src_id] = 0.
        new_node_probabilities = new_node_probabilities / np.sum(new_node_probabilities)
        dst = self.rng.choices([x for x in self.topology.nodes()], weights=new_node_probabilities)[0]
        dst_id = self.topology.graph['node_indices'].index(dst)
        return src, src_id, dst, dst_id


    """
    what is actually in the topology object? We need to know what information the agent is getting...
    """

    def observation(self):
        """ this method defines the state representation for the RL agent
        :return: topology and service """


        return {'topology': self.topology,
                'service': self.service}


    def reward(self):
        """ method defines the reward for RL agent
        :return: acceptance status
        """
        return 1 if self.service.accepted else 0

    def reset(self):

        # this method resets the network state to initial state
        self._events = []
        self.current_time = 0
        self.services_processed = 0
        self.services_accepted = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        self.topology.graph['available_spectrum'] = np.full((self.topology.number_of_edges()),
                                                            fill_value=self.num_spectrum_resources,
                                                            dtype=int)
        self.topology.graph["services"] = []
        self.topology.graph["service_wavelengths"] = []
        self.topology.graph["running_services"] = []
        self.topology.graph["running_service_wavelengths"] = []

        self.topology.graph["last_update"] = 0.
        for idx, lnk in enumerate(self.topology.edges()):
            self.topology[lnk[0]][lnk[1]]['utilization'] = 0.
            self.topology[lnk[0]][lnk[1]]['last_update'] = 0.
            self.topology[lnk[0]][lnk[1]]['services'] = []
            self.topology[lnk[0]][lnk[1]]['service_wavelengths'] = []
            self.topology[lnk[0]][lnk[1]]['running_services'] = []
            self.topology[lnk[0]][lnk[1]]['running_service_wavelengths'] = []


    def seed(self, seed=None):
        # this method assigns a seed for the run or generates a random seed if not provided as input
        if seed is not None:
            self.rand_seed = seed
        else:
            self.rand_seed = 41
        self.rng = random.Random(self.rand_seed)
