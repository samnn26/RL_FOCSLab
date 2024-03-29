"""
This script generates a binary file containing a NetworkX graph with all the initializations necessary to work with Optical RL-Gym.
It facilitates when it comes to training a RL agent.
"""

import optical_rl_gym
from optical_rl_gym.utils import Path

from itertools import islice
import numpy as np
import networkx as nx
import math
import pickle
from xml.dom.minidom import parse
import xml.dom.minidom

import matplotlib.pyplot as plt

from graph_utils import read_sndlib_topology, read_txt_file, get_k_shortest_paths, get_path_weight


def get_topology(file_name, topology_name, k_paths=2):
    k_shortest_paths = {}
    if file_name.endswith('.xml'):
        topology = read_sndlib_topology(file_name)
    elif file_name.endswith('.txt'):
        topology = read_txt_file(file_name)
    else:
        raise ValueError('Supplied topology is unknown')
    idp = 0
    # edge_counter = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            # print(n1)
            # print(n2)
            # if idn1 != idn2:
            if idn1 < idn2:
                paths = get_k_shortest_paths(topology, n1, n2, k_paths, weight='length')
                weights = [get_path_weight(topology, path, weight='length') for path in paths]
                lengths = [get_path_weight(topology, path, weight='length') for path in paths]
                objs = []
                for path, weight,length in zip(paths, weights, lengths):
                    objs.append(Path(idp, path,weight,length))
                    print(idp, weight,length, path)
                    idp += 1
                k_shortest_paths[n1, n2] = objs
                # edge_counter += 1
                k_shortest_paths[n2, n1] = objs
    topology.graph['name'] = topology_name
    topology.graph['ksp'] = k_shortest_paths
    topology.graph['k_paths'] = k_paths
    topology.graph['node_indices'] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph['node_indices'].append(node)
        topology.nodes[node]['index'] = idx
    # print(edge_counter)
    return topology


k_paths = 1
# base_topology_name = 'nsfnet_chen.txt'
# base_topology_name = 'nsfnet_chen_rounded.txt'
# base_topology_name = 'nsfnet_chen_directional.txt'
# base_topology_name = '3_node_network_asym.txt'
# base_topology_name = '3_node_network_asym.txt'
base_topology_name = 'dtag.txt'
# base_topology_name = 'gb.txt'
# new_topology_name = 'nsfnet_chen_'+str(k_paths)+'-paths.h5'
# new_topology_name = 'nsfnet_chen_'+str(k_paths)+'-paths_rounded.h5'
# new_topology_name = 'nsfnet_chen_'+str(k_paths)+'-paths_directional.h5'
# new_topology_name = '3_node_network_asym.h5'
# new_topology_name = '3_node_network_sym.h5'
new_topology_name = 'dtag_'+str(k_paths)+'-paths.h5'
# new_topology_name = 'gb_'+str(k_paths)+'-paths.h5'

topology = get_topology('./topologies/'+base_topology_name, 'DTAG', k_paths=k_paths)

with open(f'./topologies/'+new_topology_name, 'wb') as f:
    pickle.dump(topology, f)
print('done for', topology)
