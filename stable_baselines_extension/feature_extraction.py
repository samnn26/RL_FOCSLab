from __future__ import print_function

from itertools import zip_longest
from typing import List

import networkx as nx
import numpy as np
import tensorflow as tf
from graph_nets.graphs import GraphsTuple
from stable_baselines.common.tf_layers import linear
from stable_baselines.common.tf_layers import ortho_init, _ln
from stable_baselines_extension.graph_networks import GraphNetworkCustom
from stable_baselines_extension.graph_networks import EncodeProcessDecode
import sonnet as snt
import graph_nets
import sys



def vf_builder(vf_arch: str, latent: tf.Tensor,
               act_fun: tf.function, shared_graph: GraphsTuple = None,
               input_graph: GraphsTuple = None, layer_size: int = 64,
               layer_count: int = 3,
               iterations: int = 10) -> tf.Tensor:
    """
    Builds the value function network for
    Args:
        vf_arch: arch to use as a string
        latent: the observation input
        act_fun: activation function
        shared_graph: the gnn output from the policy
        input_graph: GraphTuple before any processing
        iterations: number of iterations of message passing
    Returns:
        A tensor which will hold the value
    """
    if vf_arch == "shared":
        output_globals_vf = tf.reshape(shared_graph.globals, [-1, layer_size])
        latent_vf = output_globals_vf
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc0", 128, init_scale=np.sqrt(2)))
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc1", 128, init_scale=np.sqrt(2)))
    elif vf_arch == "graph":
        model_vf = GraphNetworkCustom(layer_size=layer_size)
        output_graph_vf = model_vf(input_graph, iterations)
        output_globals_vf = tf.reshape(output_graph_vf.globals,
                                       [-1, layer_size])
        latent_vf = output_globals_vf
    elif vf_arch == "mlp":
        latent_vf = latent
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc0", 128, init_scale=np.sqrt(2)))
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc1", 128, init_scale=np.sqrt(2)))
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc2", 128, init_scale=np.sqrt(2)))
    else:
        raise Exception("No such vf network")

    return latent_vf


def mlp_extractor(flat_observations, net_arch, act_fun,network_graphs,iterations: int = 5, layer_size: int = 128,
                  layer_count: int = 3,
                  vf_arch: str = "mlp"):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value



def gnn_extractor(flat_observations: tf.Tensor, act_fun: tf.function,
                  network_graphs: List[nx.DiGraph],
                  iterations: int = 5, layer_size: int = 128,
                  layer_count: int = 3,
                  vf_arch: str = "mlp"):
    """
    Constructs a graph network from the graph passed in. Then inputs are
    traffic demands, placed on nodes as feature vectors. The output policy
    tensor is built from the edge outputs (in line with the k shortest path routing
    approach). The value function can be switched between mlp and graph net
    using the net_arch argument.

    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the
    specified network. If all layers are shared, then ``latent_policy ==
    latent_value``

    self.action_space = gym.spaces.Discrete(self.k_paths * self.j + self.reject_action)
     shape = 1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 3) * self.k_paths
    self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))

    """

    # get graph info
    sorted_edges_list = [sorted(network_graph.edges()) for network_graph in
                         network_graphs]
    num_edges_list = [len(l) for l in sorted_edges_list]
    num_nodes_list = [network_graph.number_of_nodes() for network_graph in
                      network_graphs]
    max_edges_len = max(num_edges_list)
    sorted_edges_list = [edges + [(0, 0)] * (max_edges_len - len(edges)) for
                         edges in sorted_edges_list]
    sorted_edges = tf.constant(sorted_edges_list)
    num_edges = tf.constant(num_edges_list)
    num_nodes = tf.constant(num_nodes_list)

    latent = flat_observations

    # prepare sender and reciever nodes for edges for GraphTuple creation
    graph_idxs = tf.cast(latent[:, 0], np.int32)
    sender_nodes = tf.map_fn(lambda i: sorted_edges[i][:, 0], graph_idxs)
    sender_nodes = tf.RaggedTensor.from_tensor(sender_nodes,
                                               lengths=num_edges)
    sender_nodes = sender_nodes.flat_values
    receiver_nodes = tf.map_fn(lambda i: sorted_edges[i][:, 1], graph_idxs)
    receiver_nodes = tf.RaggedTensor.from_tensor(receiver_nodes,
                                                 lengths=num_edges)
    receiver_nodes = receiver_nodes.flat_values

    #todo: fix extracting node, edge and global features from flat_observations
    node_features = tf.zeros((1, layer_size), np.float32)
    edge_features = tf.zeros((1, layer_size), np.float32)
    global_features = tf.zeros((1, layer_size), np.float32)

    #create graph tuple
    input_graph = GraphsTuple(nodes=node_features,
                              edges=edge_features,
                              globals=global_features,
                              senders=sender_nodes,
                              receivers=receiver_nodes,
                              n_node=num_nodes,
                              n_edge=num_edges
                             )

    # Create the graph network.
    model = graph_nets.modules.GraphNetwork(
        edge_model_fn=lambda: snt.nets.MLP([64, 64]),
        node_model_fn=lambda: snt.nets.MLP([64, 64]),
        global_model_fn=lambda: snt.nets.MLP([64, 64]))

    #generate output graph
    output_graph = model(input_graph)


    #todo:double check extracting output edges and globals from output graph
    output_edges = tf.RaggedTensor.from_row_lengths(output_graph.edges, #k paths*J
                                                    num_edges)
    output_globals = tf.reshape(output_graph.globals, (-1, layer_size))  # global output is  reject action


    #create latent policy gnn from outputs
    latent_policy_gnn = tf.concat([output_edges, output_globals], axis=1)
    # build value function network
    latent_vf = vf_builder(vf_arch, flat_observations, act_fun,
                           output_graph, input_graph, layer_size, layer_count,
                           iterations)

    return latent_policy_gnn, latent_vf

