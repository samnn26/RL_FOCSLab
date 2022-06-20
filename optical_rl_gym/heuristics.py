from typing import Sequence
import gym
import heapq
import logging
import numpy as np
import GN_model
from optical_rl_gym.utils import Service, Path, LightPath
from .envs.optical_network_env import OpticalNetworkEnv
from .envs.rwa_env_focs_v2 import RWAEnvFOCSV2

# def kSP_FF(env) -> Sequence[int]:
#     # best_hops = np.finfo(0.0).max  # in this case, shortest means least hops
#     best_length = np.inf
#     decision = (env.k_paths+100, env.num_spectrum_resources+100)  # stores current decision, initilized as "reject"
#     for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
#
#         if path.length < best_length:  # if path is shorter
#             # checks all wavelengths
#             for wavelength in range(env.num_spectrum_resources):
#                 if env.is_lightpath_free(path, wavelength) and env.get_available_lightpath_capacity(path,
#                 wavelength) > env.service.bit_rate:  # if new viable lightpath is found
#
#                     # stores decision and breaks the wavelength loop (first fit)
#                     best_length = path.length
#                     decision = (idp, wavelength)
#                     break
#                 elif env.does_lightpath_exist(path,wavelength) and env.get_available_lightpath_capacity(path,
#                 wavelength) > env.service.bit_rate: # viable lightpath exists
#                     # stores decision and breaks the wavelength loop (first fit)
#                     best_length = path.length
#                     decision = (idp, wavelength)
#                     break
#     return decision

def kSP_FF(env) -> Sequence[int]:

    #print("source, dest ", env.service.source, env.service.destination)
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):

        #if path.length < best_length:  # if path is shorter
            # checks all wavelengths
        for wavelength in range(env.num_spectrum_resources):
            if env.is_lightpath_free(path, wavelength) and env.get_available_lightpath_capacity(path,
            wavelength) > env.service.bit_rate:  # if new viable lightpath is found
                    # stores decision and breaks the wavelength loop (first fit)
                    #best_length = path.length
                    decision = (idp, wavelength)
                    # print("get_available_lightpath_capacity:", env.get_available_lightpath_capacity(path,wavelength)/1e9)
                    # print("calculate_lightpath_capacity:", GN_model.calculate_lightpath_capacity(path.length, wavelength))
                    # print("length", path.length)
                    # # breakpoint()
                    # print("new lightpath")
                    # print("decision", decision)
                    # print(env.topology.graph['ksp'][env.service.destination, env.service.source][idp].node_list)
                    return decision
            elif env.does_lightpath_exist(path,wavelength) and env.get_available_lightpath_capacity(path,
            wavelength) > env.service.bit_rate: # viable lightpath exists
                    # stores decision and breaks the wavelength loop (first fit)
                    #best_length = path.length
                    decision = (idp, wavelength)
                    # print("reused lightpath")
                    # print(decision)
                    # print(env.topology.graph['ksp'][env.service.destination, env.service.source][idp].node_list)

                    return decision

    return decision



# def FF_kSP(env: RWAEnvFOCSV2) -> Sequence[int]:
#     # best_hops = np.finfo(0.0).max  # in this case, shortest means least hops
#     best_length = np.inf
#     decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
#     found_path = False
#     for wavelength in range(env.num_spectrum_resources):
#         if found_path:
#             return decision
#         for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
#             if path.length < best_length:  # if path is shorter
#                 breakpoint()
#                 if env.is_lightpath_free(path, wavelength) and env.get_available_lightpath_capacity(path,
#                 wavelength) > env.service.bit_rate:  # if new viable lightpath is found
#                     # stores decision and breaks the path loop (first fit)
#                     best_length = path.length
#                     decision = (idp, wavelength)
#                     found_path = True
#                 elif env.does_lightpath_exist(path,wavelength) and env.get_available_lightpath_capacity(path,
#                 wavelength) > env.service.bit_rate: # viable lightpath exists
#                     # stores decision and breaks the wavelength loop (first fit)
#                     best_length = path.length
#                     decision = (idp, wavelength)
#                     found_path = True
#     return decision

def FF_kSP(env) -> Sequence[int]:
    # best_hops = np.finfo(0.0).max  # in this case, shortest means least hops

    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    # found_path = False
    for wavelength in range(env.num_spectrum_resources):
        # if found_path:
        #     return decision
        for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
            # breakpoint()
            if env.is_lightpath_free(path, wavelength) and env.get_available_lightpath_capacity(path,
            wavelength) > env.service.bit_rate:  # if new viable lightpath is found
                # stores decision and breaks the path loop (first fit)
                decision = (idp, wavelength)
                return decision
                # found_path = True
                # break
            elif env.does_lightpath_exist(path,wavelength) and env.get_available_lightpath_capacity(path,
            wavelength) > env.service.bit_rate: # viable lightpath exists
                # stores decision and breaks the wavelength loop (first fit)
                best_length = path.length
                decision = (idp, wavelength)
                return decision
                # found_path = True
                # break
    return decision


def kSP_MU(env) -> Sequence[int]:

    mu_wavelength = -1
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        for wavelength in range(env.num_spectrum_resources): # checks all wavelengths

            if env.is_lightpath_free(path, wavelength) and env.get_available_lightpath_capacity(path,
            wavelength) > env.service.bit_rate:  # if new viable lightpath is found
                # get usage of wavelength across the whole network (i.e. for all path IDs)
                if np.sum(env.lightpath_service_allocation[:, wavelength]) > mu_wavelength:

                    mu_wavelength = wavelength
                    decision = (idp, wavelength)

            elif env.does_lightpath_exist(path,wavelength) and env.get_available_lightpath_capacity(path,
            wavelength) > env.service.bit_rate: # viable lightpath exists

                if np.sum(env.lightpath_service_allocation[:, wavelength]) > mu_wavelength:

                    mu_wavelength = wavelength
                    decision = (idp, wavelength)

    return decision

# def kSP_MU(env: RWAEnvFOCSV2) -> Sequence[int]:
#     # best_hops = np.finfo(0.0).max  # in this case, shortest means least hops
#     best_length = np.inf
#     mu_wavelength = -1
#     decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
#     for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
#
#         if path.length < best_length:  # if path is shorter
#             # checks all wavelengths
#             for wavelength in range(env.num_spectrum_resources):
#
#                 if env.is_lightpath_free(path, wavelength) and env.get_available_lightpath_capacity(path,
#                 wavelength) > env.service.bit_rate:  # if new viable lightpath is found
#                     # get usage of wavelength across the whole network (i.e. for all path IDs)
#                     if np.sum(env.lightpath_service_allocation[:, wavelength]) > mu_wavelength:
#                         best_length = path.length
#                         mu_wavelength = wavelength
#                         decision = (idp, wavelength)
#
#                 elif env.does_lightpath_exist(path,wavelength) and env.get_available_lightpath_capacity(path,
#                 wavelength) > env.service.bit_rate: # viable lightpath exists
#
#                     if np.sum(env.lightpath_service_allocation[:, wavelength]) > mu_wavelength:
#                         best_length = path.length
#                         mu_wavelength = wavelength
#                         decision = (idp, wavelength)
#
#     return decision

def CA_MU(env: RWAEnvFOCSV2) -> Sequence[int]:
    # best_hops = np.finfo(0.0).max  # in this case, shortest means least hops
    best_weight = -1
    mu_wavelength = -1
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        weight = 0
        for i in range(len(path.node_list) - 1):
            frac_unoccupied_wavelengths = (env.num_spectrum_resources - np.count_nonzero(env.topology.graph['available_wavelengths'][
                      env.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                      :])) / env.num_spectrum_resources
            weight += frac_unoccupied_wavelengths/path.length # currently not multiplying by capacity of path - need to speak to Nikita about how this is being modelled
        if weight > best_weight:  # find the wavelength for a given path with the best weight
            # checks all wavelengths
            for wavelength in range(env.num_spectrum_resources):

                if env.is_lightpath_free(path, wavelength) and env.get_available_lightpath_capacity(path,
                wavelength) > env.service.bit_rate:  # if new viable lightpath is found

                    if np.sum(env.lightpath_service_allocation[:, wavelength]) > mu_wavelength:
                        best_weight = weight # only assign the best weight for viable lightpaths
                        mu_wavelength = wavelength
                        decision = (idp, wavelength)

                elif env.does_lightpath_exist(path,wavelength) and env.get_available_lightpath_capacity(path,
                wavelength) > env.service.bit_rate: # viable lightpath exists

                    if np.sum(env.lightpath_service_allocation[:, wavelength]) > mu_wavelength:
                        best_weight = weight
                        mu_wavelength = wavelength
                        decision = (idp, wavelength)

    return decision


def kSP_FF_binary(env) -> Sequence[int]:

    #print("source, dest ", env.service.source, env.service.destination)
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):

        #if path.length < best_length:  # if path is shorter
            # checks all wavelengths
        for wavelength in range(env.num_spectrum_resources):
            if env.is_lightpath_free(path, wavelength):  # if new viable lightpath is found
                    # stores decision and breaks the wavelength loop (first fit)
                    #best_length = path.length
                    decision = (idp, wavelength)
                    # print("get_available_lightpath_capacity:", env.get_available_lightpath_capacity(path,wavelength)/1e9)
                    # print("calculate_lightpath_capacity:", GN_model.calculate_lightpath_capacity(path.length, wavelength))
                    # print("length", path.length)
                    # # breakpoint()
                    # print("new lightpath")
                    # print("decision", decision)
                    # print(env.topology.graph['ksp'][env.service.destination, env.service.source][idp].node_list)
                    return decision
    return decision


def FF_kSP_binary(env) -> Sequence[int]:
    # best_hops = np.finfo(0.0).max  # in this case, shortest means least hops

    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    # found_path = False
    for wavelength in range(env.num_spectrum_resources):
        # if found_path:
        #     return decision
        for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
            # breakpoint()
            if env.is_lightpath_free(path, wavelength):  # if new viable lightpath is found
                # stores decision and breaks the path loop (first fit)
                decision = (idp, wavelength)
                return decision

    return decision


def kSP_MU_binary(env) -> Sequence[int]:

    mu_wavelength = -1
    decision = (env.k_paths, env.num_spectrum_resources)  # stores current decision, initilized as "reject"
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        for wavelength in range(env.num_spectrum_resources): # checks all wavelengths

            if env.is_lightpath_free(path, wavelength):  # if new viable lightpath is found
                # get usage of wavelength across the whole network (i.e. for all path IDs)
                if np.sum(env.lightpath_service_allocation[:, wavelength]) > mu_wavelength:

                    mu_wavelength = wavelength
                    decision = (idp, wavelength)

    return decision

def CA_MU_TEST(env) -> Sequence[int]:
    #init the varialbes for CA, MU and decision
    ca_highest_weight = -1
    mu_highest_weight = -1
    mu_wavelength = -1
    ca_path = -1
    decision = (env.k_paths, env.num_spectrum_resources)

    #first loop through the paths
    for path_id in range(env.k_paths):
        path = env.k_shortest_paths[env.service.source,env.service.destination][path_id]
        path_weight = 0
        #loop through the links in each path
        for node_id in range(len(path.node_list) - 1):
            link_weight = 0
            #calculate the propotion of unused wavelengths for  a link
            wavelen_on_link = np.count_nonzero(env.topology.graph['available_wavelengths'][
                      env.topology[path.node_list[node_id]][path.node_list[node_id + 1]]['index'],
                      :])
            unused_wavelen_on_link = env.num_spectrum_resources - wavelen_on_link
            propotion_of_unused_wavelen = unused_wavelen_on_link/env.num_spectrum_resources

            #calculate the link weight for the link using propotion of usused wavelengths / link length
            link_len = env.topology[path.node_list[node_id]][path.node_list[node_id + 1]]['weight']
            link_weight = propotion_of_unused_wavelen/link_len
            #add link weight to path weight
            path_weight = path_weight + link_weight
        #end loop links
        # if path weight > CA highest weight then update the highest CA weight variable
        if path_weight > ca_highest_weight:
            ca_highest_weight = path_weight
            ca_path = path_id
            print(" least congested path so far is ", ca_path, " and the congestion weight is ", ca_highest_weight)
            #loop through wavelenths
            for wavelength in range(env.num_spectrum_resources):
                
                #if path is already used and leftover capacity avaiilable then ccalculate wavelen_weight
                if env.does_lightpath_exist(path,wavelength) and env.get_available_lightpath_capacity(path,
                wavelength) > env.service.bit_rate:
                    wavelen_weight = np.sum(env.lightpath_service_allocation[:,wavelength])

                    # if wavelen_weight is greater than mu_weight update mu_weight, update mu_wavelenth
                    if wavelen_weight > mu_highest_weight:
                        mu_wavelength = wavelength
                        mu_highest_weight = wavelen_weight
                        print("wavelen weight ", wavelen_weight)
                        decision = (ca_path, mu_wavelength)
                    
                    
                #else if w#if path is free and capacity avaiilable then ccalculate wavelen_weight
                elif env.is_lightpath_free(path, wavelength) and env.get_available_lightpath_capacity(path, wavelength) > env.service.bit_rate:
                    wavelen_weight = np.sum(env.lightpath_service_allocation[:,wavelength])

                    # if wavelen_weight is greater than mu_weight update mu_weight, update mu_wavelenth
                    if wavelen_weight > mu_highest_weight:
                        mu_wavelength = wavelength
                        mu_highest_weight = wavelen_weight
                        print("wavelen weight ", wavelen_weight)
                        decision = (ca_path, mu_wavelength)

               



        #end loop wavelengths
    #end loop paths
    return decision
