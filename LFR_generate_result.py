# -*- coding: utf-8 -*-
"""

This module test the results on LFR networks

Example:
    Execute the code to test on LFR network::

        $ python LFR.py

Research article is available at:
   http://...

"""

import networkx as nx
import os
import time
import pickle
from networkx.algorithms.community.modularity_max import * 
from networkx.algorithms.community import LFR_benchmark_graph
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from heatmap import hist
import numpy as np

from pylouvain import PyLouvain
from run import multiscale


def LFR(n, tau1, tau2, mu, min_com_size, force = False):
    # enforce regeneration if force==True  
    path = "data/LFR/LFR_%d_%.2f_%.2f_%.2f.gpickle" %(n, tau1, tau2, mu)
    if not force and os.path.isfile(path): 
        G = nx.read_gpickle(path)
        return G
    else:
        G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree = 8, min_community = min_com_size, seed=0)
        print("write gpickle file", path)
        nx.write_gpickle(G, path)
        return G

if __name__ == "__main__":

    verbose = True 

    #=========== Global Parameters ===========#
    _network_size = 5000
    _min_com_size = 5   

    #=========== Generate Graph ==============#

    G = LFR(n = _network_size, tau1 = 3.0, tau2 = 1.5, mu = 0.25, min_com_size = _min_com_size, force = True)

    #G = LFR(n = 100, tau1 = 2.5, tau2 = 1.2, mu = 0.15)
    print(nx.info(G))

    print("get ground truth")
    gnc = {frozenset(G.nodes[v]['community']) for v in G}
    map_comm = {v:i for i, c in enumerate(gnc) for v in c}
    gnc_list = [map_comm[k] for k in G.nodes()]

    sizes = [len(i) for i in gnc]
    gnc_sizes = sorted(sizes)
    verbose and print("ground truth community sizes=", gnc_sizes)

    nodes = list(G.nodes())
    edges = [((a, b), 1) for a, b in G.edges()] 

    #=========== Benchmark Fast Greedy ===============#
    print("Start Fast Greedy community detection")

    #start = time.time()
    #commsFG = greedy_modularity_communities(G)
    #end = time.time()

    #commsFG_sizes = sorted([len(commsFG[i]) for i in range(len(commsFG))])
    #verbose and print(commsFG_sizes)

    #map_comm = {v:i for i, c in enumerate(commsFG) for v in c}
    #a = [map_comm[k] for k in G.nodes()]
    #print("FastGreedy Algorithm ARI=", adjusted_rand_score(a, b), "NMI=", normalized_mutual_info_score(a, b))
    #print("which takes", end - start, "seconds")

    #=========== Benchmark Louvain ===============#
    '''
    print("Start Louvain community detection")

    #x, y1, y2, y3, y4 = [], [], [], [], []
    for gamma in np.linspace(0.5, 6.0, num=12):
        start = time.time()
        pyl = PyLouvain(nodes, edges)
        commsLV, q = pyl.apply_method(gamma)
        end = time.time()

        map_comm = {v:i for i, c in enumerate(commsLV) for v in c}
        LV_list = [map_comm[k] for k in G.nodes()]
        print(gamma)
        print("Louvain Algorithm ARI=", adjusted_rand_score(LV_list, gnc_list), "NMI=", normalized_mutual_info_score(LV_list, gnc_list))
        print("which takes", end - start, "seconds")
        print()

        #x.append(gamma)
        #y1.append(adjusted_rand_score(LV_list, gnc_list))
        #y2.append(normalized_mutual_info_score(LV_list, gnc_list))
        #y3.append(end - start)

    #pickle.dump({'x':x,'y1':y1,'y2':y2,'y3':y3}, open("tmp/save_louvain_generalized_modularity.p", "wb" ) )
    '''

    #=========== Multi-scale Community Detection ===============#

    print("Start Multi-scale Community Detection")

    x, y1, y2, y3, y4 = [], [], [], [], []
    for gamma in np.linspace(0.4, 0.9, num=11):

        print(gamma)

        start = time.time()
        commsMS = multiscale(nodes, edges, gamma, verbose = False)
        end = time.time()

        commsMS_sizes = sorted([len(commsMS[i]) for i in range(len(commsMS))])
        verbose and print(commsMS_sizes)
        verbose and print(len(commsMS_sizes))

        map_comm = {v:i for i, c in enumerate(commsMS) for v in c}
        MS_list = [map_comm[k] for k in G.nodes()]
        print("Multi-scale Algorithm ARI=", adjusted_rand_score(MS_list, gnc_list), "NMI=", normalized_mutual_info_score(MS_list, gnc_list))
        print("which takes", end - start, "seconds")
        print()

        x.append(gamma)
        y1.append(adjusted_rand_score(MS_list, gnc_list))
        y2.append(normalized_mutual_info_score(MS_list, gnc_list))
        y3.append(end - start)
   
    pickle.dump({'x':x,'y1':y1,'y2':y2,'y3':y3}, open("fig/save_louvain_generalized_modularity.p", "wb" ) )
    exit(0)


    #============ Plot community sizes ==============#

    #print("Plot histogram of community sizes")
    #sizes_distri = {"Ground Truth": gnc_sizes, "Modularity": commsFG_sizes, "Multiscale": commsMS_sizes}

    #pickle.dump(sizes_distri, open('save%d.p' % _network_size, 'wb'))
    #hist(sizes_distri, _network_size)
