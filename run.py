#!/usr/bin/env python3

import math
from collections import defaultdict
import numpy as np
import scipy
from scipy.stats import entropy
from pylouvain import PyLouvain, in_order

def bayes_model_selection(nodes, edges, partition):
    N, E = len(nodes), len(edges)
    comm = {n:i for i, ns in enumerate(partition) for n in ns}
    k_r, k_r_in = defaultdict(int), defaultdict(int)
    for e in edges:
       u, v = e[0][0], e[0][1]
       k_r[comm[u]] += 1
       k_r[comm[v]] += 1
       if comm[u] == comm[v]:
          k_r_in[comm[u]] += 2 
    a = sum(k_r_in.values())
    b = sum([v * v for v in k_r.values()]) / (2. * E)
    c, d = 2. * E - a, 2. * E - b
    assert a >= 0
    assert b >= 0
    assert c >= 0
    assert d >= 0

    res = 0.
    if a > 0: 
        #res += a * (np.log(a) - np.log(b))
        res += a * np.log(a) - (a + 1) * np.log(b + 1)
    if c > 0:
        #res += c * (np.log(c) - np.log(d))
        res += c * np.log(c) - (c + 1) * np.log(d + 1)

    res += (2. * E + 1) * np.log(2. * E + 1) - 2. * E * np.log(2. * E)

    # entropy of len(c) nodes in each blocks (and sum(c) = N)
    ent1 = scipy.stats.entropy([float(len(c))/N for c in partition])

    # entropy of B blocks (B in range [1, N])
    ent2 = scipy.stats.entropy([len(partition)/N, 1.-len(partition)/N])
    
    return res / N - (ent1 + ent2)

def multiscale(nodes, edges, gamma, deg = None, depth = 1, verbose = False, max_depth = 4):
    '''
    Multi-scale community detection.
    Recursively split sub-graph by maximizing generalized modularity.
    Terminate at each level of recursion by bayes model selection.

    Args:
        nodes: a list of nodes
        edges: a list of edges ((src, dst), weight)
        gamma: the resolution parameter for the generalized modularity
    Return:
        a list of lists, each contains the nodes in a community
    '''

    if depth >= max_depth or len(nodes) < 2:
        return [nodes]

    verbose and print("    " * depth, "***", "depth=", depth, "N=", len(nodes))

    nodes.sort()
    d = {n:i for i, n in enumerate(nodes)}
    rd = {i:n for n, i in d.items()}
    nodes = list(range(len(d))) 
    edges = [((d[e[0][0]], d[e[0][1]]), e[1]) for e in edges]
    
    if deg is None:
        deg = {i:0 for i in nodes}
        for e in edges:
            deg[e[0][0]] += e[1]
            deg[e[0][1]] += e[1]

    pyl = PyLouvain(nodes, edges, deg)

    # execution
    partition, q = pyl.apply_method(gamma)
    verbose and print("    " * depth, "gamma=", gamma, "comm=", len(partition))

    if len(partition) < 2: return [list(map(rd.get, nodes))]
    odds = bayes_model_selection(pyl.nodes, pyl.edges, partition)
    verbose and print("    " * depth, "odds=", odds)
    if odds <= 1. or math.isnan(odds): return [list(map(rd.get, nodes))]

    comm = {n:i for i, ns in enumerate(partition) for n in ns}
    edge_list = [[] for _ in range(len(partition))]
    for e in edges:
        u, v = e[0][0], e[0][1] 
        if comm[u] == comm[v]: 
            edge_list[comm[u]].append(e)

    R = []
    for nodes_, edges_ in zip(partition, edge_list):
        groups = multiscale(nodes_, edges_, gamma, deg, depth + 1, verbose, max_depth)
        for grp in groups :
            R.append([rd[n] for n in grp])

    return R